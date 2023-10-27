# import sys
# sys.path.append('/home/mike/anaconda3/bin/')
import serial
import time
import os
from scipy.stats import linregress
from scipy.signal import savgol_filter, argrelextrema
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np 
from datetime import datetime
import lmfit
from humanfriendly import format_timespan
import matplotlib.pyplot as plt
from XRR.plotly_layouts import plotly_layouts as layouts
from XRR.utilities.file_operations import *
from XRR.helpers import *
import plotly.graph_objs as go
layouts = layouts()

def temp(v):
	
   m, b = 49.99991636019922, -0.1450371167625235
   return m * v +  b

def pressure(v, offset = 0):
	m = (700-offset) / (10)
	b = offset
	return m * v + b

def record(savepath = None, samplename = None, sleeptime = 1, verbose = True, pressure_offset = 0, file_ext = '.csv'):
	'''
	Record temperature and pressure during experiment.
	---------
	Parameters:
	savepath: Path, where the data file (date, voltage_temp, temp, voltage_press, press) should be saved. If not given, the current working directory is used.
	samplename: Name of the data file. No fileending needed.
	sleeptime: float, time in seconds to wait for next record. Default is 1
	verbose: If true, the recorded data is printed in the terminal.
	pressure_offset: float, offset of pressure, e. g. ambient = 0 bar (offset = 0) or ambient = 1 bar (offset = 1))
	----------	
	Returns:
	txt-file with data: date, voltage, temperature.
	'''


	# Variables for time

	now = time.strftime("%Y%m%d%H%M%S",time.localtime())
	if not savepath:
		savepath = os.getcwd()
		print(f'{savepath} is used as savepath')
	if not os.path.isdir(savepath):
		os.mkdir(savepath)
		print(f'Created {savepath}.')
	# Create filename
	if samplename == None:
		filename = os.path.join(savepath, now + file_ext)
	else:
		filename = os.path.join(savepath, samplename + file_ext)

	# Read data
	counter = 0
	ser=serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
	while True:
		now = time.strftime("%Y%m%d%H%M%S",time.localtime())
		nowToPrint = time.strftime("%H:%M:%S", time.localtime())
		# read out channel #0 (temperature measurement)
		ser.write(b'#010\r')
		data=b''
		while not (b'\r' in data):
			data += ser.read(1)
		voltage_temperature_float = float(data.decode().strip('>+'))
		tempFloat = temp(voltage_temperature_float)
		
		# read out channel #1 (pressure measurement)
		ser.write(b'#011\r')
		data = b''
		while not (b'\r' in data):
			data += ser.read(1)
		voltage_pressure_float = float(data.decode().strip('>+'))
		pressFloat = pressure(voltage_pressure_float, offset=pressure_offset)

		if counter == 0:
			if not os.path.isfile(filename):
				with open(filename, 'a+') as f:
					f.write('time [YYYYMMDDHHMMSS], ' +  'voltage_temp [V], ' + 'temperature [°C], ' + 'voltage_pressure [V], ' + 'pressure [bar]' + '\n')
				counter += 1
		if counter == 1:
			with open(filename, 'a+') as f:
				f.write(f'{now}, {voltage_temperature_float:5.3f}, {tempFloat:6.3f}, {voltage_pressure_float:5.3f}, {pressFloat:6.3f}\n')
			counter = 0
		if counter == 0 and os.path.isfile(filename):
			with open(filename, 'a+') as f:
				f.write(f'{now}, {voltage_temperature_float:5.3f}, {tempFloat:6.3f}, {voltage_pressure_float:5.3f}, {pressFloat:6.3f}\n')
		if verbose:
			print(f'time: {nowToPrint}, voltage(t): {voltage_temperature_float:.3f} V, temperature: {tempFloat:6.3f} °C, voltage(p): {voltage_pressure_float:.3f} V, pressure: {pressFloat:6.3f} bar' )
		time.sleep(sleeptime)


def temperature_to_dataframe(tempfile, concat_files = False, fileext = '.txt'):
	'''
	Load data recorded with recordTemp ("time, volts, temperature") in pandas.Dataframe.
	Parameters: 
	-----------
	tempfile: full path of file recordes with recordTemp.
	concat_files:If True and multiple files were created while recording the temperature, all files in the path of tempfile ending with fileext will be used. The data is concatenated.
	'''
	if concat_files:
		tempData_single = []
		filepath = os.path.dirname(tempfile)
		tempfiles = [os.path.join(filepath, t) for t in os.listdir(filepath) if t.endswith(fileext)]
		for t in tempfiles:
			anz_cols = read_lines_file(t, onlyFirstLine=True).split()
			tempData_single.append(pd.read_csv(t, header = 1, names = ['date', 'volts', 'temp']))
		tempData = pd.concat(tempData_single, ignore_index = True)
	else:
		colnames = [e.strip(' ').strip('\n') for e in read_lines_file(tempfile, onlyFirstLine=True).split(',')]
		tempData =pd.read_csv(tempfile, header = 1, names = colnames)	
	tempData[colnames[0]] = pd.to_datetime(tempData[colnames[0]], format="%Y%m%d%H%M%S")
	starttime = tempData[colnames[0]][0]
	duration = tempData[colnames[0]] - starttime
	total_seconds = []
	for i in duration:
		total_seconds.append(i.total_seconds())
	tempData['total_seconds'] = total_seconds	
	return tempData

def dprime_temp_curve(tempData, window_length = 100, polyorder = 4, plot_derivates = False, extrema_order = 300, calc_constant_time = False,
	plotwith_plotly = False):
	# ignore part of heating up by choosing only values where plateau/ target temp is reached:
	# tempData.temp = tempData.temp[tempData.temp.idxmax()-30:]
	diff_t = np.gradient(tempData.temp)
	ddiff_t = np.gradient(diff_t)

	temp_prime = savgol_filter(diff_t, window_length=window_length, polyorder=polyorder)
	temp_dprime = savgol_filter(ddiff_t, window_length=window_length, polyorder=polyorder)
	
	tp_inds = argrelextrema(temp_dprime, np.less, order = extrema_order)[0]
	# turning_points = go.Scatter(x = df.total_seconds[tp_inds] / 60, y = df.temp[tp_inds], mode = 'markers', marker = dict(size = 8, color = 'red',), name = 'turning_points')
	# constant_time = (tempData.total_seconds[tp_inds[1]] -tempData.total_seconds[tp_inds[0]]) /60
	if plot_derivates and plotwith_plotly:
		marker_number = np.arange(0,6,1)
		temp_plot = go.Scattergl(x = tempData.total_seconds / 60, y = tempData.temp, mode = 'lines+markers',
			marker = dict(symbol = marker_number[0]),
			name ='temperature')
		# deriv_plot = go.Scattergl(x = tempData.total_seconds / 60, y = diff_t * 200, mode = 'lines+markers',
		# 	marker = dict(symbol = marker_number[1]), name='prime')
		deriv_savgol_plot = go.Scattergl(x = tempData.total_seconds / 60, y = temp_prime * 50, mode = 'lines', line = dict(width = 3),
			name='prime savgol')
		# dderiv_plot = go.Scatter(x = tempData.total_seconds / 60, y = ddiff_t * 1000, mode = 'lines+markers', 
			# marker = dict(symbol = marker_number[3]),name = 'double prime')
		dderiv_savgol_plot = go.Scattergl(x = tempData.total_seconds / 60, y= temp_dprime * 1000,mode = 'lines', line = dict(width=3),
			name = 'double prime savgol')
		turning_points_plot = go.Scattergl(x = tempData.total_seconds[tp_inds] / 60, y = tempData.temp[tp_inds], mode = 'markers',
			marker = dict(symbol = 4, color = 'red'),name = 'turning_points')
		plot_data = [temp_plot, deriv_savgol_plot, dderiv_savgol_plot, turning_points_plot]
		# [temp_plot, deriv_plot, deriv_savgol_plot, dderiv_plot, dderiv_savgol_plot, turning_points_plot]
		fig = go.Figure(data = plot_data)
		fig.show()
	elif plot_derivates and not plotwith_plotly:
		print('	')
		plt.plot(tempData.total_seconds / 60, tempData.temp, '.-', label='temperature')
		# plt.plot(tempData.total_seconds / 60, diff_t * 200, '-.', label='prime')
		plt.plot(tempData.total_seconds / 60, temp_prime * 1000, '-.', linewidth = 2, label='prime savgol')
		# plt.plot(tempData.total_seconds / 60, ddiff_t * 1000,'-.', label = 'double prime')
		plt.plot(tempData.total_seconds / 60, temp_dprime * 1000,'-.', linewidth = 2, label = 'double prime savgol')
		plt.plot(tempData.total_seconds[tp_inds] / 60, tempData.temp[tp_inds], 'x',label = 'turning_points')
		plt.legend()
		plt.show()		
	if calc_constant_time:
		constant_time = (tempData.total_seconds[tp_inds[1]] -tempData.total_seconds[tp_inds[0]])
		print(f'Temperature kept constant for {format_timespan(constant_time)}.')
	return tp_inds

def cooling_model(t, T_env, T_0, tau):
	return T_env + (T_0 - T_env) * np.exp(-t/tau)

def fit_temperature_curve(tempData, turning_point_indices, fit_model = cooling_model, show_fit_result = True):
	cooling_time = tempData.total_seconds.iloc[turning_point_indices[1]:] / 60
	starttime = cooling_time.iloc[0]
	cooling_time -= starttime
	cooling_temp = tempData.temp.iloc[turning_point_indices[1]:]
    
	cust_model = lmfit.Model(fit_model)
	# opt, cov = curve_fit(fit_model, cooling_time, cooling_temp, p0=[25,344, 50 ])
	cust_model.set_param_hint('T_env', value = 25, min = 10, max = 150)
	cust_model.set_param_hint('T_0', value = cooling_temp.iloc[0], vary = True)
	cust_model.set_param_hint('tau', value = 50, vary = True)	
	params = cust_model.make_params()
	result = cust_model.fit(cooling_temp, params, t = cooling_time)
	# print(result.params['T_env'])

	if show_fit_result:
		fig_fit, ax_fit = plt.subplots()
		fig_fit.set_size_inches(12, 8)
		ax_fit.plot(tempData.total_seconds / 60, tempData.temp, '.-', label='temperature')
		ax_fit.plot((cooling_time + starttime),result.best_fit, '-', label = 'Fit')
		# ax_fit.annotate(r'$T_{\text{env}} = T_env$', xy=(0.5, 0.5)))
		T_env = "(""{:.2f}".format(result.params['T_env'].value) + "\u00B1" + "{:.2f}".format(result.params['T_env'].stderr) + ") °C"
		T_0 = "(""{:.2f}".format(result.params['T_0'].value) + "\u00B1" + "{:.2f}".format(result.params['T_0'].stderr) + ") °C"
		tau = "(""{:.2f}".format(result.params['tau'].value) + "\u00B1" + "{:.2f}".format(result.params['tau'].stderr) + ") s"
		print(f'{T_env}\n{T_0}\n{tau}')
		ax_fit.annotate(r"$T_{\mathrm{env}}$ = " + f"{T_env}\n" + r"$T_{0}$ = " + f"{T_0}\n" + r"$\tau$ = " + f"{tau}", xy=(0.5, 0.2),  xycoords='axes fraction',
			xytext=(0.3, 0.2), textcoords='axes fraction')

		left, bottom, width, height = [0.47, 0.5, 0.4, 0.2]
		ax_res = fig_fit.add_axes([left, bottom, width, height])
		residuals = result.residual
		ax_res.plot((cooling_time + starttime), residuals, 'o', mec='r', mfc='r', label = 'residual')
		ax_res.axhline(y = 0, color = 'k', )
		lines_labels = [ax.get_legend_handles_labels() for ax in fig_fit.axes]
		lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
		fig_fit.legend(lines, labels,)
		plt.show()
	# print(opt, cov)
	return result

def calc_remaining_cooling_time(fit_result, startTemp, targetTemp):
    T_env = fit_result.params['T_env'].value
    T_0 = fit_result.params['T_0'].value
    tau = fit_result.params['tau'].value
    remainingMinutes = tau * np.log((startTemp-T_env)/(targetTemp-T_env))
    print(f'Starting from {startTemp} °C. Remaining time to reach {targetTemp} °C: {format_timespan(remainingMinutes * 60)}')
