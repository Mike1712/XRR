import sys
import os
# import XRR_datatreatment_class as x
# from XRR_datatreatment_class import XRR_datatreatment as xd
from XRR.plotly_layouts import plotly_layouts as layouts
from XRR.helpers import new_listdir
from XRR.utilities.conversion_methods import *
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 5)
from matplotlib import pyplot as plt, patches, animation as animation, cm
from matplotlib.colors import LogNorm, rgb2hex
import plotly.offline as py
import plotly.graph_objs as go
import plotly_express as px
import plotly.io as pio
import scipy as sc
from scipy import constants
from scipy.signal import find_peaks, argrelmin
from collections import defaultdict
from scipy import integrate
import shutil
from glob import glob
import re
from PIL import Image, ImageEnhance, ImageFilter
# import cv2
from itertools import chain
import math
import time
from datetime import datetime
import re
from natsort import natsorted
import h5py
import fabio
from lmfit import Parameters, minimize, report_fit

import pickle

layout = layouts(transpPaper=False, transpPlot = False)

class ID31:

	'''
	Class to evaluate data recorded at ID31 (ESRF, Grenoble). The class contains functions for reading and evaluating data.
	'''
	def __init__(self, sample_path, sample_name, savepath, wavelength = None, prefix = None):

		self.sample_path = sample_path
		self.sample_name = sample_name
		self.savepath = savepath
		self.wavelength = wavelength
		if not os.path.isdir(self.savepath):
			os.mkdir(self.savepath)
			print(f'Created path {self.savepath}')
		if not wavelength:
			self.wavelength = 0.177120
			print('No wavelength was specified. ' + '\u03BB = ' + "{:.4f}".format(self.wavelength) + ' \u00C5' +' is used as default.')

	def get_sample_name_prefix(self, datasetnumber):
		'''
		The h5files contain the information about the sample name. This function returns the prefix from the sample name. 
		E.g.: Sample name: 'OTS_C2F6_1_alignment' <-- initiated by class init function
			  datasetnumber: 1
			  Returns: 'OTS_C2F6_1_alignment_0001'
			  			With datasetnumber = None the sample name is returned (OTS_C2F6_alignment)

			  -------------
			  Parameters:
			  	datasetnumber: None or int, number of dataset. If None, no second dataset is assumed.
			  -------------
			  Returns:
			  	 str, Joined sample name and formatted datasetnumber with underscore


		'''
		if not datasetnumber == None:
			dsn = "%04d"%datasetnumber
			prefix = '_'.join([self.sample_name , dsn])	
		else:
			# dsn = "{:04d}".format(1)
			dsn = ''
			prefix = self.sample_name	

		return prefix

	@staticmethod
	def min_function(par, x0, y0, x1, y1, overlap_area, step = 1000):
		'''
		Function to shift different loops to match each other, if XRR curve is recorded with overlap and different scan ranges.
		This functions is minimized in ID31.read_from_tif, if "fit_absorber_correction" is True.
		-----------
		Parameters:
			* par: shift value to shift one scan range. 
			* x0, y0: list, array; x- and y-values for the first scan range.
			* x1, y1: list, array; x- and y-values for the second, neighbouring scan range.
			* overlap_area: list, range in q, where the neighbouring scan ranges overlap
			* step: int: number of points for interpolation of y0 and y1. Default is 1000.
		-----------
		Returns:
			residual: float; summed up difference between between the first scan range and the second, shifted scan range. 

		'''
		ver_shift = par['shift'].value
		qrange = overlap_area[1] - overlap_area[0]
		q_dist = qrange / step
		q_interp = np.arange(overlap_area[0], overlap_area[1], q_dist)

		y1_interp_func = sc.interpolate.interp1d(x1, y1, fill_value = 'extrapolate')
		ref_y1_interp = y1_interp_func(q_interp)

		y0_interp_func = sc.interpolate.interp1d(x0, y0, fill_value = 'extrapolate')
		ref_y0_interp = y0_interp_func(q_interp)
		
		y1_interp_shifted = [refy1 * ver_shift for refy1 in ref_y1_interp]

		residual = ((np.array(ref_y0_interp)- np.array(y1_interp_shifted))**2).sum()

		return residual

	def read_from_tif(self, start_scan:int, macroname:str, datasetnumber = None, absorber_correc:list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], show = False,
		roi:list = [(159,135), 6, 3], bgRoiDist = 1, anz_aborted_scans = 0, scaling_factor = 1, save = False, pre_peak_eval=True, background = True,
		error_plot = False, weight_on_error = True, srcurr_norm = False, plot_seperate_loops = False, show_mondio_signal = False, show_srcur = False,
		fit_absorber_correction = False, counts2consider = 'roi_counts_norm2mondio_bg_and_atten_corrected'):

		'''
		Read XRR data from detector images.
		-----------
		Parameters:

			*start_scan: First scan number of XRR curve.
			*macroname: Name of macro used. This macro defines the scan ranges and the steptime. 
			*datasetnumber: Number of dataset. Default is None.
			*absorber_correc: When adjacent overlapping scan ranges do not match in y a correction factor can be given to shift the scan ranges to match each other.
			*show: bool; Plot the reflectivitiy. Default is False.
			*roi: [(x0, y0), width, height]: Region of interest on detector, in which counts are summed up. 
			*bgROiDist: int. Horizontal distance in pixel between the roi and the rois for the background substraction. Default is 1.
			*anz_aborted_scans: Number of aborted scans during the recording of 1 XRR curve. Default is 0. This parameter should not be needed anymore.
			* save: bool; Default is False. If True, a '*.dat' file is saved containing q and intensity values. The intensity values depend on the value of "counts2consider".
			* counts2consider:
				+ roi_counts_unnorm: Summed up counts in ROI. Not normalized.
				+ roi_counts_unnorm_bg_corrected: Summed up counts in ROI. Not normalized. Corrected by background signal.
				+ roi_counts_norm2mondio: Summed up counts in ROI. Normalized to diode signal.
				+ roi_counts_norm2mondio_bg_and_atten_corrected: Summed up counts in ROI, background and absorber corrected. The counts are normalized to the diode signal.
				+ roi_counts_norm2mondio_bg_corrected: Summed up counts in ROI, background corrected. The counts are normalized to the diode signal.
				+ roi_counts_norm2mondio_atten_corrected: Summed up counts in ROI, absorber corrected. The counts are normalized to the diode signal.
				+ roi_counts_bg_and_atten_corrected: Summed up counts in ROI, background and absorber corrected. 
			* pre_peak_eval: bool; Default True. If the primary beam below the critical is inside the ROI, an increased intensity can be observed at the beginning of the measurement.
							 If True, this part is cut off.
			* background: bool; Default True. If True, the background signal is substracted from the roi counts.
			* error_plot: bool; Default False. If True, a plot with the unnormalized XRR curve and its errors is shown.
			* weignt_on_error: bool; Default True. If True, the points of the reflectivity curve a weighted based on their statistics.
			* srcurr_norm: bool; Default Faslse. If True, the signal is normalized by the ring current.
			* plot_seperate_loops: bool; Default False. If True, a plot of the different scan ranges is shown (sorted correctly). Useful to see if overlapping scan ranges match each other.
			* show_mondio_signal: bool; Default False. If True, a plot of the diode signal during the measurement is shown.
			* show_srcur: bool; Default False. If True, a plot of the ring current during the measurement is shown.
		
		------------
		Returns:
			pandas.DataFrame containing counts, angles, qz, etc. The dataframe is sorted by the column "q".
		'''


		bgL_roi = [(roi[0][0] - bgRoiDist - math.floor(roi[1] / 2), roi[0][1]), math.floor(roi[1] / 2), roi[2]]
		bgR_roi = [(roi[0][0] + bgRoiDist + roi[1] , roi[0][1]), math.floor(roi[1] / 2), roi[2]]
		anz_scans, attenuators, d_att, mu= [], [], 1.25, 0.2144 * 2.203

		ret_data = {'data':None, 'data_figure': None, 'absorber': None, 'error_figure':None, 'sep_loop_figure': None, 'roi': {'bgL_roi': bgL_roi, 'bgR_roi':bgR_roi, 'roi':roi}}

		# dark current from mondio:
		I_0_mondio = 2.4e-10 # only noted once, can vary between p/T-series
		macrofile = '/'.join(self.sample_path.split('/')[0:-1]) + '/sc5084.py'
		with open(macrofile, 'r') as f:
			lines = f.readlines()
		loopcounter, matchcounter, newlines = 0, 0, [] 
		for l in lines:
			if macroname == l.replace(' ','').replace(':','').replace('def','').replace('\n','') and loopcounter ==0 and matchcounter == 0:
				matchcounter +=1
			if not macroname in l and matchcounter == 1:
				if 'def' in l:
					matchcounter += 1
				newlines.append(l)	

		for l in newlines:
			if not "def" in l:
				if not 'mv' in l and 'att' in l:
					att = int(re.findall(r"\(\s*\+?(-?\d+)\s*\)", l)[0])
					attenuators.append(att)
					loopcounter +=1
		anz_scans = loopcounter
		anz_scans = loopcounter - anz_aborted_scans
		
		qual_tab10 = cm.get_cmap('tab10', anz_scans)
		qual_tab10 = [rgb2hex(c) for c in qual_tab10.colors]

		normsignal = [scaling_factor * val for val in attenuators]

		prefix = self.get_sample_name_prefix(datasetnumber = datasetnumber)

		h5file = [os.path.join(self.sample_path,f) for f in os.listdir(self.sample_path) if os.path.isfile(os.path.join(self.sample_path,f)) and not '-' in f][0]
		end_scan = start_scan + loopcounter  - anz_aborted_scans

		dect_counts_unnorm, bgL_counts_unnorm, bgR_counts_unnorm, roi_counts_unnorm, roi_counts_norm2mondio = list(), list(), list(), list(), list()
		angles, bg_mean_counts_norm2mondio, mondi, normsignal, loop, title_dataset, scaled_mondio = list(), list(), list(), list(), list(), list(), list()
		mondio, srcur, steptime, absorber_correc_rep = list(), list(), list(), list()
		with h5py.File(h5file, 'r') as f:
			for i, scan in enumerate(range(start_scan, start_scan + anz_scans - anz_aborted_scans)):
				try:
					title_dataset_loop = f[f'{prefix}_{scan}.1/title'][()].decode('utf-8')

					# print(f'{prefix}_{scan}, {self.sample_name}')
				except AttributeError:
					title_dataset_loop = f[f'{prefix}_{scan}.1/title'][()]
				try:
					angle_loop = f[f'{prefix}_{scan}.1/instrument/mu/value'][()]
					srcur_loop =  f[f'{prefix}_{scan}.1/instrument/srcur/data'][()]
					mondio_loop = f[f'{prefix}_{scan}.1/instrument/mondio/data'][()]
					scaled_mondio_loop = f[f'{prefix}_{scan}.1/instrument/scaled_mondio/data'][()]
					I_zero_mondio_h5data = f[f'{prefix}_{scan}.1/instrument/I0 value'][()]
					time_loop = f[f'{prefix}_{scan}.1/measurement/sec'][()]
				except Exception as e:
					print(f'{e}\nScan {scan} seems not to be a scan of the measurement or was aborted during measurment.')
					angle_loop, srcur_loop, mondio_loop, scaled_mondio_loop, I_zero_mondio_h5data = np.nan, np.nan, np.nan, np.nan, np.nan
				mondio_fixed_noted = mondio_loop - I_0_mondio
				mondio_fixed_h5data = mondio_loop - I_zero_mondio_h5data

				if not isinstance(angle_loop, np.ndarray):
					numbOfReps = 0
				else:
					numbOfReps = len(angle_loop)
				# print(scan, numbOfReps)
				atten = np.repeat(attenuators[i], numbOfReps)
				normsignal_loop = np.exp(-d_att * atten * mu)
				normsignal_loop/= time_loop 
				if srcurr_norm:
					normsignal_loop/=srcur_loop
				edfpath = os.path.join(self.sample_path, prefix, 'scan'  + 	str(scan).zfill(4))
				edffile = [f for f in new_listdir(edfpath)][0]
				edfdata = fabio.open(edffile)

				for j in range(numbOfReps):
					# print(j)
					angles.append(angle_loop[j])
					loop.append(i)
					steptime.append(time_loop[j])
					srcur.append(srcur_loop[j])
					title_dataset.append(title_dataset_loop)
					data = edfdata.getframe(j).data
					dect_counts_unnorm_dummy = np.sum(data, axis = (0,1))
					roi_counts_unnorm_dummy = np.sum(data[roi[0][1]:roi[0][1] + roi[2], roi[0][0]:roi[0][0] + roi[1]], axis = (0,1))
					bgL_counts_unnorm_dummy = np.sum(data[roi[0][1]:roi[0][1] + roi[2], roi[0][0] - bgRoiDist - roi[1]:roi[0][0] - bgRoiDist], axis=(0,1))
					bgR_counts_unnorm_dummy = np.sum(data[roi[0][1]:roi[0][1] + roi[2], roi[0][0] + roi[1] + bgRoiDist:roi[0][0] + roi[1] + bgRoiDist + roi[1]], axis = (0,1))
					bg_mean_counts_unnorm_dummy = np.mean([bgL_counts_unnorm_dummy, bgR_counts_unnorm_dummy])

					dect_counts_unnorm.append(dect_counts_unnorm_dummy)
					bgL_counts_unnorm.append(bgL_counts_unnorm_dummy)
					bgR_counts_unnorm.append(bgR_counts_unnorm_dummy)
					roi_counts_unnorm.append(roi_counts_unnorm_dummy)
					roi_counts_norm2mondio.append(roi_counts_unnorm_dummy / mondio_fixed_noted[j])
					bg_mean_counts_norm2mondio.append(bg_mean_counts_unnorm_dummy / mondio_fixed_noted[j])
					
					mondio.append(mondio_fixed_noted[j])
					scaled_mondio.append(scaled_mondio_loop[j])
					normsignal.append(normsignal_loop[j])
					absorber_correc_rep.append(absorber_correc[i])

				edfdata.close()
		if fit_absorber_correction:
			for i in range(len(absorber_correc_rep)):
				absorber_correc_rep[i] = 1
		bg_mean_counts_unnorm = .5 * (np.array(bgL_counts_unnorm) + np.array(bgR_counts_unnorm))
		
		bg_mean_counts_norm2mondio_atten_corrected = bg_mean_counts_norm2mondio / np.array(normsignal) * np.array(absorber_correc_rep)
		# bg_mean_counts_norm2mondio_atten_corrected = bg_mean_counts_norm2mondio * np.array(normsignal) * np.array(absorber_correc_rep)


		roi_counts_unnorm_bg_corrected = np.array(roi_counts_unnorm) - bg_mean_counts_unnorm

		roi_counts_norm2mondio_bg_corrected = np.array(roi_counts_unnorm_bg_corrected) / np.array(mondio) * np.array(absorber_correc_rep)	
		roi_counts_norm2mondio_atten_corrected = np.array(roi_counts_norm2mondio) / np.array(normsignal) * np.array(absorber_correc_rep)
		roi_counts_norm2mondio_bg_and_atten_corrected = np.array(roi_counts_norm2mondio_atten_corrected) - np.array(bg_mean_counts_norm2mondio_atten_corrected)

		q = th2q(self.wavelength, angles)

		columns = ['loop', 'angles', 'q','dect_counts_unnorm', 'bgL_counts_unnorm', 'bgR_counts_unnorm', 'bg_mean_counts_unnorm', 'roi_counts_unnorm',
		'roi_counts_norm2mondio', 'bg_mean_counts_norm2mondio', 'mondio', 'normsignal',
		'roi_counts_unnorm_bg_corrected', 'roi_counts_norm2mondio_bg_corrected', 'roi_counts_norm2mondio_bg_and_atten_corrected', 'title_dataset', 'srcur',
		'roi_counts_norm2mondio_atten_corrected', 'bg_mean_counts_norm2mondio_atten_corrected', 'absorber_correc_rep']
		df_unsort = pd.DataFrame(columns = columns, index = np.arange(0, len(q)))
		for col in df_unsort:
			df_unsort[col] = eval(col)

		df_unsort = df_unsort.loc[~df_unsort['title_dataset'].isin(['a2scan mu'])]
		df_unsort['counts2consider'] = df_unsort[counts2consider]
		elems_in_loop = df_unsort.pivot_table(columns=['loop'], aggfunc='size')
		for i in range(len(elems_in_loop)):
			if elems_in_loop[i] == 1:
				loop_to_drop = i
				df_unsort.drop(df_unsort.index[df_unsort['loop'] == i], inplace = True)

		if pre_peak_eval:
			counts = np.array(df_unsort.counts2consider[df_unsort.q < 0.03])
			firstmin = argrelmin(counts, order=8, mode = 'wrap')[0][0]
			truncate_indices = list(range(0, firstmin + 1))
			df_unsort = df_unsort.drop(truncate_indices).reset_index(drop=True)

		if fit_absorber_correction:
			par = Parameters()
			par.add('shift', value=1.1)
			# print(df_unsort)
			idx = 0
			absorber_shifts = list()
			while idx in range(len(df_unsort.angles)):
				if 0 <= idx < len(df_unsort.angles) -1:
					if df_unsort.angles[idx] > df_unsort.angles[idx + 1]:
						loop_val = df_unsort.at[idx, 'loop']
						next_loop_val = loop_val + 1
						first_loop_values = df_unsort[df_unsort.loop == loop_val]
						next_loop_values = df_unsort[df_unsort.loop == next_loop_val]
						overlap_area = [next_loop_values.q.iloc[0], first_loop_values.q.iloc[-1]]
						first_loop_overlap_df = first_loop_values.loc[first_loop_values['q'].between(overlap_area[0], overlap_area[1], inclusive='both'), first_loop_values.columns.tolist()]
						next_loop_overlap_df = next_loop_values.loc[next_loop_values['q'].between(overlap_area[0], overlap_area[1], inclusive='both'), next_loop_values.columns.tolist()]
						# print(f'{loop_val}: first_loop_overlap: {len(first_loop_overlap_df.q)} points, next_loop_overlap: {len(next_loop_overlap_df.q)} points')
						
						if len(next_loop_overlap_df.q) < 2 and not len(first_loop_overlap_df.q) < 2:
							print(f'Between loops {loop_val} - {next_loop_val}: Second loop has not enough overlapping points')
							q_first_index = next_loop_overlap_df.q.index[0]
							q_first = df_unsort.q.iloc[q_first_index]
							q_second = df_unsort.q.iloc[q_first_index + 1]
							counts2consider_first = df_unsort.counts2consider.iloc[q_first_index]
							counts2consider_second = df_unsort.counts2consider.iloc[q_first_index + 1]
							# print(df_unsort.counts2consider.iloc[q_first_index], df_unsort.counts2consider.iloc[q_first_index + 1])
							result = minimize(fcn = ID31.min_function, params = par,
								args = (first_loop_overlap_df.q, first_loop_overlap_df.counts2consider, [q_first, q_second], [counts2consider_first, counts2consider_second], overlap_area),
								kws = {'step':20}, method = 'leastsq')
						# print(f'{loop_val}: first_loop_overlap: {len(first_loop_overlap_df.q)} points, next_loop_overlap: {len(next_loop_overlap_df.q)} points')
						
						elif len(first_loop_overlap_df.q) < 2 and not len(next_loop_overlap_df.q) <2:
							print(f'Between loops {loop_val} - {next_loop_val}: First loop has not enough overlapping points')
							q_second_index = first_loop_overlap_df.q.index[0]
							q_second = df_unsort.q.iloc[q_second_index]
							q_first = df_unsort.q.iloc[q_second_index - 1]
							counts2consider_first = df_unsort.counts2consider.iloc[q_second_index - 1]
							counts2consider_second = df_unsort.counts2consider.iloc[q_second_index]
							result = minimize(fcn = ID31.min_function, params = par,
								args = (first_loop_overlap_df.q, first_loop_overlap_df.counts2consider, [q_first, q_second], [counts2consider_first, counts2consider_second], overlap_area),
								kws = {'step':20}, method = 'leastsq')
						
						elif len(next_loop_overlap_df.q) < 2 and len(first_loop_overlap_df.q) <2:
							print(f'Between loops {loop_val} - {next_loop_val}: Both loops have not enough overlapping points')
							# first_loop:
							q_second_index_fl = first_loop_overlap_df.q.index[0]
							q_second_fl = df_unsort.q.iloc[q_second_index_fl]
							q_first_fl = df_unsort.q.iloc[q_second_index_fl - 1]
							counts2consider_first_fl = df_unsort.counts2consider.iloc[q_second_index_fl - 1]
							counts2consider_second_fl = df_unsort.counts2consider.iloc[q_second_index_fl]

							# second loop
							q_first_index_nl = next_loop_overlap_df.q.index[0]
							q_first_nl = df_unsort.q.iloc[q_first_index_nl]
							q_second_nl = df_unsort.q.iloc[q_first_index_nl + 1]
							counts2consider_first_nl = df_unsort.counts2consider.iloc[q_first_index_nl]
							counts2consider_second_nl= df_unsort.counts2consider.iloc[q_first_index_nl + 1]

							result = minimize(fcn = ID31.min_function, params = par,
								args = ([q_first_fl, q_second_fl], [counts2consider_first_fl, counts2consider_second_fl], 
									[q_first_nl, q_second_nl], [counts2consider_first_nl, counts2consider_second_nl], overlap_area),
								kws = {'step':20}, method = 'leastsq')

						else:
							result = minimize(fcn = ID31.min_function, params = par,
								args = (first_loop_overlap_df.q, first_loop_overlap_df.counts2consider, next_loop_overlap_df.q, next_loop_overlap_df.counts2consider, overlap_area),
								kws = {'step':20}, method = 'leastsq')
							# report_fit(result)
						absorber_shift = result.params.valuesdict()['shift']
						absorber_shifts.append(absorber_shift)
						for col in df_unsort[['counts2consider', 'bg_mean_counts_norm2mondio_atten_corrected']].columns:
							df_unsort.loc[df_unsort.loop == next_loop_val, col] = df_unsort[col] * absorber_shift
						# df_unsort.loc[df_unsort.loop == next_loop_val, 'counts2consider'] = df_unsort.counts2consider * absorber_shift
				idx += 1
			absorber_text = ["{:.2f}".format(a) if a >=0.01 else "{:.2e}".format(a) for a in absorber_shifts]
			absorber_text.insert(0,1)
			print(f'Loops shifted with {absorber_text}')
			# loops = list()
			# for i in range(max(df_unsort.loop) + 1):
			# 	loops.append(go.Scatter(x = df_unsort[df_unsort.loop == i].q, y = df_unsort[df_unsort.loop == i].counts2consider, mode = 'lines',
			# 		line = dict(color = qual_tab10[i], width = 4), name = f'loop {i}: {"{:.4f}".format(min(df_unsort.q[df_unsort.loop == i]))} - {"{:.4f}".format(max(df_unsort.q[df_unsort.loop == i]))} \u00C5'))
 
			# fig = go.Figure(data = loops, layout = layout.refl())
			# fig.update_layout(hovermode ='x', title_text = str(absorber_text))
			# fig.show()
		
#############################################
		df = df_unsort.sort_values(by='q')###
#############################################
		ret_data['data'] = df

		if plot_seperate_loops:
			absorber_shifts.insert(0,1)
			single_loops, traces, bg_traces =dict(), list(), list()
			# customdata = customdata
			
			for i in range(anz_scans):
			    single_loops[i] = df_unsort[['q', 'angles', 'roi_counts_norm2mondio_atten_corrected', 'roi_counts_norm2mondio_bg_and_atten_corrected', 'bg_mean_counts_norm2mondio_atten_corrected']][df_unsort.loop == i]
			    # customdata = np.stack((single_loops[i]['steptime'], single_loops[i]['atten_position'], single_loops[i]['angles']), axis = -1)
			    if background:
			    	counts2consider = single_loops[i]['roi_counts_norm2mondio_bg_and_atten_corrected'] * absorber_shifts[i]
			    	bg2consider = single_loops[i]['bg_mean_counts_norm2mondio_atten_corrected'] * absorber_shifts[i]
			    	bg_traces.append(go.Scatter(x = single_loops[i]['q'], y = bg2consider,
			    	mode = 'lines+markers', marker = dict(color = qual_tab10[i]), line = dict(width = 4, color = qual_tab10[i]),
			    	name = f'background loop {i}', legendgroup = f'{i}', showlegend = False))
			    else:
			    	counts2consider = single_loops[i]['roi_counts_norm2mondio_atten_corrected'] * absorber_shifts[i]
			    traces.append(go.Scatter(x = single_loops[i]['q'], y = counts2consider,
			    	mode = 'lines+markers', marker = dict(color = qual_tab10[i]), line = dict(width = 4, color = qual_tab10[i]),
			    	name = f'loop {i}', legendgroup = f'{i}'
			        # hovertemplate = '<b>q</b> = %{x:.3f} <br><b>th</b> = %{customdata[2]:.3f}<br><b>t</b> = %{customdata[0]:.1f}<br><b>atten_pos</b> = %{customdata[1]:.1f}'
			        ))
			fig = go.Figure(data = traces, layout = layout.refl())
			# if background: 
				# fig.add_traces(bg_traces)
			fig.update_layout(legend = dict(title_text = f'scans {start_scan} - {start_scan + anz_scans - anz_aborted_scans} ', tracegroupgap=0), title = dict(text = self.sample_name),)
			# fig.show()		
			ret_data['sep_loop_figure'] = fig

		if weight_on_error:
			weights = []
			if background:
				error_diff_y = abs((np.sqrt(df.bg_mean_counts_unnorm + df.roi_counts_unnorm) / df.roi_counts_unnorm_bg_corrected))
			else:
				error_diff_y = abs((np.sqrt(df.roi_counts_unnorm)) / df.roi_counts_unnorm)

			for i, err in enumerate(error_diff_y): 
				if not df.roi_counts_unnorm_bg_corrected[i] < 0.00000:
					if error_diff_y[i] <= 0.100000:
						weights.append(float(1))
					if 0.100000 < error_diff_y[i] < 0.500000:
						weights.append(float(0.5))
					if 0.500000 <= error_diff_y[i] <= 0.800000:
						weights.append(float(0.2))
					if 0.800000 < error_diff_y[i] < 1.00000:
						weights.append(0.1)
					if error_diff_y[i] >= 1.000000:
						weights.append(float(0))
				else:
					weights.append(float(0))
		else:
			weights = np.ones(len(df.counts2consider.tolist())).tolist()
		df['weights'] = weights

		if show:
			if background:
				# mw_trace = go.Scatter(x = df.q, y = df.roi_counts_norm2mondio_bg_and_atten_corrected, mode = 'lines', line = dict(color = 'green', width = 4), name = 'background corrected')
				# bg_tace = go.Scatter(x = df.q, y = df.bg_mean_counts_norm2mondio_atten_corrected, mode = 'lines', line = dict(color = 'red', width = 4), name = 'background')
				# mw_uncorrected_trace = go.Scatter(x = df.q, y = df.roi_counts_norm2mondio_atten_corrected, mode = 'lines', line = dict(color = 'red', width = 4), name = 'uncorrected')
				mw_trace = go.Scatter(x = df.q, y = df.counts2consider, mode = 'lines', line = dict(color = 'green', width = 4), name = 'background corrected')
				bg_tace = go.Scatter(x = df.q, y = df.bg_mean_counts_norm2mondio_atten_corrected, mode = 'lines', line = dict(color = 'red', width = 4), name = 'background')
				mw_uncorrected_trace = go.Scatter(x = df.q, y = df.counts2consider, mode = 'lines', line = dict(color = 'red', width = 4), name = 'uncorrected')
				traces = [mw_trace, bg_tace, mw_uncorrected_trace]
			else:
				# mw_trace = go.Scatter(x = df.q, y = df.roi_counts_norm2mondio_atten_corrected, mode = 'lines', line = dict(color = 'green', width = 4))
				mw_trace = go.Scatter(x = df.q, y = df.counts2consider, mode = 'lines', line = dict(color = 'green', width = 4))
				traces = [mw_trace]
			fig = go.Figure(data = traces, layout = layout.refl())
			fig.update_layout(title = dict(text = self.sample_name),
				legend_title= f'scan {start_scan} - {start_scan + anz_scans - anz_aborted_scans}')
			# fig.show()
			ret_data['data_figure'] = fig
		
		if error_plot:
			rawdata = go.Scatter(x = df_unsort['q'], y = df_unsort['roi_counts_unnorm'], customdata = np.sqrt(df_unsort['roi_counts_unnorm'].tolist()),
			        error_y = dict(array = np.sqrt(df_unsort['roi_counts_unnorm'].tolist()),visible=True),
			        mode='markers',name = 'raw data',
			        hovertemplate = '<b>q<b> = %{x:.3f} <br> <b>I</b> = %{y:} <span>&#177;</span> %{customdata:.2f}')

			raw_bg =  go.Scatter(x = df_unsort['q'],y = df_unsort['bg_mean_counts_unnorm'], customdata = np.sqrt(df_unsort['bg_mean_counts_unnorm'].tolist()),
			    error_y = dict(array = np.sqrt(df_unsort['bg_mean_counts_unnorm'].tolist()), visible=True),mode='markers', name = 'raw background',
			    hovertemplate = '<b>q<b> = %{x:.3f} <br> <b>I</b> = %{y:} <span>&#177;</span> %{customdata:.2f}')

			raw_diff = go.Scatter(x = df_unsort['q'], y = df_unsort['roi_counts_unnorm_bg_corrected'],
			    customdata = np.sqrt(df_unsort['bg_mean_counts_unnorm'].tolist() + df_unsort['roi_counts_unnorm'].tolist()),
			    error_y = dict(array = np.sqrt(df_unsort['bg_mean_counts_unnorm'].tolist() + df_unsort['roi_counts_unnorm'].tolist()), visible=True),
			    mode='markers', name = 'raw - raw background',
			    hovertemplate = '<b>q<b> = %{x:.3f} <br> <b>I</b> = %{y:} <span>&#177;</span> %{customdata:.2f}')
			fig = go.Figure(data = [rawdata, raw_bg, raw_diff], layout = layout.refl())

			fig.update_layout(autosize=True, showlegend=True, legend_title='', legend_borderwidth=1.0, title=dict(
			    text = f'{self.sample_name} scan {start_scan} - {end_scan}'), xaxis_mirror=True, yaxis_mirror=True, hovermode = 'x')
			# fig.show()

			ret_data['error_figure'] = fig

		if show_mondio_signal:
			traces, single_loops = list(), dict()
			for i in range(anz_scans):
				single_loops[i] = df_unsort[['q', 'angles', 'mondio']][df_unsort.loop == i]
				traces.append(go.Scatter(x = single_loops[i].q, y = single_loops[i].mondio, mode = 'lines+markers', line = dict(color = qual_tab10[i], width = 4),
					marker = dict(symbol = 'circle', color = qual_tab10[i], size = 8),
					name = f'loop{i}')
				)
			fig = go.Figure(data = traces, layout = layout.mondio())
			# fig.show()
			ret_data['mondio_figure'] = fig
		
		if show_srcur:
			traces, single_loops = list(), dict()
			for i in range(anz_scans):
				single_loops[i] = df_unsort[['q', 'angles', 'srcur']][df_unsort.loop == i]
				traces.append(go.Scatter(x = single_loops[i].q, y = single_loops[i].srcur, mode = 'lines+markers', line = dict(color = qual_tab10[i], width = 4),
					marker = dict(symbol = 'circle', color = qual_tab10[i], size = 8),
					name = f'loop{i}')
				)
			fig = go.Figure(data = traces, layout = layout.mondio())
			# fig.show()

			ret_data['srcur_figure'] = fig
		if save:
			filename = os.path.join(self.savepath, prefix) +f'_{start_scan:04}_{end_scan:04}.dat'
			dict2save={'q':df.q, 'counts':df.counts2consider, 'weights':df.weights}
			df2save = pd.DataFrame.from_dict(dict2save, orient = 'columns')
			df2save.to_string(filename, header=False, index=False)
			print(f'Saved {filename}')
		return ret_data


	def save_data_to_pickle(self, obj, fname = None):
		if fname is None:
		    name = 'pickle_file'
		else:
		    name = fname
		if not name.endswith('.pkl'):
			pickle_file = os.path.join(self.savepath , f'{name}.pkl')
		else:
			pickle_file = os.path.join(self.savepath , name)
		file_obj = open(pickle_file, 'wb')
		pickle.dump(obj, file_obj)
		file_obj.close()

	@staticmethod
	def load_from_pickle(file):
		file_obj = open(file, 'rb')
		data = pickle.load(file_obj)
		file_obj.close()
		return data



	def dectimage(self, scan_number, th_ind, roi = None, bgRoiDist = 1, margin = 10, sleeptime=0.01, series = False, save_imgs = False,
		datasetnumber = None):
		'''
		Show detector images or series of detector images to e.g. check the positioning of the ROI.
		----------
		Parameters:
			* scan_number: int; Scan number of the scan to show.
			* th_ind: int; index of theta value, to which the detector image should be plotted. If series is True, this value has no impact.
			* roi: list; Default None. Region of Interest in format: [(x0, y0), width, height]. (x0, y0) describing the bottom left corner of the ROI.
			* bgRoiDist: int; Default 1. Horizontal distance in pixel between the ROI and the ROIs for the background.
			* margin: int; Default 10. Plot margin in pixel
			* sleeptime: float; Default 0.01. Time in seconds between to detector images in a series. Only has impact if series = True.
			* series: bool; Default False. If True, all detectorimages belonging to the given scan_number are shown one after another.
			* save_imgs: bool; Default False. If True, the detectorimage or detectorimages are saved in "self.savepath/dect_imgs". The files are named using the scan and frame number.
		'''
		prefix = self.get_sample_name_prefix(datasetnumber = datasetnumber)
		h5file = [os.path.join(self.sample_path,f) for f in os.listdir(self.sample_path) if os.path.isfile(os.path.join(self.sample_path,f)) and not '-' in f][0]

		with h5py.File(h5file, 'r') as f:
			angle = f[f'{prefix}_{scan_number}.1/instrument/mu/value'][()]
			q = th2q(wavelength = self.wavelength, th = angle)	
		
		edfpath = os.path.join(self.sample_path, prefix, 'scan'  + 	str(scan_number).zfill(4))
		edffile = [f for f in new_listdir(edfpath)][0]
		edfdata = fabio.open(edffile)
		imarrays = list()

		for j in range(int(edfdata.header["acq_nb_frames"])):
			data = edfdata.getframe(j).data
			imarrays.append(data)
		edfdata.close()
		print(imarrays[0].shape)
		if save_imgs:
			img_path = os.path.join(self.savepath, 'dect_imgs')
			if not os.path.isdir(img_path):
				os.mkdir(img_path)
			img_names = [os.path.join(img_path, 'scan'  + str(scan_number).zfill(4) + '_frame_' + str(j).zfill(3) + '.png') for j in range(int(edfdata.header["acq_nb_frames"]))]
		
		if not roi is None:
			roi_height = roi[2]
			roi_width = roi[1]
			bgL_roi = [(roi[0][0] - bgRoiDist - math.floor(roi[1] / 2), roi[0][1]), math.floor(roi[1] / 2), roi[2]]
			bgR_roi = [(roi[0][0] + bgRoiDist + roi[1] , roi[0][1]), math.floor(roi[1] / 2), roi[2]] 
			bg_height = roi_height
			

			if series:
				plt.ion()
				f = plt.figure(figsize = (12,32))
				ax = f.gca()
				ax.set(xlim=[roi[0][0] - margin, roi[0][0] + roi[1] + margin], ylim = [roi[0][1] + roi[2] + margin, roi[0][1] - margin])
				plt.xlabel('x', fontsize=16)
				plt.xticks(fontsize=16)
				plt.ylabel('y',fontsize=16)
				plt.yticks(fontsize=16)

				draw_roi = patches.Rectangle(roi[0], roi[1], roi[2], fc='None', ec='g', lw=2) 
				draw_bgL_roi = patches.Rectangle((roi[0][0] - bgRoiDist - roi[1] / 2, roi[0][1]), roi[1] / 2, roi[2] , fc='None', ec='r', lw=2)
				draw_bgR_roi = patches.Rectangle((roi[0][0] + bgRoiDist + roi[1] , roi[0][1]), roi[1] / 2, roi[2] , fc='None', ec='r', lw=2)
				ax.add_patch(draw_bgR_roi)
				ax.add_patch(draw_roi)
				ax.add_patch(draw_bgL_roi)
				f.suptitle('Detector image of scan ' + str(scan_number) + ' th = ' + "{:.3f}".format(angle[0]) + '°\nq = ' + "{:.3f}".format(q[0]) + r' $\AA^{-1}$', fontsize=16)
				im = ax.imshow(imarrays[0], norm = LogNorm())
				for i in range(1, len(imarrays)):
					im.set_data(imarrays[i])
					f.suptitle('Detector image of scan ' + str(scan_number) + ' th = ' + "{:.3f}".format(angle[i]) + '°\nq = ' + "{:.3f}".format(q[i]) + r' $\AA^{-1}$', fontsize=16)
					if save_imgs:
						f.savefig(img_names[i], dpi = 300)
					f.canvas.flush_events()
					plt.pause(sleeptime)
				plt.close()

			else:

				f = plt.figure(figsize = (12,32))
				ax = f.gca()
				ax.set(xlim=[roi[0][0] - margin, roi[0][0] + roi[1] + margin], ylim = [roi[0][1] + roi[2] + margin, roi[0][1] - margin])
				plt.xlabel('x', fontsize=16)
				plt.xticks(fontsize=16)
				plt.ylabel('y',fontsize=16)
				plt.yticks(fontsize=16)
				im = ax.imshow(imarrays[th_ind], norm = LogNorm())
				f.suptitle('Detector image of scan ' + str(scan_number) + ' th = ' + "{:.3f}".format(angle[th_ind]) + '°\nq = ' + "{:.3f}".format(q[th_ind]) + r' $\AA^{-1}$', fontsize=16)
				draw_roi = patches.Rectangle(roi[0], roi[1], roi[2], fc='None', ec='g', lw=2) 
				draw_bgL_roi = patches.Rectangle((roi[0][0] - bgRoiDist - roi[1] / 2, roi[0][1]), roi[1] / 2, roi[2] , fc='None', ec='r', lw=2)
				draw_bgR_roi = patches.Rectangle((roi[0][0] + bgRoiDist + roi[1] , roi[0][1]), roi[1] / 2, roi[2] , fc='None', ec='r', lw=2)
				ax.add_patch(draw_bgR_roi)
				ax.add_patch(draw_roi)
				ax.add_patch(draw_bgL_roi)
				plt.tight_layout()
				plt.show()

	@staticmethod
	def makedirlsfit(filepath, lsfitpath, ext = '.dat'):
		'''
		Create subdirectories in filepath and copy all files necessary to run lsfit into this subderictories. The filenames in the "*.par" and "LSFIT.OPT" files are changed automatically.
		---------
		Parameters:
			* filepath: Path where the data files with q, intensity (, weights) are saved. 
			* lsfitpath: Path with lsfit template files. This path should contain:
				+ LSFIT.OPT
				+ reflek.exe
				+ testcli.exe
				+ pardat.par
				+ condat.con
			* ext: str; Default ".dat". File ending of the datafiles.
		'''
		files = [f for f in new_listdir(filepath) if f.endswith(ext)]
		lsfitfiles = [lsf for lsf in new_listdir(lsfitpath)]
		subfolders = [f.split(ext)[0] for f in files]
		[print(sd) for sd in subfolders]
		for f, sd in zip(files, subfolders):
			try:
				filename = sd.split('/')[-1]
				confilename = os.path.join(sd, filename + '.con')
				parfilename = os.path.join(sd, filename + '.par')
				os.mkdir(sd)
				shutil.move(f, sd)
			except:
				print('Subdirectories already created.')

			for lsfile in lsfitfiles:
				if lsfile.endswith('.con'):
					confile_template = lsfile 
					shutil.copyfile(confile_template, confilename)

				elif lsfile.endswith('.par'):
					parfile_template = lsfile
					shutil.copyfile(parfile_template, parfilename)
					with open(parfilename, 'r') as f:
						pardata = f.readlines()
						pardata[-1] = filename + ext
					with open(parfilename, 'w') as f:
						f.writelines(pardata)
			
				elif lsfile.endswith('.OPT'):
					file =os.path.join(sd, lsfile.split('/')[-1])
					shutil.copyfile(lsfile, file)
					with open(file,'r') as f:
						optdata = f.readlines()
						optdata[0] = filename + '\n'
					with open(file,'w') as f:
						f.writelines(optdata)

				else:
					file =os.path.join(sd, lsfile.split('/')[-1])
					shutil.copyfile(lsfile, file)

	def read_md_table_data(self, skipStartScans = list(), selectStartScans=list()):
	    '''
		Read data from markdown table in jupyter notebook. The mean pressure and temperature for each measurement are calculated using numpy.nanmean.
		Example table:
		|              macro              | scan number | say position | <I>p</I><sub>before</sub> [bar] | <I> p </I><sub> after </sub>  [bar] | <I> T  </I><sub> before </sub>  [°C] | <I> T  </I><sub> after </sub>  [°C] |              Comments              |
		|:-------------------------------:|:-----------:|:------------:|:-------------------------------:|:-----------------------------------:|:------------------------------------:|:------------------------------------:|:----------------------------------:|
		|        doref_lesspoints()       |     1-7     |     -1.3     |               air               |                 air                 |                  25                  |                  25                  |                                    |
		|        doref_lesspoints()       |    10-16    |     -1.1     |               0.0               |                 0.0                 |                  25                  |                  25                  |                                    |
		|      doref_lp_lesspoints()      |    19-25    |     -0.9     |               24.6              |                 24.6                |                  24                  |                  24                  |                                    |
		|      doref_hp_lesspoints()      |    29-34    |     -0.7     |               47.4              |                 47.6                |                  24                  |                  25                  |                                    |
		|      doref_hp_lesspoints()      |    38-43    |     -0.5     |               52.3              |                 52.4                |                  50                  |                  50                  |                                    |
		|      doref_hp_lesspoints()      |     6-11    |     -0.3     |               58.8              |                 58.8                |                  100                 |                  100                 |     <b>NEW DATASET:</b> 0002!!!    |
		|      doref_hp_lesspoints()      |    17-22    |     -0.1     |               58.8              |                 58.8                |                  100                 |                  100                 |                                    |
		
		The table data has to be saved in "self.savepath" with the fileending ".md".
		-----------
		Parameters:
			* skipStartScans: list; start scan numbers that should not be considered

		-----------
		Returns:
			* pandas.DataFrame. The column names are taken from the headerline of the table. Extra columns for mean temperature and pressure are added.

		'''

	    try:
	        mdfile = [f for f in new_listdir(self.savepath) if f.endswith('.md')][0]
	    except:
	        print(f'No md file in {self.savepath}.')
	        pass
	    with open(mdfile, 'r') as f:
	        lines = f.readlines()
	    headerline = lines[0].split('|')
	    headerline = ["".join(elem.split()) for elem in headerline if not any(elem == x for x in ['', '\n'])]
	    lines = lines[2:]
	    lines = [x for x in lines if x]
	    # lines = [l for l in lines[2:] if l != '']
	    index_df = [i for i in range(len(lines))]
	    df = pd.DataFrame(columns = headerline, index = index_df)
	    nested_lines = list()
	    for l in lines:
	        splitline = l.split('|')
	        splitline = [elem.strip(' ') for elem in splitline if not any(elem == x for x in ['', '\n'])]
	        nested_lines.append(splitline)
	    nested_lines = [lst for lst in nested_lines if lst != []]
	    
	    for nl, ind in zip(nested_lines, index_df):
	    	df.iloc[ind] = nl
	    df.dropna(axis = 'index',inplace = True)
	    pressure, temperature = list(), list()
	    for pv, pn in zip(df['<I>p</I><sub>before</sub>[bar]'], df['<I>p</I><sub>after</sub>[bar]']):
	    	if 'air' in pv:
	    		pressure.append(0)
	    	else:
	    		try:
	    			pvF = float(pv)
	    		except:
	    			pvF =np.nan
	    		try:
	    			pnF = float(pn)
	    		except:
	    			pnF = np.nan
	    		pressure.append(np.nanmean([pvF, pnF]))	

	    for tv, tn in zip(df['<I>T</I><sub>before</sub>[°C]'], df['<I>T</I><sub>after</sub>[°C]']):
	    	try:
	    		tvF = float(tv)
	    	except:
	    		tvF =np.nan
	    	try:
	    		tnF = float(tn)
	    	except Exception as e:
	    		tnF = np.nan
	    	temperature.append(np.nanmean([tvF, tnF]))
	    
	    anz_scans = [elem.split('-') for elem in df.scannumber]
	    start_scans = [int(sn[0]) for sn in anz_scans]
	    anz_scans = [int(sn[1]) - int(sn[0]) + 1 for sn in anz_scans]
	    df['pressure'], df['temperature'], df['start_scans'], df['anz_scans']= pressure, temperature, start_scans, anz_scans
	    if not skipStartScans == []:
	    	df = df.loc[~df['start_scans'].isin(skipStartScans)].reset_index()
	    if not selectStartScans == []:
	    	df = df.loc[df['start_scans'].isin(selectStartScans)].reset_index()

	    return df

	def read_macrofile(self, macroname):
		attenuators, newlines = [], []
		I0 = 2.4e-10
		macrofile = '/'.join(self.sample_path.split('/')[0:-1]) + '/sc5084.py'
		with open(macrofile, 'r') as f:
			lines = f.readlines()
		loopcounter, matchcounter, newlines, anz_points, lp = 0, 0, [], [], []
		for l in lines:
			if macroname == l.replace(' ','').replace(':','').replace('def','').replace('\n','') and loopcounter ==0 and matchcounter == 0:
				matchcounter +=1
			if not macroname in l and matchcounter == 1:
				if 'def' in l:
					matchcounter += 1
				newlines.append(l)	
		newlines = [l.strip() for l in newlines]
		for l in newlines:
			if not "def" in l:
				if not 'mv' in l and 'att' in l:
					att = int(re.findall(r"\(\s*\+?(-?\d+)\s*\)", l)[0])
					anz_p = int(l.split()[-1].strip(';||a2scan|(|)').split(',')[-2])
					attenuators.append(att)
					anz_points.append(anz_p)
					lp.append(loopcounter)
					loopcounter +=1
		anz_scans = loopcounter
		loop_params = pd.DataFrame(list(zip(lp, anz_points, attenuators)), columns = ['loop','anz_points', 'attenuator'])
		return loop_params
	
	def rename_files_from_listOfNames(self, names:list, files = list(), ext='.dat', keepScanNumbers=True, shownames=True, rename=False):
		if files == []:
			files = [f for f in new_listdir(self.savepath) if f.endswith(ext)]
			prefix = [os.path.basename(f).strip(ext) + '_' for f in files]
			if not keepScanNumbers:
				prefix = [p[:-11] + '_' for p in prefix]

		new_names = [p + n.replace(' ','_').replace('@_','').replace('.','p').replace('p0','').replace('_°C','C').replace('°C','C')  + ext for p, n in zip(prefix,names)]
		# print(os.path.basename(new_names[0]), names[0].replace(' ','_').replace('@_','').replace('.','p').replace('_°C','C').replace('°C','C').replace('p0',''))
		if not os.path.isdir(names[0]):
			new_names = [os.path.join(self.savepath,os.path.basename(f) ) for f in new_names]
		if shownames:
			if not names[0].replace(' ','_').replace('@_','').replace('.','p').replace('_°C','C').replace('°C','C').replace('p0','') in files[0]:
				{print(n) for n in new_names}
			else:
				print('Already renamed files.')
		if rename:
			if not names[0].replace(' ','_').replace('@_','').replace('.','p').replace('_°C','C').replace('°C','C').replace('p0','') in files[0]:
				[os.rename(f, n) for f, n in zip(files, new_names)]
			else:
				print('Already renamed files.')


	# def find_h5file(self):
	# 	h5file = [os.path.join(self.sample_path,f) for f in os.listdir(self.sample_path) if os.path.isfile(os.path.join(self.sample_path,f)) and not '-' in f][0]
	# 	return h5file

	# def edfdata(self, scan, datasetnumber = None, roi = None):

	# 	if not datasetnumber == None:
	# 		dsn = "%04d"%datasetnumber
	# 		prefix = '_'.join([self.sample_name , dsn])
	# 	else:
	# 		dsn = ''
	# 		prefix=self.sample_name


		
	# 	edfpath = os.path.join(self.sample_path, prefix, 'scan'  + str(scan).zfill(4))
	# 	try:
	# 		edffile = [f for f in new_listdir(edfpath)][0]
	# 		edfdata = fabio.open(edffile)
	# 		numb_imgs = int(edfdata.header['acq_nb_frames'])
	# 		img_index = np.arange(0,numb_imgs)
			
	# 		df = pd.DataFrame(columns = ['detector_counts', 'roi_counts'], index = img_index)
	# 		df.detector_counts = [np.sum(frame.data, axis = (0,1)) for frame in edfdata.frames()]
	# 		# df.roi_counts
	# 	except Exception as e:
	# 		print(f'{edffile} not there.')
	# 		print(e)
		
	# 	print(df)

	