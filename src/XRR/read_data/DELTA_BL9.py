import os, sys
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import argrelmin
import pandas as pd
import math 
from lmfit import minimize, Parameters
import re
from XRR.utilities.conversion_methods import kev2angst, th2q, q2th, inch2cm, cm2inch
from XRR.helpers import scan_numbers_arr, new_listdir
from XRR.utilities.file_operations import read_lines_file
from XRR.plotly_layouts import plotly_layouts, create_colormap
layout = plotly_layouts(transpPlot = False, transpPaper = False, unit_mode = 'slash', locale_settings='EN')

from shutil import move as smove, copy as scopy, copyfileobj as scopy_obj
from pathlib import Path
from PIL import Image
from datetime import datetime, timedelta
from time import perf_counter, sleep
import pickle
from collections import defaultdict
# plotting
from pprint import pprint
from matplotlib import pyplot as plt, patches, cm
from matplotlib.colors import LogNorm, rgb2hex
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Cursor
from itertools import count
import plotly.graph_objs as go
import plotly_express as px

class einlesenDELTA:
    '''
    Read XRR data recorded at DELTA BL9.
    '''
    def __init__(self, fiopath, tifpath, savepath, wavelength=None, prefix=None):
        '''
        create .dat file with q and counts of XRR experiment (at BL 9, DELTA). The arguments consist of the path, where the detectorimages are saved,
        the path of the fio-files, and the path where the .dat-files should be saved,

        parameters:
            * fp: path, where the fiofiles are saved:
            * tp: path, where the tiffiles are saved
            * sp: path, where you want to save your datafiles
            * start_scan: first scan number of your measurement
            * anz_scan: number of scans belonging to your measurement
            * roi: roi of your measurement
        optional parameters:
            *aborber_correc: correction of absorber, default is 1.
            * save: if true, '.dat' files will be saved in sp (default: False)
            * show: if true, a plot of the reflectivity curve will be displayed (default: False)
            * wavelength: wavelength in Angstrom of the xrays, if None 0.4592 A is used.
            * prefix: beginning of the savename of the files to be saved. After the prefix, the scan number of the first scan will be attached.
        '''
        self.fiopath = fiopath
        self.tifpath = tifpath
        self.savepath = savepath
        self.wavelength = wavelength
        self.prefix = prefix
        if not wavelength:
            self.wavelength = kev2angst(27)
            print(f'No wavelength was specified. \u03BB = {"{:.4f}".format(self.wavelength)} \u00C5 is used as default.')
        if not os.path.isdir(self.savepath):
            os.mkdir(self.savepath)
            print(f'Created {self.savepath}')
    
    @staticmethod
    def min_function(par, x0, y0, x1, y1, overlap_area, step = 1000):
        ver_shift = par['shift'].value
        qrange = overlap_area[1] - overlap_area[0]
        q_dist = qrange / step
        q_interp = np.arange(overlap_area[0], overlap_area[1], q_dist)

        y1_interp_func = interp1d(x1, y1, fill_value = 'extrapolate')
        ref_y1_interp = y1_interp_func(q_interp)

        y0_interp_func = interp1d(x0, y0, fill_value = 'extrapolate')
        ref_y0_interp = y0_interp_func(q_interp)
        
        y1_interp_shifted = [refy1 * ver_shift for refy1 in ref_y1_interp]

        residual = ((np.array(ref_y0_interp)- np.array(y1_interp_shifted))**2).sum()

        return residual


    def list_measurement_FIOS(self, scan_numbers):

        fiofiles = [f for f in os.listdir(self.fiopath) if all(x in f for x in [self.prefix, '.FIO'])]  
        measurement_files = []        
        for i, f in enumerate(fiofiles):
            for scan in scan_numbers:
                    string = int(re.split('_|\.', f)[-2])
                    if string == scan:
                        measurement_files.append(os.path.join(self.fiopath, f))
        measurement_files = sorted(
            measurement_files, key=lambda x: os.path.basename(x).partition('_')[-1])
        return measurement_files


    def StartTimeFIO(self, fio):
        lines = read_lines_file(fio)
        for index, l in enumerate(range(0, 7)):
            if 'date' in lines[index]:
                dateline = "".join(lines[index].split()).replace('date=','').replace('time=','')
                dummy = dateline.split('-')
                day = f'{int(dummy[0]):02}'
                month_name = dummy[1]
                datetime_object = datetime.strptime(month_name, "%b")
                month_number = f'{datetime_object.month:02d}'
                t = "".join(re.split('[0-9]{4}', dummy[2])[1].split(':'))
                year = re.split('[0-9]{2}:[0-9]{2}:[0-9]{2}', dummy[2])[0]
        startTime = year + month_number + day + t
        
        return startTime
    
    def EndTimeFIO(self, fio):
        startTime = self.StartTimeFIO(fio)
        date_obj_start = datetime.strptime(startTime, '%Y%m%d%H%M%S')
        df = self.readFioData2df(fio)
        scanDuration = len(df.steptime) * int(df.steptime[0])
        date_obj_end = timedelta(seconds = scanDuration)
        endTime_obj = date_obj_start + date_obj_end
        endTime = endTime_obj.strftime('%Y%m%d%H%M%S')
        return endTime

    def StartTimeFirstFIO(self, fio):
        TimeStampFile = self.StartTimeFIO(fio)
        date_obj_TimeStampFile = datetime.strptime(TimeStampFile, '%Y%m%d%H%M%S')
        df = self.readFioData2df(fio)
        scanDuration = len(df.steptime) * int(df.steptime[0])
        date_obj_scanDuration = timedelta(seconds = scanDuration)
        startTimeScan_obj = date_obj_TimeStampFile - date_obj_scanDuration
        StartTimeScan = startTimeScan_obj.strftime('%Y%m%d%H%M%S')
        return StartTimeScan

    def readFioData2df(self, fio):
        absorber, q, columnheaders, rangeparameters = defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict), 
        q_dummy, th, line_counter, I0, dataLineCounter, normfactor= [], [], [], [], [], []
        I_roi1, I_roi2, I_roi3, I_roi1_with_bg, I_background, absorber_factor, steptime, tif_df, tifPref = [], [], [], [], [], [], [], [], []
        current_scan = str(fio.split('_')[-1].strip('.FIO').strip('_'))
        filepref = os.path.basename(fio).strip('.FIO').strip(current_scan)
        current_file = filepref + current_scan
        tifprefix = f'{os.path.join(self.tifpath, self.prefix)}{current_scan}'
        lines = read_lines_file(fio)
        counter = 0
        for index, l in enumerate(range(7, len(lines))):
            if not r'%d' in lines[l] and counter == 0:
                if ' = ' in lines[l]:
                    dummy = lines[l].strip('\n').split(' = ')
                    parameter = dummy[0].replace(' ', '')
                    if parameter == 'SAMPLE_TIME':
                        steptimeDummy = dummy[1].replace(' ', '')
                    value = dummy[1].replace(' ', '')
                    rangeparameters.update({parameter: value})
            elif r'%d' in lines[l]:
                counter += 1
            elif not r'%d' in lines[l] and not counter == 0:
                if 'Col' in lines[l]:
                    dummy2 = lines[l].strip('\n').replace('Col', '').split()
                    colnumber = dummy2[0]
                    colvalue = ' '.join(dummy2[1:])
                    columnheaders.update({colnumber: colvalue})
                else:
                    dataLineCounter.append(index)
                    data = lines[l].strip('\n').split()
                    data = [float(d) for d in data]
                    tifPref.append(tifprefix)
                    th_dummy = data[0]
                    th.append(th_dummy)
                    I0_dummy = data[3]
                    I0.append(I0_dummy)
                    q_dummy.append(th2q(self.wavelength, th_dummy))
                    absorber_factor.append(data[16])
                    # normfactor.append(float(data[16]) / float(steptimeDummy) / absorber_correc[i])
                    steptime.append(steptimeDummy)
        dataLineCounter = np.arange(1, len(dataLineCounter) + 1,1)
        dataLineCounter = [str("{:05d}".format(j)) + '.tif' for j in dataLineCounter]
        df = pd.DataFrame(list(zip(q_dummy, th, absorber_factor, steptime, tifPref)), 
            columns= ['q', 'theta', 'absorber_factor','steptime','tif'])
        df.tif = [f'{pref}_{t}' for pref, t in zip(df.tif, dataLineCounter)]
        return df


    @staticmethod
    def tif2array(f):
        im = Image.open(f)
        return np.array(im)

    def calculate_vals_from_tif(*args):
        bg1_counts = np.sum(imarray[bg_left[0]:bg_left[1], bg_left[2]:bg_left[3]], axis=(0,1)) * self.df_unsort['normfactor'][i]
        bg2_counts = np.sum(imarray[bg_right[0]:bg_right[1], bg_right[2]:bg_right[3]], axis=(0,1)) * self.df_unsort['normfactor'][i]
        roi_counts = np.sum(imarray[roi[0]:roi[1], roi[2]:roi[3]], axis=(0,1)) * self.df_unsort['normfactor'][i]
        mw = roi_counts - (bg1_counts + bg2_counts)
        mw_bg = bg1_counts + bg2_counts
        mw_uncorr = roi_counts
        if not self.df_unsort.normfactor[i] == 0:
            mw_unnorm = roi_counts / self.df_unsort.normfactor[i] - (bg1_counts / self.df_unsort.normfactor[i] + bg2_counts / self.df_unsort.normfactor[i]) 
            if mw_unnorm <= 0: mw_unnorm = 1e-10
            mw_bg_unnorm = bg1_counts / self.df_unsort.normfactor[i] + bg2_counts / self.df_unsort.normfactor[i]
            if mw_bg_unnorm <= 0: mw_bg_unnorm = 1e-10
            mw_unnorm_uncorr = roi_counts / self.df_unsort.normfactor[i]
            if mw_unnorm_uncorr <= 0: mw_unnorm_uncorr = 1e-10    
        else: mw_unnorm, mw_bg_unnorm, mw_unnorm_uncorr = np.nan, np.nan, np.nan

        fill_elements = pd.Series({'mw':mw, 'uncorr':mw_uncorr, 'detector_counts':mw_unnorm, 'bg':mw_bg, 'bg_unnorm':mw_bg_unnorm,
            'detector_counts_uncorrected':mw_unnorm_uncorr})
    
    def read_XRR(self, start_scan_number, anz_scans, absorber_correc=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), roi=None, save=False, show=False,
                 background=True, error_plot=False, bgRoiDist = 1,readtime = False, weight_on_error_plot = False, pre_peak_eval = False,
                 norm_to_I0 = False, calc_mean_temp = False, return_dataframe = False, plot_seperate_loops = False, fit_absorber_correction = False,
                 counts2consider = 'mw', verbose=False):
        """
        parameters:
            * start_scan_number: The number of the first scan of the measurement
            * anz_scans: The total number of scans performed for the reflectivity curve
        optional parameters:
            * absorber_correc: the pilatus detector has a timedependent offset. Use factors to shift parts of the XRR curve to match each other
            * roi: region of interest of the measurement(rectangle). Can't be changed (yet) between the scans
                    roi[0]: horizontal lower pixel
                    roi[1]: horizontal upper pixel
                    roi[2]: vertical lower pixel
                    roi[3]: vertical upper pixel
            * save: saves a q-counts-file in the savepath
            * background: if True, two areas beside the roi are defined and the measured counts substracted from the counts in the roi             
            * error_plot: if True, the detector counts of thesingle parts of the reflectivity are plotted uncorrected with error bars
            * weight_on_error_plot: Default False. If True, the absoulte counts with errors will be used to weight the datapoints.
                err <= 0.1 * counts --> weight = 1
                0.1 *counts < err < 0.5 * counts --> weight = 0.5
                0.5 * counts <= err <= 0.8 * counts --> weight = 0.2
                0.8 * counts < err < 1.0 * counts --> weight = 0.1
                err >= 1.0 * counts --> weight = 0
        """

        ret_data = {'data':None, 'data_figure': None, 'absorber': None, 'error_figure':None, 'sep_loop_figure': None, 'data_unsort': None}

        tstart = perf_counter()
        bg_width = (roi[1] - roi[0]) / 2
        bg_left = np.array( 
            [math.floor(roi[0] - bgRoiDist - bg_width), roi[0] - bgRoiDist, roi[2], roi[3]])
        bg_right = np.array(
            [roi[1] + bgRoiDist, math.ceil(roi[1] + bgRoiDist + bg_width), roi[2], roi[3]])

        scan_numbers = scan_numbers_arr(start_scan = start_scan_number, anz_scans = anz_scans)
        ret_data['scan_numbers'] = scan_numbers
        norm_factor, dataframes = [], []

        measurement_files = self.list_measurement_FIOS(scan_numbers = scan_numbers)
        

        qual_tab10 = cm.get_cmap('tab10', anz_scans)
        qual_tab10 = [rgb2hex(c) for c in qual_tab10.colors]
        
        if fit_absorber_correction:
            for i in range(len(absorber_correc)):
                absorber_correc[i] = 1
        for i, fio in enumerate(measurement_files):
            single_dfs = self.readFioData2df(fio = fio)
            single_dfs['normfactor'] = [float(af) / float(st) for af, st in zip(single_dfs.absorber_factor,single_dfs.steptime)] 
            single_dfs['normfactor'] /= absorber_correc[i]
            single_dfs['absorber_correc'] = np.repeat(absorber_correc[i], len(single_dfs.absorber_factor))
            single_dfs['loop'] = i
            dataframes.append(single_dfs)
        self.df_unsort = pd.concat(dataframes, ignore_index = True)
       
        t1 = perf_counter()  
        shape = len(self.df_unsort.q)
        mw_arr, mw_uncorr_arr, mw_unnorm_arr, mw_bg_arr, mw_bg_unnorm_arr, mw_unnorm_uncorr_arr = np.empty(shape=shape), np.empty(shape = shape), np.empty(shape=shape), np.empty(shape=shape), np.empty(shape = shape), np.empty(shape=shape)

        for i in range(len(self.df_unsort.q)):
            try:
                imarray = self.tif2array(self.df_unsort.tif[i])
                sumtime1 = perf_counter()
                bg1_counts = np.sum(imarray[bg_left[0]:bg_left[1], bg_left[2]:bg_left[3]], axis=(0,1)) * self.df_unsort['normfactor'][i]
                bg2_counts = np.sum(imarray[bg_right[0]:bg_right[1], bg_right[2]:bg_right[3]], axis=(0,1)) * self.df_unsort['normfactor'][i]
                roi_counts = np.sum(imarray[roi[0]:roi[1], roi[2]:roi[3]], axis=(0,1)) * self.df_unsort['normfactor'][i]
        
                mw = roi_counts - (bg1_counts + bg2_counts)
                mw_bg = bg1_counts + bg2_counts
                mw_uncorr = roi_counts
                if not self.df_unsort.normfactor[i] == 0:
                    mw_unnorm = roi_counts / self.df_unsort.normfactor[i] - (bg1_counts / self.df_unsort.normfactor[i] + bg2_counts / self.df_unsort.normfactor[i]) 
                    if mw_unnorm <= 0: mw_unnorm = 1e-10
                    mw_bg_unnorm = bg1_counts / self.df_unsort.normfactor[i] + bg2_counts / self.df_unsort.normfactor[i]
                    if mw_bg_unnorm <= 0: mw_bg_unnorm = 1e-10
                    mw_unnorm_uncorr = roi_counts / self.df_unsort.normfactor[i]
                    if mw_unnorm_uncorr <= 0: mw_unnorm_uncorr = 1e-10    
                else: mw_unnorm, mw_bg_unnorm, mw_unnorm_uncorr = np.nan, np.nan, np.nan

                mw_arr[i], mw_bg_arr[i], mw_uncorr_arr[i], mw_unnorm_arr[i], mw_bg_unnorm_arr[i], mw_unnorm_uncorr_arr[i] = mw, mw_bg, mw_uncorr, mw_unnorm, mw_bg_unnorm, mw_unnorm_uncorr
            except Exception as e:
                print(f'{self.df_unsort.tif[i]} not there or damaged. Value with index {i} for q = ' + "{:.4f}".format(self.df_unsort.q[i]) + ' \u00C5' +  ' will be skipped.')
                mw_arr[i], mw_bg_arr[i], mw_uncorr_arr[i], mw_unnorm_arr[i], mw_bg_unnorm_arr[i], mw_unnorm_uncorr_arr[i] = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        columns, values = ['mw', 'bg', 'uncorr', 'detector_counts', 'bg_unnorm', 'detector_counts_uncorrected'], [mw_arr, mw_bg_arr, mw_uncorr_arr, mw_unnorm_arr, mw_bg_unnorm_arr, mw_unnorm_uncorr_arr]
        col_val_mapping = dict(zip(columns, values))
        self.df_unsort = self.df_unsort.assign(**col_val_mapping)

        self.df_unsort = self.df_unsort.dropna()
        self.df_unsort['counts2consider'] = self.df_unsort[counts2consider]

        # reset index to start from zero if there were missing tif-files. Otheriwse fit_absorber_correction does not work.
        self.df_unsort = self.df_unsort.reset_index()        
        
        if fit_absorber_correction:
            par = Parameters()
            par.add('shift', value=1.1)
            absorber_shifts = list()
            idx = 0
            while idx in range(len(self.df_unsort.theta)):
                if 0 <= idx < len(self.df_unsort.theta) -1:
                    if self.df_unsort.theta[idx] >= self.df_unsort.theta[idx + 1]:
                        loop_val = self.df_unsort.at[idx, 'loop']
                        next_loop_val = loop_val + 1
                        first_loop_values = self.df_unsort[self.df_unsort.loop == loop_val]
                        next_loop_values = self.df_unsort[self.df_unsort.loop == next_loop_val]
                        overlap_area = [next_loop_values.q.iloc[0], first_loop_values.q.iloc[-1]]
                        first_loop_overlap_df = first_loop_values.loc[first_loop_values['q'].between(overlap_area[0], overlap_area[1], inclusive='both'), first_loop_values.columns.tolist()]
                        next_loop_overlap_df = next_loop_values.loc[next_loop_values['q'].between(overlap_area[0], overlap_area[1], inclusive='both'), next_loop_values.columns.tolist()]
                        if len(next_loop_overlap_df.q) < 2 and not len(first_loop_overlap_df.q) < 2:
                            if verbose:
                                print(f'Between loops {loop_val} - {next_loop_val}: Second loop has not enough overlapping points')
                            q_first_index = next_loop_overlap_df.q.index[0]
                            q_first = self.df_unsort.q.iloc[q_first_index]
                            q_second = self.df_unsort.q.iloc[q_first_index + 1]
                            counts2consider_first = self.df_unsort.counts2consider.iloc[q_first_index]
                            counts2consider_second = self.df_unsort.counts2consider.iloc[q_first_index + 1]
                            if overlap_area[0] == overlap_area[1]: overlap_area = [q_first, q_second]
                            # print(self.df_unsort.counts2consider.iloc[q_first_index], self.df_unsort.counts2consider.iloc[q_first_index + 1])
                            result = minimize(fcn = einlesenDELTA.min_function, params = par,
                                args = (first_loop_overlap_df.q, first_loop_overlap_df.counts2consider, [q_first, q_second], [counts2consider_first, counts2consider_second], overlap_area),
                                kws = {'step':20}, method = 'leastsq')
                        # print(f'{loop_val}: first_loop_overlap: {len(first_loop_overlap_df.q)} points, next_loop_overlap: {len(next_loop_overlap_df.q)} points')
                        
                        elif len(first_loop_overlap_df.q) < 2 and not len(next_loop_overlap_df.q) <2:
                            if verbose:
                                print(f'Between loops {loop_val} - {next_loop_val}: First loop has not enough overlapping points')
                            q_second_index = first_loop_overlap_df.q.index[0]
                            q_second = self.df_unsort.q.iloc[q_second_index]
                            q_first = self.df_unsort.q.iloc[q_second_index - 1]
                            counts2consider_first = self.df_unsort.counts2consider.iloc[q_second_index - 1]
                            counts2consider_second = self.df_unsort.counts2consider.iloc[q_second_index]
                            if overlap_area[0] == overlap_area[1]: overlap_area = [q_first, q_second]
                            result = minimize(fcn = einlesenDELTA.min_function, params = par,
                                args = (first_loop_overlap_df.q, first_loop_overlap_df.counts2consider, [q_first, q_second], [counts2consider_first, counts2consider_second], overlap_area),
                                kws = {'step':20}, method = 'leastsq')
                        
                        elif len(next_loop_overlap_df.q) < 2 and len(first_loop_overlap_df.q) <2:
                            if verbose:
                                print(f'Between loops {loop_val} - {next_loop_val}: Both loops have not enough overlapping points')
                            # first_loop:
                            q_second_index_fl = first_loop_overlap_df.q.index[0]
                            q_second_fl = self.df_unsort.q.iloc[q_second_index_fl]
                            q_first_fl = self.df_unsort.q.iloc[q_second_index_fl - 1]
                            counts2consider_first_fl = self.df_unsort.counts2consider.iloc[q_second_index_fl - 1]
                            counts2consider_second_fl = self.df_unsort.counts2consider.iloc[q_second_index_fl]

                            # second loop
                            q_first_index_nl = next_loop_overlap_df.q.index[0]
                            q_first_nl = self.df_unsort.q.iloc[q_first_index_nl]
                            q_second_nl = self.df_unsort.q.iloc[q_first_index_nl + 1]
                            counts2consider_first_nl = self.df_unsort.counts2consider.iloc[q_first_index_nl]
                            counts2consider_second_nl= self.df_unsort.counts2consider.iloc[q_first_index_nl + 1]
                            if overlap_area[0] == overlap_area[1]: overlap_area = [q_first_fl, q_second_nl]
                            result = minimize(fcn = einlesenDELTA.min_function, params = par,
                                args = ([q_first_fl, q_second_fl], [counts2consider_first_fl, counts2consider_second_fl], 
                                    [q_first_nl, q_second_nl], [counts2consider_first_nl, counts2consider_second_nl], overlap_area),
                                kws = {'step':20}, method = 'leastsq')

                        else:
                            result = minimize(fcn = einlesenDELTA.min_function, params = par,
                                args = (first_loop_overlap_df.q, first_loop_overlap_df.counts2consider, next_loop_overlap_df.q, next_loop_overlap_df.counts2consider, overlap_area),
                                kws = {'step':20}, method = 'leastsq')
                            # report_fit(result)
                        absorber_shift = result.params.valuesdict()['shift']
                        absorber_shifts.append(absorber_shift)
                        self.df_unsort.loc[self.df_unsort.loop == next_loop_val, 'counts2consider'] = self.df_unsort.counts2consider * absorber_shift
                        # self.df_unsort.loc[self.df_unsort.loop == next_loop_val, 'counts2consider'] = self.df_unsort.counts2consider * absorber_shift
                        for col in self.df_unsort[['mw', 'bg','uncorr', 'detector_counts']].columns:
                            self.df_unsort.loc[self.df_unsort.loop == next_loop_val, col] = self.df_unsort[col] * absorber_shift
                idx += 1
            absorber_text = ["{:.2f}".format(a) if a >=0.01 else "{:.2e}".format(a) for a in absorber_shifts]
            absorber_text.insert(0,1)
            if verbose:
                print(f'Loops shifted with {absorber_text}')
            ret_data['absorber'] = absorber_text
   
        

###########################################################
        self.df = self.df_unsort.sort_values(by = 'q')#####
###########################################################




        t2 = perf_counter()
        tiftime = t2 - t1
        if readtime:
            print(f'Read TIF-images in ' + "{:.4f}".format(tiftime) +' seconds.')

        if pre_peak_eval:
            counts = np.array(self.df.mw)
            firstmin = argrelmin(counts, order=8)[0][0]
            truncate_indices = list(range(0, firstmin + 1))
            self.df = self.df.drop(truncate_indices).reset_index(drop=True)

        if plot_seperate_loops:
            qual_tab10 = cm.get_cmap('tab10', len(measurement_files))
            qual_tab10 = [rgb2hex(c) for c in qual_tab10.colors]
            single_loops, traces = dict(), list()
            # [print(q, th, ap, st) for q, th, ap, st in zip(self.df_unsort.q, self.df_unsort.th, self.df_unsort.atten_position, self.df_unsort.steptime)]
            for i in range(0, len(measurement_files)):
                single_loops[i] = self.df_unsort[['q', 'theta', 'mw', 'steptime', 'absorber_factor']][self.df_unsort.loop == i]
                customdata = np.stack((single_loops[i]['steptime'], single_loops[i]['absorber_factor'], single_loops[i]['theta']), axis = -1)
                traces.append(go.Scatter(x = single_loops[i]['q'], y = single_loops[i]['mw'], mode = 'lines+markers',
                    marker = dict(color = qual_tab10[i]),
                    line = dict(width = 4, color = qual_tab10[i]), customdata = customdata, name = f'loop {i}',
                    hovertemplate = '<b>q</b> = %{x:.3f} <br><b>th</b> = %{customdata[2]:.3f}<br><b>I</b> = %{y:.2f}<br><b>t</b> = %{customdata[0]:.1f}<br><b>atten_pos</b> = %{customdata[1]:.1f}'
                    ))
            fig_sep_loops = go.Figure(data = traces, layout = layout.refl())
            ret_data['sep_loop_figure'] = fig_sep_loops
            # fig_sep_loops.show()


        if show:
            traces = []
            reflplotlay = layout.refl()
            if background:
                trace1 = go.Scatter(x = self.df['q'], y = self.df['uncorr'], mode = 'lines+markers', name = 'uncorrected')
                trace2 = go.Scatter(x = self.df['q'], y = self.df['bg'], mode = 'lines+markers', name = 'background')
                trace3 = go.Scatter(x = self.df['q'], y = self.df['mw'], mode = 'lines+markers', name = 'corrected')

                # dummy = go.Scatter(x = self.df['q'], y = self.df['mw'], mode = 'none', yaxis = 'y2', showlegend = False, hoverinfo = 'skip')
                # tts = [trace1, trace2, trace3, dummy]
                tts = [trace1, trace2, trace3]
                fig = go.Figure(data=tts, layout=layout.refl())
                # fig.update_layout(legend_title= 'scan ' + str(scan_numbers[0]) + '-' + str(scan_numbers[-1]), legend_title_font_size = 16)
            else:
                trace1  = go.Scatter(x = self.df['q'], y = self.df['uncorr'], mode = 'lines+markers', name = 'data', showlegend=True)
                # dummy = go.Scatter(x = self.df['q'], y = self.df['uncorr'], mode = 'none', yaxis = 'y2', showlegend = False, hoverinfo = 'skip')
                # tts = [trace1, dummy]
                tts = [trace1]
                fig = go.Figure(data = tts, layout = layout.refl())
            min_x, max_x = min(self.df.q), max(self.df.q) + 0.05
            min_y, max_y = min(self.df.uncorr) - 10, max(self.df.uncorr) + 10
            fig.update_layout(title = None,
                legend = dict(title_text =  'scan ' + str(scan_numbers[0]) + '-' + str(scan_numbers[-1])),
                xaxis = dict(range = [min_x, max_x]),
                yaxis = dict(range = [min_y, max_y]),
                hovermode='x', yaxis_range = [0,9])
            # fig.show()
            ret_data['data_figure'] = fig
# This should be backed up!!
        if error_plot:
            t1 = perf_counter()
            start_scan = os.path.basename(self.df_unsort['tif'][0]).strip('.tif').split('_')[-2].lstrip('0')
            end_scan = os.path.basename(self.df_unsort['tif'][len(self.df_unsort['q']) - 1]).strip('.tif').split('_')[-2].lstrip('0')
            sample_system = os.path.basename(self.df_unsort['tif'][0]).strip('.tif').split('_')
            sample_system = ' '.join(sample_system[0:-2])

            error_raw = np.sqrt(self.df_unsort['detector_counts_uncorrected'])
            error_bg = np.sqrt(self.df_unsort['bg_unnorm'])
            error_diff = np.sqrt(self.df_unsort['bg_unnorm'] + self.df_unsort['detector_counts_uncorrected'])
            th_arr = np.array(self.df_unsort.theta)
            customdata = np.stack((error_raw, error_bg, error_diff, th_arr), axis = -1)
            rawdata = go.Scatter(x = self.df_unsort['q'], y = self.df_unsort['detector_counts_uncorrected'], customdata = customdata,
                    error_y = dict(array = error_raw,visible=True), mode='markers',name = 'raw data',
                    hovertemplate = '<b>q</b> = %{x:.3f} <br><b>th</b> = %{customdata[3]:.3f}<br><b>I</b> = %{y:} <span>&#177;</span> %{customdata[0]:.2f}')

            raw_bg =  go.Scatter(x = self.df_unsort['q'],y = self.df_unsort['bg_unnorm'], customdata = customdata,
                error_y = dict(array = error_bg, visible=True),mode='markers', name = 'raw background',
                hovertemplate = '<b>q</b> = %{x:.3f} <br><b>th</b> = %{customdata[3]:.3f}<br><b>I</b> = %{y:} <span>&#177;</span> %{customdata[1]:.2f}')

            raw_diff = go.Scatter(x = self.df_unsort['q'], y = self.df_unsort['detector_counts_uncorrected'] - self.df_unsort['bg_unnorm'] , customdata = customdata,
                error_y = dict(array = error_diff, visible=True), mode='markers', name = 'raw - raw background',
                hovertemplate = '<b>q</b> = %{x:.3f} <br><b>th</b> = %{customdata[3]:.3f} <br><b>I</b> = %{y:} <span>&#177;</span> %{customdata[2]:.2f}')

            fig_err = go.Figure(data = [rawdata, raw_bg, raw_diff], layout = layout.refl())
            max_y= max(self.df_unsort['detector_counts_uncorrected'])
            max_y, min_y = np.ceil(np.log10(max_y)), -1
            max_x, min_x = max(self.df_unsort.q) +0.05, 0
            fig_err.update_layout(showlegend=True, legend_title='', legend_borderwidth=0.0,
                title=dict(text = f'{sample_system} scan {start_scan} - {end_scan}'),
                yaxis = dict(range = [min_y, max_y]),
                xaxis = dict(range = [min_x, max_x]))
            ret_data['error_figure'] = fig_err
            # fig_err.show()
        t2 = perf_counter()
        error_time = t2 - t1
        if readtime: print(f'Created error plot in {error_time}')
        if weight_on_error_plot:
            if background:
                error_diff_y = abs(np.sqrt((self.df['bg_unnorm'] + self.df['detector_counts_uncorrected']).tolist()) / self.df['detector_counts'].tolist())
            else:
                error_diff_y = abs((np.sqrt(self.df['detector_counts_uncorrected'].tolist())) / self.df['detector_counts_uncorrected'].tolist())
            
            weights = []
            for i, err in enumerate(error_diff_y):
                if background:
                    counts2consider = self.df['detector_counts'].tolist()
                else:
                    counts2consider = self.df['detector_counts_uncorrected'].tolist()
                if not counts2consider[i] < 0.00000:
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

            self.df['weights'], self.df['errors'] = weights, error_diff_y
        # if pre_peak_eval:
        #     counts = np.array(self.df.mw)
        #     firstmin = argrelmin(counts, order=8)[0][0]
        #     truncate_indices = list(range(0, firstmin + 1))
        #     self.df = self.df.drop(truncate_indices).reset_index(drop=True)

        if save:

            # filename =self.savepath + '/' + measurement_files[0].split('/')[-1].strip('.fio').strip('.FIO') + '.dat'
            filename = os.path.join(self.savepath, os.path.basename(measurement_files[0]).strip('.fio').strip('.FIO') + '.dat')
            if weight_on_error_plot:
                df2save = pd.DataFrame(list(zip(self.df.q, self.df.mw, self.df.weights)), columns = ['q', 'mw', 'weights'])
                # print('with errors')
            else:
                df2save = pd.DataFrame(list(zip(self.df.q, self.df.mw)))
            df2save.to_string(filename, header=False, index=False)
            print('File ' + filename + ' has been saved.')
        ret_data['data']= self.df
        ret_data['data_unsort'] = self.df_unsort


        # if pickle_data:
        #     pickle_file = os.path.join(self.savepath , f'{start_scan_number}_pklfile.pickle')
        #     obj = ret_data
        #     file_obj = open(pickle_file, 'wb')
        #     pickle.dump(obj, file_obj)
        #     file_obj.close()
        
        return ret_data

    def save_data_to_pickle(self, obj, fname = None):
        if not fname:
            print('no name given')
            name =  'pickle_file'
        else:
            name = fname

        pickle_file = os.path.join(self.savepath , f'{fname}.pkl')
        file_obj = open(pickle_file, 'wb')
        pickle.dump(obj, file_obj)
        file_obj.close()

    @staticmethod
    def load_from_pickle(file):
        file_obj = open(file, 'rb')
        data = pickle.load(file_obj)
        file_obj.close()
        return data

    def calc_mean_temp(self, tempfile, start_scan, anz_scans, concat_files = False):
        ''' Calculate mean temperature during XRR scan from file

        '''
        if concat_files:
            tempData_single = []
            filepath = os.path.dirname(tempfile)
            tempfiles = [os.path.join(filepath, t) for t in os.listdir(filepath) if t.endswith('.txt')]
            for t in tempfiles:
                tempData_single.append(pd.read_csv(t, header = 1, names = ['date', 'volts', 'temp']))
            tempData = pd.concat(tempData_single, ignore_index = True)
        else:
            tempData =pd.read_csv(tempfile, header = 1, names = ['date', 'volts', 'temp'])

        tempData.date = [str(d) for d in tempData.date]
        tempData['date_ints'] = [int(d) for d in tempData.date]
        tempData['date_objects'] = [datetime.strptime(d,'%Y%m%d%H%M%S') for d in tempData.date]
        tempData['date_pd'] = [pd.to_datetime(do) for do in tempData.date_objects]        

        scan_numbers = scan_numbers_arr(start_scan= start_scan, anz_scans = anz_scans)
        measurement_files = self.list_measurement_FIOS(scan_numbers = scan_numbers)  
        
        # starttime = self.StartTimeFIO(measurement_files[0])
        starttime = self.StartTimeFirstFIO(measurement_files[0])
        starttime_obj = datetime.strptime(starttime,'%Y%m%d%H%M%S')
        starttime_pd = pd.to_datetime(starttime_obj)
        
        # endtime = self.EndTimeFIO(measurement_files[-1])
        endtime = self.StartTimeFIO(measurement_files[-1])
        endtime_obj = datetime.strptime(endtime,'%Y%m%d%H%M%S')
        endtime_pd = pd.to_datetime(endtime_obj)

        tempData_trunc = tempData[(tempData['date_pd'] > starttime_pd) & (tempData['date_pd'] < endtime_pd)]
        mean_temp = np.mean(tempData_trunc.temp)
        std = np.std(tempData_trunc.temp)
        mean_temp = np.round(mean_temp, 3)
        std = np.round(std, 3)
        # print(f'starttime: {starttime_pd}, endtime: {endtime_pd}')
        return mean_temp, std
                
    def dectimage(self, scan_number, th_ind, roi=None, bgRoiDist = 1, margin=10, series=False, sleeptime=0.1, save_imgs = False, return_fig = False,
        axis = 'off', figsize = None, suptitle = False, show = True, saveFig = False, rotate_axes=True):
        '''
        function shows the detectorimage. The scan number and the number of the measurement point (value) are needed. 
        -----------
        Parameters:
            scan_number: number of scan of the detectorimage, you want to look at
            th_ind: index of scan angle
            roi: Region of interest
            margin: area around the roi you want to see
            series: boolean, if True, all scans with the chosen scan number will be shown one after another
            sleeptime: time between the update in a series
        '''
        if not figsize:
            figsize = (800/109, 566/109)
        measurement_file = self.list_measurement_FIOS(scan_numbers = [scan_number])[0]
        df = self.readFioData2df(fio = measurement_file)
        imarrays = list()
        for i in range(len(df.tif)):
            imarray = self.tif2array(df.tif[i])
            if imarray.size == 1:
                imarray = np.nan
            else:
                imarray = imarray
            imarrays.append(imarray)
        df['imarrays'] = imarrays
        df = df.dropna().reset_index(drop = True)

        if save_imgs:      
            # tifnames = [t.split('/')[-1] for t in tifs_sorted]
            tifnames = [os.path.basename(t) for t in df.tif]
            imgs2save = [os.path.join(self.savepath, t.replace('.tif', '.png').replace('.TIF', '.png')) for t in tifnames]

        if not roi is None:
            roi_height = roi[1] - roi[0]
            roi_width = roi[3]-roi[2]

            bg_height = (roi[1] - roi[0]) / 2
            
            bg_left = np.array([roi[2], roi[0] - bgRoiDist - bg_height, roi[3], roi[0] - bgRoiDist])

            bg_right = np.array([roi[2], roi[1] + bgRoiDist, roi[3], roi[1] + bgRoiDist + bg_height])

        if series:
            plt.ion()
            f = plt.figure(figsize = figsize)
            ax = f.gca()
            ax.set(xlim=[roi[2] - margin, roi[3] + margin], ylim=[roi[0] - margin, roi[1] + margin])
            if axis == 'off':
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

            draw_roi = patches.Rectangle((roi[2], roi[0]),roi_width, roi_height, fc='None', ec='g', lw=2)
            draw_bgl = patches.Rectangle((bg_left[0], bg_left[1]),roi_width, bg_height, fc='None', ec='r', lw=2)
            draw_bgr = patches.Rectangle((bg_right[0], bg_right[1]),roi_width, bg_height, fc='None', ec='r', lw=2)
            rect = ax.add_patch(draw_roi)
            rect_bgl = ax.add_patch(draw_bgl)
            rect_bgr = ax.add_patch(draw_bgr)

            im = ax.imshow(df.imarrays.loc[0], norm = LogNorm())
            for i in range(1, len(df.tif)):
                im.set_data(df.imarrays.loc[i])
                if suptitle:
                    f.suptitle('Detector image of scan ' + str(scan_number) + ' th = ' + "{:.3f}".format(df.theta.loc[i]) + '°\nq = ' + "{:.3f}".format(th2q(self.wavelength, df.theta.loc[i])) + r' $\AA^{-1}$', fontsize=16)
                    ax.set_xlabel('x [px]')
                    ax.set_ylabel('y [px]')
                f.canvas.flush_events()
                plt.pause(sleeptime)
                plt.tight_layout()
            plt.close()
        else:
            imarray = self.tif2array(df.tif.loc[th_ind])
            f = plt.figure(figsize = figsize)
            ax = f.add_subplot(111)
            if axis == 'off':
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            im = ax.imshow(imarray, norm = LogNorm())
            if suptitle:
                f.suptitle('Detector image, scan: ' + str(scan_number) + ' th = ' + "{:.4f}".format(df.theta.loc[th_ind]) + '°\nq = ' + "{:.3f}".format(th2q(self.wavelength, df.theta.loc[th_ind])) + r' $\AA^{-1}$', fontsize=16)
            ax.set(xlim=[roi[2] - margin, roi[3] + margin], ylim=[roi[0] - margin, roi[1] + margin])
            draw_roi = patches.Rectangle((roi[2], roi[0]),roi_width, roi_height, fc='None', ec='g', lw=2)
            draw_bgl = patches.Rectangle((bg_left[0], bg_left[1]),roi_width, bg_height, fc='None', ec='r', lw=2)
            draw_bgr = patches.Rectangle((bg_right[0], bg_right[1]),roi_width, bg_height, fc='None', ec='r', lw=2)
            rect = ax.add_patch(draw_roi)
            rect_bgl = ax.add_patch(draw_bgl)
            rect_bgr = ax.add_patch(draw_bgr)
            plt.tight_layout()
            if show:
                plt.show()
            if saveFig:
                fname = os.path.join(self.savepath, f'{"{:.3f}".format(th2q(self.wavelength, df.theta.loc[th_ind]))}.png')
                f.savefig()
        if return_fig:
            return f
   
    def dectimage2(self, scan_number, th_ind, roi=None, bgRoiDist = 2, pref_tif='run07_20', margin=10, series=False, sleeptime=0.1,
            save_imgs = False, detector='p100k', return_figure = False):
        '''
        function shows the detectorimage. The scan number and the number of the measurement point (value) are needed. 
        -----------
        Parameters:
            scan_number: number of scan of the detectorimage, you want to look at
            th_ind: index of scan angle
            roi: Region of interest
            pref_tif: string, prefix of the scan files, e.g. 'run11_19'
            margin: area around the roi you want to see
            series: boolean, if True, all scans with the chosen scan number will be shown one after another
            sleeptime: time between the update in a series
            bgRoiDist: Distance in pixel between the background ROIs and the counting ROI. Default is 2.
        '''
        measurement_file = self.list_measurement_FIOS(scan_numbers = [scan_number])[0]
        df = self.readFioData2df(fio = measurement_file)        
        df = df.dropna().reset_index(drop= True)

        if save_imgs:               
            tifnames = [t.split('/')[-1] for t in tifs_sorted]
            imgs2save = [os.path.join(self.savepath, t.replace('.tif', '.png').replace('.TIF', '.png')) for t in tifnames]
        # # roi = np.array([237, 249, 101, 105])
        if not roi is None:
            roi_height = roi[1] - roi[0]
            roi_width = roi[3]-roi[2]

            bg_height = (roi[1] - roi[0]) / 2
            bg_left = np.array([roi[2], roi[0] - bgRoiDist - bg_height, roi[3], roi[0] - bgRoiDist])
            bg_right = np.array([roi[2], roi[1] + bgRoiDist, roi[3], roi[1] + bgRoiDist + bg_height])

            roi_rect = go.layout.Shape(type = 'rect', x0 = roi[2], x1 = roi[3], y0 = roi[0], y1 = roi[1], line = dict(width = 3, color = 'green'))
            bgl_rect = go.layout.Shape(type = 'rect', x0 = bg_left[2], y0 = bg_left[3], x1 = bg_left[0], y1 = bg_left[1], line = dict(width = 3, color = 'red'))
            bgr_rect = go.layout.Shape(type = 'rect', x0 = bg_right[2], y0 = bg_right[3], x1 = bg_right[0], y1 = bg_right[1], line = dict(width = 3, color = 'red'))

            if not series:
                tiffile = df.tif.loc[th_ind]
                imarray = self.tif2array(tiffile)
                # imarray = df.imarrays.loc[th_ind]
                log_imarray =  np.where(imarray > 0.00, np.log10(np.clip(imarray, a_min=1e-10, a_max=None)), 0.0)
                fig = px.imshow(log_imarray,
                    labels = dict(x ='<i>x</i>&#8201;[px]',y = '<i>y</i>&#8201;[px]'),
                    width = 800, height = 566,
                    origin = 'lower',
                    aspect = 'auto',
                    # xaxis = dict(range = [roi[0]-margin, roi[1]+margin]),
                    # yaxis = dict(range = [roi[2]-margin, roi[3]+margin]), 
                    )
                # fig.update_xaxes(range = [roi[0]-margin, roi[1]+margin])
                # fig.update_yaxes(range = [roi[2]-margin, roi[3]+margin], autorange = False)
                # fig.update_xaxes(range = [roi[0]-margin, roi[1]+margin])
                # fig.update_yaxes(range = [roi[2]-margin, roi[3]+margin], autorange = False)
                title_text = f'Detector image of scan: {scan_number}<br>\u03B8 = {df.theta.loc[th_ind]:.3f}&#8201;°; <i>q</i><sub>z</sub> = {th2q(self.wavelength, df.theta.loc[th_ind]):.3f}&#8201;\u212B '
                # roi_rect = go.layout.Shape(type = 'rect', x0 = roi[0], x1 = roi[1], y0 = roi[2], y1 = roi[3], line = dict(width = 3, color = 'green'))
                # bgl_rect = go.layout.Shape(type = 'rect', x0 = bg_left[0], x1 = bg_left[1], y0 = bg_left[2], y1 = bg_left[3], line = dict(width = 3, color = 'red'))
                # bgr_rect = go.layout.Shape(type = 'rect', x0 = bg_right[0], x1 = bg_right[1], y0 = bg_right[2], y1 = bg_right[3], line = dict(width = 3, color = 'red'))
                fig.add_shape(roi_rect)
                fig.add_shape (bgl_rect)
                fig.add_shape (bgr_rect)

                fig.show()
            else:
                def log_array(arr):
                    return np.where(arr > 0.00, np.log10(np.clip(arr, a_min=1e-10, a_max=None)), 0.0)

                all_imarrays = np.stack(tuple([log_array(self.tif2array(df.tif.loc[i])) for i in range(len(df.tif))]))
                # all_imarrays = np.stack(tuple([self.tif2array(df.tif.loc[i]) for i in range(len(df.tif))]))
                fig = px.imshow(all_imarrays, animation_frame=0, binary_string=False,
                    color_continuous_scale='RdBu_r',
                    labels = dict(x ='<i>x</i>&#8201;[px]',y = '<i>y</i>&#8201;[px]'),
                    width = 800, height = 566, aspect = 'auto',
                    origin = 'lower',)
                fig.update_xaxes(range = [roi[2]-margin, roi[3]+margin], autorange=False)
                fig.update_yaxes(range = [roi[0]-margin, roi[1]+margin], autorange = False)
                fig.add_shape(roi_rect)
                fig.add_shape (bgl_rect)
                fig.add_shape (bgr_rect)
                fig.update_layout(coloraxis_showscale = True, transition_duration = sleeptime*1e3,
                    updatemenus = {
                    'buttons': [{'args': [None, {'frame': {'duration': sleeptime, 'redraw': False},
                                          'mode': 'immediate', 'fromcurrent': True, 'transition':
                                          {'duration': sleeptime, 'easing': 'linear'}}],
                                 'label': '&#9654;',
                                 'method': 'animate'},
                                {'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                          'mode': 'immediate', 'fromcurrent': True, 'transition':
                                          {'duration': 0, 'easing': 'linear'}}],
                                 'label': '&#9724;',
                                 'method': 'animate'}],
                    'direction': 'left',
                    'pad': {'r': 10, 't': 70},
                    'showactive': False,
                    'type': 'buttons',
                    'x': 0.1,
                    'xanchor': 'right',
                    'y': 0,
                    'yanchor': 'top'
                }
                    )
                fig.show()
        if return_figure:
            return fig



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
        # [print(sd) for sd in subfolders]
        for f, sd in zip(files, subfolders):
            try:
                filename = os.path.basename(sd)
                confilename = os.path.join(sd, filename + '.con')
                parfilename = os.path.join(sd, filename + '.par')
                os.mkdir(sd)
                smove(f, sd)
            except:
                print('Subdirectories already created.')

            for lsfile in lsfitfiles:
                if lsfile.endswith('.con'):
                    confile_template = lsfile 
                    scopy(confile_template, confilename)

                elif lsfile.endswith('.par'):
                    parfile_template = lsfile
                    scopy(parfile_template, parfilename)
                    with open(parfilename, 'r') as f:
                        pardata = f.readlines()
                        pardata[-1] = filename + ext
                    with open(parfilename, 'w') as f:
                        f.writelines(pardata)
            
                elif lsfile.endswith('.OPT'):
                    optfilename = os.path.basename(lsfile)
                    file = os.path.join(sd, optfilename)
                    scopy(lsfile, file)
                    with open(file,'r') as f:
                        optdata = f.readlines()
                        optdata[0] = filename + '\n'
                    with open(file,'w') as f:
                        f.writelines(optdata)

                else:
                    file = os.path.join(sd, os.path.basename(lsfile))
                    # file =os.path.join(sd, lsfile.split('/')[-1])
                    scopy(lsfile, file)

    def delete_neg_values(self, file_ext = '.dat', saveOriginalFiles = False, replaceWithZeroWeightVals = False):    
        '''
        Delte negative counts from a file. File at least has to contain two coloumns: incident angles and counts. If counts should be weighted, weights should be last column
        '''
        colnames = ['q', 'counts', 'weigths']
        all_dat_files_full_path = []
        for root, dirnames , f_names in os.walk(self.savepath):
          for f in f_names:
            if f.endswith(file_ext) and not any(x in f for x in ['orig', 'unweighted', 'unshifted']):
              oldname_fp=os.path.join(root,f)
              newname_fp = os.path.join(root,f).strip(file_ext) + '_orig' + file_ext
              firstline = read_lines_file(oldname_fp, onlyFirstLine=True).split()
              anz_cols = len(firstline)

              df = pd.read_csv(oldname_fp, header=None,delimiter='\s+', names=colnames[0:anz_cols])
              if anz_cols < 3:
                df[colnames[-1]] = np.repeat(1, len(df.q))
              try:
                no_neg_values = df[df.counts >=10**(-12)]
                neg_val_inds = df.index[df.counts <0]        
                zeroWeightDf = df
                zeroWeightDf.loc[neg_val_inds, ['counts', 'weigths']] = 1e-10, 0
                if len(no_neg_values) < len(df):
                    if saveOriginalFiles:
                        smove(oldname_fp, newname_fp)
                    else:
                        os.remove(oldname_fp)
                    if not replaceWithZeroWeightVals:
                        no_neg_values.to_string(oldname_fp, index =False, header=False)
                    else:
                        zeroWeightDf.to_string(oldname_fp, index =False, header = False)
                    print( f' File: {os.path.basename(oldname_fp)} contained {len(df.q) - len(no_neg_values.q)} negative values.')
                    if saveOriginalFiles:
                        print(f'Old file with negative values saved as:\n{os.path.basename(newname_fp)}\nNew file saved as:\n{os.path.basename(oldname_fp)}')
                else:
                    print(f'File {os.path.basename(oldname_fp)} does not contain negative values')
              except Exception as e:
                print(f'File{os.path.basename(oldname_fp)} has not the correct format.')
                print(e)
                continue

    def read_md_table_data(self):
        try:
            mdfile = [f for f in new_listdir(self.savepath) if f.endswith('.md')][0]
        except:
            print(f'No md file in {self.savepath}.')
            pass
        with open(mdfile, 'r') as f:
            lines = f.readlines()
        headerline = lines[0].split('|')
        # headerline = [elem.strip(' ') for elem in headerline if not any(elem == x for x in ['', '\n'])]
        headerline = ["".join(elem.split()) for elem in headerline if not any(elem == x for x in ['', '\n'])]
        lines = lines[2:]
        index_df = [i for i in range(len(lines))]
        df = pd.DataFrame(columns = headerline, index = index_df)
        nested_lines = list()
        for l in lines:
            splitline = l.split('|')
            splitline = [elem.strip(' ') for elem in splitline if not any(elem == x for x in ['', '\n'])]
            nested_lines.append(splitline)

        for nl, id in zip(nested_lines, index_df): df.loc[id] = nl
        
        return df

    def animate(self, i, roi):
        # fig = plt.figure()
        # ax = fig.add_subplot(1,1,1)
        junkfile = os.path.join(self.fiopath, r'csjunk.dat')
        lines = read_lines_file(junkfile)
        anz_lines = len(lines)
        # if len(lines) == 0:
        if len(lines) >= 1:
            fioprefix = lines[0].rstrip().lstrip()
        if len(lines) >= 2:
            motor = lines[1].rstrip().lstrip()

            if motor == 'MOT62':
                motor_name = 'MOT62, TH'
                logscale_y = True
            else:
                motor_name = motor
                logscale_y = False
        motpositions, roi_counts, detector_counts = list(), list(), list()
        if len(lines) >= 3:
            for l in range(2, len(lines)):
                dat = lines[l].split()
                dat = [float(elem) for elem in dat]
                motpos = dat[0]
                motpositions.append(motpos)
                steptime = float(dat[4])
            number_array = list(np.arange(1, anz_lines-2, step = 1))
            all_img_numbers = ["{:05d}".format(int(numb)) for numb in number_array]
            all_tifs, roi_counts, detector_counts = list(), list(), list()
            for i in range(len(all_img_numbers)):
                tif = os.path.join(self.tifpath, f'{fioprefix}_{all_img_numbers[i]}.tif')
                imarray = self.tif2array(tif)
                rc = np.sum(imarray[roi[0]:roi[1], roi[2]:roi[3]], axis=(0,1))
                dc = np.sum(imarray, axis=(0,1))
                roi_counts.append(rc)
                detector_counts.append(dc)
        # if logscale_y:
        #     ax.set_yscale('log')
        plt.cla()
        # ax.clear()    
        plt.plot(motpositions[:-1], roi_counts, label = 'roi counts', marker = 'o', linestyle = '-', color = 'green',)
        plt.plot(motpositions[:-1], detector_counts, label = 'detector counts', marker = 'o', linestyle = '-', color = 'black',)
        plt.legend(loc='upper right')
        # plt.set_xlabel(motor)
        ax = plt.gca()
        ax.set_xlabel(motor_name)
        ax.set_ylabel('Intensity [arb. u.]')
        if logscale_y:
            ax.set_yscale('log')
        plt.tight_layout()
    def live_plot(self, roi):

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        cursor = Cursor(ax, useblit=True, color='red', linewidth=2)
        # ax.set_xlabel(motor)
        ani = FuncAnimation(fig, self.animate, fargs =[roi], interval = 250, repeat = False)
        plt.tight_layout()
        plt.show()
