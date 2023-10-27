import sys
import os
import numpy as np

import pandas as pd
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 10000)
pd.set_option('display.max_colwidth', 1000)

from matplotlib import pyplot as plt, patches, cm 
from matplotlib.colors import LogNorm, rgb2hex

import plotly.graph_objs as go
import plotly_express as px
import plotly.io as pio
from XRR.utilities.conversion_methods import kev2angst, th2q, q2th, inch2cm, cm2inch
from XRR.helpers import scan_numbers_arr, new_listdir
from XRR.utilities.file_operations import read_lines_file
from XRR.plotly_layouts import plotly_layouts as layouts, create_colormap
layout = layouts(transpPlot = False, transpPaper = False, unit_mode = 'bracket', locale_settings='de_DE.utf8')

import scipy as sc
from scipy import constants
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from collections import defaultdict
from scipy import integrate
from lmfit import minimize, Parameters
from shutil import move as smove, copy as scopy, copyfileobj as scopy_obj
from glob import glob
import re
from PIL import Image, ImageEnhance, ImageFilter
from itertools import chain
import math
import time
from datetime import datetime
import re
from natsort import natsorted
# import XRR_datatreatment_class as x
r_e_angstrom = constants.physical_constants['classical electron radius'][0] * 1e10
# layout = layouts(unit_mode='bracket')
class P08:

    def __init__(self, fiopath, tifpath, savepath, wavelength=None, prefix=None):
        """
        create .dat file with q and counts of XRR experiment (at BL 9, DELTA). The arguments consist of the path, where the detectorimages are saved,
        the path of the fio-files, and the path where the .dat-files should be saved,

        parameters:
            * fiopath: path, where the fiofiles are saved:
            * tifpath: path, where the tiffiles are saved
            * savepath: path, where you want to save your datafiles
            * start_scan: first scan number of your measurement
            * anz_scan: number of scans belonging to your measurement
            * roi: roi of your measurement
        optional parameters:
            *aborber_correc: correction of absorber, default is 1.
            * save: if true, '.dat' files will be saved in sp (default: False)
            * show: if true, a plot of the reflectivity curve will be displayed (default: False)
            * wavelength: wavelength in Angstrom of the xrays, if None 0.4592 A is used.
            * prefix: beginning of the savename of the files to be saved. After the prefix, the scan number of the first scan will be attached.
        """
        self.fiopath = fiopath
        self.tifpath = tifpath
        self.savepath = savepath
        if not os.path.isdir(self.savepath):
            os.mkdir(self.savepath)
        self.wavelength = wavelength
        self.prefix = prefix
        if not wavelength:
            # self.wavelength = 0.4959
            self.wavelength = kev2angst(25)
            print('No wavelength was specified. ' + '\u03BB = ' + "{:.4f}".format(self.wavelength) + ' \u00C5' +' is used as default.')


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
        # fiofiles = [f for f in os.listdir(self.fiopath) if all(x in f for x in ['.FIO'])]
        # print(len(fiofiles))
        fiofiles = [f for f in os.listdir(self.fiopath) if all(x in f for x in ['.fio'])]  
        measurement_files = []        
        for i, f in enumerate(fiofiles):
            for scan in scan_numbers:
                    string = int(re.split('_|\.', f)[-2])
                    if string == scan:
                        measurement_files.append(os.path.join(self.fiopath, f))
        measurement_files = sorted(
            measurement_files, key=lambda x: x.partition('/')[-1].partition('_')[-1])
        return measurement_files

    def readFio2df(self, fio, detector = 'p100k', norm2monitor_counts = False):    
        absorber, columnheaders, rangeparameters = defaultdict(dict), defaultdict(dict), defaultdict(dict) 

        q_dummy, th = [], []
        I_roi1, I_roi2, I_roi3, I_roi1_with_bg, I_background, atten_position, steptime, normfactor, tif_df, petra_current, timeVar, monitor_counts = [], [], [], [], [], [], [], [], [], [], [], []
        
        current_scan = str(fio.split('_')[-1].strip('.fio').strip('_'))
        filepref = fio.split('/')[-1].strip('.fio').strip(current_scan)
        current_file = filepref + current_scan
        tifpref = os.path.join(self.tifpath, current_file, detector)
        openfio = open(fio, 'r')
        lines = openfio.readlines()
        openfio.close()
        counter = 0
        for l in range(9, len(lines)):
            if not r'%d' in lines[l] and counter == 0:
                if ' = ' in lines[l]:
                    dummy = lines[l].strip('\n').split(' = ')
                    parameter = dummy[0].replace(' ', '')
                    value = dummy[1].replace(' ', '')
                    rangeparameters.update(
                        {parameter: value})
                    # rangeparameters[scan_numbers[i]].update(
                        # {parameter: value})

            elif r'%d' in lines[l]:
                counter += 1
            elif not r'%d' in lines[l] and not counter == 0:
                if 'Col' in lines[l]:
                    dummy2 = lines[l].strip(
                        '\n').replace('Col', '').split()
                    colnumber = dummy2[0]
                    colvalue = ' '.join(dummy2[1:])
                    columnheaders.update(
                        {colnumber: colvalue})
                    # columnheaders[scan_numbers[i]].update(
                        # {colnumber: colvalue})
                else:
                    if not '! Acq' in lines[l]:
                        data = lines[l].strip('\n').split()
                        data = [float(d) for d in data]
                        th_dummy = data[0]
                        th.append(th_dummy)
                        q_dummy.append(4 * sc.pi / (self.wavelength) * np.sin(th_dummy * sc.pi / 180))
                        
                        petra_curr = float(data[11])
                        petra_current.append(petra_curr)

                        attenpos = float(data[9])
                        atten_position.append(attenpos)
                        
                        stept = float(data[2])
                        steptime.append(stept)
                        
                        monitor_cts = float(data[14])
                        monitor_counts.append(monitor_cts)

                        if norm2monitor_counts:
                            normfact = attenpos / stept / monitor_cts
                        else:
                            normfact = attenpos / stept  
                        
                        # normfact = attenpos / stept / float(data[13]) float(data[13]) actually irrelevant
                        normfactor.append(normfact)
                        
                        I_roi2.append(float(data[5]) * data[9] / data[2] / float(data[13]))
                        I_roi3.append(float(data[6]) * data[9] / data[2] / float(data[13]))

                        timeVar.append(float(data[16]))

                        tif_ind = str("{:05d}".format(int(data[7])))
                        tif_dummy = os.path.join(tifpref,filepref + current_scan + '_' + tif_ind + '.tif')
                        tif_df.append(tif_dummy)
                        
                        I_roi1.append((float(data[4]) * attenpos / stept) - 0.5 * (float(data[5]) * attenpos / stept + float(data[6]) * attenpos / stept))
                        I_roi1_with_bg.append(float(data[4]) * attenpos / stept)
                        I_background.append(0.5*(float(data[5]) * attenpos / stept + float(data[6]) * attenpos / stept))

        df = pd.DataFrame(list(zip(q_dummy, I_roi1, I_roi2, I_roi3, I_background, I_roi1_with_bg,th, atten_position, steptime, normfactor, petra_current, timeVar, monitor_counts, tif_df)),
             columns=['q', 'I_roi1', 'I_roi2', 'I_roi3','I_background', 'I_roi1_with_bg','th', 'atten_position', 'steptime', 'normfactor', 'petra_current', 'time', 'monitor_counts','tif'])

        return df

    @staticmethod
    def tif2array(f):
        im = Image.open(f)
        return np.array(im)

    def read_XRR(self, start_scan_number, anz_scans, absorber_correc=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), roi=None, bgRoiDist = 2,save=False, show=False,
                 background=False, detector='p100k', detector_series=False, error_plot=True, current_plot = False, counts2consider = 'mw',
                 weight_on_error_plot = True, plot_seperate_loops = True, fit_absorber_correction = True, verbose = True, **kwargs):
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
            * bgRoiDist: Distance in pixel between the background ROIs and the counting ROI. Default is 2.
            * save: saves a q-counts-file in the savepath
            * tifread: Read data from detector images.
            * background: if True, two areas beside the roi are defined and the measured counts substracted from the counts in the roi             
            * error_plot: if True, all the loops will be plotted with absolute counts and error bars (1/sqrt(N)) are added.
            * read_time: if True, the time needed to read the detector images will be printed out. Default is False.
            * current_plot: if True, the Petra beam current will be plotted. Default is False
            * weight_on_error_plot: Default False. If True, the absoulte counts with errors will be used to weight the datapoints. Only valid if error_plot = True and tifread = True.
                err <= 0.1 * counts --> weight = 1
                0.1 *counts < err < 0.5 * counts --> weight = 0.5
                0.5 * counts <= err <= 0.8 * counts --> weight = 0.2
                0.8 * counts < err < 1.0 * counts --> weight = 0.1
                err >= 1.0 * counts --> weight = 0
        """
        # if roi.any():
        # print('No ROI specified.')
        ret_data = {'data':None, 'data_figure': None, 'absorber': None, 'error_figure':None, 'sep_loop_figure': None}
        if fit_absorber_correction:
            for i in range(len(absorber_correc)):
                absorber_correc[i] = 1

        try:kwargs['norm2monitor_counts'] = norm2monitor_counts
        except: norm2monitor_counts = False

        bg_width = (roi[1] - roi[0]) / 2
        bg_left = np.array( 
            [math.floor(roi[0] - bgRoiDist - bg_width), roi[0] - bgRoiDist, roi[2], roi[3]])
        bg_right = np.array(
            [roi[1] + bgRoiDist, math.ceil(roi[1] + bgRoiDist + bg_width), roi[2], roi[3]])
        end_scan_number = start_scan_number + (anz_scans - 1)
        scan_numbers = np.arange(
            start_scan_number, end_scan_number + 1, step=1)

        measurement_files = self.list_measurement_FIOS(scan_numbers = scan_numbers)
        dataframes = list()
        for i, fio in enumerate(measurement_files):
            single_dfs = self.readFio2df(fio=fio, norm2monitor_counts=norm2monitor_counts)
            single_dfs['normfactor_corrected'] = single_dfs.normfactor / absorber_correc[i]
            single_dfs['loop'] = i
            dataframes.append(single_dfs)
        
        self.df_unsort = pd.concat(dataframes, ignore_index = True)
        ## read data from detectorimages!!

        columns = ['mw', 'uncorr', 'detector_counts', 'bg', 'bg_unnorm', 'detector_counts_uncorrected']
        self.df_unsort['mw'], self.df_unsort['uncorr'], self.df_unsort['detector_counts'], self.df_unsort['bg'], self.df_unsort['bg_unnorm'], self.df_unsort['detector_counts_uncorrected'] = None, None, None, None, None,None,

        for i in range(len(self.df_unsort.q)):
            try:
                imarray = self.tif2array(self.df_unsort.tif[i])
                # print(imarray[bg_left[2]:bg_left[3], bg_left[0]:bg_left[1]].size, imarray[bg_right[2]:bg_right[3], bg_right[0]:bg_right[1]].size, imarray[roi[2]:roi[3], roi[0]:roi[1]].size)
                bgL_counts = np.sum(imarray[bg_left[2]:bg_left[3], bg_left[0]:bg_left[1]], axis=(0,1)) * self.df_unsort.normfactor_corrected[i]
                bgR_counts = np.sum(imarray[bg_right[2]:bg_right[3], bg_right[0]:bg_right[1]], axis=(0,1)) * self.df_unsort.normfactor_corrected[i]
                roi_counts = np.sum(imarray[roi[2]:roi[3], roi[0]:roi[1]], axis=(0,1)) * self.df_unsort.normfactor_corrected[i]
                mw = roi_counts - (bgL_counts + bgR_counts)
                mw_bg = bgL_counts + bgR_counts
                mw_uncorr = roi_counts
                if not self.df_unsort.normfactor[i] == 0:
                    mw_unnorm = (roi_counts - (bgL_counts  + bgR_counts )) / self.df_unsort.normfactor_corrected[i] 
                    if mw_unnorm <= 0: mw_unnorm = 1e-10
                    mw_bg_unnorm = (bgL_counts + bgR_counts) / self.df_unsort.normfactor_corrected[i] 
                    if mw_bg_unnorm <= 0: mw_bg_unnorm = 1e-10
                    mw_unnorm_uncorr = roi_counts / self.df_unsort.normfactor_corrected[i]
                    if mw_unnorm_uncorr <= 0: mw_unnorm_uncorr = 1e-10    
                else: mw_unnorm, mw_bg_unnorm, mw_unnorm_uncorr = np.nan, np.nan, np.nan
                fill_elements = pd.Series({'mw':mw, 'uncorr':mw_uncorr, 'detector_counts':mw_unnorm, 'bg':mw_bg, 'bg_unnorm':mw_bg_unnorm,
                    'detector_counts_uncorrected':mw_unnorm_uncorr})
                for k in fill_elements.keys():
                    self.df_unsort.loc[i,k] = fill_elements[k]
            except Exception as e:
                print(e)
                print(f'{self.df_unsort.tif[i]} not there or damaged. Value with index {i} for q = {self.df_unsort.q[i]} will be skipped.')
                for k in columns:
                    self.df_unsort.loc[i, k] = None

        self.df_unsort = self.df_unsort.dropna()
        self.df_unsort['counts2consider'] = self.df_unsort[counts2consider]
        self.df_unsort = self.df_unsort.reset_index()

        if fit_absorber_correction:
            par = Parameters()
            par.add('shift', value=1.1)
            absorber_shifts = list()
            idx = 0
            while idx in range(len(self.df_unsort.th)):
                if 0 <= idx < len(self.df_unsort.th) -1:
                    if self.df_unsort.th[idx] >= self.df_unsort.th[idx + 1]:
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
                            result = minimize(fcn = P08.min_function, params = par,
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
                            result = minimize(fcn = P08.min_function, params = par,
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
                            result = minimize(fcn = P08.min_function, params = par,
                                args = ([q_first_fl, q_second_fl], [counts2consider_first_fl, counts2consider_second_fl], 
                                    [q_first_nl, q_second_nl], [counts2consider_first_nl, counts2consider_second_nl], overlap_area),
                                kws = {'step':20}, method = 'leastsq')

                        else:
                            result = minimize(fcn = P08.min_function, params = par,
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
        ret_data['data'] = self.df
############################################################

        start_scan = self.df_unsort['tif'][0].split('/')[-1].strip('.tif').split('_')[-2].lstrip('0')
        end_scan = self.df_unsort['tif'][len(self.df_unsort['q']) - 1].split('/')[-1].strip('.tif').split('_')[-2].lstrip('0')
        sample_system = self.df_unsort['tif'][0].split('/')[-1].strip('.tif').split('_')
        sample_system = ' '.join(sample_system[0:-2])
        
        if plot_seperate_loops:
            qual_tab10 = create_colormap(len(measurement_files), 'tab10')
            # qual_tab10 = cm.get_cmap('tab10', len(measurement_files))
            # qual_tab10 = [rgb2hex(c) for c in qual_tab10.colors]
            single_loops, traces = dict(), list()
            # [print(q, th, ap, st) for q, th, ap, st in zip(self.df_unsort.q, self.df_unsort.th, self.df_unsort.atten_position, self.df_unsort.steptime)]
            for i in range(0, len(measurement_files)):
                single_loops[i] = self.df_unsort[['q', 'th', 'mw', 'steptime', 'atten_position']][self.df_unsort.loop == i]
                customdata = np.stack((single_loops[i]['steptime'], single_loops[i]['atten_position'], single_loops[i]['th']), axis = -1)
                traces.append(go.Scatter(x = single_loops[i]['q'], y = single_loops[i]['mw'], mode = 'lines+markers',
                    marker = dict(color = qual_tab10[i]),
                    line = dict(width = 4, color = qual_tab10[i]), customdata = customdata, name = f'loop {i}',
                    hovertemplate = '<b>q</b> = %{x:.3f} <br><b>th</b> = %{customdata[2]:.3f}<br><b>t</b> = %{customdata[0]:.1f}<br><b>atten_pos</b> = %{customdata[1]:.1f}'
                    ))
            fig = go.Figure(data = traces, layout = layout.refl())
            # fig.show()
            ret_data['sep_loop_figure'] = fig


        if error_plot:
            t1 = time.perf_counter()

            rawdata = go.Scatter(x = self.df_unsort['q'], y = self.df_unsort['detector_counts_uncorrected'], customdata = np.sqrt(self.df_unsort['detector_counts_uncorrected'].tolist()),
                    error_y = dict(array = np.sqrt(self.df_unsort['detector_counts_uncorrected'].tolist()),visible=True),
                    mode='markers',name = 'raw data',
                    hovertemplate = '<b>q<b> = %{x:.3f} <br> <b>I</b> = %{y:} <span>&#177;</span> %{customdata:.2f}')

            raw_bg =  go.Scatter(x = self.df_unsort['q'],y = self.df_unsort['bg_unnorm'], customdata = np.sqrt(self.df_unsort['bg_unnorm'].tolist()),
                error_y = dict(array = np.sqrt(self.df_unsort['bg_unnorm'].tolist()), visible=True),mode='markers', name = 'raw background',
                hovertemplate = '<b>q<b> = %{x:.3f} <br> <b>I</b> = %{y:} <span>&#177;</span> %{customdata:.2f}')

            raw_diff = go.Scatter(x = self.df_unsort['q'], y = self.df_unsort['detector_counts'],
                customdata = np.sqrt(self.df_unsort['bg_unnorm'].tolist() + self.df_unsort['detector_counts_uncorrected'].tolist()),
                error_y = dict(array = np.sqrt(self.df_unsort['bg_unnorm'].tolist() + self.df_unsort['detector_counts_uncorrected'].tolist()), visible=True),
                mode='markers', name = 'raw - raw background',
                hovertemplate = '<b>q<b> = %{x:.3f} <br> <b>I</b> = %{y:} <span>&#177;</span> %{customdata:.2f}')
            fig = go.Figure(data = [rawdata, raw_bg, raw_diff], layout = layout.refl())

            fig.update_layout(autosize=True, showlegend=True, legend_title='', legend_borderwidth=1.0, title=dict(
                text = f'{sample_system} scan {start_scan} - {end_scan}'), xaxis_mirror=True, yaxis_mirror=True, hovermode = 'x')

            ret_data['error_figure'] = fig
            # fig.show()
            t2 = time.perf_counter()
            error_time = t2 - t1


        if weight_on_error_plot:
            if background:
                error_diff_y = abs(np.sqrt((self.df['bg_unnorm'] + self.df['detector_counts_uncorrected']).tolist()) / self.df['detector_counts'].tolist() )
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
            
        if save:
            filename = os.path.join(self.savepath, os.path.basename(measurement_files[0]).rstrip('.fio').rstrip('.FIO') + '.dat')
            if weight_on_error_plot:
                df2save = pd.DataFrame(list(zip(self.df.q, self.df.mw, self.df.weights)), columns = ['q', 'mw', 'weights'])
            else:
                df2save = pd.DataFrame(list(zip(self.df.q, self.df.mw)))
            df2save.to_string(filename, header=False, index=False)
            print('File ' + filename + ' has been saved.')

        # plot of reflectivity
        if show:
            traces = []
            # if not tifread:
            #     if background:
            #         trace1 = go.Scatter(x = self.df['q'], y = self.df['I_roi1'], mode = 'lines+markers', name = 'corrected')
            #         trace2 = go.Scatter(x = self.df['q'], y = self.df['I_roi1'] + 0.5 * (self.df['I_roi2'] + self.df['I_roi3']),
            #                  mode = 'lines+markers', name = 'uncorrected')
            #         trace3 = go.Scatter(x = self.df['q'], y = 0.5 * (self.df['I_roi2'] + self.df['I_roi3']), mode = 'lines+markers', name = 'background')
                    # fig = go.Figure(data=[trace1, trace2, trace3], layout=layout.refl())
                    # fig.update_layout(legend_title= 'scan ' + str(scan_numbers[0]) + '-' + str(scan_numbers[-1]), legend_title_font_size = 16)
            #     else:
            #         trace1  = go.Scatter(x = self.df['q'], y = self.df['I_roi1'], mode = 'lines+markers', name = 'data', showlegend=True)
            #         fig = go.Figure(data = [trace1], layout = layout.refl())
            #         # fig.update_layout(legend_title= 'scan ' + str(scan_numbers[0]) + '-' + str(scan_numbers[-1]), legend_title_font_size = 16)
            # else:
            if not background: 
                trace1 = go.Scatter(x = self.df['q'], y = self.df['uncorr'],  mode='lines+markers', name = 'data', showlegend=True)
                tts = [trace1]
            else:
                trace1 = go.Scatter(x = self.df['q'], y = self.df['uncorr'],  mode='lines+markers', name = 'uncorrected', line_color = 'blue')
                trace2 = go.Scatter(x = self.df['q'], y =self.df['bg'], mode='lines+markers', name = 'background', line_color = 'red')
                trace3 = go.Scatter(x = self.df['q'], y = self.df['mw'],  mode='lines+markers', name = 'corrected', line_color = 'green')
                tts = [trace1, trace2, trace3]
            fig = go.Figure(data=tts, layout=layout.refl())
            fig.update_layout(title_text=f'{sample_system}', legend_title_text= 'scan ' + str(scan_numbers[0]) + '-' + str(scan_numbers[-1]))
            ret_data['data_figure'] = fig
            # fig.show()

        if current_plot:
            timeReference = df_unsort['time']
            timeReference = [time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t)) for t in timeReference]
            current = df_unsort['petra_current'].round(decimals=3)
            monitor_counts = data_from_fio.monitor_counts
            # mean_current, stdev_current = np.mean(current), np.std(current)
            current_data = go.Scatter(
                x = timeReference, y = current,
                mode = 'lines + markers',
                name = 'PETRA current',
                showlegend=True,
                line_color = 'blue'
          )
            monitor_data = go.Scatter(
                x = timeReference, y = monitor_counts,
                mode = 'lines + markers',
                name = 'monitor counts',
                line_color = 'red',
                showlegend=True,
                yaxis = 'y2'
          )
            # [print(q, t) for q, t in zip(df_unsort.q, timeReference)]
            fig = go.Figure(data = [current_data, monitor_data], layout = layout.blank())
            fig.update_layout(xaxis = dict(title='time', title_font_size = 18, showgrid=False), yaxis = dict(color='blue', showgrid=False,title= 'beam current / mA', title_font_size = 18), yaxis2 = dict(color='red',
                title_font_size = 18, title = 'monitor counts / a.u.', overlaying = 'y', side = 'right', showline=True, showgrid=False,linewidth=2, linecolor='red',ticks = 'outside', showticklabels=True))
            fig.show()

        return ret_data
        # if return_dataframe:
        #     return self.df

    def dectimage(self, scan_number, th_ind, roi=None, bgRoiDist = 2, pref_tif='run07_20', margin=10, series=False, sleeptime=0.1,
        save_imgs = False, detector='p100k', plotly_series = True):
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
        df = self.readFio2df(fio = measurement_file)        
        imarrays = list()
        for i in range(len(df.tif)):
            imarray = self.tif2array(df.tif[i])
            if imarray.size == 1:
                imarray = np.nan
            else:
                imarray = imarray
            imarrays.append(imarray)
        df['imarrays'] = imarrays
        df = df.dropna().reset_index(drop= True)
            
        if save_imgs:               
            tifnames = [t.split('/')[-1] for t in tifs_sorted]
            imgs2save = [os.path.join(self.savepath, t.replace('.tif', '.png').replace('.TIF', '.png')) for t in tifnames]
        # # roi = np.array([237, 249, 101, 105])
        if not roi is None:
            roi_height = roi[3] - roi[2]
            roi_width = roi[1]-roi[0]

            bg_width = (roi[1] - roi[0]) / 2
            bg_left = np.array( 
                [math.floor(roi[0] - bgRoiDist - bg_width), roi[0] - bgRoiDist, roi[2], roi[3]])
            bg_right = np.array(
                [roi[1] + bgRoiDist, math.ceil(roi[1] + bgRoiDist + bg_width), roi[2], roi[3]])

            if series: 
                imarray = df.imarrays.loc[0]
                plt.ion()
                f = plt.figure(figsize = (12,32))
                ax = f.gca()
                ax.set(xlim=[roi[0] - margin, roi[1] + margin], ylim=[roi[2] - margin, roi[3] + margin])

                draw_roi = patches.Rectangle((roi[0], roi[2]),roi_width, roi_height, fc='None', ec='g', lw=2)
                draw_bgl = patches.Rectangle((bg_left[0], roi[2]),bg_width, roi_height, fc='None', ec='r', lw=2)
                draw_bgr = patches.Rectangle((bg_right[0], roi[2]),bg_width, roi_height, fc='None', ec='r', lw=2)
                rect = ax.add_patch(draw_roi)
                rect_bgl = ax.add_patch(draw_bgl)
                rect_bgr = ax.add_patch(draw_bgr)
                
                im = ax.imshow(df.imarrays.loc[0], norm = LogNorm())
                for i in range(1, len(df.tif)):
                    im.set_data(df.imarrays.loc[i])
                    f.suptitle('Detector image of scan ' + str(scan_number) + ' th = ' + "{:.3f}".format(df.th.loc[i]) + '°\nq = ' + "{:.3f}".format(th2q(self.wavelength, df.th.loc[i])) + r' $\AA^{-1}$', fontsize=16)
                    f.canvas.flush_events()
                    plt.pause(sleeptime)
                plt.close()

            else:
                # imarray = self.tif2array(df.tif.loc[th_ind])
                imarray = df.imarrays.loc[th_ind]

                f = plt.figure()
                ax = f.add_subplot(111)
                im = ax.imshow(imarray, norm = LogNorm())
                f.suptitle('Detector image of scan ' + str(scan_number) + ' th = ' + "{:.3f}".format(df.th.loc[th_ind]) + '°\nq = ' + "{:.5f}".format(th2q(self.wavelength, df.th.loc[th_ind])) + r' $\AA^{-1}$', fontsize=16)
                ax.set(xlim=[roi[0] - margin, roi[1] + margin], ylim=[roi[2] - margin, roi[3] + margin])
                draw_roi = patches.Rectangle((roi[0], roi[2]),roi_width, roi_height, fc='None', ec='g', lw=2)
                draw_bgl = patches.Rectangle((bg_left[0], roi[2]),bg_width, roi_height, fc='None', ec='r', lw=2)
                draw_bgr = patches.Rectangle((bg_right[0], roi[2]),bg_width, roi_height, fc='None', ec='r', lw=2)
                rect = ax.add_patch(draw_roi)
                rect_bgl = ax.add_patch(draw_bgl)
                rect_bgr = ax.add_patch(draw_bgr)
                plt.show()

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
        df = self.readFio2df(fio = measurement_file)        
        df = df.dropna().reset_index(drop= True)
            
        if save_imgs:               
            tifnames = [t.split('/')[-1] for t in tifs_sorted]
            imgs2save = [os.path.join(self.savepath, t.replace('.tif', '.png').replace('.TIF', '.png')) for t in tifnames]
        # # roi = np.array([237, 249, 101, 105])
        if not roi is None:
            roi_height = roi[3] - roi[2]
            roi_width = roi[1]-roi[0]

            bg_width = (roi[1] - roi[0]) / 2
            bg_left = np.array( 
                [math.floor(roi[0] - bgRoiDist - bg_width), roi[0] - bgRoiDist, roi[2], roi[3]])
            bg_right = np.array(
                [roi[1] + bgRoiDist, math.ceil(roi[1] + bgRoiDist + bg_width), roi[2], roi[3]])

            if not series:
                tiffile = df.tif.loc[th_ind]
                imarray = self.tif2array(tiffile)
                # imarray = df.imarrays.loc[th_ind]
                log_imarray =  np.where(imarray > 0.00, np.log10(np.clip(imarray, a_min=1e-10, a_max=None)), 0.0)
                fig = px.imshow(log_imarray,
                    labels = dict(x ='<i>x</i>&#8201;[px]',y = '<i>y</i>&#8201;[px]'),
                    width = 800, height = 566,
                    origin = 'lower',
                    # xaxis = dict(range = [roi[0]-margin, roi[1]+margin]),
                    # yaxis = dict(range = [roi[2]-margin, roi[3]+margin]), 
                    )
                fig.update_xaxes(range = [roi[0]-margin, roi[1]+margin])
                fig.update_yaxes(range = [roi[2]-margin, roi[3]+margin], autorange = False)
                title_text = f'Detector image of scan: {scan_number}<br>\u03B8 = {df.th.loc[th_ind]:.3f}&#8201;°; <i>q</i><sub>z</sub> = {th2q(self.wavelength, df.th.loc[th_ind]):.3f}&#8201;\u212B '
                roi_rect = go.layout.Shape(type = 'rect', x0 = roi[0], x1 = roi[1], y0 = roi[2], y1 = roi[3], line = dict(width = 3, color = 'green'))
                bgl_rect = go.layout.Shape(type = 'rect', x0 = bg_left[0], x1 = bg_left[1], y0 = bg_left[2], y1 = bg_left[3], line = dict(width = 3, color = 'red'))
                bgr_rect = go.layout.Shape(type = 'rect', x0 = bg_right[0], x1 = bg_right[1], y0 = bg_right[2], y1 = bg_right[3], line = dict(width = 3, color = 'red'))
                fig.add_shape(roi_rect)
                fig.add_shape (bgl_rect)
                fig.add_shape (bgr_rect)
                # print(fig)
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
                fig.update_xaxes(range = [roi[0]-margin, roi[1]+margin])
                fig.update_yaxes(range = [roi[2]-margin, roi[3]+margin], autorange = False)
                roi_rect = go.layout.Shape(type = 'rect', x0 = roi[0], x1 = roi[1], y0 = roi[2], y1 = roi[3], line = dict(width = 3, color = 'green'))
                bgl_rect = go.layout.Shape(type = 'rect', x0 = bg_left[0], x1 = bg_left[1], y0 = bg_left[2], y1 = bg_left[3], line = dict(width = 3, color = 'red'))
                bgr_rect = go.layout.Shape(type = 'rect', x0 = bg_right[0], x1 = bg_right[1], y0 = bg_right[2], y1 = bg_right[3], line = dict(width = 3, color = 'red'))
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

                # imarrays = np.vstack(tup)
                # pass
                # imarray = df.imarrays.loc[0]
                # log_imarray =  np.where(imarray > 0.00, np.log10(np.clip(imarray, a_min=1e-10, a_max=None)), 0.0)
                # img2plot = go.Heatmap(z=log_imarray)
                # fig = go.Figure(data = img2plot)
                # title_text = f'Detector image of scan: {scan_number}<br>\u03B8 = {df.th.loc[0]:.3f}&#8201;°; <i>q</i><sub>z</sub> = {th2q(self.wavelength, df.th.loc[0]):.3f}&#8201;\u212B'
                # fig.update_layout(title =dict(text = title_text, font = dict(family = 'latin modern roman', size = 20, color = 'black')),
                #     yaxis= dict(title_text = '<i>y</i>&#8201;[px]', range = [roi[2]-margin, roi[3]+margin]),
                #     xaxis= dict(title_text = '<i>x</i>&#8201;[px]', range = [roi[0]-margin, roi[1]+margin]),
                #     width = 800, height = 566
                #     )
                # roi_rect = go.layout.Shape(type = 'rect', x0 = roi[0], x1 = roi[1], y0 = roi[2], y1 = roi[3], line = dict(width = 3, color = 'green'))
                # bgl_rect = go.layout.Shape(type = 'rect', x0 = bg_left[0], x1 = bg_left[1], y0 = bg_left[2], y1 = bg_left[3], line = dict(width = 3, color = 'red'))
                # bgr_rect = go.layout.Shape(type = 'rect', x0 = bg_right[0], x1 = bg_right[1], y0 = bg_right[2], y1 = bg_right[3], line = dict(width = 3, color = 'red'))
                # fig.add_shape(roi_rect)
                # fig.add_shape (bgl_rect)
                # fig.add_shape (bgr_rect)
                # fig.show()
                # for i in range(1,len(df.tif)):
                #     title_text = f'Detector image of scan: {scan_number}<br>\u03B8 = {df.th.loc[i]:.3f}&#8201;°; <i>q</i><sub>z</sub> = {th2q(self.wavelength, df.th.loc[i]):.3f}&#8201;\u212B'
                #     imarray = df.imarrays.loc[i]
                #     log_imarray =  np.where(imarray > 0.00, np.log10(np.clip(imarray, a_min=1e-10, a_max=None)), 0.0)
                #     time.sleep(sleeptime)
                #     img2plot = go.Heatmap(z=log_imarray)
                #     fig.data[0].z = log_imarray
                #     fig.update_layout(title_text = title_text)
        if return_figure:
            return fig


    def rename_files(self, pressure, rename=False, shownames=False, prefix=str(), replace_prefix=False):
        '''
        Rename files to get pressure into the filename
        arg:q
            *pressure: array of ordered pressure values
        kwargs:
            * rename: boolean, renames the datfiles
            * shownames: shows the new names of the files
            * prefix: if given, the prefix will be eliminated from the filenames. Must be part of the filename.
            * replace prefix: New prefix at the beginning of filenames
        '''
        self.datfiles_fp = [os.path.join(self.savepath, f) for f in os.listdir(
            self.savepath) if f.endswith('.dat')]
        self.datfiles_fp = sorted(
            self.datfiles_fp, key=lambda x: x.replace('.dat', '').split('_')[2:])
        self.datfiles = [f for f in os.listdir(self.savepath) if f.endswith('.dat')]
        self.datfiles = sorted(
            self.datfiles, key=lambda x: x.replace('.dat', '').split('_')[2:])
        try:
            if not len(pressure) == len(self.datfiles_fp):
                print(
                    'Some entries for the pressure list seem to be missing. Please check.')
            else:
                self.new_names = ['_'.join(name.split('_')[
                                           :-1]) + '_' + re.sub("\s+", "_", pressure[ind],flags=re.UNICODE) + '.dat' for ind, name in enumerate(self.datfiles)]
                if prefix:
                    if type(prefix) == list:
                        for j in range(len(prefix)):
                            self.new_names = [i.replace(prefix[j], "") for i in self.new_names]
                        # self.new_names = [i.replace(prefix,"") for i in self.new_names]
                    else:
                        self.new_names = [i.replace(prefix,"") for i in self.new_names]
                    
                if replace_prefix:
                    self.new_names = [replace_prefix + i for i in self.new_names]
                self.new_names = [name.replace(',','p') for name in self.new_names]
                self.new_names = [name.strip('.dat').replace('.','p') + '.dat' for name in self.new_names]
                for i,name in enumerate(self.new_names):
                    try:
                        rep_number = re.findall('#+\d',name)[0]
                    except Exception as e:
                        rep_number='#1'
                    if not any(x in name for x in ['bar', 'air', 'luft']):
                        if not '#' in name:
                            self.new_names[i] = name.strip('.dat') + '_bar.dat'
                            # print(name.strip('.dat') + '_bar.dat')
                        else:
                            # rep_number = re.findall('#+\d',name)[0]
                            self.new_names[i] = name.strip('.dat').strip(rep_number) + 'bar_' + rep_number.strip('#') + '.dat'
                        if 'down' in name:
                            if not "#" in name:
                                self.new_names[i] = name.strip('down.dat') + 'bar_down.dat'
                            else:
                                # rep_number = re.findall('#+\d',name)[0]
                                self.new_names[i] = name.strip('.dat').strip('_down' + rep_number) + '_bar_down_' + rep_number.strip('#') + '.dat'
                                # print(name.strip('.dat').strip('_down' + rep_number) + '_bar_down_' + rep_number.strip('#'))
                    elif 'air' in name and '#' in name:
                        self.new_names[i] = name.replace(rep_number,rep_number.strip('#'))

                self.new_names_fp = []
                [self.new_names_fp.append(os.path.join(self.savepath, i)) for i in self.new_names]

            if shownames:
                [print(old, new)
                 for old, new in zip(self.datfiles, self.new_names)]
            if rename:
                checklist = []
                for i, n in enumerate(self.datfiles):
                    check = re.findall('\d{5}', n)
                    if not check == []:
                        os.rename(self.datfiles_fp[i], self.new_names_fp[i])
                    else:
                        print(f'Already renamed {self.datfiles[i]}')
        except Exception as e:
            # print('Something is going wrong.')
            print(e)



    def makedirlsfit(filepath, lsfit_path, splitpattern='.',ext='.dat'):
        '''
        Create a directory where all needed files for lsfit are copied.
        Parameters
        ----------
        filepath: path, where the I vs. q data are saved
        lsfit_path= path, where lsfit_files are saved: Needed files: 'condat.con', 'pardat.par', 'reflek.exe', 'testcli.exe','LSFIT.OPT'
        ext: extension of the I vs. q files. Default is '.dat'
        splitpattern: pattern, where to separate the files at the extension. Default is '.'
        '''
        os.chdir(filepath)
        # Erstmal Unterordner für .dat-Dateien erstellen und diese dann verschieben
        files = os.listdir(filepath)
        files = [f for f in files if os.path.isfile(os.path.join(filepath,f))]

        for i, entry in enumerate(files):
            if entry.endswith(ext): 
                subfolders = entry.split(splitpattern)[0]
                subfolders_with_path = os.path.join(filepath, subfolders)
                # print(subfolders_with_path)
                try:
                    os.mkdir(subfolders_with_path)
                    smove(entry, subfolders_with_path)
                except:
                    print('Subdirectories already created')


        # LSFit-Dateien aus Vorlagen-Ordner in die Unterordner kopieren und par- und con-Datei umbenennen.
        lsfitfiles = [f.name for f in os.scandir(lsfit_path) if f.is_file()]
        subfolders = [f.path for f in os.scandir(filepath) if f.is_dir()]
        for i, entry in enumerate(subfolders):
            filenames = os.listdir(entry)
            for file in filenames:
                if file.endswith(ext):
                    extfilename = file[:-4]
                    confilename = extfilename + '.con'
                    parfilename = extfilename + '.par'
            os.chdir(lsfit_path)
            for j, lsfile in enumerate(lsfitfiles):
                scopy(lsfile, entry)

            os.chdir(entry)
            old_con_name = ''.join([entry, '/', 'condat.con'])
            new_con_name = ''.join([entry, '/', confilename])
            old_par_name = ''.join([entry, '/', 'pardat.par'])
            new_par_name = ''.join([entry, '/', parfilename])
            try:
                if os.path.exists('condat.con'):
                    os.rename(old_con_name, new_con_name)
                if os.path.exists('pardat.par'):
                    os.rename(old_par_name, new_par_name)

            except FileExistsError:
                os.remove(old_con_name)
                os.remove(old_par_name)

            parfile = new_par_name
            new_last_parline = extfilename + '.dat'
            parlines = open(parfile, 'r').readlines()  # reads all lines of the parfile
            parlines[-1] = new_last_parline
            open(parfile, 'w').writelines(parlines)

            optfile = ''.join([entry, '/', 'LSFIT.OPT'])
            optdata = open(optfile, 'r+')
            firstline = optdata.readline()
            firstline = extfilename + '\n'
            optdata2 = open(optfile, 'w')
            optdata2.write(firstline)
            scopy_obj(optdata, optdata2)
            optdata.close()
            optdata2.close()


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