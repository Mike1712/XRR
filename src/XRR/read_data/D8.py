import sys
import os
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_colwidth', 1000)
from matplotlib import pyplot as plt, patches, cm
from matplotlib.colors import LogNorm, rgb2hex
# from plotly_things import plotly_layouts as layouts
from XRR.plotly_layouts import plotly_layouts as layouts
import plotly.graph_objs as go
import scipy as sc
from scipy import constants
from scipy.signal import find_peaks, argrelextrema, peak_widths, argrelmin
from scipy import integrate, special, interpolate
import re
from lmfit import Parameters, minimize, report_fit

layout = layouts()
def new_listdir(filepath):
    try:
        files = [os.path.join(filepath, f) for f in os.listdir(filepath)]
    except TypeError:
        files = []
        for f in os.listdir(filepath):
            file = os.path.join(filepath, f.decode('utf-8'))
            files.append(file)
    return files

def kev2angst(energy):
    '''
    Convert energy to wavelength.

    Parameters:
    -----------
    energy: Beam energy in keV

    Returns:
    --------
    wavelength: wavelength in Angstrom
    '''
    return sc.constants.h * sc.constants.c / (energy * 1e3 * sc.constants.e) *1e10

def th2q(wavelength, th):
    # input is numpy array with theta values
    # function returns q in inverse Angstrom
    if not isinstance(th, list):
        q = 4 * sc.pi / (wavelength) * np.sin(th * sc.pi / 180)
    else:
        q = [4 * sc.pi / (wavelength) * np.sin(t * sc.pi / 180) for t in th]
    return q

class einlesenD8:
    '''
        parameters:
            * Filepath: The absolute path of the uxd-file you want to read
        optional parameters
            * background: if underground measurement was performed, the underground will be fitted and subtracted from the data
                           default=false
            * save: save the transformed data in '*.dat' Filepath; default=False
            * normalize: if True, the counts will be divided by the maximun of the measured counts, default=False
            * show: plot the data, default=False
            * pre_peak_eval: should delete first values if there are datapoints too close to the primary beam (not working right now)
            #### problems with normalize and background, since data is normalized before the background substraction 
    '''

    def __init__(self, filepath, wavelength = None):
        self.filepath = filepath
        self.wavelength = wavelength
        if not wavelength:
            self.wavelength = kev2angst(8.048)
            print(f'No wavelength was specified. \u03BB = {"{:.4f}".format(self.wavelength)} \u00C5 is used as default.')

    def find_uxd_file(self):
        try:
            uxdfile = [f for f in new_listdir(self.filepath) if f.endswith('.uxd')][0]
        except:
            print(f'No "*.uxd file" in {self.filepath}')
            uxdfile = None
        return uxdfile

    @staticmethod
    def readlines_file(file):
        with open(file, 'r') as f:
            lines = f.readlines()
        return lines

    @staticmethod
    def metadata_uxd_file(lines):
        counter = 0
        index, value = list(), list()
        while counter == 0:
            for l in range(len(lines)):
                if counter == 0 and not "Data for range" in lines[l]:
                    splitlines = lines[l].strip().split('=')
                    if len(splitlines) > 1:
                    # print(len(splitlines), splitlines)
                        i, v = splitlines[0], splitlines[1]
                        index.append(i), value.append(v)
                elif l == len(lines):
                    counter += 1
                elif 'Data for range' in lines[l]:
                    counter += 1
        metadata = pd.DataFrame(list(zip(index, value)), columns  = ['parameter', 'value'])
        return metadata

    @staticmethod
    def get_numb_ranges(lines):
        counter = 0
        for l in range(len(lines)):
            if 'Data for range' in lines[l]:
                counter += 1
        range_numbers = list(range(1, counter + 1))
        return range_numbers

    @staticmethod
    def range_parameters(lines, range_number, wavelength, absorber = [10450, 100.966, 9.59459, 1, 1, 1, 1]):
        parameter, value, twotheta, counts = list(), list(), list(), list()
        counter = 0
        for l in range(len(lines)):
            if f'Data for range {str(range_number)}' in lines[l]:
                counter +=1
            elif f'Data for range {str(range_number + 1)}' in lines[l]:
                counter += 1
            if counter == 1:
                if lines[l].strip().startswith('_'):
                    splitlines = lines[l].strip().lstrip('_').split('=')
                    if len(splitlines) == 2:
                        p, v = splitlines[0].strip(), splitlines[1].strip()
                        parameter.append(p), value.append(v)
                else:
                    dataline = lines[l].strip().split()
                    if not dataline == [] and not dataline[0] == ';':
                        if len(dataline) == 2:
                            twotheta.append(float(dataline[0])), counts.append(float(dataline[1]))

        range_parameters = pd.DataFrame(list(zip(parameter, value)), columns = ['parameter', 'value'])
        
        start_th = range_parameters.loc[range_parameters.parameter == 'THETA', 'value'].iloc[0]
        start_tt = range_parameters.loc[range_parameters.parameter == '2THETA', 'value'].iloc[0]
        if not float(start_tt) == 2 * float(start_th):
            comment = 'offset'
        else:
            comment = 'scan'  
        comment_rep = np.repeat(comment, len(twotheta))
        
        steptime = range_parameters.loc[range_parameters.parameter == 'STEPTIME', 'value'].iloc[0]
        steptime = float(steptime)
        steptime_rep = np.repeat(steptime, len(twotheta))

        normfactor = absorber[range_number - 1] / steptime
        normfactor_rep = np.repeat(normfactor, len(twotheta))
        
        range_number_rep = np.repeat(range_number, len(twotheta))

        theta = [tt / 2 for tt in twotheta]
        q = th2q(wavelength, theta)
        single_df = pd.DataFrame(list(zip(twotheta, theta, q, steptime_rep, counts, normfactor_rep, comment_rep, range_number_rep)),
            columns = ['twotheta', 'theta', 'q', 'steptime', 'counts', 'normfactor','comment', 'range_number'])
        single_df['counts_norm2steptime'] = [c/s for c,s in zip(single_df.counts, single_df.steptime)]
        single_df['counts_norm2normfactor'] = [c * nf for c,nf in zip(single_df.counts, single_df.normfactor)]

        return single_df

    @staticmethod
    def min_function(par, x0, y0, x1, y1, overlap_area, step = 1000):
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
         
    def read_XRR(self, background = True,save=False, normalize=False, show=False, pre_peak_eval=False,
        absorber_correc = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), fit_absorber_correction = False, counts2consider = 'counts_norm2normfactor', plot_separate_loops=False,
        error_plot = False, weight_on_error_plot = False):

        
        if fit_absorber_correction:
            for i in range(len(absorber_correc)):
                absorber_correc[i] = 1
        uxdfile = self.find_uxd_file()
        lines = einlesenD8.readlines_file(uxdfile)
        metadata = einlesenD8.metadata_uxd_file(lines)
        range_numbers = einlesenD8.get_numb_ranges(lines)
        
        dataframes = []
        for r in range_numbers:
            single_df = einlesenD8.range_parameters(lines = lines, range_number = r, wavelength = self.wavelength)
            dataframes.append(single_df)
        df_unsort = pd.concat(dataframes, ignore_index = True )

        df_unsort['counts2consider'] = df_unsort[counts2consider]
        if background:
            bg_counts = df_unsort.loc[df_unsort.comment=='offset', 'counts2consider'].values
            bg_q = df_unsort.loc[df_unsort.comment=='offset', 'q'].values
            fit_bg = sc.interpolate.PchipInterpolator(bg_q, bg_counts)
            bg_counts_fit = fit_bg(df_unsort.q)
            df_unsort['bg_counts_fit'] = bg_counts_fit
            new_key = str(counts2consider) + '_bg_corrected'
            bg_corrected = [c2c - bgcfit for c2c, bgcfit in zip(df_unsort.counts2consider, df_unsort.bg_counts_fit)]
            df_unsort[str(new_key)] = [c2c - bgcfit for c2c, bgcfit in zip(df_unsort.counts2consider, df_unsort.bg_counts_fit)]
            df_unsort['counts2consider_without_bg_subst'] = df_unsort['counts2consider']
            df_unsort['counts2consider'] = df_unsort[str(new_key)]
            df_unsort['bg_counts_fit'] = bg_counts_fit
            # [print(a, b) for a, b in zip(df_unsort.counts2consider_without_bg_subst, df_unsort.counts2consider)]
            
        df_bg = df_unsort.loc[df_unsort.comment=='offset']
        
        df_unsort.drop(df_unsort.index[df_unsort.comment=='offset'], inplace=True)

        if fit_absorber_correction:
            
            par = Parameters()
            par.add('shift', value=1.1)
            absorber_shifts = list()
            idx = 0
            while idx in range(len(df_unsort.theta)):
                if 0 <= idx < len(df_unsort.theta) -1:
                    if df_unsort.theta[idx] >= df_unsort.theta[idx + 1]:
                        loop_val = df_unsort.at[idx, 'range_number']
                        next_loop_val = loop_val + 1
                        first_loop_values = df_unsort[df_unsort.range_number == loop_val]
                        next_loop_values = df_unsort[df_unsort.range_number == next_loop_val]
                        overlap_area = [next_loop_values.q.iloc[0], first_loop_values.q.iloc[-1]]
                        first_loop_overlap_df = first_loop_values.loc[first_loop_values['q'].between(overlap_area[0], overlap_area[1], inclusive='both'), first_loop_values.columns.tolist()]
                        next_loop_overlap_df = next_loop_values.loc[next_loop_values['q'].between(overlap_area[0], overlap_area[1], inclusive='both'), next_loop_values.columns.tolist()]
                        if len(next_loop_overlap_df.q) < 2 and not len(first_loop_overlap_df.q) < 2:
                            print(f'Between loops {loop_val} - {next_loop_val}: Second loop has not enough overlapping points')
                            q_first_index = next_loop_overlap_df.q.index[0]
                            q_first = df_unsort.q.iloc[q_first_index]
                            q_second = df_unsort.q.iloc[q_first_index + 1]
                            counts2consider_first = df_unsort.counts2consider.iloc[q_first_index]
                            counts2consider_second = df_unsort.counts2consider.iloc[q_first_index + 1]
                            if overlap_area[0] == overlap_area[1]: overlap_area = [q_first, q_second]
                            # print(self.df_unsort.counts2consider.iloc[q_first_index], self.df_unsort.counts2consider.iloc[q_first_index + 1])
                            result = minimize(fcn = einlesenD8.min_function, params = par,
                                args = (first_loop_overlap_df.q, first_loop_overlap_df.counts2consider, [q_first, q_second], [counts2consider_first, counts2consider_second], overlap_area),
                                kws = {'step':20}, method = 'leastsq')
                        # print(f'{loop_val}: first_loop_overlap: {len(first_loop_overlap_df.q)} points, next_loop_overlap: {len(next_loop_overlap_df.q)} points')
                        
                        elif len(first_loop_overlap_df.q) < 2 and not len(next_loop_overlap_df.q) <2:
                            print(f'Between loops {loop_val} - {next_loop_val}: First loop has not enough overlapping points')
                            q_second_index = first_loop_overlap_df.q.index[0]
                            q_second = self.df_unsort.q.iloc[q_second_index]
                            q_first = self.df_unsort.q.iloc[q_second_index - 1]
                            counts2consider_first = df_unsort.counts2consider.iloc[q_second_index - 1]
                            counts2consider_second = df_unsort.counts2consider.iloc[q_second_index]
                            if overlap_area[0] == overlap_area[1]: overlap_area = [q_first, q_second]
                            result = minimize(fcn = einlesenD8.min_function, params = par,
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
                            if overlap_area[0] == overlap_area[1]: overlap_area = [q_first_fl, q_second_nl]
                            result = minimize(fcn = einlesenD8.min_function, params = par,
                                args = ([q_first_fl, q_second_fl], [counts2consider_first_fl, counts2consider_second_fl], 
                                    [q_first_nl, q_second_nl], [counts2consider_first_nl, counts2consider_second_nl], overlap_area),
                                kws = {'step':20}, method = 'leastsq')

                        else:
                            result = minimize(fcn = einlesenD8.min_function, params = par,
                                args = (first_loop_overlap_df.q, first_loop_overlap_df.counts2consider, next_loop_overlap_df.q, next_loop_overlap_df.counts2consider, overlap_area),
                                kws = {'step':20}, method = 'leastsq')
                            # report_fit(result)
                        absorber_shift = result.params.valuesdict()['shift']
                        absorber_shifts.append(absorber_shift)
                        df_unsort.loc[df_unsort.range_number == next_loop_val, 'counts2consider'] = df_unsort.counts2consider * absorber_shift
                        # self.df_unsort.loc[self.df_unsort.loop == next_loop_val, 'counts2consider'] = self.df_unsort.counts2consider * absorber_shift
                        # for col in self.df_unsort[['mw', 'bg','uncorr', 'detector_counts']].columns:
                            # self.df_unsort.loc[self.df_unsort.loop == next_loop_val, col] = self.df_unsort[col] * absorber_shift
                idx += 1
            absorber_text = ["{:.2f}".format(a) if a >=0.01 else "{:.2e}".format(a) for a in absorber_shifts]
            absorber_text.insert(0,1)
            print(f'Loops shifted with {absorber_text}')


        
###########################################################
        df = df_unsort.sort_values(by = 'q')#####
###########################################################

        if plot_separate_loops:
            qual_tab10 = cm.get_cmap('tab10', len(range_numbers) - 1)
            qual_tab10 = [rgb2hex(c) for c in qual_tab10.colors]
            single_loops, traces = dict(), list()
            # [print(q, th, ap, st) for q, th, ap, st in zip(self.df_unsort.q, self.df_unsort.th, self.df_unsort.atten_position, self.df_unsort.steptime)]
            for i in range(1, len(range_numbers)):
                if background:
                    single_loops[i] = df_unsort[['q', 'theta', 'counts2consider', 'steptime']][df_unsort.range_number == i]
                else:
                    single_loops[i] = df_unsort[['q', 'theta', 'counts2consider', 'steptime']][df_unsort.range_number == i]

                customdata = np.stack((single_loops[i]['steptime'], single_loops[i]['theta']), axis = -1)
                
                traces.append(go.Scatter(x = single_loops[i]['q'], y = single_loops[i]['counts2consider'], mode = 'lines+markers',
                    marker = dict(color = qual_tab10[i-1]),
                    line = dict(width = 4, color = qual_tab10[i-1]), name = f'loop {i}',
                    customdata = customdata,
                    hovertemplate = '<b>q</b> = %{x:.3f} <br><b>th</b> = %{customdata[1]:.3f}<br><b>I</b> = %{y:.2f}<br><b>t</b> = %{customdata[0]:.1f}'
                    ))
            fig = go.Figure(data = traces, layout = layout.refl())
            fig.show()

        if show:
            traces = []
            if background:
                trace1 = go.Scatter(x = df['q'], y = df['counts2consider_without_bg_subst'], mode = 'lines+markers', name = 'uncorrected')
                trace2 = go.Scatter(x = df['q'], y = df['bg_counts_fit'], mode = 'lines+markers', name = 'background')
                trace3 = go.Scatter(x = df['q'], y = df['counts2consider'], mode = 'lines+markers', name = 'corrected')

                dummy = go.Scatter(x = df['q'], y = df['counts2consider'], mode = 'none', yaxis = 'y2', showlegend = False, hoverinfo = 'skip')
                fig = go.Figure(data=[trace1, trace2, trace3, dummy], layout=layout.refl())
                # fig.update_layout(legend_title= 'scan ' + str(scan_numbers[0]) + '-' + str(scan_numbers[-1]), legend_title_font_size = 16)
            else:
                trace1  = go.Scatter(x = df['q'], y = df['counts2consider'], mode = 'lines+markers', name = 'data', showlegend=True)
                dummy = go.Scatter(x = df['q'], y = df['counts2consider'], mode = 'none', yaxis = 'y2', showlegend = False, hoverinfo = 'skip')
                fig = go.Figure(data = [trace1, dummy], layout = layout.refl())
            fig.update_layout(title = None,hovermode='x')
            fig.show()

        if error_plot:
            error_raw = np.sqrt(df_unsort['counts'])
            th_arr = np.array(df_unsort.theta)
            if background:
                error_bg = np.sqrt(df_unsort['bg_counts_fit'])
                error_diff = np.sqrt(df_unsort['bg_counts_fit'] + df_unsort['counts'])
                customdata = np.stack((error_raw, error_bg, error_diff, th_arr), axis = -1)
            else:
                customdata = np.stack((error_raw, th_arr), axis = -1)


            rawdata = go.Scatter(x = df_unsort['q'], y = df_unsort['counts'], customdata = customdata,
                    error_y = dict(array = error_raw,visible=True), mode='markers',name = 'raw data',
                    hovertemplate = '<b>q</b> = %{x:.3f} <br><b>th</b> = %{customdata[3]:.3f}<br><b>I</b> = %{y:} <span>&#177;</span> %{customdata[0]:.2f}')
            if background:
                raw_bg =  go.Scatter(x = df_bg['q'],y = df_bg['counts'], customdata = customdata,
                    error_y = dict(array = error_bg, visible=True),mode='markers', name = 'raw background',
                    hovertemplate = '<b>q</b> = %{x:.3f} <br><b>th</b> = %{customdata[3]:.3f}<br><b>I</b> = %{y:} <span>&#177;</span> %{customdata[1]:.2f}')

                raw_diff = go.Scatter(x = df_unsort['q'], y = df_unsort['counts2consider'], customdata = customdata,
                    error_y = dict(array = error_diff, visible=True), mode='markers', name = 'raw - raw background',
                    hovertemplate = '<b>q</b> = %{x:.3f} <br><b>th</b> = %{customdata[3]:.3f} <br><b>I</b> = %{y:.2e} <span>&#177;</span> %{customdata[2]:.2f}')
            
                fig = go.Figure(data = [rawdata, raw_bg, raw_diff], layout = layout.refl())
            else:
                fig = go.Figure(data = [rawdata], layout = layout.refl())
            fig.update_layout(autosize=True, showlegend=True, legend_title='', legend_borderwidth=1.0,
                # title=dict(text = f'{sample_system} scan {start_scan} - {end_scan}'),
                xaxis_mirror=True, yaxis_mirror=True)

            fig.show()

        if weight_on_error_plot:
            if background:
                error_diff_y = abs(np.sqrt((df['bg_counts_fit'] + df['counts']).tolist()) / df['counts2consider'].tolist() )
            else:
                error_diff_y = abs((np.sqrt(df['counts'].tolist())) / df['counts2consider'].tolist())
            
            weights = []
            for i, err in enumerate(error_diff_y):
                counts2consider = df.counts2consider
                # if background:
                    # counts2consider = df[''].tolist()
                # else:
                    # counts2consider = self.df['detector_counts_uncorrected'].tolist()
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

            df['weights'], df['errors'] = weights, error_diff_y
            # print(df.weights)

        if save:
            filename =self.filepath + '/' + self.filepath.split('/')[-1] + '.dat'
            if weight_on_error_plot:
                df2save = pd.DataFrame(list(zip(df.q, df.counts2consider, df.weights)), columns = ['q', 'mw', 'weights'])
                print('with errors')
            else:
                df2save = pd.DataFrame(list(zip(df.q, df.counts2consider)))     
            df2save.to_string(filename, header=False, index=False)
            print('File ' + filename + ' has been saved.')

        return df
        