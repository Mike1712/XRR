import os
import re
from pprint import pprint
import plotly.graph_objects as go
from XRR.plotly_layouts import plotly_layouts as layouts, create_colormap
layout = layouts(transpPaper=False, transpPlot=False, transpLegend=True, bigTitles=True, locale_settings='DE')
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (12,10)
from XRR.utilities.file_operations import *
from XRR.utilities.conversion_methods import *
from XRR.utilities.math import *
from XRR.helpers import *
import XRR.FluidProps as FP
import XRR.XRR_analysis as XRRa
import numpy as np
import math

from scipy.interpolate import interp1d
from scipy.signal import argrelextrema, find_peaks, savgol_filter, argrelmin, peak_widths, peak_prominences
from scipy import constants
from scipy import integrate
from scipy import intersect1d
from scipy.ndimage import gaussian_filter1d

from pandas import DataFrame, read_csv, read_fwf,concat as pd_concat
from datetime import datetime, timedelta
from time import perf_counter
from multiprocessing.pool import ThreadPool as Pool

__all__ = ['VolumeFractionProfile', 'angshift', 'shift_eldens', 'enlarge_zrange_eldens', 'qweights', 'restore_orig_weights', 'roughness_from_VFP']
class XRR_datatreatment:
    '''
    Evaluate data from an XRR experiment.
    '''
    def __init__(self, Filepath, wavelength=None, fresnel_ext = '_RF.out', eldens_ext = '_ed.rft', distfile_ext = 'dist.rft'):        
        self.fresnel_ext = fresnel_ext
        self.eldens_ext = eldens_ext
        self.distfile_ext = distfile_ext
        self.thickness = []
        self.Filepath = Filepath
        self.wavelength = wavelength
        self.r_e_angstrom = constants.physical_constants['classical electron radius'][0] * 1e10
        if not wavelength:
            self.wavelength = 0.4592
            print('No wavelength was specified. ' + '\u03BB = ' + "{:.4f}".format(self.wavelength) + ' \u00C5' +' is used as default.')

        rftfiles, datfiles, outfilename, distfile, fresnelfile, headerlen_outfile, headerlen_ff= [], [], '', '', '', np.nan, np.nan
        files = new_listdir(self.Filepath)
        for file in files:
            if file.endswith('.out') and not any(j in file for j in [self.fresnel_ext, 'lserror.out']):
                outfilename = file
                headerlen_outfile = determine_headerlen_outfile(outfilename)
            if file.endswith(self.fresnel_ext):
                fresnelfile = file
                headerlen_ff = determine_headerlen_outfile(fresnelfile)
            if file.endswith(self.eldens_ext):
                rftfiles.append(file)
            if file.endswith(self.distfile_ext):
                distfile = file
            if file.endswith('.dat'):
                datfiles.append(file)

        self.outfilename, self.fresnelfile, self.headerlen_outfile, self.headerlen_ff, self.rftfiles, self.distfile, self.datfiles = outfilename, fresnelfile, headerlen_outfile, headerlen_ff, rftfiles, distfile, datfiles

    def read_XRR_data(self, timeit = False):
            ''' Read XRR-data. Function returns following values in the order mentioned:
            q, YOBS_norm, YCALC_norm, height, rho (, q_fresnel, measured_values, fit, measured_values_fresnel, fit_fresnel)
            '''
            t1 = perf_counter()
            datfile = min(self.datfiles, key = len)
            datfile_data = readDatFileData(datfile)
            if len(datfile_data.keys()) == 2:
                datfile_data['weights'] = np.repeat(1, len(datfile_data.q))
            anz_vals = len(datfile_data[datfile_data.keys()[0]])

            data_refl = DataFrame(np.nan, index = np.arange(0,anz_vals,1), columns = ['XOBS', 'q', 'q_fresnel', 'YOBS', 'measured_values',  'measured_values_fresnel',
                'YCALC', 'fit', 'YCALC_norm', 'fit_fresnel', 'DELTA', 'DELTA/SIGMA', 'datfile_q', 'datfile_counts', 'datfile_weights'])
            
            data_refl['datfile_q'], data_refl['datfile_counts'], data_refl['datfile_weights'] = datfile_data[datfile_data.keys()[0]],\
            datfile_data[datfile_data.keys()[1]], datfile_data[datfile_data.keys()[2]]
            if not self.outfilename == '':
            # Messertwte aus der Out-datei
                df = read_csv(self.outfilename, header=self.headerlen_outfile, sep='\s+', usecols=[
                    1, 2, 3, 4, 5], names=['XOBS', 'YOBS',   'YCALC', 'DELTA', 'DELTA/SIGMA'])
                q = df['XOBS']
                measured_values = 10**(df['YOBS'])
                fit = 10**(df['YCALC'])
                for key in df.keys(): data_refl[key] = df[key]
                data_refl.q, data_refl.measured_values, data_refl.fit, data_refl['theta'] = q, measured_values, fit, q2th(self.wavelength, q)
        
            if not self.fresnelfile == '':   
                # Fresnel-Daten aus der RF_Out-Datei
                fresnel = read_csv(self.fresnelfile, header=self.headerlen_ff, sep='\s+', usecols=[
                    1, 2, 3, 4, 5], names=['XOBS', 'YOBS', 'YCALC', 'DELTA', 'DELTA/SIGMA'])
                q_fresnel = fresnel['XOBS']
                measured_values_fresnel = 10**(fresnel['YOBS'])
                fit_fresnel = 10**(fresnel['YCALC'])
                if not self.outfilename == '':
                    YOBS_norm = measured_values / fit_fresnel
                    YCALC_norm = fit / fit_fresnel
                else:
                    YOBS_norm = np.repeat(np.nan, len(fit_fresnel)+1)
                    YCALC_norm = np.repeat(np.nan, len(fit_fresnel)+1)
                data_refl['fit_fresnel'], data_refl[ 'YOBS_norm'], data_refl['YCALC_norm'], data_refl['q_fresnel'], data_refl['measured_values_fresnel'] =\
                 fit_fresnel, YOBS_norm, YCALC_norm, q_fresnel, measured_values_fresnel
            if not self.rftfiles == [] and not self.rftfiles == '':
                rftfile = sorted(self.rftfiles, key = len)[0]
                    # Dipsersion aus der rft-Datei
                eldens_data = readEldensFile(rftfile)
                rho = 2 * constants.pi * eldens_data.delta / (self.wavelength**2 * self.r_e_angstrom)
                eldens_data['eldens'] = rho
            else:
                eldens_data = {}

            t2 = perf_counter()
            if timeit:
                print(f'Script took {t2-t1:3f} seconds.' )

            return data_refl, eldens_data

    def find_turning_points_eldens(self, min_x_distance = 1.5, min_y = 0.02, sum_up_values_at_index = 2, diff_threshold = 1e-4,show=False):
        ret_data = {'layer_thickness':None, 'figure':None, 'tp_inds':None, 'tp_zpos':None, 'tp_eldens':None}
        _, eldens_data = self.read_XRR_data()
        filtered_indices, filtered_x_coords, filtered_turning_points = list(), list(), list()
        # Berechne die zweite Ableitung
        dy = np.gradient(eldens_data.eldens, edge_order=2)
        ddy = np.gradient(dy, edge_order=2)
        dddy = np.gradient(ddy, edge_order=2)
        # Finde die Indizes der Nullstellen in der zweiten Ableitung
        zero_indices = np.where(np.diff(np.sign(ddy)))[0]
        # Füge den ersten und den letzten Index hinzu, um sicherzustellen, dass alle Wendepunkte abgedeckt sind
        # zero_indices = np.concatenate(([0], zero_indices, [len(y) - 1]))
        # Gib die Indizes und Werte der Wendepunkte zurück
        turning_points = eldens_data.eldens[zero_indices].to_numpy()
        tp_x = eldens_data.z[zero_indices].to_numpy()
        for i in range(1, len(turning_points)-1):
            diff = turning_points[i+1] - turning_points[i]
            # diff = turning_points[i] - turning_points[i+1]
            x_coord_dist = math.dist([tp_x[i]], [tp_x[i+1]])
            if abs(diff) > diff_threshold and x_coord_dist > 1.5 and turning_points[i] > min_y:
                # print(f'Turning point with index {zero_indices[i]} at x = {tp_x[i]:.4f}\u2009\u212B appended.')
                filtered_indices.append(zero_indices[i])
                filtered_x_coords.append(tp_x[i])
                filtered_turning_points.append(eldens_data.eldens[zero_indices[i]])
        filtered_x_coords, filtered_turning_points = filtered_x_coords[1:], filtered_turning_points[1:]
        layer_thickness = np.diff(filtered_x_coords)
        layer_thickness = np.insert(layer_thickness,sum_up_values_at_index, layer_thickness[sum_up_values_at_index] + layer_thickness[sum_up_values_at_index + 1])
        layer_thickness = np.delete(layer_thickness, sum_up_values_at_index +1)
        layer_thickness = np.delete(layer_thickness, sum_up_values_at_index +1)
        print(layer_thickness)
        # for i in range(0,len(filtered_turning_points)-1):
        #     print(i, filtered_x_coords[i])
            # print(f'Distance between neigbhouring turning points at {tp_x[i]:.4f}\u2009\u212B and {tp_x[i+1]:.4f}\u2009\u212B: {filtered_x_coords[i]}\u2009\u212B')
        fig = go.Figure(layout = layout.eldens())
        fig.add_trace(go.Scatter(x= filtered_x_coords, y=filtered_turning_points, mode = 'markers', marker = dict(color = 'rgba(0,0,0,0)', size = 10, line = dict(width = 2)),name='Wendepunkte'))
        fig.add_trace(go.Scatter(x = eldens_data.z, y = eldens_data.eldens, name = 'Elektronendichte'))
    
        ret_data['layer_thickness'], ret_data['figure'], ret_data['tp_inds'], ret_data['tp_zpos'], ret_data['tp_eldens'] = layer_thickness, fig, filtered_indices, filtered_x_coords, filtered_turning_points
        
        return ret_data
    
    def readDistFile(self, model = 'tidswell', OTS = True, layOnSub = [], convertDispersion = True, widthGasLayer = False, meanDensity = True,
        show = True, calc_layer_properties = True, shiftSubstrate2zero = False, timeit = False):
        '''
            Read data from distfile if that file is available. Filename hast to end with "dist.rft"! By specifying the system, a guess of the number of columns is made.
            --------------
            Returns: Data in pandas.DataFrame
            --------------
            optional parameters:
                * model: ['steinrueck', 'tidswell', 'none', num]. Default: tidswell. 
                    - steinrueck: an extra column is assumed for the layer between Si and SiO2.
                    - tidswell: only layer for Si and SiO2 are assumed as substrate.
                    - none: no columns for substrate assumed
                * layOnSub: adds column for e.g. an adsorbant on the substrate
                * gas: Str, if given, the columns after the substrate are named with gas.
                * OTS: Adds two columns for head and tailgroup
                * convertDispersion: Convert the dispersion values to an electron density in number of electrons per angstrom cubed.
                * meanDensity: returns maximum density of all layers in dictionary.
                * show: If True, all layers are plotted.
        '''
        t1 = perf_counter()
        ret_data = {'dist_data_lsfit':None, 'dist_data_sliced':None,'mean_density': None, 'width_adsorbate': None, 'fig_lsfit':None, 'con_params':None, 'layer_properties':None, 'weighted_dispersions':None, 'fig_dist_sliced':None}
        colnames = {'tidswell':['Si', 'SiO2'], 'steinrueck':['Si','BetweenSiSiO2','SiO2'], 'none':[]}
        if not isinstance(model, list):
            [colnames[m].extend(['OTS_head', 'OTS_tail']) for m in colnames.keys() if OTS]
            [colnames[m].extend([name for name in layOnSub]) for m in colnames.keys() if not layOnSub == []]
        else:
            colnames['custom'] = model
            model = 'custom'
        anzcols = len(colnames[model])
        colnames[model].insert(0, 'z')  
        numLayOnSub = len(layOnSub)
        
        distfile = self.distfile
        # outfile = self.outfilename
        # lines_outfile = read_lines_file(outfile)
        # for l in lines_outfile:
        #     if all(elem in l for elem in( 'mber', 'of', 'layers')):
        #         numbLayers = l.split()[-1]
        #         numbLayers = int(float(numbLayers))
        # print(f'cols: {colnames[model]} with len: {len(colnames[model])} and anz_cols:{anzcols}')

        if not distfile:
            print('No ' + 'layer distribution file found in '  + self.Filepath)
            pass
        else:
            df = read_csv(distfile, header=None, skiprows=2, sep='\s+', names = colnames[model])
            for col in df.keys():
                if not col == 'z':
                    if convertDispersion:
                        eldens = disp2edens(wavelength = self.wavelength, dispvalues = df[col])
                    else:
                        eldens = df[col]
                    df[col] = eldens
            height = df['z']

            if widthGasLayer:
                gasLayers = df[layOnSub[:]]
                halfMax = [max(df[l]) / 2 for l in gasLayers]
                halfMaxDict, idxDict, leftIdx, rightIdx, width = dict(), dict(), [], [], []
                [halfMaxDict.update({k: halfMax[i]}) for i, k in enumerate(layOnSub)]
                for i, l in enumerate(gasLayers):
                    d = np.sign(halfMax[i] - np.array(df[l][0:-1])) - np.sign(halfMax[i] - np.array(df[l][1:]))
                    leftIdx.append(np.where(d > 0)[0])
                    rightIdx.append(np.where(d < 0)[-1])
                for idx in range(0, len(leftIdx)):
                    hl = np.array(height[leftIdx[idx]])
                    hr = np.array(height[rightIdx[idx]])
                    w = np.sum(hr - hl)
                    width.append(w)
                width = "{:.4f}".format(sum(width))
                self.width = width

                ret_data['width_adsorbate'] = self.width
            
            if calc_layer_properties:
                lay_widths, roughness, eff_density, inflection_points = list(), list(), list(), list()
                outfile_data = read_csv(self.outfilename, header=self.headerlen_outfile, sep='\s+', usecols=[1, 2, 3, 4, 5], names=['XOBS', 'YOBS', 'YCALC', 'DELTA', 'DELTA/SIGMA'])
                q = outfile_data['XOBS']
                cp = self.resort_conparams(anz_layers=anzcols -1)
                # cp = self.resort_conparams()
                lay_params = list(cp['layer_parameters'].values())
                layers = [XRRa.layer(*lay_params[i]) for i in range(len(lay_params))]

                rhos, dispersions, absorptions, roughness = list(), list(), list(), list()
                for i in range(len(layers)):
                    rhos.append(disp2edens(self.wavelength, layers[i].delta))
                    dispersions.append(layers[i].delta)
                    absorptions.append(layers[i].beta)
                    roughness.append(abs(layers[i].roughness))
                # rhos = [disp2edens(self.wavelength, layers[i].delta) for i in range(len(layers))]
                # dispersions = [layers[i].delta for i in range(len(layers))]
                # absorptions = [layers[i].beta for i in range(len(layers))]
                # roughness = [abs(layers[i].roughness) for i in range(len(layers))]
                simul = XRRa.refl_multiple_layers(q, layers, self.wavelength, ind_of_refrac_profile='tanh', alpha_i_unit='Ae-1')
                EffDichteModeling = simul.EffectiveDensitySlicing(step = .05, shiftSubstrate2zero=shiftSubstrate2zero, convert2eldens = True)  
                Wfuncs = EffDichteModeling['weighted_dispersions']

                W_traces, fit_traces, tp_traces = list(), list(), list()
                cmap  = create_colormap(len(Wfuncs), 'viridis', extendRangeWith=5)
                for ind in range(0, len(Wfuncs)-1):
                    W = Wfuncs[ind]
                    z = EffDichteModeling['z']
                    cen_pos = z[np.argmax(W)]
                    amplitude = W[np.argmax(W)]
                    # print(type(W), type(z), type(cen_pos), type(amplitude))
                    if ind == 0:
                        if shiftSubstrate2zero:
                            cen_pos = 0
                        try:
                            cen_ind = find_nearest(W,amplitude/2)
                            peak = model_neg_step(x = z, y = W, amplitude = amplitude, mu = z[cen_ind], sigma = 0.5, form = 'error_fct')
                            infl = find_nearest(z, peak.params['mu'].value)

                            # peak = model_subbotin(x = z, y = W, beta = 2, sigma = 1.5, center = cen_pos, amplitude=amplitude)
                            # sigma_infl = peak.params['sigma']    
                            # infl = find_inflection_points(W, sigma = sigma_infl, filter_mode='mirror', smoothen = True)
                            # infl = infl[0]
                        except Exception as e:
                            print('Subbotin model for step not working')
                            infl = find_nearest(W, amplitude / 2)
                    else:
                        peak = skewedGaussModel(x = z, y = W, sigma = 1.5, center = cen_pos, gamma = 0, amplitude=amplitude)
                        sigma_infl = peak.params['sigma']
                        if peak.chisqr > 0.1:
                            peak = model_subbotin(x = z, y = W, beta = 2, sigma = 1.5, center = cen_pos, amplitude=amplitude)
                            sigma_infl = peak.params['sigma']

                        infl = find_inflection_points(W, sigma = sigma_infl, filter_mode='mirror')
                        
                    if not ind == 0:
                        ft = go.Scattergl(x = z, y = peak.best_fit, mode = 'lines', line = dict(color = cmap[ind], dash = 'dash'), name = f'Fit #{ind}', legendgroup=ind, showlegend=True)
                        fit_traces.append(ft)
                    # if not ind == 0:
                        p1 = z[infl[0]]
                        p2 = z[infl[1]]
                        y2, y1 = Wfuncs[ind][infl[1]], Wfuncs[ind][infl[0]]
                        tp = go.Scattergl(x = [p2, p1], y = [y2, y1], mode = 'markers', marker = dict(size=8, color = cmap[ind]), name = f'inflection points #{ind}',legendgroup=ind, showlegend=True)
                    else:
                        fit_traces.append(go.Scattergl(x = z, y = peak.best_fit, mode = 'lines', line = dict(color = cmap[ind], dash = 'dash'), name = f'Fit #{ind}', legendgroup=ind, showlegend=True))
                        tp = go.Scattergl(x = [z[infl]], y = [Wfuncs[ind][infl]], mode = 'markers', marker = dict(size=8, color = cmap[ind]), name = f'inflection points #{ind}',legendgroup=ind, showlegend=True)
                    tp_traces.append(tp)
                    
                    if ind == 0:
                        substrate_pos = z[infl]
                        # substrate_pos = z[infl[0]]
                        lay_widths.append(np.inf)
                        eff_density.append(disp2edens(self.wavelength, layers[ind].delta))
                        inflection_points.append(substrate_pos)
                    # elif 0 < ind < len(Wfuncs) -1:
                    elif 0 < ind < len(Wfuncs):
                        inflection_points.append((p2, p1))
                        # lay_widths.append(abs(abs(p1)-abs(p2)))
                        lay_widths.append(math.dist([p2], [p1]))
                        # lay_widths.append(abs(p2-p1))
                        eff_density.append(max(Wfuncs[ind]))

                lay_widths.append(np.inf)
                inflection_points.append(np.nan)
                eff_density.append(max(Wfuncs[len(Wfuncs)-1]) * rhos[len(Wfuncs)-1])
                lp_indices = colnames[model][1:] + ['reference material']
                layer_properties = DataFrame(index = lp_indices, columns = ['Effective electron density', 'Nominal electron density', 'roughness', 'thickness', 'inflection_points', 'absorption'])
                # calculated from inflection points
                layer_properties['thickness'] =  lay_widths
                layer_properties['Effective electron density'] = eff_density
                layer_properties['Nominal electron density'] =  rhos
                layer_properties['dispersion'] = dispersions
                layer_properties['absorption'] = absorptions
                layer_properties['roughness'] = roughness
                layer_properties['inflection_points'] = inflection_points
                tilt_ang = np.arccos(layer_properties['thickness'].loc['OTS_tail'] / (18*1.26)) * 180 / np.pi
                # layer_properties['tilt_angle'] = np.nan
                layer_properties['tilt_angle'] = tilt_ang
                distribution_figure = EffDichteModeling['fig_disp_profile']
                distribution_figure.add_traces(tp_traces)
                distribution_figure.add_traces(fit_traces)
                # distribution_figure.update_layout(xaxis_range = [])
                # distribution_figure.update_layout(width = 800, height = 900)

                ret_data['fig_dist_sliced'] = distribution_figure
                ret_data['layer_properties'] = layer_properties
          
                df_sliced = DataFrame.from_dict(data = Wfuncs)
                # df_sliced.set_axis(colnames[model][1:] + ['reference material'], axis=1,inplace=True)
                df_sliced = df_sliced.set_axis(colnames[model][1:] + ['reference material'], axis='columns')
                # print(colnames[model][1:])
                df_sliced.insert(0, 'z', z)
                ret_data['dist_data_sliced'] = df_sliced

            if meanDensity:
                mean_dens = dict()
                for k in df.keys():
                    if not k == 'z':
                        maxdens = max(df[k])
                        mean_dens.update({k:maxdens})
                # print(mean_dens)
            numb_layer = anzcols - 2 # substract 1 for z column and one for substrate
            con_params = self.con_params()
            sigmas = self.roughness
            sigmas = list(sigmas.values())
            reference_density = df[colnames[model][-1]].iloc[-1]
            layer_lines, intersections_ind, pos_inter, W, zeta = dict(), [], [], {},{}
            [layer_lines.update({k:(df.z, df[k])}) for k in df.keys() if not k == 'z']
            ret_data['mean_density'] = mean_dens
            ret_data['con_params'] = con_params
           
            if show:
                traces = []
                for key in df.keys():
                    if not key == 'z':
                        traces.append(go.Scatter(x = height, y = df[key], mode = 'lines', line = dict(width = 4), name = key,
                            )
                        )

                # traces.insert(len(traces),go.Scatter(x = height, y = d_oben / d_unten, mode = 'lines+markers', name = "profile",
                    # line_width = 4))

                f = go.Figure(data = traces, layout= layout.eldens())
                f.update_layout(yaxis_autorange=True, xaxis_autorange = True)   
                ret_data['fig_lsfit'] = f
                # f.show()
        if calc_layer_properties:
            ret_data['weighted_dispersions'] = Wfuncs
        ret_data['dist_data_lsfit'] = df
        # if meanDensity:
        #     return mean_dens, df
        # else:
        #     return df
        t2 = perf_counter()
        if timeit:
            print(f'Calculations from distribution file needed {t2-t1:.3f} seconds.')
        return ret_data

    def con_params(self, anz_layers = None):
        conparams = {}
        self.thickness = {}
        self.roughness = {}
        self.absorbtion = {}
        self.dispersion = {}
        param_keys = ['footprint','background', 'resolution', 'disp_reference', 'disp_layer0','absorb_layer0','sigma_layer0', 'intensity_off', 
            'disp_layer1', 'absorb_layer1', 'sigma_layer1', 'd_layer1', 'disp_layer2', 'absorb_layer2', 'sigma_layer2', 'd_layer2', 'disp_layer3', 'absorb_layer3',
             'sigma_layer3', 'd_layer3', 'disp_layer4', 'absorb_layer4', 'sigma_layer4', 'd_layer4', 'disp_layer5', 'absorb_layer5', 'sigma_layer5', 'd_layer5',
             'disp_layer6', 'absorb_layer6', 'sigma_layer6', 'd_layer6']
        try:
            kfiles = [os.path.join(self.Filepath,f) for f in os.listdir(self.Filepath) if '.k' in f]
        except TypeError:
            kfiles = []
            for f in os.listdir(self.Filepath):
                kfile = os.path.join(self.Filepath, f.decode('utf-8'))
                if '.k' in kfile:
                    kfiles.append(kfile)
        # latest_save =sorted(kfiles, key=lambda f:[int(f.split('.k')[-1]) for f in kfiles])[0]
        if not kfiles == []:
            latest_save =sorted(kfiles, key=lambda x:int(x.split('.k')[-1]))[-1]
            df = read_fwf(latest_save, header=3, skipfooter=3, widths=[
                36, 15, 14], names=['parameter', 'Value', 'Increment'], skip_blank_lines=True)
            # print(df)
            for v, p in zip(df.Value, param_keys):
                conparams.update({p: v})
                if 'disp_' in p:
                    self.dispersion.update({p: v})
                if 'absorb' in p:
                    self.absorbtion.update({p: v})
                if 'sigma_' in p:
                    self.roughness.update({p: v})
                if 'd_' in p:
                    self.thickness.update({p: v})
        if anz_layers:
            pass
            # for i in range(anz_layers):
        return conparams
    def con_params_from_outfile(self, anz_layers = None):
        conparams = {}
        self.thickness = {}
        self.roughness = {}
        self.absorbtion = {}
        self.dispersion = {}
        param_keys = ['footprint','background', 'resolution', 'disp_reference', 'disp_layer0','absorb_layer0','sigma_layer0', 'intensity_off', 
            'disp_layer1', 'absorb_layer1', 'sigma_layer1', 'd_layer1', 'disp_layer2', 'absorb_layer2', 'sigma_layer2', 'd_layer2', 'disp_layer3', 'absorb_layer3',
             'sigma_layer3', 'd_layer3', 'disp_layer4', 'absorb_layer4', 'sigma_layer4', 'd_layer4', 'disp_layer5', 'absorb_layer5', 'sigma_layer5', 'd_layer5',
             'disp_layer6', 'absorb_layer6', 'sigma_layer6', 'd_layer6']
        outfile_lines = read_lines_file(self.outfilename)
        for i in range(len(outfile_lines)):
            if 'the model function was calculated' in outfile_lines[i]:
                ind_start = i + 2
            elif 'part' in outfile_lines[i]:
                current_layer = int(outfile_lines[i].split('part')[0].strip().split()[-1])
            elif '###' in outfile_lines[i]:
                ind_end = i -2 
        anz_layers = int(float(outfile_lines[ind_end+3].split()[-1]))
        layer_lines_to_skip = 4*(current_layer-anz_layers)
        df = read_fwf(self.outfilename, header = ind_start, skipfooter = len(outfile_lines) - ind_end + (layer_lines_to_skip-1), widths = [39, 15, 11, 11], names = ['parameter', 'Value', 'Sigma', 'Last_change'], skip_blank_lines=True)
        for v, p in zip(df.Value, param_keys[:len(df.Value)-1]):
            conparams.update({p: v})
            if 'disp_' in p:
                self.dispersion.update({p: v})
            if 'absorb' in p:
                self.absorbtion.update({p: v})
            if 'sigma_' in p:
                self.roughness.update({p: v})
            if 'd_' in p:
                self.thickness.update({p: v})
        return conparams

    def resort_conparams(self, anz_layers = None):
        if not anz_layers:
            anz_layers = 100 # make sure to have enough entries
        cp = self.con_params_from_outfile()
        dispersion, absorption, thickness, roughness = list(), list(), list(), list()
        conparams = {'layer_parameters':{}, 'experimental_parameters':{}}
        # conparams = {}
        for k in cp.keys():
            if isinstance(cp[k], str):
                cp[k] = float(cp[k])
            if 'disp_reference' in k:
                ref_disp = cp[k] *1e-6
            try:
                laynumb = int(re.sub('\D','', k))
            except ValueError:
                laynumb = None
            if not laynumb == None and laynumb <= anz_layers:
                if 'disp' in k:
                    dispersion.append(cp[k] * 1e-6)
                if 'absorb' in k:
                    # absorption.append(cp[k])
                    absorption.append(dispersion[laynumb] / (cp[k]))
                if 'sigma' in k:
                    roughness.append(cp[k])
                if 'd_' in k:
                    thickness.append(cp[k])
            elif not laynumb and not 'disp_reference' in k:

                conparams['experimental_parameters'][k] = cp[k]
        thickness.insert(0, np.inf)
        roughness.insert(0, 0)
        thickness.append(0)
        dispersion.append(ref_disp)
        absorption.append(0)
        layer_keys = np.arange(0, anz_layers+2, 1)
        layer_keys = [f'layer_{str(lk)}' for lk in layer_keys]
        for i in range(len(thickness)):
            conparams['layer_parameters'].update({layer_keys[i]:(dispersion[i], absorption[i], thickness[i], roughness[i])})

        return conparams        
    

    # def first_min_refl(self, first_min_guess=0.1, nofit=False, from_outfiles = False, print_all_min_pos = False, smooth_data = True, show = False,**kwargs):
    #     ''' Die Funktion findet die Postion des ersten Minmums einer Reflektivität in q. Benötigt wird der Dateipfad,
    #             indem die Outdatei liegt. Mit first_min_guess
    #         wird eine erste Schätzung der Position abgegben.
    #            '''
    #     data_refl, _ = self.read_XRR_data()
    #     if nofit and not from_outfiles:
    #         datfile_wp = min([os.path.join(self.Filepath,erg.name) for erg in os.scandir(self.Filepath) if erg.name.endswith('.dat')])
    #         data = readDatFileData(datfile_wp)
    #         q = data.q
    #         counts = data.counts
    #     elif nofit and from_outfiles:
    #         q = data_refl.XOBS
    #         counts = data_refl.YOBS_norm
    #         if smooth_data:
    #             if 'window_size' in kwargs.keys():
    #                 window_size = kwargs['window_size']
    #             else:
    #                 window_size = 5
    #             smoothed_data = savgol_filter(data.counts, window_size, 3)
    #             minima = argrelmin(smoothed_data, order = 3)
    #             q_first_min = min(data.q[minima[0]], key=lambda x: abs(x-first_min_guess))
    #             if show:
    #                 trace_data = go.Scattergl(x = data.q, y = data.counts, mode = 'lines', line = dict(width = 3, dash = 'dash', color = 'black'), name = 'data')
    #                 trace_smooth = go.Scattergl(x = data.q, y = smoothed_data, mode = 'lines', line = dict(width = 3, dash = 'solid', color = 'green'), name = 'savgol')
    #                 fig = go.Figure(data = [trace_data, trace_smooth], layout = layout.refl())
    #                 fig.show()
    #         else:
    #             minima = find_peaks(-data['counts'])
    #             q_first_min = min(data['q'][minima[0]], key=lambda x: abs(x - first_min_guess))

    #     elif nofit and from_outfiles:
    #         pass
    #     else:
    #         # data = dict()
    #         # data, _ = self.read_XRR_data()
    #         minima = find_peaks(- data_refl.YCALC_norm)
    #         q_first_min = min(data_refl.q[minima[0]], key=lambda x: abs(x - first_min_guess))
    #     if print_all_min_pos:
    #         print(data_refl.q[minima[0]])
    #     return q_first_min

    def first_min_refl(self, first_min_guess=0.1, second_min_guess = 0.4, nofit=False, from_outfiles = False, print_all_min_pos = False, smooth_data = True, show = False, min_q = 0.08,**kwargs):
        ''' Die Funktion findet die Postion des ersten Minmums einer Reflektivität in q. Benötigt wird der Dateipfad,
                indem die Outdatei liegt. Mit first_min_guess
            wird eine erste Schätzung der Position abgegben.
               '''
        # read XRR data
        data_refl, _ = self.read_XRR_data()
        # check for kwargs regarding savgol filter:
        if 'window_size' in kwargs.keys():
                window_size = kwargs['window_size']
        else:
            window_size = 5
    
        if 'poly_order' in kwargs.keys():
            poly_order = kwargs['poly_order']
        else:
            poly_order = 3

        if nofit and not from_outfiles:
            datfile_wp = min([os.path.join(self.Filepath,erg.name) for erg in os.scandir(self.Filepath) if erg.name.endswith('.dat')])
            data = readDatFileData(datfile_wp)
            q = data.q
            counts = data.counts

        elif nofit and from_outfiles:
            q = data_refl.q
            counts = data_refl.YOBS_norm

        elif not nofit:
            q = data_refl.q
            counts = data_refl.YCALC_norm
        # cut off beginning of refl to avoid getting minima at critical angle 
        
        if smooth_data:
            smoothed_data = savgol_filter(counts, window_size, poly_order)
            minima = argrelmin(smoothed_data, order = 3)
            q_first_min = min(q[minima[0]], key=lambda x: abs(x-first_min_guess))
            # q_second_min = min(q[minima[0]], key=lambda x: abs(x-second_min_guess))
            # print(q_second_min)
            # print(f'Layer thickness:{2*np.pi / (q_second_min - q_first_min):.2f}')
            if show:
                trace_all_mins = go.Scattergl(x = q[minima[0]], y = smoothed_data[minima[0]], mode = 'markers', marker = dict(symbol ='x', size = 16, color = 'red'), name = 'minima')
                trace_data = go.Scattergl(x = q, y = counts, mode = 'lines', line = dict(width = 3, dash = 'dash', color = 'black'), name = 'data')
                trace_smooth = go.Scattergl(x = q, y = smoothed_data, mode = 'lines', line = dict(width = 3, dash = 'solid', color = 'green'), name = 'savgol')
                fig = go.Figure(data = [trace_data, trace_smooth, trace_all_mins], layout = layout.refl())
                fig.show()
        if not smooth_data:
            minima = find_peaks(-counts)
            q_first_min = min(q[minima[0]], key=lambda x: abs(x - first_min_guess))
        if not smooth_data and show:
            trace_all_mins = go.Scattergl(x = q[minima[0]], y = counts[minima[0]], mode = 'markers', marker = dict(symbol ='x', size = 16, color = 'red'), name = 'minima')
            trace_data = go.Scattergl(x = q, y = counts, mode = 'lines', line = dict(width = 3, dash = 'dash', color = 'black'), name = 'data')
            fig = go.Figure(data = [trace_data, trace_all_mins], layout = layout.refl())
            fig.show()
        if q_first_min < min_q: print('minima is probly at crit ang')
        if print_all_min_pos:
            print(data_refl.q[minima[0]])
        return q_first_min

def VolumeFractionProfile(referencefile, dispfile, molecule, step = .1, wavelength = None, show = False, zrange = 'All',
    header_len = 2, threshold = 1e-3, find_turning_points = True, calc_layer_width = False, anz_el = False, verbose = False, exclude_tail_part = False):
    '''Calculate volume fraction profile from electron density profile from LSFit
    Parameters
    ----------
    referencefile: File with electron density profile of substrate.
    dispfile: File with electron density profile of sample system       
    molecule: str, Type of adsorbent, e.g. CO2. Needed to calculate the integrated electron density of the adsorbed layer
    Kwargs:
    -------
    step: float,stepsize in Angstrom between the points for interpolation. Default: 0.1 Angstrom
    wavelength: float, wavelength of radiation in Angstrom. Default 0.4592 (27 keV)
    show: boolean, if True a plot of the volume fraction profile is shown
    zrange: 'All', 'auto' or [z_min, z_max]: If 'All': integration takes place over the whole range. If 'auto', the range for integration is calculated automatically. If [z_min, z_max], this range is used.
    threshold: 
    find_turning_points: boolean, if True, the 
    calc_layer_width: boolean, if True, the width of the adsorbed layer is calculated using the points of inflection of the difference between the reference and the profile with adsorbent.
    Returns:
    --------
    int_val: Integrated electron density
    df: pandas.Dataframe with interpolated electron density values of the reference, the sample system and the difference betwee them

    '''


    return_data = {'int_val': None, 'df':None, 'FWHM_adsorbate':None, 'figure':None, 'int_val_minus_ref_mat':None, 'surfac_excess':None}
    if not wavelength:
        wavelength = 0.4592
        print('No wavelength was specified. ' + '\u03BB = ' + "{:.4f}".format(wavelength) + ' \u00C5' +' is used as default.')

    if not anz_el:
        fluid_props = FP.selectMolecule(molecule)
        anz_el = fluid_props.anz_el

    referenceData = readEldensFile(referencefile)
    dispData = readEldensFile(dispfile)
    
    min_z = min([min(dispData.z), min(referenceData.z)])
    max_z = max([max(dispData.z), max(referenceData.z)])
    
    z_ref_min = np.floor(min_z)
    z_ref_max = np.ceil(max_z)
    
    referenceData.z.iloc[0] = z_ref_min
    dispData.z.iloc[0] = z_ref_min
    
    last_ind_ref = len(referenceData.z) -1
    last_ind_disp =  len(dispData.z) -1
    
    dispData.z.last_ind_disp = z_ref_max
    referenceData.z.last_ind_disp = z_ref_max
    z_interpol = np.arange(start = z_ref_min, stop = z_ref_max + step, step = step)
    interpol_ref = interp1d(x = referenceData.z, y = referenceData.delta, fill_value = 'extrapolate')
    interpol_disp = interp1d(x = dispData.z, y = dispData.delta, fill_value = 'extrapolate')
    
    ref_vals_new = interpol_ref(z_interpol)
    ref_vals_new = disp2edens(dispvalues = ref_vals_new, wavelength = wavelength)
    
    disp_vals_new = interpol_disp(z_interpol)
    disp_vals_new = disp2edens(dispvalues = disp_vals_new, wavelength = wavelength)

    difference = disp_vals_new - ref_vals_new
    reference_mat_disp = disp_vals_new[-1]
    ref_disp_deviation = disp_vals_new- reference_mat_disp

    # calculate roughness of left and right interfaces from VFP using its derivative    
    difference_prime = np.diff(difference)

    # print(pws)
    
    if not any(zrange == x for x in ['All', 'auto']):
        zrangemin = find_nearest(z_interpol, zrange[0])
        zrangemax = find_nearest(z_interpol, zrange[1])
        # diffsum = np.sum(difference[zrangemin:zrangemax+1])
    elif zrange == 'auto':
        zrangemin, zrangemax = [],[]
        # difference_prime = scdiff(difference)
        i = 0
        while i in range(len(z_interpol) - 1):
            if difference[i] > threshold:
                zrangemin = i
                break
            i += 1   
        j = 0
        while j in range(len(z_interpol) - 1):
            if ref_disp_deviation[j] < threshold:
                zrangemax = j
                break
            j += 1
        if exclude_tail_part:
            peak_diff = find_peaks(difference_prime, height = 0.1*max(difference_prime), distance = 2, width = 5)[0]
            if len(peak_diff) >=1:
                zrangemin = peak_diff[0]
        if zrangemin == [] or zrangemax == []:
            zrangemin, zrangemax = 0, len(z_interpol) -1
            print('No indices found. Integration over whole range.', zrangemin, zrangemax)
    else:
        zrangemin, zrangemax = 0, len(z_interpol) - 1

    # For total adsorbed amount 
    diffsum = np.trapz(difference[zrangemin:zrangemax], z_interpol[zrangemin:zrangemax])
    # For surface excess
    diffsum_gibbs = np.trapz(ref_disp_deviation[zrangemin:zrangemax], z_interpol[zrangemin:zrangemax])
    # diff_gibbs = difference- reference_mat_disp
    # diffsum = np.trapz(diff_gibbs[zrangemin:zrangemax], z_interpol[zrangemin:zrangemax])
    int_val = diffsum / anz_el
    int_val_excess = diffsum_gibbs / anz_el
    # print(int_val*1e2)
    a = FWHM(difference)
    
    # If reference material density is greate than half max of peak
    if a[1].size == 0 and a[0].size >0:
        a_mref_r = FWHM(ref_disp_deviation)[1]
        il, ir = a[0], a_mref_r
        zl, zr = z_interpol[a[0]], z_interpol[a_mref_r]
    # normal case for a peak
    elif a[0].size == 1 and a[1].size == 1 :
        il, ir = a[0], a[1]
        zl, zr = z_interpol[a[0]], z_interpol[a[1]]
    # multiple FWHMS
    elif a[0].size > 1 and a[1].size > 1:
        print(f'You have multiple peaks: {a[1]}')
        # il, ir =a[-2], a[-1]
        il, ir = np.array([a[0][1]]), np.array([a[1][1]])
        zl, zr = z_interpol[il], z_interpol[ir]

    # no peak, e.g. for reference measurement
    elif a[0].size == 0 and a[1].size == 0:
        il, ir = [], []
        zl, zr = np.array([]), np.array([])

    dist = math.dist(zl, zr)
    if verbose:
        print(f'{int_val * 100:.5f} molecules/nm\u00b2={int_val:.5f} molecules/\u212b\u00b2\n\u03c1={diffsum:.5f} e\u207B/\u212B\u00b2')
    df = DataFrame(list(zip(z_interpol, ref_vals_new, disp_vals_new, difference)),
        columns = ['z_interpol', 'ref_vals_new', 'disp_vals_new', 'difference'])
    if show:
        trace1 = go.Scatter(x = z_interpol, y = ref_vals_new, mode = 'lines', line_color = 'black', line_width = 4, name = 'reference')
        trace2 = go.Scatter(x = z_interpol, y = disp_vals_new, mode = 'lines', line_color = 'red', line_width = 4, name = 'profile')
        trace3 = go.Scatter(x = z_interpol, y = difference, mode = 'lines', line_color = 'blue', line_width = 4, name = 'difference')
        trace4 = go.Scatter(x = z_interpol, y = abs(difference_prime), yaxis = 'y2',mode = 'lines',line_color = 'green', line_width = 4, name = 'derivative')
        dot_zmin = go.Scatter(x = [z_interpol[zrangemin]], y = [difference[zrangemin]], mode = 'markers', marker = dict(line = dict(color = 'red'), size = 12, color = 'red'),
            showlegend = False, hoverinfo = 'skip')
        dot_zmax = go.Scatter(x = [z_interpol[zrangemax]], y = [difference[zrangemax]], mode = 'markers', marker = dict(line = dict(color = 'red'), size = 12, color = 'red'),
            showlegend = False, hoverinfo = 'skip')
        data = [trace1, trace2, trace3, trace4,dot_zmin, dot_zmax]
        # if zrange == 'auto':
        #     trace4 = go.Scatter(x = z_interpol, y = abs(difference_prime), yaxis = 'y2',mode = 'lines',line_color = 'green', line_width = 4, name = 'derivative')
        #     data = data + [trace4]
        fig = go.Figure(data = data, layout = layout.eldens())
        if not zrange == 'All':
            fig.add_trace(go.Scatter(x = z_interpol[zrangemin:zrangemax], y = difference[zrangemin:zrangemax], fill = 'tozeroy',
                fillcolor = 'rgba(149, 165, 166, .7)', mode = 'lines', showlegend = False, line = dict(width = 0, color = 'rgba(149, 165, 166, 1)'))
            )
        fig.update_layout(yaxis2 = dict(**layout.standard_linear_axis, side = 'right', overlaying='y'),
            width = 1200, height = 800, xaxis_range = [z_interpol[zrangemin] - 5, z_interpol[zrangemax] + 5])
        fig.show()

    if find_turning_points:
        difference_doubleprime = np.diff(np.diff(difference))
        height_doubleprime = z_interpol[:-2]
        tps = argrelextrema(difference_doubleprime, np.greater, order = 30)[0]
        turningpoints = [z_interpol[tp] for tp in tps]
        disp_turningpoints = [difference[tp] for tp in tps]
        # disp_turningpoints = [disp_vals_new[tp] for tp in tps]
        doubleprime = go.Scatter(x = height_doubleprime, y = 100 * difference_doubleprime, mode = 'lines', line = dict(
            width = 4, color = 'black'), name = 'doubleprime')
        # vol_frac_prof = go.Scatter(x = z_interpol, y = difference, mode = 'lines', line_color = 'blue', line_width = 4, name = 'difference')
        # make negative parts zero or not
        # y_val = np.where(difference < 0, 0, difference)
        y_val = difference
        vol_frac_prof = go.Scatter(x = z_interpol, y = y_val, mode = 'lines', line_color = 'blue', line_width = 4, name = 'difference')
        turning_points_trace = go.Scatter(x = turningpoints, y = disp_turningpoints, mode = 'markers', marker = dict(
            size = 10, color = 'red', line = dict(
                color = 'black', width = 2),
            ),
        name = 'turningpoints')
        eldens = go.Scatter(x = dispData.z, y= disp2edens(wavelength = wavelength, dispvalues = dispData.delta), mode = 'lines', line = dict(
            width = 4, color = 'green'), name = 'profile')
        # eldens = go.Scatter(x = z_interpol, y= disp_vals_new, mode = 'lines', line = dict(
        #     width = 4, color = 'green'), name = 'profile')
        eldens_reference = go.Scatter(x = referenceData.z, y= disp2edens(wavelength = wavelength, dispvalues = referenceData.delta), mode = 'lines', line = dict(
            width = 4, color = 'red'), name = 'reference')

        eldens_minus_refmat_disp = go.Scattergl(x = z_interpol, y = ref_disp_deviation,
            mode = 'lines', line = dict(width = 4, color = 'purple'), name = 'profile minus ref. mat')
        derivative_trace = go.Scatter(x = z_interpol, y = abs(difference_prime), yaxis = 'y2',mode = 'lines',line_color = 'green', line_width = 4, name = 'derivative')
        # FWHM_ind_l, FWHM_ind_r = a[0], a[1]
        FWHM_ind_l, FWHM_ind_r = il, ir
        if not FWHM_ind_l == [] and not FWHM_ind_r == []:
            z_FWHM_l, z_FWHM_r, y_FWHM_l, y_FWHM_r = z_interpol[FWHM_ind_l], z_interpol[FWHM_ind_r], difference[FWHM_ind_l], difference[FWHM_ind_r]
        else:
            z_FWHM_l, z_FWHM_r, y_FWHM_l, y_FWHM_r = [0], [0], [0], [0]
        z_FWHM = list(z_FWHM_l) + list(z_FWHM_r)
        diff_FWHM = list(y_FWHM_l) + list(y_FWHM_r)
        FWHM_trace = go.Scattergl(x = z_FWHM, y = diff_FWHM, mode = 'markers', marker = dict(size = 10, symbol = 'x', color = 'blue'), name = 'FWHM')

        # dt = go.Scattergl(x = df.z_interpol, y = df.difference, mode = 'lines', line = dict(color = 'green', width = 3), name = 'difference', showlegend = True)
        fig2 = go.Figure(data = [doubleprime,vol_frac_prof, turning_points_trace, eldens, eldens_reference, FWHM_trace, eldens_minus_refmat_disp, derivative_trace], layout = layout.eldens())
        fig2.update_layout(xaxis_range = [z_interpol[zrangemin] - 5, z_interpol[zrangemax] + 5],
            yaxis2 = dict(**layout.standard_linear_axis, side = 'right', overlaying = 'y'))
        return_data['figure'] = fig2
        # fig2.show()

    if calc_layer_width:
        try:
            cen_pos = z_interpol[difference == max(difference)]
            amplitude = max(difference)
            amplitude, cen_pos, difference, z_interpol = float(amplitude), float(cen_pos), list(difference), list(z_interpol)
            # print(type(amplitude), type(cen_pos), type(z_interpol), type(difference), len(z_interpol), len(difference))
            print(amplitude, cen_pos)
            peak = model_expGauss(x = z_interpol, y = difference, center= cen_pos, amplitude = amplitude, gamma = 1, sigma = 1, min_correl = 0.3)
            fit = peak.best_fit
            infls = find_inflection_points(fit)
            p0, p1 = z_interpol[infls[0]], z_interpol[infls[1]]
            thickness = math.dist([p1], [p0])
            fit_trace = go.Scattergl(x = z_interpol, y = fit, mode = 'lines', line = dict(color = 'green', dash = 'dash', width = 2), name = 'fit', showlegend=True)
            infl_trace = go.Scattergl(x = [p0, p1], y = [difference[infls[0]], difference[infls[1]]], mode = 'markers', name = 'inflection points')
            fig3 = go.Figure(data = [fit_trace, vol_frac_prof, infl_trace], layout = layout.eldens())
            fig3.show()
            print(thickness)
        except Exception as e:
            print(e)
            print('No adsorbed layer?!')

    if not float(difference[-1]) == 0.:
        ind_difference_above_refmat = np.where(difference > difference[-1])[0]
        # print(ind_difference_above_refmat)
        difference_above_refmat = difference[ind_difference_above_refmat]
        z_intergrate_min, z_integrate_max = z_interpol[ind_difference_above_refmat[0]], z_interpol[ind_difference_above_refmat[-1]]
        diff_interp = np.interp(z_interpol[ind_difference_above_refmat], z_interpol[ind_difference_above_refmat], difference_above_refmat)
        refmat_interp = np.interp(z_interpol[ind_difference_above_refmat], z_interpol[ind_difference_above_refmat], np.repeat(difference[-1], len(z_interpol[ind_difference_above_refmat])))
        area = np.trapz(diff_interp-refmat_interp, z_interpol[ind_difference_above_refmat])
        int_val_minus_ref_mat = area / anz_el
        return_data['int_val_minus_ref_mat'] = int_val_minus_ref_mat
    else:
        return_data['int_val_minus_ref_mat'] = int_val
    return_data['df'], return_data['int_val'], return_data['surfac_excess'] = df, int_val, int_val_excess
    # unbedingt nachher wieder einkommentieren
    return_data['FWHM_adsorbate'] = dist
    # return int_val, df
    return return_data

def Gibbs_excess_adsorption():
    pass

def roughness_from_VFP(VFP_data, verbose = False, show = False):
    '''
    Calculate roughness of left and right interface from VFP data using its derivative
    -----------
    Parameters:
        *VFP_data: data of VolumeFractionProfile calculated with XRR.data_evaluation.VolumeFractionProfile()
        *verbose: boolean, default Fsle: if True, print z-positions of peaks of VFP derivative to check, if the right peaks are chosen
    ----------
    Returns:
        * roughness: numpy.ndarray with roughnesses of left and right interface
    '''
    roughness = []
    VFP_df = VFP_data['df']
    VFP_peak_pos = VFP_df.z_interpol[VFP_df.difference.idxmax()]
    FWHM_data = peak_widths(VFP_df.difference, [VFP_df.difference.idxmax()], rel_height = 0.1)
    FWHM_l, FWHM_r = int(round(FWHM_data[2][0])), int(round(FWHM_data[3][0]))
    FWHM_width = VFP_df.z_interpol[FWHM_r]-VFP_df.z_interpol[FWHM_l]
    # Calculate derivative of VFP and make all peaks positive
    abs_deriv = abs(np.diff(VFP_df.difference))
    abs_deriv_smoth =  savgol_filter(abs_deriv, 50, 3)
    peak_inds = find_peaks(abs_deriv, height = 1e-5*max(VFP_df.difference))
    prominences = peak_prominences(abs_deriv, peak_inds[0])[0]
    peaks = abs_deriv[peak_inds[0]]
    pws = peak_widths(abs_deriv, peak_inds[0], rel_height = 0.5)
    for i in range(len(pws[3])):
        li, ri = int(round(pws[2][i])), int(round(pws[3][i]))
        lz,rz = VFP_df.z_interpol[li], VFP_df.z_interpol[ri]
        width = rz-lz
        if not width < 1 and prominences[i]>0.0005:
            roughness.append(width)
            if verbose:
                print(f'roughness is {width:.2f} at left_ind: {lz:.2f} and right_ind: {rz:.2f}. Peak prominence: {prominences[i]}')
            # else:
            #     print(f'Peak not regarded: roughness is {width:.2f} at left_ind: {lz:.2f} and right_ind: {rz:.2f}. Peak prominence: {prominences[i]}')
        # if VFP_peak_pos-lz > 3 *FWHM_width and VFP_peak_pos >lz:
        #     continue
        #     # print(f'Not the right peak {lz}, {rz}')
        # else:
        #     if not width < 1 and prominences[i]>0.0005:
        #         roughness.append(width)
        #         if verbose:
        #             print(f'roughness is {width:.2f} at left_ind: {lz:.2f} and right_ind: {rz:.2f}. Peak prominence: {prominences[i]}')
        #     # else:
        #     #     print(f'Peak not regarded: roughness is {width:.2f} at left_ind: {lz:.2f} and right_ind: {rz:.2f}. Peak prominence: {prominences[i]}')
    if show:
        VFP_trace = go.Scatter(x = VFP_df.z_interpol, y = VFP_df.difference, mode = 'lines', line = dict(color = 'green', width = 3), name = 'VFP')
        deriv_tace = go.Scatter(x = VFP_df.z_interpol, y = abs_deriv, yaxis = 'y2', mode = 'lines', line = dict(color = 'blue', width = 3), name = 'derivative')
        deriv_smooth_tace = go.Scatter(x = VFP_df.z_interpol, y = abs_deriv_smoth, yaxis = 'y2', mode = 'lines', line = dict(color = 'red', width = 3), name = 'derivative sav-gol')
        profile_trace = go.Scatter(x = VFP_df.z_interpol, y = VFP_df.disp_vals_new, mode = 'lines', line = dict(color = 'black', width = 3), name = 'profile')
        fig = go.Figure(data = [VFP_trace, deriv_tace, deriv_smooth_tace, profile_trace], layout = layout.eldens())
        fig.update_layout(yaxis2 =dict(side = 'right', overlaying = 'y'))
        fig.show()
            # elif prominences[i]<0.001:
            #     roughness.append(np.nan)
            

    # print(len(roughness))
    # if not len(roughness) == 2:
    #     print('You have more or less than two values in roughness. Check all parameters')

    return np.array(roughness)

def angshift(Filepath, shift, save=False, ext = '.dat', fit_shift = False):

        # Get the Path and the names of the .dat and .out file
        files = new_listdir(Filepath)
        all_datfiles = [f for f in files if f.endswith(ext)]
        datfile = min(all_datfiles, key=len)
        datfile_us = datfile.strip(ext) + '_unshifted' + ext
        # # try:
        all_outfiles = [f for f in files if f.endswith('.out') and not 'lserror' in f]
        outfile = min(all_outfiles, key=len)
        data_inds = []
        lines = read_lines_file(outfile, onlyFirstLine=False)
        # with open(outfile, 'r') as oof:
        #     lines = oof.readlines()
        for l in range(0, len(lines)):
            if lines[l].startswith('C') or lines[l].startswith(' C'):
                column_names = lines[l - 1].split()
            else:
                data_inds.append(l)
        column_names = column_names[2:]
        header_end = data_inds[0]
        
        # # Read data from outfile
        df = read_csv(outfile, header=header_end - 1, sep='\s+', usecols=[
            1, 2, 3, 4, 5], names=column_names[0:])
        measured_values = 10**(df['YOBS'])
        fit = np.array(10**(df['YCALC']))
        
        # # Read data from datfile
        data = readDatFileData(datfile)
        firstline = read_lines_file(datfile, onlyFirstLine=True)
        q = np.array(data.q)
        if not fit_shift:
            shifted_q = q + shift
        else:
            pass
            # print('Try to fit angshift')
            # crit_angle_index = data.q_datfile[data.counts == max(data.counts)].index[0]
            # indices_datfile = np.arange(crit_angle_index - 4, crit_angle_index + 4)

            # fit_angles_datfile = [data.q_datfile[i] for i in indices_datfile]
            # fit_counts_datfile = [data.counts[i] for i in indices_datfile]
            
            
        counts_norm = np.array(data['counts']) / max(data['counts'])


        f1 = plt.figure(num=1, figsize=(8, 6.5))
        ax1 = f1.add_subplot(111)
        ax1.set(yscale='log', ylim=[10**(-1.2), 1.05], xlim=[0.01, 0.06],
                xlabel=r'$q\, \left[\mathrm{\AA} ^ {-1}\right]$', ylabel=r'$R/R_{\mathrm{F}}$')
        ax1.plot(np.array(df['XOBS']), fit, lw=3, marker = 'o', label='Fit')  # Fit from outfile
        ax1.plot(q, counts_norm, ls='', marker='*',
                 ms=6, mec='k', mfc='None', label='Unshifted')  # measured values from datfile (normalized)
        ax1.plot(shifted_q, counts_norm, ls='', marker='d',
                 ms=6, mec='purple', mfc='None', label='Shifted')  # Shifted Values
        # ax1.plot(fit_angles_datfile, np.array(fit_counts_datfile) / max(data['counts']), marker = 'x', ms = 12, mec='purple', label ='fit q')
        ax1.legend(shadow=True)
        ax1.grid()
        # Maximize plot window
        figManager = plt.get_current_fig_manager()
        figManager.full_screen_toggle()

        plt.show()

        # save data, if the shift is ok.  
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        notefile = os.path.join(Filepath,'Notes.txt')
        if not os.path.isfile(notefile):
            open(notefile,'w+')
        lines = open(notefile).readlines()
  
        names_new_df = ['shifted_q', 'counts', 'weights']
        if (os.path.isfile(datfile_us) and save):
            print(str(datfile_us.split('/')[-1]) +' already exists. Shifted values will be changed in ' + str(datfile.split('/')[-1]))
            if len(firstline.split()) == 2:
                new_data = {'shifted_q': shifted_q, 'counts': data['counts']}
            else:
                new_data = {'shifted_q': shifted_q, 'counts': data['counts'], 'weights': data['weights']}
            os.remove(datfile)
            new_df = DataFrame(new_data, columns=names_new_df[0:len(firstline.split())])
            new_df.to_csv(datfile, index=None, header=False, lineterminator='\n', sep='\t')
            os.remove(outfile)
            print(str(outfile.split('/')[-1]) + ' is deleted.' )
            if lines == []:
                with open(notefile,'r+') as nf:
                    nf.write(dt_string + '\n' + 'shift in q created by angshift.py.\n' + 'shift = ' + str(shift))
            else:
                with open(notefile,'a')as nf:
                    nf.write('\n' + dt_string + '\n' + 'shift in q created by angshift.py.\n' + 'shift = ' + str(shift))

        elif (not os.path.isfile(datfile_us) and save):
            print('Shifted values saved in ' + str(datfile.split('/')[-1]) +'. Unshifted values saved in ' + str(datfile_us.split('/')[-1]) + '.')
            os.rename(datfile, datfile_us)
            if len(firstline.split()) == 2:
                new_data = {'shifted_q': shifted_q, 'counts': data['counts']}
            else:
                new_data = {'shifted_q': shifted_q, 'counts': data['counts'], 'weights': data['weights']}
            new_df = DataFrame(new_data, columns=names_new_df[0:len(firstline.split())])
            new_df.to_csv(datfile, index=None, header=False, lineterminator='\n',sep='\t')
            os.remove(outfile)
            print(str(outfile.split('/')[-1]) + ' is deleted.' )
            if lines == []:
                with open(notefile,'r+') as nf:
                    nf.write(dt_string + '\n' + 'shift in q created by angshift.py.\n' + 'shift = ' + str(shift))
            else:
                with open(notefile,'a')as nf:
                    nf.write('\n' + dt_string + '\n' + 'shift in q created by angshift.py.\n' + 'shift = ' + str(shift))
        else:
            print('Changes have not been saved.')



def _determine_shift_value_eldens(f, save = False):
    # save=True
    with open(f, 'r+') as file:
        cnt = 0
        lines =list()
        while cnt <=2:
            lines.append(file.readline().strip('\n'))
            cnt +=1      
    if 'normalized to reference' in lines[0]:
        shiftvalue = float(lines[0].replace(' ', '').split(':')[-1])
        if shiftvalue == 0 :
            print(f.split('/')[-1] + ': Dispersion of reference material is ' +
                str(shiftvalue) + '.' ' No shift necessary.')  
        anzcols = len(lines[2].split())
        numlayer = np.arange(anzcols - 1)
        newline1 = f'C dispersion of refernce material {shiftvalue} added to dispersion values'
        if '...' in lines[1]:
            colnames = list()
            colnames = ['z']
            [colnames.append('layer ' + str(i)) for i in numlayer]
            if not shiftvalue == 0:
                df = read_csv(f, header=2, sep=r'\s+', names=colnames)
                # for (columnName, columnData) in df.iteritems():
                for (columnName, columnData) in df.items():
                    if not columnName == 'z':
                        df[columnName] = columnData.values + shiftvalue
                        df[columnName].map("{:.4E}".format)
                    else:
                        df[columnName].map(":.3f".format)
                if save:
                    with open(f, 'w') as newfile:
                        newfile.write(newline1 + '\n' + lines[1] + '\n')
                        df.to_string(newfile, header=False, index=False)
                        print(f'Added {shiftvalue} to dispersion values in '+ 
                                str(f.split('/')[-1]))
                else:
                    print(df, '\n',f.split('/')[-1])
        else:
            colnames = ['z', 'delta', 'beta']   
            df = read_csv(f, header=2, sep=r'\s+', names=colnames)
            df['delta'] = df['delta'].values + shiftvalue
            df['delta'].map(":.4E".format)
            df['z'].map(":.3f".format)
            if save:
                with open(f, 'w') as newfile:
                    newfile.write(newline1 + '\n' + lines[1] + '\n')
                    df.to_string(newfile, header=False, index=False)
                    print(f'Added {shiftvalue} to dispersion values in '+ 
                            str(f.split('/')[-1]))
            else:
                print(f.split('/')[-1], df.head(2).to_string(header=df.columns.tolist()), df.tail(2).to_string(header=False) ,sep='\n\n')

def shift_eldens(Filepath, subdirs=False, save = False, timeit = False, eldens_ext = '.rft'):
    '''
    Shift electron density with the reference medium dispersion value.

    Parameters:
    -----------
    Filepath: path, where the ".rft-files" are saved or path where subdirs containing ".rft-files" are located
    subdirs: boolean; if True, files are searched through the subdirs of path. If not, only path is searched. Default: False 
    save: boolean, if True, the added values are saved in the same file.
    '''
    t1 = perf_counter()
    anzcores = os.cpu_count()
    if not subdirs:
        rftfile = [f for f in new_listdir(Filepath) if f.endswith(eldens_ext)]
        # rftfile = [os.path.join(Filepath, f) for f in os.listdir(Filepath) if f.endswith(eldens_ext)]
    else:
        rftfile = []
        for root, subdirs, _ in os.walk(Filepath):
            for s in subdirs:
                for f in os.listdir(os.path.join(root,s)):
                    if f.endswith(eldens_ext):
                        rftfile.append(os.path.join(root,s,f))

    saveB = np.repeat(save, len(rftfile))
    argtuple = [(rf, sb) for rf, sb in zip(rftfile, saveB)]
    with Pool(anzcores-1) as pool:
        # pool.map(XRR_datatreatment.determine_shift_value_eldens, [f for f in rftfile])
        pool.starmap(_determine_shift_value_eldens, argtuple)


    t2 = perf_counter()
    if timeit:
        print(f'Finished in {t2-t1} seconds')

def _zrange_eldens(f, save = False, minZ = -50, maxZ = 150, timeit = False, verbose = False):
    '''

    '''
    t1 = perf_counter()
    with open(f, 'r+') as file:
        cnt = 0
        lines =list()
        while cnt <=2:
            lines.append(file.readline().strip('\n'))
            cnt +=1    
    anz_cols = len(lines[2].split()) 
    numlayer = np.arange(anz_cols - 1)
    colnames = list()
    colnames = ['z']
    [colnames.append('layer ' + str(i)) for i in numlayer]
    df = read_csv(f, header=None, sep=r'\s+', names=colnames, skiprows = len(lines) -1)

    first_line = [df[k].iloc[0] for k in df.keys() if not 'z' in k] 
    first_line.insert(0, minZ)
    kvPairsMin = dict()
    [kvPairsMin.update({k:[flv]}) for k, flv in zip(df.keys(), first_line)]
    kvPairs_df_min = DataFrame.from_dict(kvPairsMin)
    last_line = [df[k].iloc[-1] for k in df.keys() if not 'z' in k]
    last_line.insert(0, maxZ)
    kvPairsMax = dict()
    [kvPairsMax.update({k:[llv]}) for k, llv in zip(df.keys(), last_line)]
    kvPairs_df_max = DataFrame.from_dict(kvPairsMax)
    
    df_new = pd_concat([df, kvPairs_df_max], ignore_index = True)
    df_new2 = pd_concat([kvPairs_df_min, df_new], ignore_index = True)
    filename = str(f.split('/')[-1])
    lines = [l + '\n' for l in lines[:-1]]
    lines = ''.join(lines)
    # headerline = pd.DataFrame(lines[:-1])
    if save:
        if df['z'].iloc[-1] == maxZ and df['z'].iloc[0] == minZ:
            # print(f'Already enlarged zrange in {filename}.')
            string = f'Already enlarged zrange in {filename}.'

        elif df['z'].iloc[-1] == maxZ and not df['z'].iloc[0] == minZ:
            string = 'Has to be enlarged at the beginning'
            # print('Has to be enlarged at the beginning')
            df_only_firstline = pd_concat([kvPairs_df_min, df], ignore_index = True)
            with open(f, 'w') as newfile:
                df_only_firstline.to_string(newfile, header = None, index = False)
            prepend_line(f, lines)
            whats_done = 'Prepended ' + '\n' + f'{lines} to {filename}'
            # print('Prepended ' + '\n' + f'{lines} to {filename}')

        elif  df['z'].iloc[0] == minZ and not df['z'].iloc[-1] == maxZ:
            string = 'Has to be enlarged at the end'
            # print('Has to be enlarged at the end')
            df_only_last_line = pd_concat([df, kvPairs_df_max], ignore_index = True)
            with open(f, 'w') as newfile:
                df_only_last_line.to_string(newfile, header = None, index = False)
            prepend_line(f, lines)
            whats_done = 'Appended ' + '\n' + f'{lines} to {filename}'
        
        elif not df['z'].iloc[0] == minZ and not df['z'].iloc[-1] == maxZ:
            with open(f, 'w') as newfile:
                df_new2.to_string(newfile, header=None, index=False)
            prepend_line(f, lines)
            whats_done = f'Prepended {first_line} and appended {last_line} to {filename}.'
            # print(f'Prepended {first_line} and appended {last_line} to {filename}.')
    
    else:
        print(filename, df_new2.head(2).to_string(header=df_new2.columns.tolist()), df_new2.tail(2).to_string(header=False) ,sep='\n\n')
    t2 = perf_counter()
    
    if timeit:
        print(f'time needed to read file, convert to dataframe and append line: ' + "{:.3f}".format(t2-t1) + ' seconds')
    if verbose:
        print(f'{string}\n{whats_done}')


def enlarge_zrange_eldens(Filepath, subdirs = False, ignore_distribution_files = False, save = False, timeit = False, minZ = -50, maxZ = 150, eldens_ext = '.rft',
    distfile_ext = 'dist.rft'):
    '''
    Increase zrange of electron density profiles. The dispersion and absorsption values are the last ones from the original dataframe. Data in file is overwritten.

    Parameters:
    -----------
    Filepath: path, where the ".rft-files" are saved or path where subdirs containing ".rft-files" are located
    subdirs: boolean; if True, files are searched through the subdirs of path. If not, only path is searched. Default: False 
    ignore_distribution_files: boolean, if True, the layer distribution files (*dist.rft) are ignored. Default: False
    save: boolean, if True, the added values are saved in the same file.
    timeit: boolean, if True. The needed time is printed for one file operation. Default: True,
    maxZ = z-value in Angstrom, up to which the electron density profile should be enlarged.
    '''
    t1 = perf_counter()
    anzcores = os.cpu_count()
    if not subdirs:
        rftfile = [f for f in new_listdir(Filepath) if f.endswith(eldens_ext)]
        # rftfile = [os.path.join(Filepath, f) for f in os.listdir(Filepath) if f.endswith(eldens_ext)]
    else:
        rftfile = []
        for root, subdirs, _ in os.walk(Filepath):
            for s in subdirs:
                for f in os.listdir(os.path.join(root,s)):
                    if f.endswith(eldens_ext):
                        rftfile.append(os.path.join(root,s,f))
    # pprint(rftfile)
    if ignore_distribution_files:
        rftfile = [rf for rf in rftfile if not any(x in rf for x in [distfile_ext, 'copy', 'substrate']) ]
    else:
        rftfile = [rf for rf in rftfile if not any(x in rf for x in ['copy', 'substrate'])]

    saveB = np.repeat(save, len(rftfile))
    minzB = np.repeat(minZ, len(rftfile))
    maxZB = np.repeat(maxZ, len(rftfile))
    timeitB = np.repeat(timeit, len(rftfile))
    argtuple = [(rf, sb, minzb,maxzb, tit) for rf, sb, minzb, maxzb, tit in zip(rftfile, saveB, minzB, maxZB, timeitB)]       
    with Pool(anzcores -1 ) as pool:
        pool.starmap(_zrange_eldens, argtuple)
    t2 = perf_counter()
    if timeit:
        print(f'Time needed for all files is {t2-t1} seconds')



def qweights(Filepath, weight, qthresh, save= False, ext = '.dat', save_orig_weights = True, leave_zeros = True, cou_thresh = 1e-10):
    '''
    Change weights of data points in file.
    Parameters:
    -----------
    *args: 
        Filepath: Path, where the file containing the data is saved
        weight: float, weights of the data points
        qthresh: float, minimum q-value from which on weight should be applied
    *kwargs:
        save: Boolean, if True, new weights will be saved.
        ext: str, extension of the file containing the data
        save_orig_weights: Boolean, if True, a file names 'orig weights.txt' is created containing the old weights in the same directory. From this file the old weights can be restored using restore_orig_weights_function.
        leave_zeros: Boolean, if True, all weights that are zero will not be changed. 
    '''
    datfile = [f for f in new_listdir(Filepath) if not any(s in f for s in ['unweighted', 'unshifted', 'unused', 'orig']) and f.endswith(ext)][0]
    notefile = os.path.join(Filepath, 'Notes.txt')
    
    columns = ['q', 'counts', 'weights']

    with open(datfile, 'r') as f:
        anz_cols = len(f.readline().split())
    columns = columns[0:anz_cols]
    
    if not os.path.isfile(notefile):
       with open(notefile,'w+') as nf:
            lines = nf.readlines()

    df = read_csv(datfile, sep='\s+', names=columns)

    first_ind = np.where(df['q'] > qthresh)[0][0]    
    if 'weights' in df.keys(): 
        if leave_zeros:
            new_weights = np.where((df.q > qthresh) & (df.weights !=0), weight, df.weights)
        else:
            new_weights = np.where((df.q > qthresh) & (df.counts > cou_thresh), weight, df.weights)
            # new_weights = np.where(df.q < qthresh, df.weights, weight)
    else:
        new_weights = np.where(df.q < qthresh, 1, weight)
    
    df['new_weights'] = new_weights

    if save_orig_weights:
        if not os.path.isfile(os.path.join(Filepath, 'orig_weights.txt')):
            df['weights'].to_string(os.path.join(Filepath, 'orig_weights.txt'), index = False, header = False)
        else:
            print('original weights already saved.')
            pass
    
    if save:
        df.drop('weights', axis = 'columns', inplace = True)
        df['q'].map(":.6f".format)
        df['counts'].map(":.4f".format)
        df.to_string(datfile, header=False, index=False)
        modtime_dat_uw = os.path.getmtime(datfile)
        modtime_dat_uw = datetime.fromtimestamp(modtime_dat_uw ).strftime("%d/%m/%Y %H:%M:%S")
        with open(notefile,'a') as nf:
            nf.write('\n' + modtime_dat_uw + '\n' + 'weigths added to ' + str(datfile.split('/')[-1]) + ' with threshold '
            + ' q =     ' + str(qthresh) + '\n' + 'weight = ' + str(weight))
    else:
        df = df.tail(len(df) - (first_ind - 2))
        print(df)

def weights(Filepath, weight, thresh, save=False):
        dateien = os.listdir(Filepath)
        datfiles = [os.path.join(Filepath, d) for d in dateien if d.endswith('.dat')]
        datfile = min(datfiles, key=len) # name of datefile with fullpath
        datfile_uw = datfile.strip('.dat') + '_unweighted.dat' # name of unweighted datfile with full path

        notefile = os.path.join(Filepath,'Notes.txt')
        if not os.path.isfile(notefile):
            open(notefile,'w+')
        lines = open(notefile).readlines()

        if not os.path.exists(datfile_uw):
            df = read_csv(datfile, sep='\s+',
                             usecols=[0, 1], names=['q', 'counts'])
            df['ones'] = 1.0
            low_weightes = np.where(df['counts'] > thresh, df['ones'], weight)
            firstind = np.where(df['counts'] < thresh)[0][0]
            df['ones'] = low_weightes
            if save:
                df['q'].map(":.6f".format)
                df['counts'].map(":.4f".format)
                os.rename(datfile, datfile_uw)
                df.to_string(datfile, index=False, header=False)
                modtime_dat_uw = os.path.getmtime(datfile)
                modtime_dat_uw = datetime.fromtimestamp(modtime_dat_uw ).strftime("%d/%m/%Y %H:%M:%S")
                with  open(notefile,'a') as nf:
                    nf.write('\n' + modtime_dat_uw + '\n' + 'weigths added to ' + str(datfile.split('/')[-1]) + ' with threshold '
                     + str(thresh) +  '\n' + 'weight = ' + str(weight))
            else:
                df.rename(columns = {'ones':'weights'}, inplace=True)
                print(df[['q','counts','weights']].tail(len(df)-(firstind-2)))

        else:
            # Wenn die weights schon da sind aber geändert werden sollen
            df = pd.read_csv(datfile, sep='\s+',
                             usecols=[0, 1, 2], names=['q', 'counts', 'weights'])
            old_weights = df.weights
            df['weights'] = 1.0
            new_weights = np.where(df['counts'] > thresh, df['weights'], weight)
            firstind = np.where(df['counts'] < thresh)[0][0]
            df['weights'] = new_weights
            if save:
                df['q'].map(":.6f".format)
                df['counts'].map(":.4f".format)
                df.to_string(datfile, header=False, index=False)
                modtime_dat_uw = os.path.getmtime(datfile)
                modtime_dat_uw = datetime.fromtimestamp(modtime_dat_uw ).strftime("%d/%m/%Y %H:%M:%S")
                with open(notefile,'a') as nf:
                    nf.write('\n' + modtime_dat_uw + '\n' + 'weigths added to ' + str(datfile.split('/')[-1]) + ' with threshold '
                     + str(thresh) + '\n' + 'weight = ' + str(weight))
            else:
                print('Last values of your dataset:')
                print(df[['q','counts','weights']].tail(len(df)-(firstind-2)))
def restore_orig_weights(Filepath, ext = '.dat'):
        try:
            weightsfile = [f for f in new_listdir(Filepath) if "orig_weights" in f][0]
            orig_weights = read_csv(weightsfile, header = None, names = ['orig_weights'])

            datfile = [f for f in new_listdir(Filepath) if not any(s in f for s in ['unweighted', 'unshifted', 'unused', 'orig']) and f.endswith(ext)][0]
            print(datfile)
            columns = ['q', 'counts', 'weights']
            with open(datfile, 'r') as f:
                anz_cols = len(f.readline().split())
            columns = columns[0:anz_cols]
            df = read_csv(datfile, sep='\s+', names=columns)
            df.weights = orig_weights.orig_weights
            df.to_string(datfile, index = False, header = False)
            modtime_dat_uw = os.path.getmtime(datfile)
            modtime_dat_uw = datetime.fromtimestamp(modtime_dat_uw ).strftime("%d/%m/%Y %H:%M:%S")
            with open(os.path.join(Filepath, 'Notes.txt'), 'a') as nf:
                nf.write('\n' + modtime_dat_uw + '\n' + 'Original weights were restored.')
            print(f'Original weights in {os.path.basename(Filepath)} were restored.')
        except Exception as e:
            print(e)
            print('Original weights were not saved. Restoring cannot be executed.')

def _fit_temps_AntonPaar(fp = None):
    '''
    Fit temperature values of Anton-Paar sample cell measured with a thermoelement against set temperature. This data is used to calculate the surface temperature values with "temp_calibration_AntonPaar"
    Parameters
    ----------
    fp: full path of file containing the measured temperature data. If None, the file is searched in the same directory, where this script is located.

    Returns
    -------
    lmfit.model.ModelResult
    '''
    path = os.path.dirname(os.path.abspath(__file__))
    try:
        if not fp:
            f = os.path.join(path, 'temp_calibration_AntonPaar.txt')
        else:
            f = fp 
        colnames = read_lines_file(f, onlyFirstLine=True).split()
        df = read_csv(f, header = 0, sep = '\s+', usecols = colnames, index_col='#', dtype = 'float32')
        # df = df.iloc[0:18]
        result = fit_linear(df['T_eingestellt'], df['T_Waferoberfläche_Thermoelement'], m = 1, b = df['T_eingestellt'][1.0])
        return result, df
    except (FileNotFoundError, TypeError):
        print(f'File\n\t {f}\nnot found.')
        return


def temp_sampleSurface_AntonPaar(t_set, fp = None, plot = False):
    '''
    Calculate temperature on top of Si-Wafer surface (thickness approx. 0.7 mm) when used in Anton-Paar sample cell.
    
    Parameters
    ----------
    args:
        t_set: list, float, int, str or np.array: set temperature from temperature control unit
    kwargs:
        fp: full path of file containing the measured temperature data. If None, the file is searched in the same directory, where this script is located.
        plot: bool, if True, a plot of the data is shown
    
    Returns
    -------
    y: np.array with surface temperature
    y_err: np.array with errors of calculated surface temperatures based on the fit parameters
    '''
    
    if isinstance(t_set, (list, np.ndarray)) :
        t_set = np.asarray([float(t) for t in t_set])
    elif isinstance(t_set, (float,int)):
        t_set = np.asarray([t_set])
    elif isinstance(t_set, str):
        t_set = np.asarray([float(t_set)])

    try:
        fit, df = _fit_temps_AntonPaar()
    except TypeError:
        path = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(path, 'temp_calibration_AntonPaar.txt')
        print(f'File with recorded temperature data not found. Save it as\n\t{f}\nand try again.')
        return

    m = fit.params['slope']
    b = fit.params['intercept']
    y = m*t_set+b
    y_err = np.sqrt((t_set*fit.params['slope'].stderr)**2 + fit.params['intercept'].stderr**2)
    fit_text = f'Fit: m = {m.value:.2f} \u00b1 {m.stderr:.2f}, b = {b.value:.2f} \u00b1 {b.stderr:.2f}'
    if plot:
        # if not len(t_set) == 1:
        t_data = go.Scattergl(x = t_set, y = y, error_y = dict(array=y_err), name = 'data', customdata = y_err,
            mode = 'markers', marker = dict(symbol = 1, size =6, color = 'blue', line = dict(width = 0, color = 'blue')),
            hovertemplate = '<b>T<sub>set</sub></b> = %{x:.1f} <br><b>T<sub>surface</sub></b> = %{y:.1f}&plusmn;%{customdata:.2f}')
        t_fit = go.Scattergl(x = df['T_eingestellt'], y =fit.best_fit, name = fit_text, mode = 'lines', line = dict(color = 'green', width = 3))
        fig = go.Figure(data = [t_data, t_fit], layout = layout.eldens())
        fig.update_layout(legend = dict(x = 0, y = 1, xanchor = 'left', yanchor='top'),
            xaxis = dict(title_text = 'T<sub>set</sub>'),
            yaxis = dict(title_text = 'T<sub>surface</sub>'))
        # fig.show()
    if plot:
        return y, y_err, fig
    else: return y, y_err

def _fit_temps_Sitec_cell():
    path = os.path.dirname(os.path.abspath(__file__))
    try:
        f = os.path.join(path, 'temp_calibration_Sitec_cell.txt')
        colnames = read_lines_file(f, onlyFirstLine=True).split()
        df = read_csv(f, header = 0, sep = '\s+', usecols = colnames, dtype = 'float32')
        result = fit_linear(df['volts'], df['temperature'], m = 1, b = 0)
        return result, df
    except (FileNotFoundError, TypeError):
        print(f'File\n\t {f}\nnot found.')
        return
def temp_Sitec_Cell(voltage, plot = False):
    '''
    Calculate temperature on top of Si-Wafer surface (thickness approx. 0.7 mm) when used in Anton-Paar sample cell.
    
    Parameters
    ----------
    args:
        voltage: list, float, int, str or np.array: measured voltages
    kwargs:
        plot: bool, if True, a plot of the data is shown
    
    Returns
    -------
    y: np.array with temperatures inside sample cell
    y_err: np.array with errors of calculated temperatures based on the fit parameters
    '''
    
    if isinstance(voltage, (list, np.ndarray)) :
        voltage = np.asarray([float(t) for t in voltage])
    elif isinstance(voltage, (float,int)):
        voltage = np.asarray([voltage])
    elif isinstance(voltage, str):
        voltage = np.asarray([float(voltage)])

    try:
        fit, df = _fit_temps_Sitec_cell()
    except TypeError:
        path = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(path, 'temp_calibration_AntonPaar.txt')
        print(f'File with recorded temperature data not found. Save it as\n\t{f}\nand try again.')
        return

    m = fit.params['slope']
    b = fit.params['intercept']
    y = m*voltage+b
    y_err = np.sqrt((voltage*0.001*m)**2 + (voltage*fit.params['slope'].stderr)**2 + fit.params['intercept'].stderr**2)
    fit_text = f'Fit: m = {m.value:.2f} \u00b1 {m.stderr:.2f}, b = {b.value:.2f} \u00b1 {b.stderr:.2f}'
    if plot:
        # if not len(voltage) == 1:
        t_data = go.Scattergl(x = voltage, y = y, error_y = dict(array=y_err), name = 'data', customdata = y_err,
            mode = 'markers', marker = dict(symbol = 1, size =6, color = 'blue', line = dict(width = 0, color = 'blue')),
            hovertemplate = '<b>U</b> = %{x:.3f} <br><b>T</b> = %{y:.1f}&plusmn;%{customdata:.2f}')
        t_fit = go.Scattergl(x = df['volts'], y =fit.best_fit, name = fit_text, mode = 'lines', line = dict(color = 'green', width = 3))
        fig = go.Figure(data = [t_data, t_fit], layout = layout.eldens())
        fig.update_layout(legend = dict(x = 0, y = 1, xanchor = 'left', yanchor='top'),
            xaxis = dict(title_text = 'U/V'),
            yaxis = dict(title_text = 'T/°C'))
        # fig.show()
    if plot:
        return y, y_err, fig
    else: return y, y_err

def calcRealTemps(lst, update_function):
    updated_temperature = list()
    for temp in lst:
        match = re.search(r'(\b\d+\b)(.*)', temp)  # Regex-Muster, um den Temperaturwert und den Rest der Zeichenkette zu extrahieren
        if match:
            value = int(match.group(1))  # Extrahierten Temperaturwert in Integer konvertieren
            new_temp = round(update_function(value)[0][0])  # Beispiel: Neue Temperatur durch Hinzufügen von 10 zum alten Wert berechnen
            updated_temp = str(new_temp) + match.group(2)  # Neuen Temperaturwert mit den zusätzlichen Zeichen kombinieren
            updated_temperature.append(updated_temp)  # Aktualisierte Temperatur zur Liste hinzufügen
        else:
            updated_temperature.append(temp)  # Zeichenketten ohne Temperaturwert unverändert zur Liste hinzufügen
    return updated_temperature

def layerPropsFromSeries(dist_data, loop_var):
    d_head, rho_head, sigma_head = list(), list(), list()
    d_tail, rho_tail, sigma_tail = list(), list(), list()
    d_gas, rho_gas, sigma_gas = list(), list(), list()
    d_ref_mat, rho_ref_mat, sigma_ref_mat = list(), list(), list()
    # print(dist_data[loop_val[i]]['layer_properties'].keys())
    # [print(dist_data[loop_val[i]]['layer_properties'].loc['OTS_head']['Effective electron density']) for i in range(len(loop_val))]
    # for i in range(1, len(loop_val)-1):
    for i in range(0, len(loop_val)):
        lay_props = dist_data[loop_val[i]]['layer_properties']
        tail_props = lay_props.loc['OTS_tail'] 
        head_props = lay_props.loc['OTS_head']
        ref_mat_props = lay_props.loc['reference material']
        
        d_head.append(head_props.thickness)
        rho_head.append(head_props['Effective electron density'])
        sigma_head.append(head_props.roughness)
        
        d_tail.append(tail_props.thickness)
        rho_tail.append(tail_props['Effective electron density'])
        sigma_tail.append(tail_props.roughness)

        d_ref_mat.append(ref_mat_props.thickness)
        rho_ref_mat.append(ref_mat_props['Effective electron density'])
        sigma_ref_mat.append(ref_mat_props.roughness)
        
        if 0 < i < len(loop_val)-1:
            CO2_props = lay_props.loc['CO2'] 
            d_gas.append(CO2_props.thickness)
            rho_gas.append(CO2_props['Effective electron density'])
            sigma_gas.append(CO2_props.roughness)
        else:
            d_gas.append(np.nan)
            rho_gas.append(np.nan)
            sigma_gas.append(np.nan)
    index = names
    head_data = {'rho': rho_head, 'sigma' : sigma_head, 'd' : d_head}
    tail_data = {'rho': rho_tail, 'sigma' : sigma_tail, 'd' : d_tail}
    gas_data = {'rho': rho_gas, 'sigma' : sigma_gas, 'd' : d_gas}
    ref_mat_data = {'rho': rho_ref_mat, 'sigma' : sigma_ref_mat, 'd' : d_ref_mat}
    print(np.nanmean(head_data['d']))
    # print(len(head_data['d']), len(head_data['rho']), len(head_data['sigma']), len(loop_val))
    # print(len(tail_data['d']), len(tail_data['rho']), len(tail_data['sigma']), len(loop_val))
    # print(len(gas_data['d']), len(gas_data['rho']), len(gas_data['sigma']), len(loop_val))
    print()
    columns = pd.MultiIndex.from_product([['Head', 'Tail', 'Gas', 'Reference_material'], ['rho', 'sigma', 'd']])
    df = pd.DataFrame(index=index, columns=columns)
    # df.loc[:, ('Head', 'rho')] = rho_head
    # df.loc[:, 'Head'][:]
    df.loc[:, ('Head', slice(None))] = head_data.values()
    df.loc[:, ('Tail', slice(None))] = tail_data.values()
    df.loc[:, ('Gas', slice(None))] = gas_data.values()
    df.loc[:, ('Reference_material', slice(None))] = ref_mat_data.values()
    rounding_dict = {('Head', 'd'): 2, ('Tail', 'd'): 2, ('Gas', 'd'): 2, ('Reference_material', 'd'): 2}
    df = df.round(3)
    # df = df.round(rounding_dict)
    # df.to_latex('/home/mike/Dokumente/Uni/Promotion/Arbeit/files/Auswertung/BeamDamage/Tabellen/BD_CO2', decimal = ',', float_format="%.2f", multicolumn_format='c', na_rep='--')
    print(df)