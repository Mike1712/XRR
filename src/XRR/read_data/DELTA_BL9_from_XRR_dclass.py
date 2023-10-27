import sys
# sys.path.append('/home/mike/Dokumente/Uni/Promotion/Python/Skripte/XRR')
# sys.path.append('/home/mike/Dokumente/Uni/Promotion/Python/Skripte/plotly')
import os
# from XRR_datatreatment import th2q
import numpy as np
import pandas as pd
# pd.set_option('display.max_rows', 10000)
from matplotlib import pyplot as plt, patches, cm
from matplotlib.colors import LogNorm, rgb2hex
from XRR.plotly_layouts import plotly_layouts as layouts
import plotly.graph_objs as go
import scipy as sc
from scipy import constants
from scipy.signal import find_peaks, argrelextrema, peak_widths, argrelmin
from collections import defaultdict
from scipy import integrate, special, interpolate
import shutil
from glob import glob
import re
from PIL import Image, ImageEnhance, ImageFilter
# from itertools import chain, product
import math
import time
from datetime import datetime, timedelta
import concurrent.futures
# from multiprocessing import Pool, Process
from multiprocessing.pool import ThreadPool as Pool
import re
from natsort import natsorted

import XRR.FluidProps as FP
from lmfit import Parameters, minimize, report_fit
import pickle

r_e_angstrom = constants.physical_constants['classical electron radius'][0] * 1e10
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

def count_words(sentence):
    sentence = sentence.split()
    wordcounts = len(sentence)
    return wordcounts

def prepend_line(file_name, line):
    """ Insert given string as a new line at the beginning of a file """
    # define name of temporary dummy file
    dummy_file = file_name + '.bak'
    # open original file in read mode and dummy file in write mode
    with open(file_name, 'r') as read_obj, open(dummy_file, 'w') as write_obj:
        # Write given line to the dummy file
        write_obj.write(line)
        # Read lines from original file one by one and append them to the dummy file
        for line in read_obj:
            write_obj.write(line)
    # remove original file
    os.remove(file_name)
    # Rename dummy file as the original file
    os.rename(dummy_file, file_name)

def point_of_intersect_lines(line_1 = (), line_2 = ()):
    '''
    Parameters:
    -----------
    line_1, line_2: tuples with arrays of x and y values of lines
    Returns:
    idx: Index of points of intersection 
    '''
    idx = np.argwhere(np.diff(np.sign(line_1[1] - line_2[1]))).flatten()
    return idx

def rhom2disp(molecule, density, wavelength = None, verbose = True):
    if wavelength == None:
        wavelength = 0.4592
        if verbose:
            print('No wavelength was specified. ' + '\u03BB = ' + "{:.4f}".format(wavelength) + ' \u00C5' +' is used as default.')
    anz_el, molar_mass = selectMolecule(molecule = molecule)
    rho_e = density / molar_mass * constants.Avogadro * anz_el * 1e-30
    dispersion = wavelength**2 / (2 * constants.pi) * r_e_angstrom * rho_e
    return dispersion

def scan_numbers_arr(start_scan, anz_scans):
    end_scan_number = start_scan + (anz_scans - 1)
    scan_numbers = np.arange(start_scan, end_scan_number + 1, step=1)
    return scan_numbers

def read_lines_file(file):
    openfile = open(file, 'r')
    lines = openfile.readlines()
    openfile.close()
    return lines

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def th2q(wavelength, th):
    '''
    Parameters:
    -----------
    wavelength: float, wavelength in anstrom
    th: float, list or array, theta in deg
    '''
    # input is numpy array with theta values
    # function returns q in inverse Angstrom
    if not isinstance(th, list):
        q = 4 * sc.pi / (wavelength) * np.sin(th * sc.pi / 180)
    else:
        q = [4 * sc.pi / (wavelength) * np.sin(t * sc.pi / 180) for t in th]
    return q

def q2th(wavelength, q):
    # input is numpy array with q values
    # function returns th in degrees
    # th = np.arcsin(180/(4*sc.pi**2) * wavelength * q)
    th = 180/sc.pi * np.arcsin((wavelength * q) / (4 * sc.pi))
    return th

def dens2edens(wavelength, mol_mass, anz_el,densvalues):
    '''
    dens2edens(wavelength, mol_mass, anz_el,densvalues)
    Calculate dispersion values and electron densities from density values:
    Check e.g. https://webbook.nist.gov/chemistry/form-ser/ for density values
    You can export the data to a file and read it with pandas 

    Parameters
    ----------
    wavelength: wavelength of beam in Angström
    mol_mass: molar mass in g/mol
    anz_el: Number of electrons of the molecule of interest
    densvalues: list or number of densities in kg/m^3.

    Returns
    -------
    Electron density and dispersion: lists or numbers, depending on input of 'densvalues'.
    
    '''

    avogadro = constants.physical_constants['Avogadro constant'][0]
    if not isinstance(densvalues, list):
        eldens = densvalues * avogadro / (mol_mass*1e-3) * anz_el * 1e-30 
        disp = wavelength**2 / (2 * constants.pi) * 2.818 * 1e-5 * eldens 
    else:
        eldens = [dval * avogadro / (mol_mass*1e-3) * anz_el * 1e-30 for dval in densvalues]
        disp = [wavelength**2 / (2 * constants.pi) * 2.818 * 1e-5 * elval for elval in eldens]
    return eldens, disp

def disp2edens(wavelength, dispvalues):
    '''
    disp2edens(wavelength,dispvalues)
    Calculate electron densities from dispersion values:
    You can export the data to a file and read it with pandas 

    Parameters
    ----------
    wavelength: wavelength of beam in Angström
    dispvalues: list or number. 

    Returns
    -------
    Electron density: list or numbers, depending on input of 'dispvalues'.
    
    '''

    # if not isinstance(dispvalues, list):
    #     eldens = dispvalues*1e-6 * 2 * np.pi  / (wavelength**2 * r_e_angstrom)
    # else:
    #     eldens = [dispval*1e-6 * 2 * np.pi  / (wavelength**2 * r_e_angstrom) for dispval in dispvalues]
    if not isinstance(dispvalues, list):
        eldens = dispvalues * 2 * np.pi  / (wavelength**2 * r_e_angstrom)
    else:
        eldens = [dispval * 2 * np.pi  / (wavelength**2 * r_e_angstrom) for dispval in dispvalues]

    return eldens

def edens2disp(wavelength, edensvalues):
    '''
    edens2disp(wavelength,dispvalues)
    Calculate dispersion values from electron densities:
    You can export the data to a file and read it with pandas 

    Parameters
    ----------
    wavelength: wavelength of beam in Angström
    eldensvalues: list or number in Angström. 

    Returns
    -------
    dispvalues: list or numbers, depending on input of 'eldensvalues'.
    
    '''        
    if not isinstance(edensvalues, list):
        dispvalues = edensvalues * wavelength**2 * r_e_angstrom / (2 * np.pi)
    else:
        dispvalues = [edensval * wavelength**2 * r_e_angstrom / (2 * np.pi) for edensval in edensvalues]
    return dispvalues

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

def angst2kev(wavelength):
    '''
    Convert wavelength to energy.

    Parameters:
    -----------
    energy: Beam energy in keV

    Returns:
    --------
    wavelength: wavelength in Angstrom
    '''
    return sc.constants.h * sc.constants.c / (wavelength) / sc.constants.e*1e-3 * 1e10

def readEldensFile(file, header_len = 2):
    colnames = ['z', 'delta', 'beta']   
    data = pd.read_csv(file, header=header_len, sep='\s+', names = colnames)
    # print(data)
    return data


def VolumeFractionProfile(referencefile, dispfile, step = .1, wavelength = None, show = False, molecule = None, zrange = 'All',
    header_len = 2, threshold = 1e-3, find_turning_points = True):
    '''Calculate volume fraction profile from electron density profile from LSFit
    Parameters
    ----------
    referencefile: File with electron density profile of substrate.
    dispfile: File with electron density profile of sample system

    Kwargs:
    -------
    step: float,stepsize in Angstrom between the points for interpolation. Default: 0.1 Angstrom
    wavelength: float, wavelength of radiation in Angstrom. Default 0.4592 (27 keV)
    show: boolean, if True the 

    Returns:
    --------
    int_val: Integrated electron density
    df: pandas.Dataframe with interpolated electron density values of the reference, the sample system and the difference betwee them

    '''
    if not wavelength:
        wavelength = 0.4592
        print('No wavelength was specified. ' + '\u03BB = ' + "{:.4f}".format(wavelength) + ' \u00C5' +' is used as default.')
    if molecule:
        anz_el, _ = FP.selectMolecule(molecule)
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
    interpol_ref = sc.interpolate.interp1d(x = referenceData.z, y = referenceData.delta, fill_value = 'extrapolate')
    interpol_disp = sc.interpolate.interp1d(x = dispData.z, y = dispData.delta, fill_value = 'extrapolate')
    
    ref_vals_new = interpol_ref(z_interpol)
    ref_vals_new = disp2edens(dispvalues = ref_vals_new, wavelength = wavelength)
    
    disp_vals_new = interpol_disp(z_interpol)
    disp_vals_new = disp2edens(dispvalues = disp_vals_new, wavelength = wavelength)
    
    difference = disp_vals_new - ref_vals_new
    difference = np.where(difference < 0, 0, difference)

    if not any(zrange == x for x in ['All', 'auto']):
        zrangemin = find_nearest(z_interpol, zrange[0])
        zrangemax = find_nearest(z_interpol, zrange[1])
        # diffsum = np.sum(difference[zrangemin:zrangemax+1])
    elif zrange == 'auto':
        zrangemin, zrangemax = [],[]
        difference_prime = sc.diff(difference)
        i = 0
        while i in range(len(z_interpol) - 1):
            if difference[i] > threshold:
                zrangemin = i
                break
            i += 1
        reference_mat_disp = disp_vals_new[-1]
        ref_disp_deviation = disp_vals_new- reference_mat_disp   
        j = 0
        while j in range(len(z_interpol) - 1):
            if ref_disp_deviation[j] < threshold:
                zrangemax = j

                break
            j += 1
        if zrangemin == [] or zrangemax == []:
            zrangemin, zrangemax = 0, len(z_interpol) -1
            print('No indices found. Integration over whole range.', zrangemin, zrangemax)
        # diffsum = np.sum(difference[zrangemin:zrangemax])
    else:
        zrangemin, zrangemax = 0, len(z_interpol) - 1
    
    diffsum = np.sum(difference[zrangemin:zrangemax])

    int_val = diffsum / anz_el
    # print(f'{int_val * 100} molecules per nm squared')
    df = pd.DataFrame(list(zip(z_interpol, ref_vals_new, disp_vals_new, difference)),
        columns = ['z_interpol', 'ref_vals_new', 'disp_vals_new', 'difference'])
    if show:
        trace1 = go.Scatter(x = z_interpol, y = ref_vals_new, mode = 'lines', line_color = 'black', line_width = 4, name = 'reference')
        trace2 = go.Scatter(x = z_interpol, y = disp_vals_new, mode = 'lines', line_color = 'red', line_width = 4, name = 'profile')
        trace3 = go.Scatter(x = z_interpol, y = difference, mode = 'lines', line_color = 'blue', line_width = 4, name = 'difference')
        dot_zmin = go.Scatter(x = [z_interpol[zrangemin]], y = [0], mode = 'markers', marker = dict(line = dict(color = 'red'), size = 12, color = 'red'),
            showlegend = False, hoverinfo = 'skip')
        dot_zmax = go.Scatter(x = [z_interpol[zrangemax]], y = [0], mode = 'markers', marker = dict(line = dict(color = 'red'), size = 12, color = 'red'),
            showlegend = False, hoverinfo = 'skip')
        data = [trace1, trace2, trace3, dot_zmin, dot_zmax]
        if zrange == 'auto':
            trace4 = go.Scatter(x = z_interpol, y = difference_prime, mode = 'lines',line_color = 'green', line_width = 4, name = 'derivative')
            data = data + [trace4]
        fig = go.Figure(data = data, layout = layout.eldens_paper())
        if not zrange == 'All':
            fig.add_trace(go.Scatter(x = z_interpol[zrangemin:zrangemax], y = difference[zrangemin:zrangemax], fill = 'tozeroy',
                fillcolor = 'rgba(149, 165, 166, .7)', mode = 'lines', showlegend = False, line = dict(width = 0, color = 'rgba(149, 165, 166, 1)'))
            )
        fig.update_layout(width = 1200, height = 800,)
        fig.show()

    if find_turning_points:
        difference_doubleprime = np.diff(np.diff(difference))
        height_doubleprime = z_interpol[:-2]
        tps = argrelextrema(difference_doubleprime, np.greater, order = 30)[0]
        turningpoints = [z_interpol[tp] for tp in tps]
        disp_turningpoints = [difference[tp] for tp in tps]
        doubleprime = go.Scatter(x = height_doubleprime, y = 100 * difference_doubleprime, mode = 'lines', line = dict(
            width = 4, color = 'black'), name = 'doubleprime')
        vol_frac_prof = go.Scatter(x = z_interpol, y = difference, mode = 'lines', line_color = 'blue', line_width = 4, name = 'difference')
        turning_points_trace = go.Scatter(x = turningpoints, y = disp_turningpoints, mode = 'markers', marker = dict(
            size = 10, color = 'red', line = dict(
                color = 'black', width = 2),
            ),
        name = 'turningpoints')
        eldens = go.Scatter(x = dispData.z, y= disp2edens(wavelength = wavelength, dispvalues = dispData.delta), mode = 'lines', line = dict(
            width = 4, color = 'green'), name = 'profile')
        
        fig2 = go.Figure(data = [doubleprime,vol_frac_prof, turning_points_trace, eldens], layout = layout.eldens())
        fig2.show()

    return int_val, df

def determine_headerlen_outfile(file):
    with open(file,'r') as f:
        lines = f.readlines()
    l = 0
    while l in range(0, len(lines)):
        if not "C" in lines[l]:
            headerlen = l - 1
            break
        l += 1
    return headerlen

class XRR_datatreatment:

    def __init__(self, Filepath, wavelength=None, fresnel_ext = '_RF.out', eldens_ext = '_ed.rft', distfile_ext = 'dist.rft'):
        self.fresnel_ext = fresnel_ext
        self.eldens_ext = eldens_ext
        self.distfile_ext = distfile_ext
        self.thickness = []
        self.Filepath = Filepath
        self.wavelength = wavelength
        if not wavelength:
            self.wavelength = 0.4592
            print('No wavelength was specified. ' + '\u03BB = ' + "{:.4f}".format(self.wavelength) + ' \u00C5' +' is used as default.')

    def read_XRR_data(self, timeit = False):
        ''' Read XRR-data. Function returns following values in the order mentioned:
        q, YOBS_norm, YCALC_norm, height, rho (, q_fresnel, measured_values, fit, measured_values_fresnel, fit_fresnel)
        '''
        t1 = time.perf_counter()
        datfile_data = self.readDatFile()
        anz_vals = len(datfile_data[datfile_data.keys()[0]])

        data_refl = pd.DataFrame(np.nan, index = np.arange(0,anz_vals,1), columns = ['XOBS', 'q', 'q_fresnel', 'YOBS', 'measured_values',  'measured_values_fresnel',
            'YCALC', 'fit', 'YCALC_norm', 'fit_fresnel', 'DELTA', 'DELTA/SIGMA', 'datfile_q', 'datfile_counts', 'datfile_weights'])
        
        data_refl['datfile_q'], data_refl['datfile_counts'], data_refl['datfile_weights'] = datfile_data[datfile_data.keys()[0]],\
        datfile_data[datfile_data.keys()[1]], datfile_data[datfile_data.keys()[2]]
        files = new_listdir(self.Filepath)
        rftfiles, outfilename, distfile, fresnelfile = [], '', '', ''
        
        for file in files:
            if file.endswith('.out') and not any(j in file for j in [self.fresnel_ext, 'lserror.out']):
                outfilename = file
                headerlen = determine_headerlen_outfile(outfilename)
            if file.endswith(self.fresnel_ext):
                fresnelfile = file
                headerlen_ff = determine_headerlen_outfile(fresnelfile)
            if file.endswith(self.eldens_ext):
                rftfiles.append(file)
            if file.endswith(self.distfile_ext):
                distfile = file

        rftfile = sorted(rftfiles, key = len)[0]
        if not outfilename == '':
        # Messertwte aus der Out-datei
            df = pd.read_csv(outfilename, header=headerlen, sep='\s+', usecols=[
                1, 2, 3, 4, 5], names=['XOBS', 'YOBS', 'YCALC', 'DELTA', 'DELTA/SIGMA'])
            q = df['XOBS']
            measured_values = 10**(df['YOBS'])
            fit = 10**(df['YCALC'])
            for key in df.keys(): data_refl[key] = df[key]
            data_refl.q, data_refl.measured_values, data_refl.fit, data_refl['theta'] = q, measured_values, fit, q2th(self.wavelength, q)
        if not fresnelfile == '':   
            # Fresnel-Daten aus der RF_Out-Datei
            fresnel = pd.read_csv(fresnelfile, header=headerlen_ff, sep='\s+', usecols=[
                1, 2, 3, 4, 5], names=['XOBS', 'YOBS', 'YCALC', 'DELTA', 'DELTA/SIGMA'])
            q_fresnel = fresnel['XOBS']
            measured_values_fresnel = 10**(fresnel['YOBS'])
            fit_fresnel = 10**(fresnel['YCALC'])
            YOBS_norm = measured_values / fit_fresnel
            YCALC_norm = fit / fit_fresnel
            data_refl['fit_fresnel'], data_refl[ 'YOBS_norm'], data_refl['YCALC_norm'], data_refl['q_fresnel'], data_refl['measured_values_fresnel'] =\
             fit_fresnel, YOBS_norm, YCALC_norm, q_fresnel, measured_values_fresnel
        if not rftfile == []:
                # Dipsersion aus der rft-Datei
            eldens_data = pd.read_csv(rftfile, header=1, sep='\s+',
                                     usecols=[0, 1, 2], names=['z', 'delta', 'beta'])
            # height = dispersion['z']
            rho = 2 * sc.constants.pi * eldens_data.delta / (self.wavelength**2 * r_e_angstrom)
            eldens_data['eldens'] = rho
        # if not distfile == '':
        #     print(f'You have a distfile in {self.Filepath}')
        t2 = time.perf_counter()
        if timeit:
            print(f'Script took {t2-t1:3f} seconds.' )
        return data_refl, eldens_data

    @staticmethod
    def angshift_shift_model():
        pass
    def angshift(Filepath, shift, save=False, ext = '.dat', fit_shift = False):

        # Get the Path and the names of the .dat and .out file
        files = os.listdir(Filepath)
        all_datfiles = [os.path.join(Filepath, f) for f in files if f.endswith(ext)]
        datfile = min(all_datfiles, key=len)  # make sure not to get wrong datfile
        datfile_us = datfile.strip(ext) + '_unshifted' + ext

        # # try:
        all_outfiles = [os.path.join(Filepath, f) for f in files if f.endswith('.out') and not 'lserror' in f]
        outfile = min(all_outfiles, key=len)
        data_inds = []
        with open(outfile, 'r') as oof:
            lines = oof.readlines()
        for l in range(0, len(lines)):
            if lines[l].startswith('C') or lines[l].startswith(' C'):
                column_names = lines[l - 1].split()
            else:
                data_inds.append(l)
        column_names = column_names[2:]
        header_end = data_inds[0]
        # Read data from outfile
        df = pd.read_csv(outfile, header=header_end - 1, sep='\s+', usecols=[
            1, 2, 3, 4, 5], names=column_names[0:])
        measured_values = 10**(df['YOBS'])
        fit = np.array(10**(df['YCALC']))
        # Read data from datfile
        with open(datfile, "r") as f:
            firstline = f.readline()
            f.close()
            names = ['q_datfile', 'counts', 'weights']
            anzcols_datfile = np.arange(0,len(firstline.split()),1).tolist()
            data = pd.read_csv(datfile, header=None, sep='\s+',
                               usecols=anzcols_datfile, names=names[0:len(firstline.split())])
            
            q = np.array(data['q_datfile'])

            if not fit_shift:
                shifted_q = q + shift
            else:
                print('Try to fit angshift')
                crit_angle_index = data.q_datfile[data.counts == max(data.counts)].index[0]
                indices_datfile = np.arange(crit_angle_index - 4, crit_angle_index + 4)

                fit_angles_datfile = [data.q_datfile[i] for i in indices_datfile]
                fit_counts_datfile = [data.counts[i] for i in indices_datfile]
                
                
                # fit_counts_outfile = 
            shifted_q = q + shift
            counts_norm = np.array(data['counts']) / max(data['counts'])
            # print(type(counts_norm))

        # Plot data
        # YCALC = go.Scatter(x = df.XOBS, y = fit, mode = 'markers', marker = dict(
        #     symbol= 1, size = 12, color = None,line = dict(
        #         color = 'black', width = 4)),
        #     name = 'Fit')
        # Unshifted = go.Scatter(x = q, y = counts_norm, mode = 'markers', marker = dict(
        #     symbol = 2, size = 12, color = None,line = dict(
        #         color = 'red', width = 4)),
        #     name = 'Unshifted')
        # Shifted = go.Scatter(x = shifted_q ,y = counts_norm, mode = 'markers', marker = dict(
        #     symbol = 3, size = 12, color = None, line = dict(
        #         color = 'green', width = 4)),
        #     name = 'Unshifted')
        # fig = go.Figure(data = [YCALC, Unshifted, Shifted], layout = layout.refl())
        # fig.update_layout(width = 1200, height = 800)
        # fig.show()

        f1 = plt.figure(num=1, figsize=(8, 6.5))
        ax1 = f1.add_subplot(111)
        ax1.set(yscale='log', ylim=[10**(-1.2), 1.05], xlim=[0.01, 0.06],
                xlabel=r'$q\, \left[\mathrm{\AA} ^ {-1}\right]$', ylabel=r'$R/R_{\mathrm{F}}$')
        ax1.plot(np.array(df['XOBS']), fit, lw=3, label='Fit')  # Fit from outfile
        ax1.plot(q, counts_norm, ls='', marker='*',
                 ms=6, mec='k', mfc='None', label='Unshifted')  # measured values from datfile (normalized)
        ax1.plot(shifted_q, counts_norm, ls='', marker='d',
                 ms=6, mec='purple', mfc='None', label='Shifted')  # Shifted Values
        # ax1.plot(fit_angles_datfile, np.array(fit_counts_datfile) / max(data['counts']), marker = 'x', ms = 12, mec='purple', label ='fit q')
        ax1.legend(shadow=True)
        ax1.grid()
        # Maximize plot window
        figManager = plt.get_current_fig_manager()
        # figManager.window.is_maximized()
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
            new_df = pd.DataFrame(new_data, columns=names_new_df[0:len(firstline.split())])
            new_df.to_csv(datfile, index=None, header=False, line_terminator='\n', sep='\t')
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
            new_df = pd.DataFrame(new_data, columns=names_new_df[0:len(firstline.split())])
            new_df.to_csv(datfile, index=None, header=False, line_terminator='\n',sep='\t')
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

        # except (ValueError, UnboundLocalError):
        #     print('No outfile in current directory')
    
    @staticmethod
    def footprint(h_beam,l_sample):
        ''' calculate the footprint f [f]=° in an XRR experiment
        args:
            * h_beam: height of the beam [microns]
            * l_sample: length of the sample in beam direction [microns]
        '''
        f = np.arcsin(h_beam / l_sample) * 180 / sc.pi
        return f
            
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
            df = pd.read_csv(datfile, sep='\s+',
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

    def qweights(Filepath, weight, qthresh, save= False, ext = '.dat', save_orig_weights = True, leave_zeros = True):
        datfile = [f for f in new_listdir(Filepath) if not any(s in f for s in ['unweighted', 'unshifted', 'unused', 'orig']) and f.endswith(ext)][0]
        notefile = os.path.join(Filepath, 'Notes.txt')
        
        columns = ['q', 'counts', 'weights']

        with open(datfile, 'r') as f:
            anz_cols = len(f.readline().split())
        columns = columns[0:anz_cols]
        
        if not os.path.isfile(notefile):
           with open(notefile,'w+') as nf:
                lines = nf.readlines()

        df = pd.read_csv(datfile, sep='\s+', names=columns)

        first_ind = np.where(df['q'] > qthresh)[0][0]
        
        if 'weights' in df.keys(): 
            if leave_zeros:
                new_weights = [weight if not w == 0 and q > qthresh else w for q, w in zip(df.q, df.weights)]
            else:
                new_weights = np.where(df.q < qthresh, df.weights, weight)
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

    def restore_orig_weights(Filepath, ext = '.dat'):
        try:
            weightsfile = [f for f in new_listdir(Filepath) if "orig_weights" in f][0]
            orig_weights = pd.read_csv(weightsfile, header = None, names = ['orig_weights'])

            datfile = [f for f in new_listdir(Filepath) if not any(s in f for s in ['unweighted', 'unshifted', 'unused', 'orig']) and f.endswith(ext)][0]
            columns = ['q', 'counts', 'weights']
            print(datfile, weightsfile)
            with open(datfile, 'r') as f:
                anz_cols = len(f.readline().split())
            columns = columns[0:anz_cols]
            df = pd.read_csv(datfile, sep='\s+', names=columns)
            df.weights = orig_weights.orig_weights
            print(len(df.weights), len(orig_weights.orig_weights))
            [print(ow, dfw) for ow, dfw in zip(orig_weights.orig_weights, df.weights)]
            df.to_string(datfile, index = False, header = False)
            modtime_dat_uw = os.path.getmtime(datfile)
            modtime_dat_uw = datetime.fromtimestamp(modtime_dat_uw ).strftime("%d/%m/%Y %H:%M:%S")
            with open(os.path.join(Filepath, 'Notes.txt'), 'a') as nf:
                nf.write('\n' + modtime_dat_uw + '\n' + 'Original weights were restored.')
        except Exception as e:
            print(e)
            print('Original weights were not saved. Restoring cannot be executed.')
        
   

    def dens2edens(wavelength, mol_mass, anz_el,densvalues):
        '''
        dens2edens(wavelength, mol_mass, anz_el,densvalues)
        Calculate dispersion values and electron densities from density values:
        Check e.g. https://webbook.nist.gov/chemistry/form-ser/ for density values
        You can export the data to a file and read it with pandas 

        Parameters
        ----------
        wavelength: wavelength of beam in Angström
        mol_mass: molar mass in g/mol
        anz_el: Number of electrons of the molecule of interest
        densvalues: list or number of densities in kg/m³.

        Returns
        -------
        Electron density and dispersion: lists or numbers, depending on input of 'densvalues'.
        
        '''

        avogadro = constants.physical_constants['Avogadro constant'][0]
        if not isinstance(densvalues, list):
            eldens = densvalues * avogadro / (mol_mass*1e-3) * anz_el * 1e-30 
            disp = wavelength**2 / (2 * constants.pi) * 2.818 * 1e-5 * eldens 
        else:
            eldens = [dval * avogadro / (mol_mass*1e-3) * anz_el * 1e-30 for dval in densvalues]
            disp = [wavelength**2 / (2 * constants.pi) * 2.818 * 1e-5 * elval for elval in eldens]
        return eldens, disp

    def disp2edens(wavelength, dispvalues):
        '''
        disp2edens(wavelength,dispvalues)
        Calculate electron densities from dispersion values:
        You can export the data to a file and read it with pandas 

        Parameters
        ----------
        wavelength: wavelength of beam in Angström
        dispvalues: list or number. 

        Returns
        -------
        Electron density: list or numbers, depending on input of 'dispvalues'.
        
        '''

        # if not isinstance(dispvalues, list):
        #     eldens = dispvalues*1e-6 * 2 * np.pi  / (wavelength**2 * r_e_angstrom)
        # else:
        #     eldens = [dispval*1e-6 * 2 * np.pi  / (wavelength**2 * r_e_angstrom) for dispval in dispvalues]
        if not isinstance(dispvalues, list):
            eldens = dispvalues * 2 * np.pi  / (wavelength**2 * r_e_angstrom)
        else:
            eldens = [dispval * 2 * np.pi  / (wavelength**2 * r_e_angstrom) for dispval in dispvalues]

        return eldens

    def edens2disp(wavelength, edensvalues):
        '''
        edens2disp(wavelength,dispvalues)
        Calculate dispersion values from electron densities:
        You can export the data to a file and read it with pandas 

        Parameters
        ----------
        wavelength: wavelength of beam in Angström
        eldensvalues: list or number in Angström. 

        Returns
        -------
        dispvalues: list or numbers, depending on input of 'eldensvalues'.
        
        '''        
        if not isinstance(edensvalues, list):
            dispvalues = edensvalues * wavelength**2 * r_e_angstrom / (2 * np.pi)
        else:
            dispvalues = [edensval * wavelength**2 * r_e_angstrom / (2 * np.pi) for edensval in edensvalues]
        return dispvalues

    @staticmethod
    def shift_refl(anz_refl,compare=False, spacing = 10, reverse = False):
        i = 1
        j = 0
        shift = np.zeros(anz_refl)
        while i in range(0, spacing**anz_refl) and j in range(0, anz_refl + 1):
            shift[j] = i
            i *= spacing
            j += 1
        if compare:
            shift = [shift[i // 2] for i in range(0, len(shift *2))]
        if reverse:
            shift = np.flip(shift)
        return shift 

    @staticmethod
    def antoine_eq(A,B,C,T):
        '''p_sat is determined by the Antoine equation. Values for A, B, C (Antoine-parameters) have to be looked up'''
        p_sat = []  
        mmhgtobar = (101.325 / 760 * 1e-2)
        [p_sat.append(10**(A-(B/(t+C)))) for t in T]
        pref_err = []
        [pref_err.append((B*np.log(10)*10**(A-(B/(C+t))))/((C+t)**2)) for t in T]
        p_sat = [mmhgtobar * p for p in p_sat]
        pref_err = [mmhgtobar * e for e in pref_err] 
        return p_sat, pref_err

    def first_min_refl(self, first_min_guess=0.1, nofit=False):
        ''' Die Funktion findet die Postion des ersten Minmums einer Reflektivität in q. Benötigt wird der Dateipfad,
                indem die Outdatei liegt. Mit first_min_guess
            wird eine erste Schätzung der Position abgegben.
               '''
        if nofit:
            datfile_wp = min([os.path.join(self.Filepath,erg.name) for erg in os.scandir(self.Filepath) if erg.name.endswith('.dat')])
             # datfile = [f for f in os.listdir(self.Filepath) if '.dat' in f and not 'unweighted' in f and not 'unshifted' in f
            #            and not 'unnorm' in f and not 'orig' in f]
            # datfile_wp = os.path.join(self.Filepath, datfile[0])
            data = pd.read_csv(datfile_wp, header=None, sep='\s+',
                               usecols=[0, 1], names=['q', 'counts'])
            minima = find_peaks(-data['counts'])
            q_first_min = min(
                data['q'][minima[0]], key=lambda x: abs(x - first_min_guess))
        else:
            data = dict()
            # minima = {}
            data, _ = self.read_XRR_data()
            minima = find_peaks(- data.YCALC_norm)
            q_first_min = min(data.q[minima[0]], key=lambda x: abs(x - first_min_guess))

            return q_first_min

    def layer_thickness(self):
        rftfile = [os.path.join(self.Filepath, f) for f in os.listdir(self.Filepath) if f.endswith(self.eldens_ext)][0]
        dispersion = pd.read_csv(rftfile, header=1, sep='\s+',
                                 usecols=[0, 1, 2], names=['z', 'delta', 'beta'])
        tp = []
        arr_delta = dispersion['delta']
        arr_z = dispersion['z']
        neg_grad = - np.gradient(arr_delta, arr_z)
        minima = find_peaks(neg_grad, height=(10**(-11), 1))
        tp = arr_z[minima[0]]
        self.thickness = np.diff(tp)


    def get_gas_layer_density(self):
        ''' Function returns height, gas layer density and the maximum electron density of the gas layer.
        '''
        pref = 2 * sc.constants.pi / (self.wavelength**2 * r_e_angstrom)
        distfile = [f for f in new_listdir(self.Filepath) if f.endswith(self.distfile_ext)][0]
        self.distribution_data = pd.read_csv(
            distfile, header=None, skiprows=2, sep='\s+')
        cols = np.arange(0, self.distribution_data.shape[1])
        height = self.distribution_data[cols[0]]
        self.distribution_data[cols[1:]
                                   ] = self.distribution_data[cols[1:]] * pref
        self.max_rho_layers = self.distribution_data[cols[0:]].max().to_dict()
        
    def peak_width(self, num_layer='last'):
        '''Determine FWHM of a single peak in data set. Input needed are the dispersion values of the layer out of the Layer-distribution-file'''
        data = self.get_gas_layer_density()
        if num_layer == 'last':
            dispersion_values = self.distribution_data.iloc[:, -1]
        else:
            dispersion_values = self.distribution_data.iloc[:,num_layer]
        height = self.distribution_data.iloc[:, 0]  
        self.half_max = max(dispersion_values) / 2.
        # find when function crosses line half_max (when sign of diff flips)
        # take the 'derivative' of signum(half_max - Y[])
        d = np.sign(self.half_max - np.array(dispersion_values[0:-1])) - \
            np.sign(self.half_max - np.array(dispersion_values[1:]))

        self.left_idx = np.where(d > 0)[0]
        self.right_idx = np.where(d < 0)[-1]    
        self.width = np.array(height[self.right_idx]) - \
            np.array(height[self.left_idx])

        return self.width

    def integrate_eldens(self, num_layer='last', errors=False, anz_layers = None):
        ''' The function returnsthe integrated electron density of a specific layer of an electron density profile.

        args:
            * num_layer: number of the column of the layer in the *_dist.rtf file. default is last (e.g. integrate electron density of adsorbed layer)
            * errors: if True, 
        '''

        pref = 2 * sc.constants.pi / (self.wavelength**2 * r_e_angstrom)

        try:
            distfile = [os.path.join(self.Filepath, f) for f in os.listdir(self.Filepath) if f.endswith(self.distfile_ext)][0]
        except TypeError:
            for f in os.listdir(self.Filepath):
                file = os.path.join(self.Filepath, f.decode('utf-8'))
                if 'dist.rft' in file and os.path.isfile(file):
                    distfile = file
        if not distfile:
            print('No ' + distfile + '_dist.rft" file found in ' + self.Filepath)
        else:
            self.data = pd.read_csv(
                distfile, header=None, skiprows=2, sep='\s+')
            anz_cols = self.data.shape[1]
            cols = np.arange(0, anz_cols)
            if num_layer == 'last':
                self.eldens = pref * self.data[cols[-1]]
                self.int_val = integrate.simps(
                    self.eldens, x=self.data[cols[0]], even='avg')
            else:
                self.eldens = pref * self.data[cols[num_layer]]

            self.int_val = integrate.simps(
                self.eldens, x=self.data[cols[0]], even='avg')

        if not anz_layers == None:
            first_layer = anz_cols - anz_layers
            sum_layers = np.sum(self.data[cols[first_layer:first_layer + anz_layers -1]], axis =1) * pref
            self.int_val = integrate.simps(sum_layers, x = self.data[cols[0]], even = 'avg')

        if errors:
            data = {}
            eldens = {}
            self.int_val = []
            for i, f in enumerate(distfiles):
                data[i] = pd.read_csv(f, header=None, skiprows=2, sep='\s+')
                anz_cols = data[i].shape[1]
                cols = np.arange(0, anz_cols)
                if num_layer == 'last':
                    eldens.update({i: pref * data[i][cols[-1]]})
                else:
                    eldens.update({i: pref * data[i][cols[num_layer]]})
                self.int_val.append(integrate.simps(
                    eldens[i], x=data[i][cols[0]], even='avg'))
            self.mean_err = (np.mean(
                [abs(self.int_val[0] - self.int_val[1]), abs(self.int_val[0] - self.int_val[2])]))

        if not errors:
            return self.int_val

    def con_params(self):
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
            df = pd.read_fwf(latest_save, header=3, skipfooter=3, widths=[
                36, 15, 14], names=['parameter', 'Value', 'Increment'], skip_blank_lines=True)
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
        return conparams

    def readDistFile(self, model = 'tidswell', OTS = True, layOnSub = [], convertDispersion = True, widthGasLayer = False, meanDensity = True,
        show = False):
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
        distfile = [f for f in new_listdir(self.Filepath) if f.endswith(self.distfile_ext)]
        if not distfile:
            print('No ' + 'layer distribution file found in '  + self.Filepath)
            pass
        else:
            distfile = distfile[0]
            df = pd.read_csv(distfile, header=None, skiprows=2, sep='\s+', names = colnames[model])
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
            if meanDensity:
                mean_dens = dict()
                for k in df.keys():
                    if not k == 'z':
                        maxdens = max(df[k])
                        mean_dens.update({k:maxdens})
                # print(mean_dens)
            numb_layer = anzcols - 2 # substract 1 for z column and one for substrate
            _ = self.con_params()
            sigmas = self.roughness
            sigmas = list(sigmas.values())
            reference_density = df[colnames[model][-1]].iloc[-1]
            layer_lines, intersections_ind, pos_inter, W, zeta = dict(), [], [], {},{}
            [layer_lines.update({k:(df.z, df[k])}) for k in df.keys() if not k == 'z']

            if show:
                traces = []
                for key in df.keys():
                    if not key == 'z':
                        traces.append(go.Scatter(x = height, y = df[key], mode = 'lines+markers', name = key, line_width=4,
                            marker=dict(size=8,)
                                ))

                # traces.insert(len(traces),go.Scatter(x = height, y = d_oben / d_unten, mode = 'lines+markers', name = "profile",
                    # line_width = 4))

                f = go.Figure(data = traces, layout= layout.eldens_paper())
                f.show()
        if meanDensity:
            return mean_dens, df
        else:
            return df
    @staticmethod
    def turningpoints(x, thresh = 0.001):
        # dy = np.gradient(x)
        # idx_thresh = np.argmax(dy > thresh)
        # return idx_thresh
        xprime = np.diff(x)
        xdoubleprime = np.diff(xprime)
        tps = argrelextrema(xdoubleprime, np.greater)
        return(tps)

    def get_mean_dens_layers(self, model = 'tidswell',OTS = True, layOnSub = [], ):
        df = self.readDistFile(model = model, OTS = OTS, layOnSub = layOnSub)
        return df
    def readDatFile(self, ext = '.dat'):
        '''
        Read data from "*.dat"-file. Filename has to end with ".dat".
        --------------
        Returns:
        Data in datfile in pandas.DataFrame
        '''
        colnames = ['q', 'counts', 'weights']
        datfile = [f for f in new_listdir(self.Filepath) if f.endswith(ext)]
        datfile = sorted(datfile, key = len)[0] 
        datfileData = pd.read_csv(datfile, header = None, sep = '\s+', names = colnames)
        return datfileData


    def readEldensFileSpecific(self, substrate = False):
        colnames = ['z', 'delta', 'beta']
        if not substrate:
            edfile = [f for f in new_listdir(self.Filepath) if f.endswith(self.eldens_ext) and not any(x in f for x in ['substrate', self.distfile_ext])][0]
        else:
            edfile = [f for f in new_listdir(self.Filepath) if all(x in f for x in ['substrate', '.rft'])][0]
        edfileData = pd.read_csv(edfile, header=2, sep='\s+', names = colnames)
        return edfileData
        
    def calculate_gas_layer(self, anz_points = 1500, show = False, return_intval = False, return_peakwidth = False):
        try:
            df = self.readEldensFileSpecific()
            df_subst = self.readEldensFileSpecific(substrate = True)
        except (TypeError, IndexError):
            print('Check if all files are there.')
            pass

        zInterpol = np.linspace(min(df.z), max(df.z), anz_points)
        disp = df.delta
        disp_substrate = df_subst.delta
        
        func = interpolate.interp1d(df.z, disp, bounds_error = False)
        func_subst = interpolate.interp1d(df_subst.z, disp_substrate, bounds_error = False, fill_value=(df.delta.iloc[0],df.delta.iloc[-1]))

        disp_Interpol = func(zInterpol)
        disp_ref_mat = disp_Interpol[-1]
        disp_substInterpol = func_subst(zInterpol)
        disp_gas_layer = disp_Interpol + disp_ref_mat - disp_substInterpol
        
        edens_Interpol = [dispval * 2 * np.pi  / (self.wavelength**2 * r_e_angstrom) for dispval in disp_Interpol]
        edens_ref_mat = disp_ref_mat * 2 * np.pi  / (self.wavelength**2 * r_e_angstrom)
        edens_subst_Interpol = [dispval * 2 * np.pi  / (self.wavelength**2 * r_e_angstrom) for dispval in disp_substInterpol]
        edens_gas_layer = [dispval * 2 * np.pi  / (self.wavelength**2 * r_e_angstrom) for dispval in disp_gas_layer]


        if show:
            edens_Int = go.Scatter(x = zInterpol, y =  edens_Interpol, mode = 'lines', line = dict(color = 'red', width = 4 ), name = 'whole system')
            edens_subst_Int = go.Scatter(x = zInterpol, y = edens_subst_Interpol, mode = 'lines', line = dict(color = 'blue', width = 4 ), name = 'substrate')
            gas_layer_Int = go.Scatter(x = zInterpol, y = edens_gas_layer, mode = 'lines', line = dict(color = 'green', width = 4 ), name = 'gas layer')
            
            fig = go.Figure(data = [edens_Int, edens_subst_Int, gas_layer_Int], layout = layout.eldens())
            fig.update_layout(title = dict(text = 'Gas layer', xanchor = 'left',x = 0.07, y = 0.91),
                             yaxis_mirror = 'all', legend = dict(x = 1.01, y = 1.02, xanchor='left', yanchor = 'top'), xaxis_automargin = False,
                             xaxis_dtick = 10, width = 800, height = 600, legend_title='')
            fig.show()

       
        data = pd.DataFrame(list(zip(zInterpol, edens_gas_layer, edens_subst_Interpol, edens_Interpol)),columns = ['z', 'edens_gas', 'edens_subst', 'edens_whole'])

        
        peak = find_peaks(edens_gas_layer, height = 0.9999 * max(edens_gas_layer))  
        _, _, left_idx, right_idx = peak_widths(edens_gas_layer, peaks = peak[0])
        leftIdx = int(np.floor(left_idx))
        rightIdx = int(np.ceil(right_idx))
        peak_width = zInterpol[rightIdx] - zInterpol[leftIdx]
        
        int_val = integrate.simps(data.edens_gas, x=data.z, even='avg')

        if return_intval and not return_peakwidth:
            return data, int_val
        elif return_intval and return_peakwidth:
            return data, int_val, peak_width
        elif return_peakwidth and not return_intval:
            return data, peak_width
        else:
            return data

    @staticmethod
    def determine_shift_value_eldens(f, save = False):
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
                    df = pd.read_csv(f, header=2, sep=r'\s+', names=colnames)
                    for (columnName, columnData) in df.iteritems():
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
                df = pd.read_csv(f, header=2, sep=r'\s+', names=colnames)
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
    @staticmethod
    def shift_eldens(Filepath, subdirs=False, save = False, timeit = False, eldens_ext = '.rft'):
        '''
        Shift electron density with the reference medium dispersion value.

        Parameters:
        -----------
        Filepath: path, where the ".rft-files" are saved or path where subdirs containing ".rft-files" are located
        subdirs: boolean; if True, files are searched through the subdirs of path. If not, only path is searched. Default: False 
        save: boolean, if True, the added values are saved in the same file.
        '''
        t1 = time.perf_counter()
        anzcores = os.cpu_count()
        if not subdirs:
            rftfile = [os.path.join(Filepath, f) for f in os.listdir(Filepath) if f.endswith(eldens_ext)]
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
            pool.starmap(XRR_datatreatment.determine_shift_value_eldens, argtuple)


        t2 = time.perf_counter()
        if timeit:
            print(f'Finished in {t2-t1} seconds')

    @staticmethod
    def zrange_eldens(f, save = False, minZ = -50, maxZ = 150, timeit = False, verbose = False):
        '''
    
        '''
        t1 = time.perf_counter()
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
        df = pd.read_csv(f, header=None, sep=r'\s+', names=colnames, skiprows = len(lines) -1)

        first_line = [df[k].iloc[0] for k in df.keys() if not 'z' in k] 
        first_line.insert(0, minZ)
        kvPairsMin = dict()
        [kvPairsMin.update({k:[flv]}) for k, flv in zip(df.keys(), first_line)]
        kvPairs_df_min = pd.DataFrame.from_dict(kvPairsMin)
        last_line = [df[k].iloc[-1] for k in df.keys() if not 'z' in k]
        last_line.insert(0, maxZ)
        kvPairsMax = dict()
        [kvPairsMax.update({k:[llv]}) for k, llv in zip(df.keys(), last_line)]
        kvPairs_df_max = pd.DataFrame.from_dict(kvPairsMax)
        
        df_new = pd.concat([df, kvPairs_df_max], ignore_index = True)
        df_new2 = pd.concat([kvPairs_df_min, df_new], ignore_index = True)
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
                df_only_firstline = pd.concat([kvPairs_df_min, df], ignore_index = True)
                with open(f, 'w') as newfile:
                    df_only_firstline.to_string(newfile, header = None, index = False)
                prepend_line(f, lines)
                whats_done = 'Prepended ' + '\n' + f'{lines} to {filename}'
                # print('Prepended ' + '\n' + f'{lines} to {filename}')

            elif  df['z'].iloc[0] == minZ and not df['z'].iloc[-1] == maxZ:
                string = 'Has to be enlarged at the end'
                # print('Has to be enlarged at the end')
                df_only_last_line = pd.concat([df, kvPairs_df_max], ignore_index = True)
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
        t2 = time.perf_counter()
        
        if timeit:
            print(f'time needed to read file, convert to dataframe and append line: ' + "{:.3f}".format(t2-t1) + ' seconds')
        if verbose:
            print(f'{string}\n{whats_done}')

    @staticmethod 
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
        t1 = time.perf_counter()
        anzcores = os.cpu_count()
        if not subdirs:
            rftfile = [os.path.join(Filepath, f) for f in os.listdir(Filepath) if f.endswith(eldens_ext)]
        else:
            rftfile = []
            for root, subdirs, _ in os.walk(Filepath):
                for s in subdirs:
                    for f in os.listdir(os.path.join(root,s)):
                        if f.endswith(eldens_ext):
                            rftfile.append(os.path.join(root,s,f))
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
            pool.starmap(XRR_datatreatment.zrange_eldens, argtuple)
        t2 = time.perf_counter()
        if timeit:
            print(f'Time needed for all files is {t2-t1} seconds')

class einlesenDELTA:

    def __init__(self, fiopath, tifpath, savepath, wavelength=None, prefix=None):
        """
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
        """
        self.fiopath = fiopath
        self.tifpath = tifpath
        self.savepath = savepath
        self.wavelength = wavelength
        self.prefix = prefix
        if not wavelength:
            self.wavelength = kev2angst(27)
            print(f'No wavelength was specified. \u03BB = {"{:.4f}".format(self.wavelength)} inverse-\u00C5 is used as default.')
        if not os.path.isdir(self.savepath):
            os.mkdir(self.savepath)
            print(f'Created {self.savepath}')
    
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
        # print(os.path.basename(fio))
        # filepref = fio.split('/')[-1].strip('.FIO').strip(current_scan)
        filepref = os.path.basename(fio).strip('.FIO').strip(current_scan)
        current_file = filepref + current_scan
        tifprefix = f'{os.path.join(self.tifpath, self.prefix)}{current_scan}'
        print(tifprefix)
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

        ret_data = {'data':None, 'data_figure': None, 'absorber': None, 'error_figure':None, 'sep_loop_figure': None}

        tstart = time.perf_counter()
        bg_width = (roi[1] - roi[0]) / 2
        bg_left = np.array( 
            [math.floor(roi[0] - bgRoiDist - bg_width), roi[0] - bgRoiDist, roi[2], roi[3]])
        bg_right = np.array(
            [roi[1] + bgRoiDist, math.ceil(roi[1] + bgRoiDist + bg_width), roi[2], roi[3]])

        scan_numbers = scan_numbers_arr(start_scan = start_scan_number, anz_scans = anz_scans)
        ret_data['scan_numbers'] = scan_numbers
        norm_factor, dataframes = [], []

        measurement_files = self.list_measurement_FIOS(scan_numbers = scan_numbers)
        print(measurement_files)

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
       
        t1 = time.perf_counter()  
        shape = len(self.df_unsort.q)
        mw_arr, mw_uncorr_arr, mw_unnorm_arr, mw_bg_arr, mw_bg_unnorm_arr, mw_unnorm_uncorr_arr = np.empty(shape=shape), np.empty(shape = shape), np.empty(shape=shape), np.empty(shape=shape), np.empty(shape = shape), np.empty(shape=shape)

        for i in range(len(self.df_unsort.q)):
            try:
                imarray = self.tif2array(self.df_unsort.tif[i])
                sumtime1 = time.perf_counter()
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




        t2 = time.perf_counter()
        tiftime = t2 - t1
        if readtime:
            print(f'Read TIF-images in ' + "{:.4f}".format(tiftime) +' seconds.')

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

                dummy = go.Scatter(x = self.df['q'], y = self.df['mw'], mode = 'none', yaxis = 'y2', showlegend = False, hoverinfo = 'skip')
                fig = go.Figure(data=[trace1, trace2, trace3, dummy], layout=layout.refl())
                # fig.update_layout(legend_title= 'scan ' + str(scan_numbers[0]) + '-' + str(scan_numbers[-1]), legend_title_font_size = 16)
            else:
                trace1  = go.Scatter(x = self.df['q'], y = self.df['uncorr'], mode = 'lines+markers', name = 'data', showlegend=True)
                dummy = go.Scatter(x = self.df['q'], y = self.df['uncorr'], mode = 'none', yaxis = 'y2', showlegend = False, hoverinfo = 'skip')
                fig = go.Figure(data = [trace1, dummy], layout = layout.refl())
            fig.update_layout(title = None, legend_title= 'scan ' + str(scan_numbers[0]) + '-' + str(scan_numbers[-1]), legend_title_font_size = 16,
                hovermode='x', yaxis_range = [0,9])
            # fig.show()
            ret_data['data_figure'] = fig
# This should be backed up!!
        if error_plot:
            t1 = time.perf_counter()
            start_scan = self.df_unsort['tif'][0].split('/')[-1].strip('.tif').split('_')[-2].lstrip('0')
            end_scan = self.df_unsort['tif'][len(self.df_unsort['q']) - 1].split('/')[-1].strip('.tif').split('_')[-2].lstrip('0')
            sample_system = self.df_unsort['tif'][0].split('/')[-1].strip('.tif').split('_')
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
            fig_err.update_layout(autosize=True, showlegend=True, legend_title='', legend_borderwidth=1.0, title=dict(
                text = f'{sample_system} scan {start_scan} - {end_scan}'), xaxis_mirror=True, yaxis_mirror=True)
            ret_data['error_figure'] = fig_err
            # fig_err.show()
        t2 = time.perf_counter()
        error_time = t2 - t1
        if readtime: print(f'Created error plot in {error_time}')
            
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
        if pre_peak_eval:
            counts = np.array(self.df.mw)
            firstmin = argrelmin(counts, order=8)[0][0]
            truncate_indices = list(range(0, firstmin + 1))
            self.df = self.df.drop(truncate_indices).reset_index(drop=True)

        if save:
            filename =self.savepath + '/' + measurement_files[0].split('/')[-1].strip('.fio').strip('.FIO') + '.dat'
            if weight_on_error_plot:
                df2save = pd.DataFrame(list(zip(self.df.q, self.df.mw, self.df.weights)), columns = ['q', 'mw', 'weights'])
                print('with errors')
            else:
                df2save = pd.DataFrame(list(zip(self.df.q, self.df.mw)))
            df2save.to_string(filename, header=False, index=False)
            print('File ' + filename + ' has been saved.')
        ret_data ['data']= self.df


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
            # t1 = time.perf_counter()
            tempfiles = [os.path.join(filepath, t) for t in os.listdir(filepath) if t.endswith('.txt')]
            for t in tempfiles:
                tempData_single.append(pd.read_csv(t, header = 1, names = ['date', 'volts', 'temp']))
            tempData = pd.concat(tempData_single, ignore_index = True)
            # t2 = time.perf_counter()
            # print(f'{t2-t1} seconds')
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
                
    def dectimage(self, scan_number, th_ind, roi=None, pref_tif='run07_20', bgRoiDist = 1, margin=10, series=False, sleeptime=0.1, save_imgs = False):
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
        '''

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
            tifnames = [t.split('/')[-1] for t in tifs_sorted]
            imgs2save = [os.path.join(self.savepath, t.replace('.tif', '.png').replace('.TIF', '.png')) for t in tifnames]

        if not roi is None:
            roi_height = roi[1] - roi[0]
            roi_width = roi[3]-roi[2]

            bg_height = (roi[1] - roi[0]) / 2
            
            bg_left = np.array([roi[2], roi[0] - bgRoiDist - bg_height, roi[3], roi[0] - bgRoiDist])

            bg_right = np.array([roi[2], roi[1] + bgRoiDist, roi[3], roi[1] + bgRoiDist + bg_height])

        if series:
            plt.ion()
            f = plt.figure(figsize = (12, 32))
            ax = f.gca()
            ax.set(xlim=[roi[2] - margin, roi[3] + margin], ylim=[roi[0] - margin, roi[1] + margin])

            draw_roi = patches.Rectangle((roi[2], roi[0]),roi_width, roi_height, fc='None', ec='g', lw=2)
            draw_bgl = patches.Rectangle((bg_left[0], bg_left[1]),roi_width, bg_height, fc='None', ec='r', lw=2)
            draw_bgr = patches.Rectangle((bg_right[0], bg_right[1]),roi_width, bg_height, fc='None', ec='r', lw=2)
            rect = ax.add_patch(draw_roi)
            rect_bgl = ax.add_patch(draw_bgl)
            rect_bgr = ax.add_patch(draw_bgr)

            im = ax.imshow(df.imarrays.loc[0], norm = LogNorm())
            for i in range(1, len(df.tif)):
                im.set_data(df.imarrays.loc[i])
                f.suptitle('Detector image of scan ' + str(scan_number) + ' th = ' + "{:.3f}".format(df.theta.loc[i]) + '°\nq = ' + "{:.3f}".format(th2q(self.wavelength, df.theta.loc[i])) + r' $\AA^{-1}$', fontsize=16)
                f.canvas.flush_events()
                plt.pause(sleeptime)
            plt.close()
        else:
            imarray = self.tif2array(df.tif.loc[th_ind])
            f = plt.figure(figsize = (12, 32))
            ax = f.add_subplot(111)
            im = ax.imshow(imarray, norm = LogNorm())
            f.suptitle('Detector image, scan: ' + str(scan_number) + ' th = ' + "{:.4f}".format(df.theta.loc[th_ind]) + '°\nq = ' + "{:.3f}".format(th2q(self.wavelength, df.theta.loc[i])) + r' $\AA^{-1}$', fontsize=16)
            ax.set(xlim=[roi[2] - margin, roi[3] + margin], ylim=[roi[0] - margin, roi[1] + margin])
            draw_roi = patches.Rectangle((roi[2], roi[0]),roi_width, roi_height, fc='None', ec='g', lw=2)
            draw_bgl = patches.Rectangle((bg_left[0], bg_left[1]),roi_width, bg_height, fc='None', ec='r', lw=2)
            draw_bgr = patches.Rectangle((bg_right[0], bg_right[1]),roi_width, bg_height, fc='None', ec='r', lw=2)
            rect = ax.add_patch(draw_roi)
            rect_bgl = ax.add_patch(draw_bgl)
            rect_bgr = ax.add_patch(draw_bgr)
            plt.show()

    def rename_files(self, variation_param, rename=False, shownames=False, prefix=str(), replace_prefix=False, variation_param_unit = 'bar', file_ext = '.dat'):
        '''
        Rename files to get variation_param into the filename
        arg:q
            *variation_param: array of ordered variation_param values
        kwargs:
            * rename: boolean, renames the datfiles
            * shownames: shows the new names of the files
            * prefix: if given, the prefix will be eliminated from the filenames. Must be part of the filename.
            * replace prefix: New prefix at the beginning of filenames
        '''


        self.datfiles_fp = [os.path.join(self.savepath, f) for f in os.listdir(
            self.savepath) if f.endswith(file_ext)]
        self.datfiles_fp = sorted(
            self.datfiles_fp, key=lambda x: x.replace(file_ext, '').split('_')[2:])
        self.datfiles = [f for f in os.listdir(self.savepath) if f.endswith(file_ext)]
        self.datfiles = sorted(
            self.datfiles, key=lambda x: x.replace(file_ext, '').split('_')[2:])
        for f in self.datfiles:
            if replace_prefix in f:
                print(f'{replace_prefix} already in files. Files were not renamed.')
                return
        try:
            if not len(variation_param) == len(self.datfiles_fp):
                print(
                    'Some entries for the variation_param list seem to be missing. Please check.')
            else:
                self.new_names = ['_'.join(name.split('_')[
                                           :-1]) + '_' + re.sub("\s+", "_", variation_param[ind],flags=re.UNICODE) + file_ext for ind, name in enumerate(self.datfiles)]
                
                if prefix:
                    self.new_names = [i.replace(prefix,"") for i in self.new_names]
                if replace_prefix:
                    self.new_names = [replace_prefix + i for i in self.new_names]
                self.new_names = [name.replace(',','p') for name in self.new_names]
                self.new_names = [name.strip(file_ext).replace('.','p') + file_ext for name in self.new_names]
                self.new_names = [name.replace('(','').replace(')','') for name in self.new_names]
                self.new_names = [name.replace('_@','') for name in self.new_names]
                gases, toAppend = ['He', 'N2'], [''] * len(variation_param)
                for i, n in enumerate(self.new_names):
                    for j, g in enumerate(gases):
                        if g in n:
                            self.new_names[i] = n.replace(g,'').strip(file_ext).rstrip('_') + file_ext
                            toAppend[i] = '_' + g
                # [print(n) for n in self.new_names]
                for i,name in enumerate(self.new_names):
                    try:
                        rep_number = re.findall('#+\d',name)[0]
                    except Exception as e:
                        rep_number='#1'
                    if not any(x in name for x in ['bar', 'air', 'luft']):
                        if not '#' in name:
                            self.new_names[i] = name.strip(file_ext) + '_'+ variation_param_unit + file_ext
                            
                        else:
                            # rep_number = re.findall('#+\d',name)[0]
                            self.new_names[i] = name.strip(file_ext).strip(rep_number) + variation_param_unit + '_' + rep_number.strip('#') + file_ext
                        if 'down' in name:
                            if not "#" in name:
                                # print(name)
                                self.new_names[i] = name.strip('down.dat').strip(' ') + variation_param_unit + '_down.dat'
                            else:
                                # rep_number = re.findall('#+\d',name)[0]
                                self.new_names[i] = name.strip(file_ext).strip('_down' + rep_number) + '_' + variation_param_unit + '_down_' + rep_number.strip('#') + file_ext
                                # print(name.strip(file_ext).strip('_down' + rep_number) + '_bar_down_' + rep_number.strip('#'))
                    elif 'air' in name and '#' in name and not 'down' in name:
                        self.new_names[i] = name.strip(file_ext).replace(rep_number,'') + f'{variation_param_unit}_' +  rep_number.strip('#') + file_ext
                        # print(f'Here are those names: {self.new_names[i]}')
                    elif 'air' in name and not any(j in name for j in ['down', '#']):
                        self.new_names[i] = name.strip(file_ext).replace(rep_number,'') + file_ext
                    elif all(x in name for x in ['air', '#' ,'down']):
                        self.new_names[i] = name.strip(file_ext).replace('_down_','').replace(rep_number, '') + f'_{variation_param_unit}' + '_down_' + rep_number.strip('#') + file_ext
                    elif all(x in name for x in ['air']) and not any(x in name for x in ['down', '#']):
                        self.new_names[i] = name.strip(file_ext) + f'_{variation_param_unit}' + file_ext    

                for i, n in enumerate(self.new_names):
                    self.new_names[i] = n.strip(file_ext) + toAppend[i] + file_ext
                
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
                    shutil.move(entry, subfolders_with_path)
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
                shutil.copy(lsfile, entry)

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
            shutil.copyfileobj(optdata, optdata2)
            optdata.close()
            optdata2.close()

    def delete_neg_values(self, file_ext = '.dat'):    
        # FilePath = r'/home/mike/Dokumente/Uni/Promotion/Messzeit_DELTA_0222/Auswertung/scCO2_2  '
        # EXT = "*.dat"
        colnames = ['q', 'counts', 'weigths']
        all_dat_files_full_path = []
        for root, dirnames , f_names in os.walk(self.savepath):
          for f in f_names:
            if f.endswith(file_ext) and not any(x in f for x in ['orig', 'unweighted', 'unshifted']):
              oldname_fp=os.path.join(root,f)
              newname_fp = os.path.join(root,f).strip(file_ext) + '_orig' + file_ext
              with open(oldname_fp, 'r') as f:
                firstline = f.readline().split()
                anz_cols = len(firstline)
              df = pd.read_csv(oldname_fp, header=None,delimiter='\s+', names=colnames[0:anz_cols])
              try:
                no_neg_values = df[df.counts >=10**(-12)]
                if len(no_neg_values) < len(df):
                  shutil.move(oldname_fp, newname_fp)
                  no_neg_values.to_string(oldname_fp, index =False, header=False)
                  print('File ' + os.path.basename(oldname_fp) + ' contained negative values. Old file with negative values saved as ' + os.path.basename(newname_fp) + 
                          ' and new file will be saved as ' + os.path.basename(oldname_fp))
                else:
                    print(f'File {os.path.basename(oldname_fp)} does not contain negative values')
              except Exception as e:
                print('File ' + os.path.basename(oldname_fp) + ' has not the correct format.')
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

    def calc_delta_from_scattering_factor(self, rho, mol_mass, scattering_factor):
        avogadro = constants.physical_constants['Avogadro constant'][0]
        delta = rho * avogadro / mol_mass * r_e_angstrom * self.wavelength**2 / (2 * sc.pi) * scattering_factor * 1e-24

        return delta

    def calc_beta_from_scattering_factor(self, rho, mol_mass, scattering_factor):
        avogadro = constants.physical_constants['Avogadro constant'][0]
        beta = rho * avogadro / mol_mass * r_e_angstrom * self.wavelength**2 / (2 * sc.pi) * scattering_factor * 1e-24

        return beta
