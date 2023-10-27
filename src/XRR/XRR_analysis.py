import sys, os
import plotly.graph_objs as go
from matplotlib import pyplot as plt
from XRR.plotly_layouts import plotly_layouts as layouts
from XRR.utilities.conversion_methods import th2q, q2th, disp2edens
from XRR.utilities.math import find_inflection_points, model_subbotin, negative_step_model, rectangle_model, model_subbotin, find_nearest, model_neg_step, FWHM
import numpy as np
from scipy import constants
from time import perf_counter
from datetime import datetime
from pandas import read_csv, to_datetime, DatetimeIndex, DataFrame ,options
import math
layout = layouts(transpPaper=False, transpPlot=False, transpLegend=True,unit_mode='slash', bigTicks=False, bigTickFont=True)

set_figure_size = (11.3, 2.5) 
set_figure_fintsize = 18

r_e_angstrom = constants.physical_constants['classical electron radius'][0] * 1e10

class single_layer_simulations:
    def __init__(self, alpha_in, wavelength, dispersion, absorption):
        '''
        Calculate Fresnel Reflectivity and transmission and the penetration depth of x-rays.
        '''
        self.alpha_in = alpha_in
        self.wavelength = wavelength
        self.dispersion = dispersion
        self.absorption = absorption
        self.alpha_crit = np.sqrt(2*self.dispersion)
        self.alpha_red = self.alpha_in / self.alpha_crit
    def Fresnel_Refl(self):
        p_plus_squared = 0.5 * (np.sqrt((self.alpha_red**2*self.alpha_crit**2-self.alpha_crit**2)**2 + 4*self.absorption**2) + \
            (self.alpha_red**2*self.alpha_crit**2 - self.alpha_crit**2))
        p_plus = np.sqrt(p_plus_squared)
        p_minus_squared = 0.5 * (np.sqrt((self.alpha_red**2*self.alpha_crit**2-self.alpha_crit**2)**2 + 4*self.absorption**2) - \
            (self.alpha_red**2*self.alpha_crit**2 - self.alpha_crit**2))
        
        R_F = ((self.alpha_red*self.alpha_crit - p_plus)**2 + p_minus_squared) / ((self.alpha_red*self.alpha_crit + p_plus)**2 + p_minus_squared)
            
        return self.alpha_red, R_F

    def Fresnel_transmission(self):
        T_F = (4 * self.alpha_in**2) / (abs(self.alpha_in + np.sqrt(self.alpha_in**2-self.alpha_crit**2 + 2j*self.absorption))**2)
        return self.alpha_red, T_F

    def pen_depth_xray(self):
        k = 2*np.pi / self.wavelength
        p_minus_squared = 0.5 * (np.sqrt((self.alpha_red**2*self.alpha_crit**2-self.alpha_crit**2)**2 + 4*self.absorption**2) - \
            (self.alpha_red**2*self.alpha_crit**2 - self.alpha_crit**2))
        p_minus = np.sqrt(p_minus_squared)
        Lambda = 1 / (k*p_minus)
        return self.alpha_red, Lambda

class layer:
    '''
    Create a layer for XRR-calculations. 
    '''

    def __init__(self, delta, beta, thickness, roughness = 0):
        '''
        Parameters:
        -----------
        delta: float, dispersion of the layer.
        beta: float, absorption  of the layer.
        thickness: float, layer thickness.
        roughness: float, default is 0. Roughness specifies the interfacial roughness between a layer and a layer below. 
        '''
        self.delta = delta 
        self.beta = beta
        self.thickness = thickness
        self.roughness = roughness
    def layer_parameters(self, delta, beta, thickness, roughness):
        '''
        Returns parameters from init function in dictionary
        '''
        return {'disp':self.delta, 'absorb':self.beta, 'd':self.thickness, 'sigma': self.roughness}

    def index_of_refraction(self):
        '''
        Returns the complex index of refraction of a layer.
        '''
        return 1-self.delta+1j*self.beta
    def __repr__(self):
        return f'{self.delta, self.beta, self.thickness, self.roughness}'

class stack_layers:
    '''
    Stack layers defined by the layer class.
    '''
    def __init__(self, lol):
        self.lol = lol
        super().__init__()
    
    def __getitem__(self, index):
        return self.lol[index]
    
    def __len__(self):
        return len(self.lol)

    def __setitem__(self):
        return self.lol
    
    def __repr__(self):

        return str(self.lol)
    def copy(self):
        return self.lol.copy()


class refl_multiple_layers:
    '''
    Several calculations such as reflectivity coefficients, wave vectors, specular reflectivity of a layer stack.
    '''

    def __init__(self, alpha_i, layer_stack,  wavelength, params_path = False, xaxis = 'qz',
        ind_of_refrac_profile = ['unity','error_fct', 'tanh', 'fisk_widom'], alpha_i_unit = ['th_deg', 'th_rad', 'Ae-1']):
        '''
        Parameters:
        -----------
        args:
            *alpha_i: list or numpy.array, angles of incidence either in degrees, radians, or inverse angstrom. The unit has to bespecified by "alpha_i_unit".
            *layer_stack: list of tuples, stack of layers created with stack_layers class
            *wavelength: floar, wavelength of the used radiation in angstrom.
        kwargs:
            *params_path: Do not use now. Boolean, if True, the parameters should be read from a lsfit-file.
            *xaxis: str, does not have a meaning yet.
            *ind_of_refrac_profile: 
            *alpha_i_unit: Unit of angles of incidence. Choose from "th_deg":degrees, "th_rad":radians, "Ae-1":inverse angstrom.    
        '''
        self.layer_stack = stack_layers(layer_stack)
        self.ind_of_refrac_profile = ind_of_refrac_profile
        input_IoR = any(self.ind_of_refrac_profile == item for item in ['unity','error_fct', 'tanh', 'fisk_widom'])
        if not input_IoR and not isinstance(self.ind_of_refrac_profile, list):
            lst = ['unity','error_fct', 'tanh', 'fisk_widom']
            newline = "\n"
            print(f'Not a valid index of refraction profile specified. Choose from:\n{newline.join(f"{item}" for item in lst)}')
        elif isinstance(self.ind_of_refrac_profile, list):
            print(f'No index of refraction profile specified. Ae-1 is used as default.')
        self.alpha_i_unit = alpha_i_unit
        self.alpha_i = alpha_i
        roughnesses = [self.layer_stack[j].roughness for j in range(len(self.layer_stack))]

        if all(r == 0 for r in roughnesses):
            self.ind_of_refrac_profile = 'unity'
        elif len(ind_of_refrac_profile) > 1 and isinstance(ind_of_refrac_profile, list) :
            self.ind_of_refrac_profile = 'tanh'
        else:
            self.ind_of_refrac_profile = ind_of_refrac_profile

        if isinstance(params_path, str):
            parameter_dict = self.params_from_file()
            self.wavelength = float(parameter_dict["wavelength_[A]_or_energy_[eV]"])
            self.x_axis = parameter_dict["x-axis_:_[0:TH,_1:2TH,_2:QZ]"]
        else:
            # self.anz_layers = anz_layers
            self.anz_layers = len(self.layer_stack) -1 
            self.wavelength = wavelength
            self.xaxis = xaxis

        if isinstance(self.alpha_i_unit, list):
            self. alpha_i_unit = 'Ae-1'
            print('No unit for incident angles declared. \u00C5 is assumed.')
            self.qz = self.alpha_i
            self.th_deg = q2th(self.wavelength, self.qz)
            self.th_rad = np.deg2rad(self.th_deg)
        if self.alpha_i_unit == 'th_deg':
            self.th_deg = self.alpha_i
            self.th_rad = np.deg2rad(self.th_deg)
            self.qz = th2q(self.wavelength, self.alpha_i)
        elif self.alpha_i_unit == 'th_rad':
            self.th_rad = self.alpha_i
            self.th_deg = np.rad2deg(self.th_rad)
            self.qz = th2q(self.wavelength, self.th_deg)
        elif self.alpha_i_unit == 'Ae-1':
            self.qz = self.alpha_i
            self.th_deg = q2th(self.wavelength, self.qz)
            self.th_rad = np.deg2rad(self.th_deg)
        # print(self.alpha_i_unit, self.qz)

        if not len(self.layer_stack) - 1 == self.anz_layers:
            print(f'Number of layers are inconsistent. Please check.')


    
    def params_from_file(self):
        lines = read_lines_file(self.params_path)
        parameters = False
        parameter_dict = {}
        for l in lines:
            if l.strip().startswith('###'):
                parameters = True
                continue
            if l.strip().startswith('***'):
                parameters=False
            if parameters:
                split = l.split()
                var_name = '_'.join(split[1:-1])
                val = split[-1]
                parameter_dict[var_name] = val
        return parameter_dict

    def calculate_z_positions(self):     
        thickness = np.asarray([lay.thickness for lay in self.layer_stack])
        pos_inter = np.zeros(len(self.layer_stack))
        thickness_all_layers = 0
        th_masked = np.ma.masked_where(abs(thickness) == np.inf, thickness)
        thickness_all_layers = np.sum(th_masked)
        # print(pos_inter)

        for i in range(len(self.layer_stack)-1, 0, -1):
            # print(f'{i}, pos inter:{pos_inter[i]}, pos_inter[{i-1}]:{pos_inter[i]-thickness[i]}\n')
            pos_inter[i-1] = pos_inter[i] - thickness[i]
        if thickness[0] == np.inf:
            pos_inter = np.insert(pos_inter, 0, - np.inf)
        if thickness[-1] == np.inf:
            pos_inter = np.append(pos_inter, np.inf)
        return pos_inter

    def calculate_specular_reflectivity(self, plot = False):
        t1 = perf_counter()
        store_rj_jp1 = dict()
        store_xj = dict()
        pos_interfaces = self.calculate_z_positions()
        refrac_indices = np.asarray([self.layer_stack[j].index_of_refraction() for j in range(len(self.layer_stack))])
        j = 0
        while j < (len(self.layer_stack)):
            if j == 0:
                store_xj[j] = np.zeros(len(self.alpha_i))
            elif 0 < j < len(self.layer_stack):
                kj = self.wave_vector(refrac_indices[j])
                kjp1 = self.wave_vector(refrac_indices[j-1])
                exp_1 = np.exp(-2j*kj*pos_interfaces[j]) 
                exp_2 = np.exp(2j* kjp1*pos_interfaces[j])
                rj_jp1 = self.refl_coeffs(j = j)
                store_rj_jp1[j] = rj_jp1
                xj = exp_1  * (store_rj_jp1[j] + store_xj[j-1] *exp_2) / (1 + store_rj_jp1[j] * store_xj[j-1] *exp_2) 
                store_xj[j] = xj
            j += 1
        if plot:
            trace = go.Scattergl(x = self.qz, y = abs(store_xj[list(store_xj.keys())[-1]])**2, mode = 'lines+markers', line = dict(width = 4, color='black'),
                marker = dict(size = 0, symbol = 2, color = 'rgba(0,0,0,0)', line = dict(width = 0, color = 'blue')), name = 'reflectivity')
            fig = go.Figure(data = trace, layout = layout.refl())
            fig.show()
        t2 = perf_counter()
        print(f'{t2-t1} seconds.')
        return abs(store_xj[list(store_xj.keys())[-1]])**2

    def wave_vector(self,ind_of_refrac):  
        return 2*np.pi / (self.wavelength) * np.sqrt(ind_of_refrac**2-np.cos(self.th_rad)**2) 

    def refl_coeffs(self, j):
        if self.layer_stack[j].roughness == 0 and self.ind_of_refrac_profile == 'tanh':
            roughness = 1e-40
        else:
            roughness = self.layer_stack[j].roughness

        kj = self.wave_vector(self.layer_stack[j].index_of_refraction())
        kjp1 = self.wave_vector(self.layer_stack[j-1].index_of_refraction())
        r = (kj - kjp1) / (kj + kjp1)
        if self.ind_of_refrac_profile == 'unity':
            return r
        elif self.ind_of_refrac_profile == 'tanh':
            numerator = np.sinh(np.sqrt(3)*roughness*(kj-kjp1))
            denominator = np.sinh(np.sqrt(3)*roughness*(kj+kjp1))
            rj_jp1_tanh = numerator /denominator
            return rj_jp1_tanh
        elif self.ind_of_refrac_profile == 'error_fct':
            rj_jp1_error_fct = r*np.exp(-2*kj*kjp1*self.layer_stack[j].roughness**2)
            return rj_jp1_error_fct
    
    def IoR_profiles_interface_self(self, anz_points = 500, margin = 10, zmin = False, zmax = False, plot = False, shiftToZero=True, convert2eldens = False):
        pre_factor = 2*np.pi / self.wavelength**2 / constants.physical_constants['classical electron radius'][0] * 1e-10
        # layer properties
        refrac_indices = np.asarray([self.layer_stack[j].index_of_refraction() for j in range(len(self.layer_stack))])
        dispersions = np.asarray([1-np.real(k) for k in refrac_indices])
        absorbtions = np.asarray([np.imag(k) for k in refrac_indices])
        rho = np.asarray([pre_factor*disp for disp in dispersions])
        thickness = np.asarray([self.layer_stack[j].thickness for j in range(len(self.layer_stack))])
        totT = np.sum(thickness) if not thickness[0] == np.inf else np.sum(thickness[1:])
        sigmas = np.asarray([self.layer_stack[r].roughness for r in range(len(self.layer_stack))])
        

        # interface positions

        pos_interfaces = self.calculate_z_positions()[:-1]
        print(pos_interfaces, sigmas)

        # if pos_interfaces[0] == - np.inf: sigmas = sigmas[1:]
        # if pos_interfaces[0] == - np.inf: pos_interfaces = np.asarray(pos_interfaces[1:])

        #  z-array to calculate density profile
        lmarg, rmarg = 10 * sigmas[0], 5* sigmas[-1]
        if lmarg == 0 or rmarg == 0:
            lmarg, rmarg = totT / 2, totT / 2
        if not zmin:
            zmin = -1.1 * totT - lmarg
        if not zmax:
            zmax = rmarg
        z = np.linspace(zmin, zmax, num = anz_points)        
        w = np.zeros((len(self.layer_stack) - 1, anz_points))
        disps = np.zeros((len(self.layer_stack) - 1, anz_points))
        data = DataFrame(columns = list(range(0,len(self.layer_stack))))
        prob_densities = DataFrame(columns = list(range(0,len(self.layer_stack))))
        for i in range(0, len(self.layer_stack)-1):
            a = constants.pi /(2*np.sqrt(3)) 
            n, probs,dat = list(), list(), list()
            for zz in z:
                nn = ((dispersions[i] + dispersions[i+1]) / 2) - (dispersions[i] - dispersions[i+1]) / 2 * math.tanh(a* (zz - pos_interfaces[i+1]) / sigmas[i+1])
                prob = a / 2 / sigmas[i+1] * 1 / (math.cosh(a * (zz - pos_interfaces[i+1]) / sigmas[i+1])**(2))
                probs.append(prob)
                dat.append(nn)
            w[i,:] = probs
            disps[i,:] = dat
            # data[i] = np.array(dat) * np.array(probs)
            data[i] = dat
            prob_densities[i] = probs
        data.insert(len(self.layer_stack), 'summe', data.sum(axis=1))
        data.insert(0,'z', z)

        # print(data.head(10))
        # data.insert(len(data.columns), 'sum', data[list(data.columns[1:len(data.columns)-1])].sum(axis = 1))
        # print(data.columns)

        traces = list()
        prob_traces = list()
        for i in range(len(self.layer_stack)):
            traces.append(go.Scattergl(x = data.z, y = data[i], mode = 'lines', name = f'{i}'))
            prob_traces.append(go.Scattergl(x = data.z, y = prob_densities[i], mode = 'lines', name = f'W{i}', yaxis = 'y2'))

        # fig = go.Figure(data = go.Scattergl(x = data.z, y = data.sum, mode = 'markers'), layout = layout.eldens())
        prof_trace = go.Scattergl(x = data.z, y = data.summe, mode = 'markers', name = 'profile')
        fig = go.Figure(data = traces + [prof_trace] + prob_traces, layout = layout.eldens())
        fig.update_layout(yaxis2 = dict(**layout.standard_linear_axis, overlaying = 'y', side = 'right'))
        fig.show()
        # return data
        #     if i == 0:
        #         if self.ind_of_refrac_profile == 'tanh':
        #            n_S[0] =  
        return data.z, data.summe
    
    def IoR_profiles_interface(self, anz_points = 500, margin = 10, zmin = False, zmax = False, plot = False, shiftToZero=True, convert2eldens = False):
            pre_factor = 2*np.pi / self.wavelength**2 / constants.physical_constants['classical electron radius'][0] * 1e-10
            
            # layer properties
            refrac_indices = np.asarray([self.layer_stack[j].index_of_refraction() for j in range(len(self.layer_stack))])
            dispersions = np.asarray([1-np.real(k) for k in refrac_indices])
            absorbtions = np.asarray([np.imag(k) for k in refrac_indices])
            rho = np.asarray([pre_factor*disp for disp in dispersions])
            thickness = np.asarray([self.layer_stack[j].thickness for j in range(len(self.layer_stack))])
            totT = np.sum(thickness) if not thickness[0] == np.inf else np.sum(thickness[1:])
            sigmas = np.asarray([self.layer_stack[r].roughness for r in range(len(self.layer_stack))])
            
            # interface positions
            pos_interfaces = self.calculate_z_positions()[:-1]
            if pos_interfaces[0] == - np.inf: sigmas = sigmas[1:]
            if pos_interfaces[0] == - np.inf: pos_interfaces = np.asarray(pos_interfaces[1:])
            
            #  z-array to calculate density profile
            lmarg, rmarg = 10 * sigmas[0], 5* sigmas[-1]
            if lmarg == 0 or rmarg == 0:
                lmarg, rmarg = totT / 2, totT / 4
            if not zmin:
                zmin = -1.1 * totT - lmarg
            if not zmax:
                zmax = rmarg
            z = np.linspace(zmin, zmax, num = anz_points)        

            w = np.zeros((len(self.layer_stack) - 1, anz_points))

            for i in range(len(self.layer_stack)-1):
                # print(i, pos_interfaces[i], sigmas[i], z)
                if math.isclose(sigmas[i], 0):
                    sup = np.heaviside(pos_interfaces[i]-z,0.5)
                else:
                    if self.ind_of_refrac_profile == 'error_fct':
                        # print('this is error fct')
                        sup = np.asarray([(1 + math.erf((pos_interfaces[i] - zz) / sigmas[i] / np.sqrt(2))) / 2 for zz in z])

                    elif self.ind_of_refrac_profile == 'tanh':
                        # print('this is tanh')
                        sup = np.asarray([(1 + math.tanh(math.pi * (pos_interfaces[i] - zz) / sigmas[i] / (2*np.sqrt(3)))) / 2 for zz in z])
                    elif self.ind_of_refrac_profile == 'unity':
                        sup = np.asarray([1 for zz in z])
                if i == 0:
                    # this should be -np.inf; roughness of substrate is always zero, bc its not the interface between substrate and first layer!!
                    zdownint = pos_interfaces[0]- thickness[0]
                    sigint = 0
                else:
                    zdownint = pos_interfaces[i-1]
                    sigint = sigmas[i-1]
                if math.isclose(sigint, 0):
                    sdown = np.heaviside(zdownint - z, 0.5)
                else:
                    if self.ind_of_refrac_profile == 'error_fct':
                        sdown = np.asarray([(1 + math.erf((zdownint - zz) / sigint / np.sqrt(2))) / 2 for zz in z])
                    elif self.ind_of_refrac_profile == 'tanh':
                        sdown = np.asarray([(1 + math.tanh(math.pi * (zdownint - zz) / sigint / (2*np.sqrt(3)))) / 2 for zz in z])
                    elif self.ind_of_refrac_profile == 'unity':
                        sdown = np.asarray([1 for zz in z])
                w[i, :] = sup - sdown
                # plt.plot(z, sup, label='sup')
                # plt.plot(z, sdown, label='sdown')
                # plt.plot(z, sup-sdown, label='diff')
                # plt.legend()
                # plt.show()

            if convert2eldens:
                rho = rho[:-1]
                layer_prof = rho[:, np.newaxis] * w[:, ...]
                prof = np.sum(layer_prof, axis=0)
            else:
                dispersions = dispersions[:-1]
                layer_prof = dispersions[:, np.newaxis] * w[:, ...]
                prof = np.sum(layer_prof, axis = 0)
            if shiftToZero:
                z = z+totT
            if plot:
                trace = go.Scattergl(x = z, y = prof, mode = 'lines', line = dict(color = 'black', width = 4))
                fig = go.Figure(data = trace, layout = layout.eldens())
                fig.show()
            return z, prof


    def EffectiveDensitySlicing(self, step = 0.1, zmin = False, zmax = False, show_profile = True, convert2eldens=True,
            shiftSubstrate2zero = True):
            '''
            Calculate electron density profile of a layer system based on the Effective Density Model.

            Parameters:
                args:
                    None
                kwargs:
                    * step: Thickness of the individual layers in angstrom.
                    * zmin/zmax: boolean, optional
                                 Default is False, If False, z-array will be calculated automatically (recommended). Else enter zmin and zmax as float for minimum and maximum z-values.
                    *show_profile: boolean, optional
                                   If True, a plot of the electron density profile is shown.
                    * convert2eledens: boolean, optional
                                       If True, dispersion will be transformed to electron density. Defauls is True
                    * shiftSubstrate2zero: boolean, optional
                                           If True, the dispersion profile will be shifted in z-direction, so that the substrate ends at z = 0.
                    * return_W_functions: boolean, optional
                                          If True, W-functions and zeta will also be returned. Default is False.
            
            Returns:
            --------
                if return_W_functions:
                z (array), disp_profile (array), sliced_layer_stack (list of layers(tuples)), W (dict), zeta (dict)
            else:
                z (array), disp_profile (array), sliced_layer_stack (list of layers(tuples))
            '''


            returnObj = {'z':None,'disp_profile': None, 'fig_disp_profile':None, 'sliced_layer_stack':None, 'W':None, 'zeta':None, 'weighted_dispersions':None}

            pre_factor = 2*np.pi / self.wavelength**2 / constants.physical_constants['classical electron radius'][0] * 1e-10
            # layer properties
            refrac_indices = np.asarray([self.layer_stack[j].index_of_refraction() for j in range(len(self.layer_stack))])
            dispersions = np.asarray([1-np.real(k) for k in refrac_indices])
            absorptions = np.asarray([np.imag(k) for k in refrac_indices])
            rho = np.asarray([pre_factor*disp for disp in dispersions])
            thickness = np.asarray([self.layer_stack[j].thickness for j in range(len(self.layer_stack))])
            totT = np.sum(thickness) if not thickness[0] == np.inf else np.sum(thickness[1:])
            sigmas = np.asarray([abs(self.layer_stack[r].roughness) for r in range(len(self.layer_stack))])
            
            # interface positions
            pos_interfaces = self.calculate_z_positions()[:-1]
            # add end of vacuum interface (np.inf), compare substrate; add sigma of interface in infnity 
            pos_interfaces = np.append(pos_interfaces,np.inf)
            # sigmas = np.append(sigmas, 0)

            #  z-array to calculate density profile
            margin = max(np.sum([r for r in sigmas]), 10 * sigmas[1], 10 * sigmas[-2])

            if not zmin:
                zmin = -1 * (totT + margin) 
            if not zmax:
                zmax = margin
            z = np.arange(zmin, zmax, step) 
            
            W, weighted_dispersions, zeta = dict(), dict(), dict()
            for i in range(len(self.layer_stack)):
                W[i] = np.zeros_like(z)
                weighted_dispersions[i] = np.zeros_like(z) 
                if i < len(self.layer_stack)-1:
                    zeta[i] = (sigmas[i]*pos_interfaces[i +1] + sigmas[i+1]*pos_interfaces[i]) / (sigmas[i]+sigmas[i+1])
                else:
                    zeta[i] = np.inf
            

            for idx_layer in range(0, len(self.layer_stack)):
                maskz = (z<=zeta[idx_layer])
                imaskz = np.logical_not(maskz)
                if self.ind_of_refrac_profile == 'error_fct':
                    W[idx_layer][maskz] =[1 / 2 * (1 + math.erf((zz - pos_interfaces[idx_layer]) / (np.sqrt(2) * sigmas[idx_layer]))) for zz in z[maskz]]
                    W[idx_layer][imaskz] =[1 / 2 * (1 - math.erf((zz - pos_interfaces[idx_layer+1]) / (np.sqrt(2) * sigmas[idx_layer+1]))) for zz in z[imaskz]]
                elif self.ind_of_refrac_profile == 'tanh':
                    W[idx_layer][maskz] = np.asarray([1 / 2 * (1 + math.tanh((zz - pos_interfaces[idx_layer]) * math.pi / (2*math.sqrt(3) * sigmas[idx_layer]))) for zz in z[maskz]])
                    W[idx_layer][imaskz] = np.asarray([1 / 2 * (1 - math.tanh((zz - pos_interfaces[idx_layer+1]) * math.pi / (2*math.sqrt(3) * sigmas[idx_layer+1]))) for zz in z[imaskz]])
                elif self.ind_of_refrac_profile == 'unity':
                    W[idx_layer][maskz] = np.asarray([1  for zz in z[maskz]])
                    W[idx_layer][imaskz] = np.asarray([1 for zz in z[imaskz]])
            wsum = np.add.reduce([w for w in W.values()])
            for k in W:
                W[k] /= wsum
            disps, absorps = dict(), dict()
            for j in range(len(self.layer_stack)):
                disps[j] = W[j] * dispersions[j]
                absorps[j] = W[j]*absorptions[j]
            disp_profile = np.asarray([sum(t) for t in zip(*disps.values())])
            absorp_profile = np.asarray([sum(t) for t in zip(*absorps.values())])
            sliced_layer_stack = list()
            for j in range(len(z)):
                sliced_layer_stack.append(layer(disp_profile[j], absorp_profile[j], step, roughness = 0))
            sliced_layer_stack[0] = layer(disp_profile[0], absorp_profile[0], np.inf, roughness = 0)
            # print(zeta, W)
            if convert2eldens:
                disp_profile = disp2edens(self.wavelength, disp_profile)

            for wfunc in W:
                if convert2eldens:
                    weighted_dispersions[wfunc] = W[wfunc]*disp2edens(self.wavelength, dispersions[wfunc])
                else:
                    weighted_dispersions[wfunc] = W[wfunc]*dispersions[wfunc]
                # deriv = np.gradient(weighted_dispersions[wfunc], z)
                # deriv_l = np.where(deriv >0, deriv, 0)
                # deriv_r = np.where(deriv<0, deriv, 0)
                # deriv_r_abs = abs(deriv_r)
                # idxs_l = FWHM(np.nan, deriv_l)
                # idxs_r = FWHM(np.nan, deriv_r_abs)
                # if not wfunc == 0:
                #     print(f'{wfunc}, left_ind:{math.dist(z[idxs_l[0]], z[idxs_l[1]])}, right_ind: {math.dist(z[idxs_r[0]], z[idxs_r[1]])}, sigma:{sigmas[wfunc]}')
                #     deriv_l_trace = go.Scattergl(x = z, y = deriv_l, mode = 'lines', name = 'deriv left')
                #     deriv_r_trace = go.Scattergl(x = z, y = deriv_r, mode = 'lines', name = 'deriv right')
                #     FWHM_l_trace = go.Scattergl(x = [z[idxs_l[0]], [idxs_l[1]]], y = [deriv_l[idxs_l[0]], deriv_l[idxs_l[1]]], mode = 'lines',
                #             line = dict(dash = 'dash', width = 3, color = 'black'), name = 'FWHM left')
                #     FWHM_r_trace = go.Scattergl(x = [z[idxs_r[0]], z[idxs_r[1]]], y = [deriv_r[idxs_r[0]], deriv_r[idxs_r[1]]], mode = 'lines',
                #             line = dict(dash = 'dash', width = 3, color = 'black'), name = 'FWHM right')
                #     fig = go.Figure(data = [deriv_l_trace, deriv_r_trace, FWHM_l_trace, FWHM_r_trace], layout = layout.eldens())
                #     fig.show()
                # else:
                #     print(f'{wfunc}, Si:{math.dist(z[idxs_r[0]], z[idxs_r[1]])}')
                #     # print(math.dist(z[idxs_r[0]], z[idxs_r[1]]))
                # # print(type(wfunc), idxs_l, idxs_r)
            if shiftSubstrate2zero:
                amplitude = weighted_dispersions[0][np.argmax(weighted_dispersions[0])] 
                cen_ind = find_nearest(weighted_dispersions[0], amplitude / 2)
                # print(cen_ind, z[cen_ind])
                # [print(i, z[i], weighted_dispersions[0][i]) for i in range(len(z))]
                cen_pos = z[np.argmax(weighted_dispersions[0])]
                fit = model_neg_step(x = z, y = weighted_dispersions[0], amplitude = amplitude, sigma = 0.5, mu = z[cen_ind], form = self.ind_of_refrac_profile)
                # fitr = rectangle_model(x = z, y = weighted_dispersions[0], amplitude = amplitude, center1 = cen_pos, center2  =cen_pos, sigma1 = 2, sigma2 = 2, mode = 'arctan')
                # print(fit.fit_report())
                # [print(f, disp) for f, disp in zip(fit.best_fit, weighted_dispersions[0])]
                # plt.plot(z, fit.best_fit, label='best fit')
                # plt.plot(z, fit.init_fit, label ='init fit')
                # plt.plot(z,weighted_dispersions[0], 'g--', label ='data')
                # plt.legend()
                # plt.show()
                zShift = fit.params['mu']
                # zShift = fitr.params['step_center2']
                # print(zShift, fit.params['mu'])
                z += abs(zShift)   
                # z += abs(zeta[1])
            if show_profile:
                traces = []
                for wfunc in W:
                    if convert2eldens:
                        # weighted_dispersions[wfunc] = W[wfunc]*disp2edens(self.wavelength, dispersions[wfunc])
                        # W[wfunc]*disp2edens(self.wavelength, dispersions[wfunc])
                        traces.append(go.Scattergl(x = z, y = weighted_dispersions[wfunc], mode = 'lines', line = dict(width = 4), name = f'W[{wfunc}]',
                            legendgroup=wfunc))
                    else:
                        # W[wfunc]*dispersions[wfunc]
                        traces.append(go.Scattergl(x = z, y = weighted_dispersions[wfunc], mode = 'lines', line = dict(width = 4), name = f'W[{wfunc}]',
                            legendgroup=wfunc))
                        # weighted_dispersions[wfunc] = W[wfunc]*dispersions[wfunc]
                traces.append(go.Scattergl(x = z, y = disp_profile, mode = 'lines', line = dict(width = 4, dash='solid'), name = 'profile', legendgroup='profile'))
                fig = go.Figure(data =traces, layout = layout.eldens())
                # fig.show()
                returnObj['fig_disp_profile'] = fig
                returnObj['weighted_dispersions'] = weighted_dispersions
            returnObj['z'], returnObj['disp_profile'], returnObj['sliced_layer_stack'], returnObj['W'], returnObj['zeta'] = z, disp_profile , sliced_layer_stack, W, zeta

            # returnObj[['z','fig_disp_profile', 'disp_profile','sliced_layer_stack', 'W','zeta']] = [z, fig, disp_profile , sliced_layer_stack, W, zeta]
            return returnObj

    def fit_refl(self, data_file):
        init_stack = self.layer_stack
        datfileData = readDatFileData(data_file)
        print(init_stack)
