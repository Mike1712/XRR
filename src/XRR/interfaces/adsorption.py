import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, simpson, trapezoid
from scipy.optimize import newton, fsolve, root
# import sympy as sp
import concurrent.futures
from scipy import constants
from scipy.signal import argrelextrema
from lmfit import Model, minimize, Parameters
from scipy.optimize import minimize as scminimize
import XRR.FluidProps as FP
from XRR.plotly_layouts import plotly_layouts as layouts
from XRR.helpers import find_closest_values_in_series
from XRR.data_evaluation import VolumeFractionProfile
from XRR.utilities.conversion_methods import celsius2kelvin, kelvin2celsius
import plotly.graph_objs as go
import pickle
from numba import jit, cuda, cfunc, njit, prange
from NumbaMinpack import lmdif, minpack_sig
# import cupy as cp
layout = layouts(locale_settings='DE')
# Critical pressure, volume and temperature
# These values are for the van der Waals equation of state for CO2:
# (p - a/V^2)(V-b) = RT. Units: p is in Pa, Vc in m3/mol and T in K.
# pc = 7.404e6
# Vc = 1.28e-4
# Tc = 304

def vdw(Tr, Vr):
    """Van der Waals equation of state.

    Return the reduced pressure from the reduced temperature and volume.

    """

    pr = 8*Tr/(3*Vr-1) - 3/Vr**2
    # print(pr)
    return pr


def vdw_maxwell(Tr, Vr, print_vrs = False):
    """Van der Waals equation of state with Maxwell construction.

    Return the reduced pressure from the reduced temperature and volume,
    applying the Maxwell construction correction to the unphysical region
    if necessary.

    """
    ret_data = {'pr':None, 'Vrmin':None, 'Vrmax':None, 'pr0':None, 'Vrcross':None}
    pr = vdw(Tr, Vr)
    if Tr >= 1:
        # No unphysical region above the critical temperature.
        ret_data['pr'] = pr
        return ret_data
        # return pr

    if min(pr) < 0:
         raise ValueError('Negative pressure results from van der Waals'
                         ' equation of state with Tr = {} K.'.format(Tr))

    # Initial guess for the position of the Maxwell construction line:
    # the volume corresponding to the mean pressure between the minimum and
    # maximum in reduced pressure, pr.
    iprmin = argrelextrema(pr, np.less)
    iprmax = argrelextrema(pr, np.greater)
    Vr0 = np.mean([Vr[iprmin], Vr[iprmax]])

    def get_Vlims(pr0):
        """Solve the inverted van der Waals equation for reduced volume.

        Return the lowest and highest reduced volumes such that the reduced
        pressure is pr0. It only makes sense to call this function for
        T<Tc, ie below the critical temperature where there are three roots.

        """

        eos = np.poly1d( (3*pr0, -(pr0+8*Tr), 9, -3) )
        roots = eos.r
        roots.sort()
        Vrmin, Vrcross, Vrmax = roots
        return Vrmin, Vrcross,Vrmax

    def get_area_difference(Vr0):
        """Return the difference in areas of the van der Waals loops.

        Return the difference between the areas of the loops from Vr0 to Vrmax
        and from Vrmin to Vr0 where the reduced pressure from the van der Waals
        equation is the same at Vrmin, Vr0 and Vrmax. This difference is zero
        when the straight line joining Vrmin and Vrmax at pr0 is the Maxwell
        construction.

        """

        pr0 = vdw(Tr, Vr0)
        Vrmin, Vrcross, Vrmax = get_Vlims(pr0)
        return quad(lambda vr: vdw(Tr, vr) - pr0, Vrmin, Vrmax)[0]

    # Root finding by Newton's method determines Vr0 corresponding to
    # equal loop areas for the Maxwell construction.
    Vr0 = newton(get_area_difference, Vr0)
    pr0 = vdw(Tr, Vr0)
    Vrmin, Vrcross, Vrmax = get_Vlims(pr0)
    # Set the pressure in the Maxwell construction region to constant pr0.
    pr[(Vr >= Vrmin) & (Vr <= Vrmax)] = pr0

    ret_data = {'pr':pr, 'Vrmin':Vrmin, 'Vrmax':Vrmax, 'Vrcross':Vrcross,'pr0':pr0}
    return ret_data

def supercritical_adsorption(x_b, epsilon_divby_kT = -2, epsilon_a_divby_kT = -1):
    x_b = np.array(x_b)
    k_b = constants.physical_constants['Boltzmann constant'][0]
    def alpha():
        alph = 1/epsilon_divby_kT * 1/(3)

def Peng_Robinson_isotherm(molecule, rho, Temp, Volume, p_units = 'bar', T_units = 'celsius', show = False):
    fluid_data = FP.selectMolecule(molecule)
    Temp, Volume = np.asarray(Temp), np.asarray(Volume)

    molar_mass, Tcrit, pcrit, acentric_factor = fluid_data.molar_mass, fluid_data.T_crit, fluid_data.p_crit, fluid_data.acentric_factor
    print(Tcrit)
    if p_units =='bar':
        pcrit *=1e5
    if T_units == 'celsius':
        Temp += 273.15
        Tcrit += 273.15
    print(Temp, Tcrit)
    Tred = Temp / Tcrit
    R = constants.physical_constants['molar gas constant'][0]
    a = (0.457235*R**2*Tcrit**2) / (pcrit)
    b = (0.077796*R*Tcrit) / (pcrit)
    # T_red = np.array(Temp) / T_crit
    # V_molar = molar_mass / rho
    V_molar = molar_mass / Volume
    
    def Peng_Robinson_reduced():
        pass
    def alpha_PR(acentric_factor, Tred):
        if acentric_factor <= 0.49:
            kappa = 0.379642 + 1.54226*acentric_factor - 0.26992*acentric_factor**2
            alpha = (1+kappa*(1-np.sqrt(Tred)))**2
        else:
            alpha = (1 + (0.379642 + (1.48503 - (1.164423 - 1.016666 * acentric_factor) * acentric_factor) * acentric_factor) * (1- np.sqrt(Tred)))**2
        return alpha
    alpha = alpha_PR(acentric_factor, Tred)
    p = ((R*Temp) / (V_molar-b)) - ((a*alpha) / (V_molar**2 + 2*b*V_molar-b**2))
    p = np.asarray(p)
    if p_units == 'bar':
        p *=1e-5
    if show:
        trace = go.Scattergl(x = V_molar, y = p, mode = 'markers', marker = dict(size = 10), name = 'PR')
        fig = go.Figure(data = trace, layout = layout.eldens())
        fig.update_layout(
            xaxis = dict(title_text = '<i>V</i>'),
            yaxis = dict(title_text = '<i>p</i>'))
        fig.show()

    def cubic_PR(p0, Temp, acentric_factor, Tred):
        alpha = alpha_PR(acentric_factor, Tred)
        a_0 = 1
        a_1 = (b- R * Temp/p0)
        a_2 = (alpha * a / p) - 3 * b**2 - 2*R*Temp*b/p0
        a_3 = b**3 +R*Temp*b**2/p0 - alpha*a*b/p0
    # return p


def vdW_p_fromT(molecule, temperatures, Vmol):
    '''
        Given pressures and temperatures are in bar and °C, respectively
    '''
    
    R_mol = constants.physical_constants['molar gas constant'][0]
    fluid_data = FP.selectMolecule(molecule)
    T_crit = celsius2kelvin(fluid_data.T_crit)
    T = celsius2kelvin(np.array(temperatures) )
    T_r = T/(kelvin2celsius(T_crit))
    a_vdW = 27 / 64 * R_mol**2 * T_crit**2 / (fluid_data.p_crit*1e5)
    b_vdW = fluid_data.molar_mass/(3*fluid_data.rho_crit)
    p = R_mol * T /(Vmol-b_vdW)-(a_vdW / (Vmol**2)) 
    trace = go.Scattergl(x = kelvin2celsius(T), y = p*1e-5, mode = 'lines')
    fig = go.Figure(data = trace, layout = layout.eldens())
    fig.add_hline(y = fluid_data.p_crit)
    fig.update_layout(xaxis_title_text = '<i>T</i>/°C', yaxis_title_text = '<i>p</i>/bar')
    fig.show()
def Peng_Robinson_p_from_T(molecule, temperatures, Vmol):
    '''
        Given pressures and temperatures are in bar and °C, respectively
    '''
    fluid_data = FP.selectMolecule(molecule)
    T_crit = celsius2kelvin(fluid_data.T_crit)
    T = celsius2kelvin(np.array(temperatures) )
    T_r = T/(kelvin2celsius(T_crit))
    p_crit = fluid_data.p_crit * 1e5
    acentric_factor = fluid_data.acentric_factor
    R_mol = constants.physical_constants['molar gas constant'][0]
    a_PR = 0.45723553 * (R_mol*T_crit)**2/ p_crit
    b_PR = 0.07779607 * (R_mol*T_crit)/p_crit
    print(a_PR, b_PR)
    if acentric_factor <=0.49:
        alpha = (1+(0.37464 + 1.54226*acentric_factor-0.26992*acentric_factor**2)*(1-np.sqrt(T_r)))**2
    else:
        alpha = (1+(0.379642+(1.48503-(1.164423-1.016666*acentric_factor)*acentric_factor)*acentric_factor)*(1-np.sqrt(T_r)))**2

    p = R_mol * T / (Vmol-b_PR)- ((a_PR * alpha)/(Vmol**2 + 2*b_PR*Vmol - b_PR**2))    
    # print(p*1e-5)
    trace = go.Scattergl(x = kelvin2celsius(T), y = p*1e-5, mode = 'lines')
    fig = go.Figure(data = trace, layout = layout.eldens())
    fig.add_hline(y = fluid_data.p_crit)
    fig.update_layout(xaxis_title_text = '<i>T</i>/°C', yaxis_title_text = '<i>p</i>/bar')
    fig.show()
# def Peng_Robinson_T_from_p(molecule, pressures, Vmol, temperature_start):
#     '''
#         Given pressures and temperatures are in bar and °C, respectively
#     '''
#     fluid_data = FP.selectMolecule(molecule)
#     T_crit = celsius2kelvin(fluid_data.T_crit)
#     temperature_start = celsius2kelvin(np.array(temperature_start) )
#     T_r = temperature_start/(kelvin2celsius(T_crit))
#     p_crit = fluid_data.p_crit * 1e5
#     acentric_factor = fluid_data.acentric_factor
#     R_mol = constants.physical_constants['molar gas constant'][0]
#     a_PR = 0.45723553 * (R_mol*T_crit)**2/ p_crit
#     b_PR = 0.07779607 * (R_mol*T_crit)/p_crit

#     if acentric_factor <=0.49:
#         alpha = (1+(0.37464 + 1.54226*acentric_factor-0.26992*acentric_factor**2)*(1-np.sqrt(T_r)))**2
#     else:
#         alpha = (1+(0.379642+(1.48503-(1.164423-1.016666*acentric_factor)*acentric_factor)*acentric_factor)*(1-np.sqrt(T_r)))**2
#     p = R_mol * T / (Vmol-b_PR)- ((a_PR * alpha)/(Vmol**2 + 2*b_PR*Vmol - b_PR**2))    
#     print(p*1e-5)

# def Peng_Robinson_reduced(molecule, Vr, Tr, p_units = 'bar', T_units = 'celsius', show = False):
#     fluid_data = FP.selectMolecule(molecule)
    
#     Tr, Vr = np.asarray(Tr), np.asarray(Vr)
#     # Rho_r = 
#     molar_mass, Tcrit, pcrit, acentric_factor = fluid_data.molar_mass, fluid_data.T_crit, fluid_data.p_crit, fluid_data.acentric_factor
#     if p_units =='bar':
#         pcrit *=1e5
#     if T_units == 'celsius':
#         Tcrit += 273.15
#     # print(Tcrit)
#     R = constants.physical_constants['molar gas constant'][0]
#     a = (0.457235*R**2*Tcrit**2) / (pcrit)
#     b = (0.077796*R*Tcrit) / (pcrit)

#     def alpha_PR(acentric_factor, Tr):
#         kappa = 0.379642 + 1.54226*acentric_factor - 0.26992*acentric_factor**2
#         alpha = (1+(kappa)*(1-np.sqrt(Tr)))**2
#         return kappa, alpha
#         # if acentric_factor <= 0.49:
#         #     kappa = 0.379642 + 1.54226*acentric_factor - 0.26992*acentric_factor**2
#         #     alpha = (1+(kappa)*(1-np.sqrt(Tred)))**2
#         # else:
#         #     alpha = (1 + (0.379642 + (1.48503 - (1.164423 - 1.016666 * acentric_factor) * acentric_factor) * acentric_factor) * (1- np.sqrt(Tred)))**2
#         # return alpha
#     kappa, alpha = alpha_PR(acentric_factor, Tr)
#     pr = (Tr / Vr) / (1-b) - (a*alpha*Tr/Vr) / (np.sqrt(Tr) * (1+kappa*(1-np.sqrt(Tr)))*(1-b))
#     if show:
#         trace = go.Scattergl(x = Vr, y = pr, mode = 'markers', marker = dict(size = 10), name = 'PR')
#         fig = go.Figure(data = trace, layout = layout.eldens())
#         fig.update_layout(
#             xaxis = dict(title_text = '<i>V</i><sub>red</sub>'),
#             yaxis = dict(title_text = '<i>p</i><sub>red</sub>'))
#         fig.show()
    
#     return pr

def surface_coverage_guess(int_dens, radgyr_molec, A_wafer = 0.00018*1e20):
    # all lengths given in angstrom!
    int_dens = np.asarray(int_dens) *1e-2
    numb_molecs = int_dens * A_wafer
    A_molec = np.pi*radgyr_molec**2 
    Theta = numb_molecs * A_molec / A_wafer
    return Theta

# def BET_adsorption_isotherm_reduced(p_red, v_m, c_BET):
#     Theta_BET = (v_m * c_BET * p_red) / ((1- p_red) * (1 + (c_BET - 1) * p_red))
#     return Theta_BET

def BET_adsorption_isotherm_reduced(x, v_m, c):
    x = np.array(x)
    # Theta_BET = K*(v_m * c * x) / ((1- x) * (1 + (c - 1) * x))
    Theta_BET = (v_m * c  * x) / ((1-x)*(1-x+c*x))
    return Theta_BET

# das ist nicht Anderson! Das "**n" war ausversehen nicht nur beim x, sondern um das (1-x)
def Anderson(x, a, c, n):
    bm = a*((c*x)*(1-x)**n) / ((1-x)*(1+(c-1)*x))
    return bm

def Anderson_real(x, a, c, n):
    x = np.array(x)
    bm = a*((c*x)*(1-x**n)) / ((1-x)*(1+(c-1)*x))
    return bm

def n_BET_adsorption_isotherm(x, v_m, c, n):
    v = v_m*c*x / (1-x) * (1-(n+1)*x**n+n*x**(n+1)) / (1+(c-1)*x-c*x**(n+1))
    return v


def Toth_adsorption(x, v_m, K, m):
    x = np.array(x)
    v = v_m * ((1+1/K)**(1/m)*x) / ((1/K+x**m)**(1/m))
    return v
# def bashiri_drouji(p_red, v_m, c, n):
#     p_red = np.asarray(p_red)
#     v = (v_m*(c*p_red)**(1/n)) / ((1-p_red)* (1-p_red+(p_red*c)**(1/n)))
#     return v

def bashiri_drouji(x, a, v_m, c, n):
    x = np.asarray(x)
    v = a + v_m * (c * x)**n / ((1-x) * (1-x + (c * x)**n))
    return v 

def mahle(x, v_m, A, B, D):
    v = v_m / D * (np.arctan((x-A)/B) - np.arctan(-A/B)) 
    return V

def new_BET_science_direct(x, v_m, c, k):
    v = v_m * c* k * x / ((1-k*x)*(1+(c-1)*k*x))
    return v

def ward_wu(x, M, c, alpha, gamma):
    v = ((M*c*alpha*x)*((1-(1+gamma)*(alpha*x)**gamma) + gamma*(alpha*x)**(1+gamma))) / \
    ((1-alpha*x)*(1+(c-1)*alpha*x-c*(alpha*x)**(1+gamma)))
    return v

def GAB(x, v_m, C, K):
    v = v_m * C * K * x /((1 - K*x) * (1+(C-1)*K*x))
    return v
def Zou_et_al(x, v_m, C, K, alpha):
    v = v_m * C * K * x**alpha /((1 - K*x**alpha) * (1+(C-1)*K*x**alpha))
    return v

def Sips(x, v_m, K, n, m):
    v = v_m * (K*x / (1+(K*x)**n))**m
    return v

def Halsey(x, a, v_molec, b):
    v  = a* v_molec * x /((1-x)*np.exp(b*x))
    return v



@cfunc(minpack_sig)
def eq2solve(x, fvec, args):
    # args[0],args[1],args[2],args[3],args[4],args[5],args[6],  = fb, a, b, psi, temperature, k_B, R
    fvec[0] = np.log(x[0]*args[6]*args[4] / (args[0]*(1-x[0]*args[2]))) + x[0]*args[2]/(1-x[0]*args[2])-2*args[1]*x[0]/(args[6]*args[4]) + args[3]/(args[5]*args[4])

funcptr = eq2solve.address

@jit(nopython=True, fastmath=True)
# @jit(fastmath=True, nopython=False)
def solver_j(molar_density_bulk, a_bulk, a_from_z, psi, b, z, rho_atoms, epsilon_fs, sigma_fs, sigma_ff, interplanar_spacing, Tcrit, temperature, k_B, avogadro, R):
    # k_B = constants.physical_constants['Boltzmann constant'][0] # J/K
    # avogadro = constants.physical_constants['Avogadro constant'][0] # Teilchen/mol
    # R = constants.physical_constants['molar gas constant'][0] #J/(mol K)
    
    factorial_1 = molar_density_bulk * R * temperature / (1 - molar_density_bulk * b)
    exponential = molar_density_bulk * b /(1 - molar_density_bulk * b) - 2*a_bulk * molar_density_bulk /(R * temperature)
    fb = factorial_1 * np.exp(exponential)
    # funcptr = eq2solve.address
    neqs=1
    rho_from_z = list()
    # rho_from_z, a_from_z, psi = list(), list(), list()
    prefactor = 4 * np.pi * rho_atoms * epsilon_fs * sigma_fs**2
    for i in prange(len(z)):
        # find roots
        args = np.array([fb, a_from_z[i], b, psi[i], temperature, k_B, R])
        if temperature >= Tcrit:
            rt, fvec, success, info = lmdif(funcptr, np.array([1e4]), neqs, args = args)
            if success == False:
                rt, fvec, success, info = lmdif(funcptr, np.array([1e3]), neqs, args = args)
            else:
                rt = rt
            root_val = rt[0]
        else:
            rt, vec, success, info = lmdif(funcptr, np.linspace(1, 1e5, num = 10)[::-1], neqs, args = args)
            # rt = root(self.eq2gettozero, np.linspace(1, 1e5, num = 10)[::-1], args = (a_from_z[i], psi[i]), method = 'lm')
            if success == False:
                rt, vec, success, info = lmdif(funcptr, np.linspace(1, 1e3, num = 10)[::-1], neqs, args = args)
            else:
                rt = rt
            root_val = rt[0]
        rho_from_z.append(root_val)
    # rho_from_z = np.array(rho_from_z)
    return rho_from_z

class SLD_rangarajan:
    '''
    Calculate SLD-profiles and Gibbs surface excess for the adsorption of molecules on a flat wall at specicific temperatues and pressures using the SLD-model from Rangarajan et al. (DOI:10.1002/aic.690410411).
    '''
    # def __init__(self, molecule, temperature, pressure, epsilon_fs, interplanar_spacing, molar_density_bulk, rho_atoms,**kwargs):
    def __init__(self, molecule, temperature, pressure, epsilon_fs, rho_atoms,**kwargs):
        '''
        Parameters:
        -----------
            * molecule: str, chemical formula of molecule. Data has to be in XRR.Fluidprops. If it is not, add data manually in this file
            * temperature: float, temperature in celsius
            * pressure: float, pressure in bar
            * epsilon_fs: float, solid-fluid interaction energy parameter. Determines depth of interaction potential

        Kwargs:
        -----------
            Note: Not all kwargs are optional as they should be. If calculation does not succeed, check which parameter is missing in error message and add it as argument!
            * molar_density_bulk: float, unit is mol/m^3. If not given and VFP_data (XRR.data_evaluation.VolumeFractionProfile) exists, molar_density_bulk is calculated from VFP
            * VFP_data: dict, properties of VolumeFractionProfile calculated with XRR.data_evaluation.VolumeFractionProfile. Use this method to make sure that data types are correct
            * dist_data: dict, properties of layer distribution file from LSFit calculated with XRR.data_evaluation.readDistFile. Needed to calculate interplanar spacing of substrate (works only for OTS).
            * interplanar_spacing: float, unit is m. Spacing of planes of solid substrate. For OTS it can be calculated from distribution data. Otherwise it has to be passed explicitly.
            * sigma_ff: float, unit is m. Diameter of fluid molecules. If not given, 2*radius_of_gyration is used.
            * sigma_fs: float, unit is m. Solid-fluid diameter. If not given, it is calculated as (sigma_ff + interplanar_spacing)/2.
            * z_coords: ndarray, list: array with z-coordinates. Have to start at sigma_ff/2. If not given and no VFP_data available, z ranges from sigma_ff/2 to 10 sigma_ff with 500 points.
                        If VFP_data given, z is used from there (from max of VFP onwards)
            * NistFluidProps: os.path, isothermal properties of NIST chemistry webbook (https://webbook.nist.gov/chemistry/). If given, molar_density_bulk is read out from NIST File

        '''
        # constants
        self.k_B = constants.physical_constants['Boltzmann constant'][0] # J/K
        self.avogadro = constants.physical_constants['Avogadro constant'][0] # Teilchen/mol
        self.R = constants.physical_constants['molar gas constant'][0] #J/(mol K)

        self.molecule_data = FP.selectMolecule(molecule)
        self.pcrit, self.Tcrit = self.molecule_data.p_crit*1e5, celsius2kelvin(self.molecule_data.T_crit)
        self.temperature = celsius2kelvin(temperature)
        self.pressure = pressure * 1e5
        self.rho_atoms = rho_atoms

        if not 'VFP_data' in kwargs.keys():
            self.interplanar_spacing = kwargs['interplanar_spacing']
            if 'molar_density_bulk' in kwargs.keys():
                self.molar_density_bulk = kwargs['molar_density_bulk']
        
        elif all(x in kwargs.keys() for x in ['VFP_data', 'dist_data']):
            self.VFP_data = kwargs['VFP_data']
            self.VFP_df = self.VFP_data['df']
            self.dist_data = kwargs['dist_data']
            self.interplanar_spacing = 1.54 * np.sin(109.5/2 * (np.pi/180)) * np.cos(self.dist_data['layer_properties']['tilt_angle'].loc['OTS_tail'] * np.pi / 180)*1e-10
            self.molar_density_bulk = self.VFP_df.difference.iloc[-1]* 1e30 /(self.molecule_data.anz_el * self.avogadro)
        else:
            print('If no distribution_data and VFP_data provided please enter values for: rho_atoms, interplanar_spacing, molar_density_bulk')
        
        self.epsilon_fs = epsilon_fs
        
        if 'sigma_ff' in kwargs.keys():
            self.sigma_ff = kwargs['sigma_ff']
        else:
            self.sigma_ff = 2*self.molecule_data.radius_of_gyration * 1e-10

        if 'z_coords' in kwargs.keys() and not 'VFP_data' in kwargs.keys():
            self.z = kwargs['z_coords']
        elif 'VFP_data' in kwargs.keys():
            start_ind = self.VFP_df['difference'].idxmax() + 10
            self.VFP_df = self.VFP_df.iloc[start_ind:]
            self.z = np.array((self.VFP_df.z_interpol - self.VFP_df.z_interpol.iloc[0])*1e-10 + self.sigma_ff/2)
            self.VFP_profile_molar_units = self.VFP_df.difference*1e30/(self.molecule_data.anz_el * self.avogadro)
            # print(self.z)
            # fig = go.Figure(data = go.Scatter(x = self.z / self.sigma_ff, y = self.VFP_df.difference, mode = 'markers'), layout = layout.eldens())
            # fig.show()
        else:
            self.z = np.linspace(self.sigma_ff/2, 20 * self.sigma_ff/2, num = 500)
            # z1 = np.linspace(self.sigma_ff/2, 5 * self.sigma_ff, num = 600)
            # z2 = np.linspace(5*self.sigma_ff, 20*self.sigma_ff, num = 100)
            # self.z = np.concatenate([z1, z2], axis = None)
        if not 'sigma_fs' in kwargs.keys():
            self.sigma_fs = (self.interplanar_spacing + self.sigma_ff)/2
        else:
            self.sigma_fs = kwargs['sigma_fs']
        
        # vdW Parameters
        self.a_bulk = 27/64 * self.R**2 * self.Tcrit**2 / self.pcrit
        self.b = self.R * self.Tcrit/(8*self.pcrit)

        if 'NistFluidProps' in kwargs.keys():
            self.NistData,_ = FP.readFluidProps(kwargs['NistFluidProps'])
            closest_pressure,cp = find_closest_values_in_series(self.NistData['pressure(bar)'], self.pressure*1e-5)
            # self.molar_density_bulk = self.NistData['density(mol/m3)'][self.NistData['pressure(bar)'] == self.pressure*1e-5]
            self.molar_density_bulk = self.NistData['density(mol/m3)'].iloc[closest_pressure]

        self.kwargs = kwargs

    def fugacity_bulk(self):
        # VFP_df = self.VFP_data['df']
        a_bulk = 27 * self.R**2 * self.Tcrit**2 / (64 * self.pcrit)
        factorial_1 = self.molar_density_bulk * self.R * self.temperature / (1 - self.molar_density_bulk * self.b)
        exponential = self.molar_density_bulk * self.b /(1 - self.molar_density_bulk * self.b) - 2*a_bulk * self.molar_density_bulk /(self.R * self.temperature)
        fugacity_bulk = factorial_1 * np.exp(exponential)
        # unit check
        # print(self.R_mol, self.temperature, self.pressure)
        return fugacity_bulk

    def calc_a_from_z(self, show = False):
        a_from_z = list()

        for zz in self.z:
            # with sigma_ff as 38.1e-9 both conditions never fullfilled--> sigma_ff has to be diameter of fluid molecule, here measured with radgyr
            # Fitzgerald et al. for adjusted values
            if 0.5 <= zz/(self.sigma_ff)<=1.5:
                az = self.a_bulk*(5/16 + 6/16 * (zz/(self.sigma_ff)))
            elif 1.5 <= zz/(self.sigma_ff)< np.inf:
                bracket = 8*(zz/self.sigma_ff - 0.5)**3
                az = self.a_bulk*(1-1/bracket)
            a_from_z.append(az)
        a_from_z = np.array(a_from_z)
        if show:
            fig = go.Figure(data = go.Scatter(x = self.z / self.sigma_ff,y = a_from_z, mode = 'markers', name = '<i>a</i><sub>z</sub>', showlegend = True), layout = layout.eldens())
            fig.show()
        return a_from_z

    def eq2gettozero(self, rho, a, psi):
        fb = self.fugacity_bulk()
        eq = np.log(rho*self.R*self.temperature / (fb*(1-rho*self.b))) + rho*self.b/(1-rho*self.b)-2*a*rho/(self.R*self.temperature) + psi/(self.k_B*self.temperature)
        return eq
    
    def lee_10_4_potential(self, show = False, numb_planes = 4, **kwargs):
        '''
        Calculate solid-fluid interaction potential. **kwargs are added to use function in fitting routine.
        kwargs:
        -------
            * epsilon_fs
            * sigma_fs
            * rho_atoms
        For definition see init signature
        '''
        for k in kwargs.keys():
            if k == "epsilon_fs":
                self.epsilon_fs = kwargs['epsilon_fs']
            elif k == "interplanar_spacing":
                self.interplanar_spacing = kwargs['interplanar_spacing']
            elif k == "rho_atoms":
                self.rho_atoms = kwargs['rho_atoms']
            elif k == 'z_coords':
                self.z = kwargs['z_coords']
        psi = list()
        # prefactor = 4 * np.pi * self.rho_atoms * self.epsilon_fs * self.sigma_fs**2
        prefactor = 4 * np.pi * self.rho_atoms * self.epsilon_fs * self.sigma_fs**6
        for zz in self.z:
            # first_part_of_bracket = self.sigma_fs**10/(5*(zz+self.sigma_ff/2)**10)
            first_part_of_bracket = self.sigma_fs**6/(5*(zz+self.sigma_ff/2)**10)
            xval = 0
            for i in range(1,numb_planes+1):
                # xval += self.sigma_fs**4 / ((zz+self.sigma_ff/2 + (i-1)*self.interplanar_spacing)**4)
                xval += 1 / ((zz+self.sigma_ff/2 + (i-1)*self.interplanar_spacing)**4)
                # print(f'plane {i}: denominator:{(zz+self.sigma_ff/2 + (i-1)*self.interplanar_spacing)**4:.2e}')
            psi_val = prefactor * (first_part_of_bracket - 0.5 * xval)
            psi.append(psi_val)
        psi = np.array(psi)
        if show:
            trace_psi = go.Scattergl(x = self.z/(self.sigma_ff), y = psi, mode = 'lines')
            fig_psi = go.Figure(data = trace_psi, layout = layout.eldens())
            fig_psi = fig_psi.update_layout(
                xaxis = dict(title_text = '<i>z</i>/&#963;<sub>ff</sub>', minor_nticks = 5, nticks = 20),
                yaxis = dict(title_text = '<i>&#936;</i>(<i>z</i>)'),
                legend = dict(x = 0, y = 1, xanchor = 'left', yanchor = 'top'),
                showlegend = True)
            fig_psi.show()
        return psi

    def solver(self, show = False):
        rho_from_z, a_from_z, psi = list(), list(), list()
        prefactor = 4 * np.pi * self.rho_atoms * self.epsilon_fs * self.sigma_fs**2
        a_from_z = self.calc_a_from_z()
        psi = self.lee_10_4_potential()
        for i in range(len(self.z)):                # find roots
            if self.temperature >= self.Tcrit:
                rt = root(self.eq2gettozero, 1e4, args = (a_from_z[i], psi[i]), method = 'lm')
                if rt['success'] == False:
                    rt = root(self.eq2gettozero, 1e3, args = (a_from_z[i], psi[i]), method = 'lm')
                else:
                    rt = rt
                root_val = rt['x'][0]
            else:
                rt = root(self.eq2gettozero, np.linspace(1, 1e5, num = 10)[::-1], args = (a_from_z[i], psi[i]), method = 'lm')
                if rt['success'] == False:
                    rt = root(self.eq2gettozero, 1e3, args = (a_from_z[i], psi[i]), method = 'lm')
                else:
                    rt = rt
                root_val = rt['x'][0]
            rho_from_z.append(root_val)
        rho_from_z = np.array(rho_from_z)
        # print(rho_from_z)
        if show:
            trace_rho_from_z = go.Scattergl(x = self.z/self.sigma_ff, y = np.array(rho_from_z), mode = 'markers', name = '<i>&#961;</i>(<i>z</i>)')
            pdata = trace_rho_from_z
            if 'VFP_data' in self.kwargs.keys():
                profile_trace = go.Scattergl(x = self.z/self.sigma_ff, y = self.VFP_df.difference*1e30/(self.molecule_data.anz_el * self.avogadro), yaxis = 'y')
                pdata  = [profile_trace,pdata]
            fig = go.Figure(data = pdata, layout = layout.eldens())
            fig = fig.update_layout(
                xaxis = dict(title_text = '<i>z</i>/&#963;<sub>ff</sub>', minor_nticks = 5, nticks = 20),
                yaxis = dict(title_text = '<i>&#961;</i>(<i>z</i>)'),
                yaxis2 = dict(side = 'right', anchor = 'free',),
                legend = dict(x = 1, y = 0, xanchor = 'left', yanchor = 'top'),
                showlegend = True)
            fig.show() 
        # int_val = simpson(rho_from_z-self.molar_density_bulk, self.z)
        rho_minus_bulk = rho_from_z - self.molar_density_bulk
        int_val = trapezoid(rho_minus_bulk, self.z)
        return rho_from_z, int_val


    def use_solver_jit(self, show = False, **kwargs):
        '''
        Calculate SLD-profile, Gibbs surface excess and absoulte adsorbed amount.
        kwargs:
        ---------
            * numb_planes: int, number of planes of solid substrate layers to take into account when calculating interaction potential. Default is 4.
            * show: boolean, if True a plot of the SLD profile is shown. If VFP_data is given, this profile is also plotted. Plotting backend is plotly 
        '''
        
        if 'numb_planes' in kwargs.keys():
            numb_planes = kwargs['numb_planes']
        else:
            numb_planes = 4
        if 'relative_density_bulk_cutoff' in kwargs.keys():
            rel_cutoff = kwargs['relative_density_bulk_cutoff']
        else:
            rel_cutoff = 0.07
        if 'interaction_potential' in kwargs.keys():
            pot_sf = kwargs['interaction_potential']
        else:
            pot_sf = self.lee_10_4_potential(numb_planes = numb_planes)

        rho_from_z = solver_j(self.molar_density_bulk, self.a_bulk, self.calc_a_from_z(), pot_sf, self.b, self.z,
            self.rho_atoms, self.epsilon_fs, self.sigma_fs, self.sigma_ff, self.interplanar_spacing, self.Tcrit, self.temperature, self.k_B, self.avogadro, self.R)
        gamma_ex = simpson(rho_from_z-self.molar_density_bulk, self.z)
        rho_short_for_rhoA, z_short_for_rhoA = np.where((rho_from_z- self.molar_density_bulk)/rho_from_z > rel_cutoff, rho_from_z,np.nan), np.where((rho_from_z- self.molar_density_bulk)/rho_from_z > rel_cutoff, self.z,np.nan)

        absolute_adsorbed_amount = simpson(rho_short_for_rhoA[~np.isnan(rho_short_for_rhoA)], z_short_for_rhoA[~np.isnan(z_short_for_rhoA)])
        # print(np.isclose(rho_from_z-self.molar_density_bulk, rho_from_z, rtol = 1e-2))
        # absolute_adsorbed_amount = simpson(rho_from_z, self.z)
        if show:
            trace_rho_from_z = go.Scattergl(x = self.z/self.sigma_ff, y = np.array(rho_from_z), mode = 'markers', name = '<i>&#961;</i>(<i>z</i>)')
            trace_rho_short_from_z = go.Scattergl(x = z_short_for_rhoA/self.sigma_ff, y = np.array(rho_short_for_rhoA), mode = 'markers', name = '<i>&#961;</i>(<i>z</i>)<sub>short</sub>')
            pdata = [trace_rho_from_z, trace_rho_short_from_z]
            if 'VFP_data' in self.kwargs.keys():
                profile_trace = go.Scattergl(x = self.z/self.sigma_ff, y = self.VFP_df.difference*1e30/(self.molecule_data.anz_el * self.avogadro), xaxis = 'x',yaxis = 'y', name = 'VFP')
                # pdata  += profile_trace
                pdata = [trace_rho_from_z, trace_rho_short_from_z, profile_trace]
            fig = go.Figure(data = pdata, layout = layout.eldens())
            fig = fig.update_layout(
                xaxis = dict(title_text = '<i>z</i>/&#963;<sub>ff</sub>', minor_nticks = 5, nticks = 20),
                yaxis = dict(title_text = '<i>&#961;</i>(<i>z</i>)'),
                # yaxis2 = dict(side = 'right', overlaying=None),
                # xaxis2 = dict(side = 'top', overlaying = None),
                legend = dict(x = 1, y = 0, xanchor = 'left', yanchor = 'top'),
                showlegend = True)
            fig.show() 
        return rho_from_z, gamma_ex, absolute_adsorbed_amount

    def calc_interaction_potential_from_VFP(self, show = False):
        if not "VFP_data" in self.kwargs.keys():
            print('Specify VFP data to perform calculation')
            return
        rho =self.VFP_profile_molar_units
        a_from_z = np.array(self.calc_a_from_z())
        fugacity_b = self.fugacity_bulk()
        fugacity_ff = rho * self.R*self.temperature / (1-rho*self.b)*np.exp(rho *self.b/(1-rho*self.b)-2*a_from_z*rho/(self.R*self.temperature))
        psi = self.k_B*self.temperature*np.log(fugacity_b/fugacity_ff)
        if show:
            psi_trace = go.Scattergl(x = self.z/self.sigma_ff, y = psi, mode = 'lines', line = dict(color = 'blue', width = 3), name = '&#936;')
            fig = go.Figure(data = psi_trace, layout = layout.eldens())
            fig.update_layout(
                yaxis = dict(title_text ='&#936; (<i>z</i>)'),
                xaxis = dict(title_text = '<i>z</i>/&#963;<sub>ff</sub>'))
            fig.show()
        return np.array(psi)


    @staticmethod
    def psi_model(z, epsilon_fs, rho_atoms, sigma_fs, sigma_ff, interplanar_spacing):
        k_b = constants.physical_constants['Boltzmann constant'][0]
        prefactor = 4 * np.pi * rho_atoms * epsilon_fs * sigma_fs**6
        first_part_of_bracket = sigma_fs**6/(5*(z+sigma_ff/2)**10)
        xval = 0
        for i in range(1,5):
            xval += 1 / ((z+sigma_ff/2 + (i-1)*interplanar_spacing)**4)
        psi_val = prefactor * (first_part_of_bracket - 0.5 * xval)
        return psi_val /k_b
    
    def lmfit_model_for_psi(self):
        model_psi = Model(SLD_rangarajan.psi_model)
        
        data2fit = self.calc_interaction_potential_from_VFP() / self.k_B
        # [print(d,p) for d, p in zip(data2fit, SLD_rangarajan.psi_model(self.z, self.epsilon_fs, self.rho_atoms, self.sigma_fs, self.sigma_ff, self.interplanar_spacing))]
        epsilon_fs_init = self.epsilon_fs / self.k_B
        sigma_fs_init = self.sigma_fs*1e10
        rho_atoms_init = self.rho_atoms*1e-20
        

        min_epsilon_fs, max_epsilon_fs = 0.001 * epsilon_fs_init, 1e3 * epsilon_fs_init
        min_rho_atoms, max_rho_atoms = 0.001 * rho_atoms_init, 1e4 * rho_atoms_init
        min_sigma_fs, max_sigma_fs = 0.001*sigma_fs_init, 1e3 * sigma_fs_init

        model_psi.set_param_hint('epsilon_fs', value = epsilon_fs_init, min = min_epsilon_fs, max = max_epsilon_fs)
        model_psi.set_param_hint('rho_atoms', value = rho_atoms_init, min = min_rho_atoms, max = max_rho_atoms)
        model_psi.set_param_hint('sigma_fs', value = sigma_fs_init, min = min_sigma_fs, max = max_sigma_fs)
        model_psi.set_param_hint('sigma_ff', value = self.sigma_ff * 1e10, vary = True)
        model_psi.set_param_hint('interplanar_spacing', value = self.interplanar_spacing * 1e10, vary = True)
        pars = model_psi.make_params()
        
        result = model_psi.fit(data2fit,  params = pars, z = self.z)
        print(result.fit_report())
        