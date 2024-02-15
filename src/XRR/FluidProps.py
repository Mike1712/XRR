import os
from pprint import pprint
# import XRR.utilities.conversion_methods
# import XRR.utilities.conversion_methods as cmet
# from XRR.utilities.conversion_methods import rhom2disp
import numpy as np
from pandas import read_csv
from scipy import constants
# from pymol import cmd
import pandas as pd
import math
r_e_angstrom = constants.physical_constants['classical electron radius'][0] * 1e10


def rhom2disp(molecule, density, wavelength = None, verbose = True):
    if wavelength == None:
        wavelength = 0.4592
        if verbose:
            print('No wavelength was specified. ' + '\u03BB = ' + "{:.4f}".format(wavelength) + ' \u00C5' +' is used as default.')
    fluid_props = selectMolecule(molecule = molecule)
    anz_el, molar_mass = fluid_props['anz_el'], fluid_props['molar_mass']
    # anz_el, molar_mass = selectMolecule(molecule = molecule)
    rho_e = density / molar_mass * constants.Avogadro * anz_el * 1e-30
    # dispersion = wavelength**2 / (2 * constants.pi) * r_e_angstrom * rho_e
    dispersion = edens2disp(wavelength, rho_e)
    return dispersion
# def selectMolecule(molecule = None):
#     molecules = {'C2F6':False, 
#                 'C3F8': False,
#                 'C4F10': False,
#                 'CO2': False,
#                 'Ar': False,
#                 'C4H10': False,
#                 'H2O': False
#             }

#     for key, val in zip(molecules.keys(), molecules.values()):
#         if val:
#             print(key + ' is your selected molecule')
#             molecule = key

#     # wavelength, constants and properties of atom

#     # C2F6
#     anz_el_C2F6 = 66
#     molar_mass_C2F6 = 138.01*1e-3 # kg/mol

#     # C3F8
#     anz_el_C3F8 = 90
#     molar_mass_C3F8 = 188.02*1e-3

#     # C4F10
#     anz_el_C4F10 = 114
#     molar_mass_C4F10 = 238.03*1e-3

#     # CO2
#     anz_el_CO2 = 22
#     molar_mass_CO2 = 44.01*1e-3

#     # Ar
#     anz_el_Ar = 18
#     molar_mass_Ar = 39.948*1e-3

#     # C4H10
#     anz_el_C4H10 = 34
#     molar_mass_C4H10 = 58.12*1e-3

#     # H20
#     anz_el_H2O = 10
#     molar_mass_H2O = 18.01528*1e-3
    
#     anz_el = eval('anz_el_' + str(molecule))
#     molar_mass = eval('molar_mass_' + str(molecule))

#     return anz_el, molar_mass
def fluidprops_df():
    '''
    Data from NIST
    '''
    molecules = {'C2F6':False, 
                'C3F8': False,
                'C4F10': False,
                'CO2': False,
                'Ar': False,
                'C4H10': False,
                'H2O': False,
                'C2H4':False
            }
    # C2F6
    anz_el_C2F6 = 66
    molar_mass_C2F6 = 138.01*1e-3 # kg/mol
    p_crit_C2F6 = 30.48 # bar
    T_crit_C2F6 = 19.88 # °C
    rho_crit_C2F6 = 613.3 # kg/m^3
    dipole_moment_C2F6 = 0 # Debye
    acentric_factor_C2F6 = 0.2566
    radius_of_gyration_C2F6 = 3.419 #Angstrom

    # C3F8
    anz_el_C3F8 = 90
    molar_mass_C3F8 = 188.02*1e-3
    p_crit_C3F8 = 26.4 # bar
    T_crit_C3F8 = 71.87 # °C
    rho_crit_C3F8 = 628 # kg/m^3
    dipole_moment_C3F8 = 0 # Debye
    acentric_factor_C3F8 = 0.3172    
    radius_of_gyration_C3F8 = 3.736 #Angstrom
    
    # C4F10
    anz_el_C4F10 = 114
    molar_mass_C4F10 = 238.03*1e-3
    p_crit_C4F10 = 23.224 # bar
    T_crit_C4F10 = 113.176 # °C
    rho_crit_C4F10 = 627.7 # kg/m^3
    dipole_moment_C4F10 = 0 # Debye
    acentric_factor_C4F10 = 0.372
    radius_of_gyration_C4F10 = 4.574 #Angstrom
    
    # CO2
    anz_el_CO2 = 22
    molar_mass_CO2 = 44.01*1e-3
    p_crit_CO2 = 73.773 # bar
    T_crit_CO2 = 30.9782 # °C
    rho_crit_CO2 = 467.6 # kg/m^3
    dipole_moment_CO2 = 0 # Debye
    acentric_factor_CO2 = 0.22394
    radius_of_gyration_CO2 = 1.040 #Angstrom

    # Ar
    anz_el_Ar = 18
    molar_mass_Ar = 39.948*1e-3
    p_crit_Ar = 48.63 # bar
    T_crit_Ar = -122.463 # °C
    rho_crit_Ar = 535.599 # kg/m^3
    dipole_moment_Ar = 0 # Debye
    acentric_factor_Ar = -0.00219
    radius_of_gyration_Ar = np.nan #Angstrom
    # C4H10 (Iso)
    anz_el_C4H10 = 34
    molar_mass_C4H10 = 58.12*1e-3
    p_crit_C4H10 = 36.29 # bar
    T_crit_C4H10 = 134.66 # °C
    rho_crit_C4H10 = 225.5 # kg/m^3
    dipole_moment_C4H10 = 0 # Debye
    acentric_factor_C4H10 = 0.184
    radius_of_gyration_C4H10 = 2.948 # Angstrom

    # H20
    anz_el_H2O = 10
    molar_mass_H2O = 18.01528*1e-3
    p_crit_H2O = 220.64 # bar
    T_crit_H2O = 373.946 # °C
    rho_crit_H2O = 322 # kg/m^3
    acentric_factor_H2O = 0.3443
    dipole_moment_H2O = 1.885
    radius_of_gyration_H2O = np.nan # Angstrom

    # C2H4
    anz_el_C2H4 = 16
    molar_mass_C2H4 = 28.05*1e-3 #kg/mol
    p_crit_C2H4 = 50.418 # bar
    T_crit_C2H4 = 9.2 # °C
    rho_crit_C2H4 = 214.2 # kg/m^3
    acentric_factor_C2H4 = 0.0866
    dipole_moment_C2H4 = 0
    radius_of_gyration_C2H4 = 1.548 # Angstrom

    params = ['anz_el_', 'molar_mass_', 'p_crit_', 'T_crit_', 'rho_crit_', 'acentric_factor_', 'dipole_moment_', 'radius_of_gyration_']
    fluid_properties = dict()
    
    for m in molecules.keys():    
        props = dict()
        for par in params:
            par = par.rstrip('_')
            val = eval(f'{par}_{m}')
            props[par] = val
        fluid_properties[m] = props

    df_fp = pd.DataFrame.from_dict(fluid_properties, orient = 'index')
    df_fp['rho_crit_molar'] = df_fp.rho_crit / df_fp.molar_mass 
    return df_fp

def selectMolecule(molecule = None):
    molecules = {'C2F6':False, 
                'C3F8': False,
                'C4F10': False,
                'CO2': False,
                'Ar': False,
                'C4H10': False,
                'H2O': False,
                'C2H4': False
            }

    for key, val in zip(molecules.keys(), molecules.values()):
        if val:
            print(key + ' is your selected molecule')
            molecule = key
    
    df = fluidprops_df()
    return df.loc[molecule]
    # return anz_el, molar_mass


def readFluidProps(file):
    '''
    Read fluid
    '''
    with open(file,'r') as f:
        firstline = f.readline().replace(' ','').lower().split()
        new_firstline= firstline + ['dispersion']
    df = read_csv(file, sep='\t', header=0, names=firstline)
    return df, new_firstline

def showKeys(file):
	df,_ = readFluidProps(file)
	[print(k) for k in df.keys()]

def _AddDispValues(file, molecule, wavelength, add_absorption_values = False):
    data, newheader = readFluidProps(file)
    fluid_props = selectMolecule(molecule = molecule)
    anz_el, molar_mass = fluid_props[f'anz_el'], fluid_props[f'molar_mass']
    for elem in newheader:
    	if 'density'.casefold() in elem.casefold():
    		denskey = elem

    mass_dens = data[denskey]
    dispersion = [rhom2disp(molecule = molecule, density = d, wavelength = wavelength, verbose = True) for d in mass_dens]

    data['dispersion'] = dispersion
    return data

def saveDispValues(file, molecule, wavelength, newfilename = None, add_absorption_values = False):
    data = _AddDispValues(file = file, molecule = molecule, wavelength  = wavelength)
    if not newfilename:
        if not '.' in file:
            newfilename = file + '_with_disp'
        elif '.' in file: 
            fileending = file.split('.')[-1]
            newfilename = file.split('.')[0:-1][0] + '_with_disp.' + fileending
    if not os.path.isfile(newfilename):
        data.to_string(newfilename, index=False)
    else: print(f'{newfilename} already exists')
    

   
# def rgyrate(selection='(all)', quiet=1):
#     '''
# DESCRIPTION

#     Radius of gyration

# USAGE

#     rgyrate [ selection ]
#     '''
#     try:
#         from itertools import izip
#     except ImportError:
#         izip = zip
#     quiet = int(quiet)
#     model = cmd.get_model(selection).atom
#     x = [i.coord for i in model]
#     mass = [i.get_mass() for i in model]
#     xm = [(m*i,m*j,m*k) for (i,j,k),m in izip(x,mass)]
#     tmass = sum(mass)
#     rr = sum(mi*i+mj*j+mk*k for (i,j,k),(mi,mj,mk) in izip(x,xm))
#     mm = sum((sum(i)/tmass)**2 for i in izip(*xm))
#     rg = math.sqrt(rr/tmass - mm)
#     if not quiet:
#         print("Radius of gyration: %.2f" % (rg))
#     return rg

# cmd.extend("rgyrate", rgyrate)

# def peng_robinson(molecule, Temp, rho, T_crit, p_crit, acentric_factor, p_unit = 'bar', T_unit = 'celsius'):
#     _, molar_mass = selectMolecule(molecule)
#     R = constants.physical_constants['molar gas constant'][0]
#     a = (0.457235*R**2*T_crit**2) / (p_crit)
#     b = (0.077796*R*T_crit) / (p_crit)
#     T_red = np.array(Temp) / T_crit
#     V_molar = molar_mass / rho
#     if acentric_factor <= 0.49:
#         alpha = (1+(0.379642 + 1.54226*acentric_factor - 0.26992*acentric_factor**2)*(1-np.sqrt(T_red)))**2
#     p = ((R*Temp) / (V_molar-b)) - ((a*alpha) / (V_molar**2 + 2*b*V_molar-b**2))
#     if p_unit == 'bar':
#         p *=1e-5
#     return p
