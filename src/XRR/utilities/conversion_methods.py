from scipy import constants
import numpy as np
from XRR.FluidProps import selectMolecule
r_e_angstrom = constants.physical_constants['classical electron radius'][0]*1e10

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
        q = 4 * constants.pi / (wavelength) * np.sin(th * constants.pi / 180)
    else:
        q = [4 * constants.pi / (wavelength) * np.sin(t * constants.pi / 180) for t in th]
    return q

def q2th(wavelength, q):
    # input is numpy array with q values
    # function returns th in degrees
    # th = np.arcsin(180/(4*constants.pi**2) * wavelength * q)
    th = 180/constants.pi * np.arcsin((wavelength * q) / (4 * constants.pi))
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
    return constants.h * constants.c / (energy * 1e3 * constants.e) *1e10

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
    return constants.h * constants.c / (wavelength) / constants.e*1e-3 * 1e10
    

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

def cm2inch(val):
    if isinstance(val, (list, np.ndarray)):
        return np.asarray([v / 2.54 for v in val])
    else:
        return val / 2.54

def inch2cm(val):
    if isinstance(val, (list, np.ndarray)):
        return np.asarray([v * 2.54 for v in val])
    else:
        return val * 2.54

def kelvin2celsius(t_kelvin):
    '''
    Convert temperature in kelvin to temperature in celsius.

    Parameters:
    -----------
        t_kelvin: list, numpy array, int or float

    Returns:
    --------
        t_celsius: list, numpy array, int or float, depending on input type
    '''
    k_factor = 273.15
    if isinstance(t_kelvin, list):
        t_celsius = [tk - k_factor for tk in t_kelvin]
    elif isinstance(t_kelvin, np.ndarray):
        t_celsius = t_kelvin - k_factor
    elif isinstance(t_kelvin, int) or isinstance(t_kelvin, float):
        t_celsius = t_kelvin - k_factor
    return t_celsius


def celsius2kelvin(t_celsius):
    '''
    Convert temperature in celsius to temperature in kelvin.

    Parameters:
    -----------
        t_celsius: list, numpy array, int or float

    Returns:
    --------
        t_kelvin: list, numpy array, int or float, depending on input type
    '''
    k_factor = 273.15
    if isinstance(t_celsius, list):
        t_kelvin = [tk + k_factor for tk in t_celsius]
    elif isinstance(t_celsius, np.ndarray):
        t_kelvin = t_celsius + k_factor
    elif isinstance(t_celsius, int) or isinstance(t_celsius, float):
        t_kelvin = t_celsius + k_factor
    return t_kelvin