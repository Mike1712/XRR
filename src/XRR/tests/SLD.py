import sys, os
from time import perf_counter
from XRR.interfaces import adsorption as ads
from scipy import constants
from XRR.helpers import *
import XRR.FluidProps as FP
from pprint import pprint
import cupy as cp
# constants
k_B = constants.physical_constants['Boltzmann constant'][0]

avogadro = constants.physical_constants['Avogadro constant'][0]

C3F8_props = FP.selectMolecule('C3F8')
temp_ind = 3

nistdata = FP.readFluidProps('/home/mike/Dokumente/Uni/Promotion/Arbeit/files/plots/Theroy_SCFA/C3F8/sub_critical/C3F8_65C.txt')[0]
pressure, density = nistdata['pressure(bar)'], nistdata['density(mol/m3)']
pressure_ind = 1
pressure_single, density_single = nistdata['pressure(bar)'].iloc[pressure_ind], nistdata['density(mol/m3)'].iloc[pressure_ind]
print(pressure_single, density_single)
SLD_C3F8 = ads.SLD_rangarajan('C3F8', 65, pressure_single, epsilon_fs = 900 * k_B, rho_atoms=0.382*1e20, interplanar_spacing = 3.35e-10, molar_density_bulk = density_single,
                              # z_coords = np.linspace(C3F8_props.radius_of_gyration*1e-10, 20*C3F8_props.radius_of_gyration*1e-10,num = 100)
                             )
rho, gam = SLD_C3F8.solver_parallel(show=True)


		

