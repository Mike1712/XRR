from .. import XRR
import numpy as np
import plotly.graph_objects as go
from .. plotly_layouts import plotly_layouts as layouts
layout = layouts(transpPaper=False, transpPlot=False, transpLegend=True,unit_mode='slash', bigTicks=False, bigTickFont=True)

a_sub, a_SiO2, a_PS = 8e-8, 7e-8, 4e-8
sub = XRR.layer(7.56e-6, a_sub, np.inf, roughness=0)
SiO2 = XRR.layer(6.8e-6, a_SiO2, 15, roughness = 7)
PS_lay = XRR.layer(3.5e-6, a_PS, 50, roughness=13)
vac = XRR.layer(0,0,0,roughness=8)

ls = XRR.stack_layers([sub, SiO2, PS_lay, vac])

SiO2_sharp = XRR.layer(6.8e-6, 7e-8, 15, )
PS_lay_sharp = XRR.layer(3.5e-6, 4e-8, 50, )
vac_sharp = XRR.layer(0,0,0,)

ls_sharp = XRR.stack_layers([sub, SiO2_sharp, PS_lay_sharp, vac_sharp])

ai = np.linspace(0, 4, 500)
wl_D8 = XRR.kev2angst(8.048)
q = XRR.th2q(wl_D8, ai)

laystack = XRR.refl_multiple_layers(q, ls, wl_D8, ind_of_refrac_profile='tanh', alpha_i_unit='Ae-1')
laystack_sharp = XRR.refl_multiple_layers(q, ls_sharp, wl_D8, ind_of_refrac_profile='tanh', alpha_i_unit='Ae-1')

refl_indep_layers = laystack.calculate_specular_reflectivity()
refl_sharp = laystack_sharp.calculate_specular_reflectivity()

z_sharp, disp_sharp = laystack_sharp.IoR_profiles_interface(shiftToZero=False)
z_indep, disp_indep = laystack.IoR_profiles_interface()
z, disp_profile, ls_sliced, W, zeta = laystack.EffectiveDensitySlicing(step = .1, return_W_functions=True, shiftSubstrate2zero=False)
refl_eff_densities = XRR.refl_multiple_layers(q, ls_sliced, wl_D8, ind_of_refrac_profile='tanh', alpha_i_unit='Ae-1').calculate_specular_reflectivity(plot = False)

r_indep = go.Scattergl(x = q, y = refl_indep_layers, mode = 'lines', line = dict(width = 4, color = 'red'), name = 'independent layers')
r_effdichte = go.Scattergl(x = q, y = refl_eff_densities, mode = 'lines', line = dict(width = 4, color = 'black'), name = 'Eff. density')
r_sharp = go.Scattergl(x = q, y = refl_sharp, mode = 'lines', line = dict(width = 4, color = 'blue'), name = 'Sharp interfaces')
fig = go.Figure(data = [r_indep, r_effdichte, r_sharp], layout = layout.refl())
fig.show()

# # plot dispersion profiles
d_sharp = go.Scattergl(x = z_sharp, y = disp_sharp, mode = 'lines', line = dict(color='red', width = 4), name = 'Ideal glatte Grenzflächen')
d_effdichte_model = go.Scattergl(x = z, y = disp_profile, mode = 'lines', line = dict(color='green', width = 4), name = 'Effektives Dichtemodel')
traces = []
refrac_indices = np.asarray([laystack.layer_stack[j].index_of_refraction() for j in range(0, len(laystack.layer_stack))])
dispersions = np.asarray([1-np.real(k) for k in refrac_indices])
for wfunc in W:
    traces.append(go.Scattergl(x = z, y = W[wfunc]*XRR.disp2edens(wl_D8, dispersions[wfunc]), mode = 'lines', line = dict(width = 4, color = 'black', dash = 'dash'),showlegend = False))
traces[0].name = '$\delta_{\mathrm{j}}W_{\mathrm{j}}$'
traces[0].showlegend=True

# zeta_trace = go.Scattergl(x = list(zeta.values()), y = np.repeat(.5, len(zeta)), mode = 'markers', marker = dict(size = 12, symbol = 'cross'))
fig = go.Figure(data = [d_sharp, d_effdichte_model] + traces, layout=layout.eldens(x0 = -100, dtick_x=10))
fig.update_layout(xaxis = dict(range = [-100,50]),
                 legend = dict(tracegroupgap = 100))
fig.show()

