import numpy as np
from scipy import constants
from scipy.optimize import curve_fit
from math import erf, gamma as math_gamma
from scipy.special import gamma as sc_gamma
    
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from lmfit.models import *
from lmfit import Model
from pprint import pprint

from XRR.plotly_layouts import plotly_layouts as layouts, create_colormap
from XRR.helpers import *
import plotly.graph_objects as go

from babel.numbers import format_decimal
def find_nearest(array, value):
    '''
    Find index of closest value in an array.
    
    Parameters
    ----------
    array: list or numpy array. Array, where index should be searched.
    value: value, to which the closest value should be found.
    
    Returns
    -------
    idx: int, Index of closest value
    '''
    array = np.asarray(array)
    idx = np.argmin(np.abs(array - value))
    return idx

def extractValuesInRange(lst:list, min_val, max_val):
    min_ind = find_nearest(lst, min_val)
    max_ind = find_nearest(lst, max_val)
    rangelist = lst[min_ind:max_ind]
    return rangelist

def FWHM(Y, X = None):
    half_max = max(Y) / 2.
    # find when function crosses line half_max (when sign of diff flips)
    # take the 'derivative' of signum(half_max - Y[])
    d = np.sign(half_max - np.array(Y[0:-1])) - \
        np.sign(half_max - np.array(Y[1:]))
    left_idx = np.where(d > 0)[0]
    right_idx = np.where(d < 0)[-1]  
    return left_idx, right_idx

def find_first_float(lst, val):
    if isinstance(lst, list):
        arr = np.array(lst)
    else:
        arr = lst
    ind = np.where(arr == float(val))
    if ind[0].size == 0:
        ind = arr.size
    else: 
        ind = ind[0][0]
    return ind

def shift_with_power(anz_shifts,compare=False, spacing = 10, reverse = False):
    '''
    Create an array with ascending/descending numbers to the power of spacing.

    Parameters:
    -----------
    args:
        * anz_shifts: int, equals the length of the array
    kwargs:
        * compare: boolean, repeat every entry. Default is True
        * spacing: base of what gets powered.
        * reverse: boolean, sort array starting with descending numbers. Default is False

    Returns:
    --------
    numpy.array
    '''
    i = 1
    j = 0
    shift = np.zeros(anz_shifts)
    while i in range(0, spacing**anz_shifts) and j in range(0, anz_shifts + 1):
        shift[j] = i
        i *= spacing
        j += 1
    if compare:
        shift = [shift[i // 2] for i in range(0, len(shift *2))]
    if reverse:
        shift = np.flip(shift)
    return np.asarray(shift)

def footprint(h_beam,l_sample):
    ''' calculate the footprint f [f]=° in an XRR experiment
    args:
        * h_beam: height of the beam [microns]
        * l_sample: length of the sample in beam direction [microns]
    '''
    f = np.arcsin(h_beam / l_sample) * 180 / constants.pi
    return f

def antoine_eq(A,B,C,T):
    '''
    Determine saturation pressure by Antoine equation.
    Parameters
    ----------
        A, B, C: Antoine parameters.
        T: Temperature in Kelvin
    Returns
    -------
     *p_sat: list, saturation pressure
     *pref:_err: list, dont know anymore
    '''
    p_sat = []  
    mmhgtobar = (101.325 / 760 * 1e-2)
    [p_sat.append(10**(A-(B/(t+C)))) for t in T]
    pref_err = []
    [pref_err.append((B*np.log(10)*10**(A-(B/(C+t))))/((C+t)**2)) for t in T]
    p_sat = [mmhgtobar * p for p in p_sat]
    pref_err = [mmhgtobar * e for e in pref_err] 
    return p_sat, pref_err

def turning_points(array):
    ''' turning_points(array) -> min_indices, max_indices
    Finds the turning points within an 1D array and returns the indices of the minimum and 
    maximum turning points in two separate lists.
    '''
    idx_max, idx_min = [], []
    if (len(array) < 3): 
        return idx_min, idx_max

    NEUTRAL, RISING, FALLING = range(3)
    def get_state(a, b):
        if a < b: return RISING
        if a > b: return FALLING
        return NEUTRAL

    ps = get_state(array[0], array[1])
    begin = 1
    for i in range(2, len(array)):
        s = get_state(array[i - 1], array[i])
        if s != NEUTRAL:
            if ps != NEUTRAL and ps != s:
                if s == FALLING: 
                    idx_max.append((begin + i - 1) // 2)
                else:
                    idx_min.append((begin + i - 1) // 2)
            begin = i
            ps = s
    # print(idx_min, idx_max, array[idx_min], array[idx_max])
    return idx_min, idx_max

def find_inflection_points(arr, filter_mode = 'mirror', sigma = 1, truncate = 4, smoothen = True):
    if smoothen:
        smooth = gaussian_filter1d(arr, sigma = sigma, mode = filter_mode, truncate = truncate)
    else:
        smooth = arr
    # smooth = np.where(arr>1e-2, arr, 0)
    smooth_d2 = np.gradient(np.gradient(smooth))
    max_ind = np.argmax(arr)
    # if max_ind == 0: max_ind +=10
    # print(f'max ind: {max_ind}, value: {arr[max_ind]}')
    infls = np.where(np.diff(np.sign(smooth_d2)))
    infls_arr = infls[0]
    # print(infls_arr)
    if len(infls_arr)>2:
        l_ind = infls_arr[infls_arr<=max_ind].max()
        r_ind = infls_arr[infls_arr>=max_ind].min()
        inds = (l_ind, r_ind)
    elif len(infls_arr) == 2:
        l_ind = infls[0][0]
        r_ind = infls[0][1]
        inds = (l_ind, r_ind)
    elif 0 < len(infls_arr)<2:
        inds = infls_arr[0]
    elif len(infls_arr) == 0:
        ind = np.nan
    return inds


def subbotin_distribution(x, beta, sigma, center, amplitude):
    y = np.asarray([amplitude * (beta) / (2*sigma*sc_gamma(1/beta))*np.exp(-((abs(xx-center))/(2*sigma))**beta) for xx in x])
    return y

def model_subbotin(x, y, beta, sigma, center, amplitude, min_correl = 0.1):
    model = Model(subbotin_distribution)
    model.set_param_hint('beta', value = beta, min = 0.01, max = 100, vary = True)
    model.set_param_hint('sigma', value = sigma, min = 0, max = 50, vary = True)
    model.set_param_hint('center', value = center, min = -100, max = 200, vary = False)
    model.set_param_hint('amplitude', value = amplitude, min = 0.1, max = 1e6, vary = True)
    pars = model.make_params()
    result = model.fit(y, pars, x = x)
    return result
    
    
def model_expGauss(x, y, center, amplitude, gamma, sigma, min_correl = 0.3):
    model = ExponentialGaussianModel(prefix='eGauss_')
    model.set_param_hint('amplitude', value = amplitude, min = 0.1, max = 1e6, vary = True)
    model.set_param_hint('gamma', value = gamma, min = 0.1, max = 10, vary = True)
    model.set_param_hint('sigma', value = sigma, min = 0.1, max = 10, vary = True)
    model.set_param_hint('center', value = center, min = -200, max = 200, vary = True)
    pars = model.make_params()

    result = model.fit(y, pars, x=x)
    return result

def negative_step_model(x, y, amplitude, center, sigma, mode = 'atan'):
    mstep = StepModel(indipendent_vars = ['x'], form = mode, prefix = 'step_')
    mconst = LinearModel(prefix = 'lin_')
    model =  mstep + mconst
    model.set_param_hint('step_amplitude', value = amplitude, min = -5, max = 5)
    model.set_param_hint('step_center', value = center, min = -100, max = 10)
    model.set_param_hint('step_sigma', value = sigma, min = -20, max = 20)
    model.set_param_hint('lin_intercept', value = 0, min = 0, max = 10)
    model.set_param_hint('lin_slope', value = 0, min = -5, max = 5)

    pars = model.make_params()
    result = model.fit(y, pars, x = x)
    return result



def rectangle_model(x, y , amplitude, center1, center2, sigma1, sigma2, mode = 'atan'):
    mrec = RectangleModel(indipendent_vars = ['x'], form = mode, prefix = 'step_')
    mconst = LinearModel(prefix = 'lin_')
    model =  mrec + mconst
    # model.set_param_hint('step_amplitude', value = amplitude, min = -5, max = 5)
    # model.set_param_hint('step_center1', value = center1, min = -100, max = 10)
    # model.set_param_hint('step_center2', value = center2, min = -100, max = 10)
    # model.set_param_hint('step_sigma2', value = sigma1, min = -20, max = 20)
    # model.set_param_hint('step_sigma2', value = sigma2, min = -20, max = 20)
    # model.set_param_hint('lin_intercept', value = y.min(), min = 0, max = 10)
    # model.set_param_hint('lin_slope', value = 0, min = -5, max = 5)
                
    # function ok
    model.set_param_hint('step_amplitude', value = amplitude, min = -5, max = 5)
    model.set_param_hint('step_center1', value = center1, min = -100, max = 10)
    model.set_param_hint('step_center2', value = center2, min = -100, max = 10)
    model.set_param_hint('step_sigma2', value = sigma1, min = -20, max = 20)
    model.set_param_hint('step_sigma2', value = sigma2, min = -20, max = 20)
    model.set_param_hint('lin_intercept', value = y.min(), min = 0, max = 10)
    model.set_param_hint('lin_slope', value = 0, min = -5, max = 5)

    pars = model.make_params()
    result = model.fit(y, pars, x = x)
    # print(result.fit_report())
    return result
def skewedGaussModel(x, y, amplitude, center, sigma, gamma):
    model = SkewedGaussianModel(indipendent_vars = ['x'])
    model.set_param_hint('amplitude', value = amplitude, )
    model.set_param_hint('center', value = center, min = -100, max = 100)
    model.set_param_hint('sigma', value = sigma, min = -100, max = 100)
    model.set_param_hint('gamma', value = gamma)

    pars = model.make_params()
    result = model.fit(y, pars, x = x)
    return result



def subbotin_skewed_gauss_combi(x, y, beta, sigma, center, amplitude, gamma):
    m_log = SkewedGaussianModel(prefix='skewedGauss_')
    m_subot = Model(subbotin_distribution, prefix = 'subott_')
    model = m_log * m_subot
    model.set_param_hint('subbot_beta', value = beta, min = 1, max = 100, vary = True)
    model.set_param_hint('subbot_sigma', value = sigma, min = 0, max = 50, vary = True)
    model.set_param_hint('subbot_center', value = center, min = -100, max = 200, vary = True)
    model.set_param_hint('subbot_amplitude', value = amplitude, min = 0.1, max = 1e6, vary = True)

    model.set_param_hint('skewedGauss_gamma', value = gamma, min = -10, max = 10, vary = True)
    model.set_param_hint('skewedGauss_sigma', value = sigma, min = 0, max = 50, vary = True)
    model.set_param_hint('skewedGauss_center', value = center, min = -100, max = 200, vary = True)
    model.set_param_hint('skewedGauss_amplitude', value = amplitude, min = 0.1, max = 1e6, vary = True)
    pars = model.make_params()
    print(pars)
    result = model.fit(y, pars, x = x)
    return result

def splitLorenzianModel(x, y, amplitude, center, sigma, sigma_r):
    model = SplitLorentzianModel()
    model.set_param_hint('sigma_r', value = sigma_r, min = 0.01, max = 100, vary = True)
    model.set_param_hint('sigma', value = sigma, min = 0, max = 50, vary = True)
    model.set_param_hint('center', value = center, min = -100, max = 200, vary = False)
    model.set_param_hint('amplitude', value = amplitude, min = 0.1, max = 1e6, vary = True)
    pars = model.make_params()
    result = model.fit(y, pars, x = x)
    return result

def subbotin_spltL_combi(x, y, beta, sigma, center, amplitude, sigma_r):
    mLor = SplitLorentzianModel(prefix='spltL_')
    m_subot = Model(subbotin_distribution, prefix = 'subott_')
    model = mLor + m_subot
    model.set_param_hint('subbot_beta', value = beta, min = 1, max = 100, vary = True)
    model.set_param_hint('subbot_sigma', value = sigma, min = 0, max = 50, vary = True)
    model.set_param_hint('subbot_center', value = center, min = -100, max = 200, vary = True)
    model.set_param_hint('subbot_amplitude', value = amplitude, min = 0.1, max = 1e6, vary = True)

    model.set_param_hint('spltL_sigma_r', value = sigma_r, min = 0.01, max = 100, vary = True)
    model.set_param_hint('spltL_sigma', value = sigma, min = 0, max = 50, vary = True)
    model.set_param_hint('spltL_center', value = center, min = -100, max = 200, vary = False)
    model.set_param_hint('spltL_amplitude', value = amplitude, min = 0.1, max = 1e6, vary = True)
    pars = model.make_params()
    print(pars)
    result = model.fit(y, pars, x = x)
    return result

# def neg_step(x, amplitude, sigma):
#     y_step = np.asarray([amplitude / 2 * (1 - np.tanh(xx * constants.pi / (2*np.sqrt(3) * sigma))) for xx in x])
#     return y_step

def neg_step_tanh(x, amplitude, mu, sigma):
    # y_step = np.asarray([amplitude / 2 * (1 - np.arctan((xx-mu) / sigma) / np.pi) for xx in x])
    y_step = np.asarray([amplitude * (1/2 - np.arctan((xx-mu) / sigma) / np.pi) for xx in x])
    return y_step

def neg_step_erf(x, amplitude, mu, sigma):
    y_step = np.asarray([amplitude /2 * (1 - erf((xx-mu) / sigma)) for xx in x])    
    return y_step
    
def model_neg_step(x, y, amplitude, mu, sigma, form = 'tanh'):
    if form == 'tanh':
        model = Model(neg_step_tanh)
    if form == 'error_fct':
        model = Model(neg_step_erf)
    model.set_param_hint('amplitude', value = amplitude)
    model.set_param_hint('sigma', value = sigma)
    model.set_param_hint('mu', value = mu)
    pars = model.make_params()
    result = model.fit(y, pars, x = x)
    return result

def model_spline(xknots, x, y):
    model = SplineModel()
    pars = model.make_params()
    result = model.fit(y, pars, x = x)
    return result

def fit_linear(x, y, m, b):
    model = LinearModel()
    model.set_param_hint('intercept', value = b, vary = True)
    pars = model.make_params()
    result = model.fit(y, pars, x = x)
    return result

def rel_deviation_two_vals(a, b, errors = False):
    if not errors:
        return (a-b)/a
    else:
        dev = dev = (a[0]-b[0])/a[0]
        err_a, err_b = a[1], b[1]
        err_dev = np.sqrt(((b[0]*err_a)/(a[0]**2))**2 + ((err_b)/(a[0]))**2)
        return (dev, err_dev)

def intersect_two_lines(first_mins, xvals, first_part_start_ind, first_part_end_ind, second_part_end_ind, show = False, print_fit_reports = False,
    xvals_overlap_up = 10, xvals_overlap_down = 10, m1_vary = True, numb_references = 3,
    **kwargs):
    """
        Description:
        This function calculates the intersection point of two lines based on given parameters and optional arguments.

        Args:
            first_mins (list): List y-values.
            xvals (list): List of x-values.
            first_part_start_ind (int): Index of the starting point of the first part of the line.
            first_part_end_ind (int): Index of the ending point of the first part of the line.
            second_part_end_ind (int): Index of the ending point of the second line.
            show (bool, optional): Indicates whether to display the linear fits and data points. Default is False.
            print_fit_reports (bool, optional): Indicates whether to print the fit reports. Default is False.
            xvals_overlap_up (int, optional): Extend x-values above the intersection point to expand the overlap. Default is 10.
            xvals_overlap_down (int, optional): Extend x-values below the intersection point to expand the overlap. Default is 10.
            m1_vary (bool, optional): Indicates whether to vary the slope of the first line. Default is True.
            numb_references (int, optional): Number of reference lines for the intersection point. Default is 3.

        Returns:
            dict: Dictionary containing the figure, fits, line properites, x-coords of intersection points and error, etc



        Kwargs:
            plot_from_first_minpos (bool): Exclude data points in plot from index lower than first_part_start_ind
            colmap (list): list of colors, that can be used to plot the datapoints
    """

    for k in kwargs:
        print(k)
        if not 'weights' in k:
            weights = np.repeat(1, len(first_mins))
        else:
            weights = kwargs[k]
    first_part_fm = first_mins[first_part_start_ind:first_part_end_ind]
    second_part_fm = first_mins[first_part_end_ind:second_part_end_ind+1]
    print(len(weights), len(first_mins))
    if 'verbose' in kwargs.keys() and kwargs['verbose']:
        print('first range:')
        printElementsFromMultipleLists([xvals[first_part_start_ind:first_part_end_ind], first_part_fm])
        print(f'second range:')
        printElementsFromMultipleLists([xvals[first_part_end_ind:second_part_end_ind+1], second_part_fm])
    
    model1 = LinearModel(prefix = 'm1_')
    model1.set_param_hint('m1_slope', value=0, vary = m1_vary)
    model1.set_param_hint('m1_intercept', value=.11,  max = 1, vary = True)
    params1 = model1.make_params()
    
    fit_m1 = model1.fit(first_part_fm, x = xvals[first_part_start_ind:first_part_end_ind], weights = weights[first_part_start_ind:first_part_end_ind])


    slope2_guess = (second_part_fm[1] - second_part_fm[0]) / (xvals[first_part_end_ind+1] - xvals[first_part_end_ind])
    model2 = LinearModel(prefix = 'm2_')
    model2.set_param_hint('m2_slope', value=slope2_guess, min = 1e-8, max = 1e6, vary = True)
    model2.set_param_hint('m2_intercept', value=.1,  max = 1, vary = True)
    params2 = model2.make_params()

    fit_m2 = model2.fit(second_part_fm, x = xvals[first_part_end_ind:second_part_end_ind+1], weights = weights[first_part_end_ind:second_part_end_ind+1])
    
    m1, b1, m1_err, b1_err = fit_m1.params['m1_slope'].value, fit_m1.params['m1_intercept'].value, fit_m1.params['m1_slope'].stderr, fit_m1.params['m1_intercept'].stderr
    m2, b2, m2_err, b2_err = fit_m2.params['m2_slope'].value, fit_m2.params['m2_intercept'].value, fit_m2.params['m2_slope'].stderr, fit_m2.params['m2_intercept'].stderr

    x_intersect = (b2-b1) / (m1-m2)
    # Berechnung der partiellen Ableitungen
    partial_x_partial_m1 = -(b2 - b1) / (m1 - m2)**2
    partial_x_partial_b1 = 1 / (m2 - m1)
    partial_x_partial_m2 = (b2 - b1) / (m1 - m2)**2
    partial_x_partial_b2 = -1 / (m1 - m2)

    # Berechnung von Delta x
    x_intersect_err = np.sqrt((partial_x_partial_m1 * m1_err)**2 +
                       (partial_x_partial_b1 * b1_err)**2 +
                       (partial_x_partial_m2 * m2_err)**2 +
                       (partial_x_partial_b2 * b2_err)**2)


    # x_intersect_err = np.sqrt(1 / ((m1-m2)**2) * (b1_err**2 + b2_err**2) + ((b2-b1)**2) /((m1-m2)**4)*(m1_err**2+m2_err**2))
    y_intersect = m2 * x_intersect + b2
    x_max_up = (b2-b2_err-b1+b1_err)/(m1+m1_err-m2-m2_err)
    # x_max_down = (abs(b2+b2_err-(b1+b1_err))) / (abs(m1-m1_err-(m2-m2_err)))
    x_max_down = (b2+b2_err-b1-b1_err)/(m1-m1_err-m2+m2_err)
    if 'colmap' in kwargs.keys():
        color = kwargs['colmap']
        # if not 'plot_from_first_ind' in kwargs.keys():
        if 'plot_from_first_minpos' in kwargs.keys() and not kwargs['plot_from_first_minpos']:
            color = color[first_part_start_ind:]
    else:
        color = create_colormap(len(first_mins), 'magma', extendRangeWith=2 + len(first_mins) - (second_part_end_ind+ (numb_references - 1)))
    
    layout_de = layouts(transpPaper=False, transpPlot=False, transpLegend=True,unit_mode='bracket', locale_settings='DE', bigTicks=False, bigTickFont=True)
    
    # trace with first minima data
    # fm_trace = go.Scattergl(x = xvals[first_part_start_ind:second_part_end_ind+1], y = first_mins[first_part_start_ind:second_part_end_ind+1],
    #     mode = 'markers', marker = dict(symbol = 'circle', color = color, size = 12,
    #         line = dict(width = .15, color = 'rgba(0,0,0,0)')),
    #                name = '<i>q</i><sub>z,fm</sub>', legendgroup = 'fm', showlegend = False)

    if 'plot_from_first_minpos' in kwargs.keys() and kwargs['plot_from_first_minpos']:
        fm_xvals, fm_yvals = xvals[:second_part_end_ind +1], first_mins[:second_part_end_ind + 1] 
    else:
        fm_xvals, fm_yvals = xvals[first_part_start_ind:second_part_end_ind +1], first_mins[first_part_start_ind:second_part_end_ind + 1] 
    # fm_trace = go.Scattergl(x = xvals[:second_part_end_ind+1], y = first_mins[:second_part_end_ind+1],
    
    fm_trace = go.Scattergl(x = fm_xvals, y = fm_yvals,
        mode = 'markers', marker = dict(symbol = 'circle', color = color, size = 12,
            line = dict(width = .15, color = 'rgba(0,0,0,0)')),
                   name = '<i>q</i><sub>z,fm</sub>', legendgroup = 'fm', showlegend = False)
    
    fig_fm = go.Figure(data = fm_trace, layout = layout_de.eldens())
    
    x_first_range = np.arange(xvals[first_part_start_ind] - xvals_overlap_down, xvals[first_part_end_ind] + xvals_overlap_up, 0.1)
    # here not to second_part_end_ind +1, since its not a slice of a list !!
    x_second_range = np.arange(xvals[first_part_end_ind] - xvals_overlap_down, xvals[second_part_end_ind] + xvals_overlap_up, 0.1)
    

    # first linear
    fig_fm.add_trace(go.Scattergl(x = x_first_range, y = linear(x_first_range, m1, b1),
                                   mode = 'lines', line = dict(color = 'black', width = 3), xaxis = 'x1', yaxis = 'y1', showlegend = False)
                     )
    # second linear
    fig_fm.add_trace(go.Scattergl(x = x_second_range, y = linear(x_second_range, m2, b2),
                                   mode = 'lines', line = dict(color = 'black', width = 3), xaxis = 'x1', yaxis = 'y1', showlegend = False, name = 'g2')
                     )

    # vertical line at T_d
    fig_fm.add_shape(type = 'line', xref = 'x1', yref = 'y1', x0 = x_intersect, x1 = x_intersect, y0 = min(first_mins)-.1, y1 = y_intersect, line = dict(width = 3, dash = 'dash', color = 'black'))
    # fig_fm.add_annotation(xref = 'x2', yref = 'y2',x = 85, y = 0.17, text = f'<i>T</i><sub>d</sub>&#8201;=&#8201;({x_intersect:.2f}&#8201;&#177;&#8201;{x_intersect_err:.2f})&#8201;°C',
    #                        showarrow = False, font = dict(size = 24, color = 'black', family = 'Latin Modern Roman'))

    # annotation T_d = ....
    # annot_text = f'<i>T</i><sub>d</sub>&#8201;=&#8201;({format_decimal(round(x_intersect, 1))}&#8201;&#177;&#8201;{format_decimal(round(x_intersect_err,1))})&#8201;°C'
    # annot_text = f'<i>T</i><sub>d</sub>&#8201;=&#8201;({"{:.1f}".format(x_intersect).replace(".",",")}&#8201;&#177;&#8201;{"{:.1f}".format(x_intersect_err).replace(".", ",")})&#8201;°C'
    annot_text = f'<i>T</i><sub>d</sub>&#8201;=&#8201;({"{:.1f}".format(x_intersect).replace(".", ",")}<sub>-{"{:.1f}".format(x_intersect-x_max_down).replace(".", ",")}</sub><sup>+{"{:.1f}".format(x_max_up-x_intersect).replace(".", ",")}</sup>)&#8201;°C'
    # annot_text = f'<i>T</i><sub>d</sub>&#8201;=&#8201;({x_intersect:.1f}<sup>+{x_max_up-x_intersect:.1f}</sup><span text-indent="10em"><sub>-{x_intersect-x_max_down:.1f}</sub></span>)&#8201;°C'

    fig_fm.add_annotation(xref = 'x1', yref = 'y1',x = 85, y = 0.17, showarrow = False, font = dict(size = 24, color = 'black', family = 'Latin Modern Roman'),
                           text = annot_text,
                           )
    fig_fm.update_layout(
        legend = dict(x = 0, y = 1, xanchor = 'left', yanchor = 'top',),
        xaxis = dict(title_text = '<i>T</i>&#8201;[°C]'),
        yaxis = dict(title_text = '<i>q</i><sub>z</sub>&#8201;[&#8491;<sup>-1</sup>]'),
        showlegend = False,
        )
    if show:
        fig_fm.show()
    if print_fit_reports:
        print(fit_m1.fit_report(), '\n', fit_m2.fit_report())
    return_data = {'fit_m1': None, 'fit_m2': None, 'x_intersect': None, 'x_intersect_err':None, 'g1_props': None, 'g2_props':None, 'figure':None}
    return_data['fit_m1'], return_data['fit_m2'], return_data['x_intersect'], return_data['x_intersect_err'], return_data['x_intersect_err_from_g_i'] = fit_m1, fit_m2, x_intersect, x_intersect_err,(x_max_up - x_intersect,x_intersect-x_max_down)
    return_data['g1_props'] = {'m1': m1, 'b1': b1, 'm1_err': m1_err, 'b1_err': b1_err}
    return_data['g2_props'] = {'m2': m2, 'b2': b2, 'm2_err': m2_err, 'b2_err': b2_err}
    return_data['figure'] = fig_fm
    return return_data
