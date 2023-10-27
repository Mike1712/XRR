import os
import numpy as np
from pandas import read_csv, DataFrame
from lmfit.models import LinearModel
import plotly.graph_objects as go

def fit_linear(x, y, m, b):
    model = LinearModel()
    model.set_param_hint('intercept', value = b, vary = True)
    pars = model.make_params()
    result = model.fit(y, pars, x = x)
    return result

def read_lines_file(file, onlyFirstLine=False):
    openfile = open(file, 'r')
    if onlyFirstLine:
        lines = openfile.readline()
    else:
        lines = openfile.readlines()
    openfile.close()
    return lines


def _fit_temps_AntonPaar(fp = None):
    '''
    Fit temperature values of Anton-Paar sample cell measured with a thermoelement against set temperature. This data is used to calculate the surface temperature values with "temp_calibration_AntonPaar"
    Parameters
    ----------
    fp: file containing the measured temperature data.

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
        fp: file containing the measured temperature data.
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
        fit, df = _fit_temps_AntonPaar(fp = fp)
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
        fig = go.Figure(data = [t_data, t_fit], layout = layout.eldens_new())
        fig.update_layout(legend = dict(x = 0, y = 1, xanchor = 'left', yanchor='top'),
            xaxis = dict(title_text = 'T<sub>set</sub>'),
            yaxis = dict(title_text = 'T<sub>surface</sub>'))
        fig.show()
    return y, y_err

