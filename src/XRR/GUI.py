import sys, os
import numpy as np
import PySimpleGUI as sg
from PIL import Image
from pprint import pprint

from matplotlib import pyplot as plt
# from matplotlib.ticker import NullFormatter  # useful for `logit` scale
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.cm as cm
from matplotlib.colors import rgb2hex
# import matplotlib
# matplotlib.use('TkAgg')

import plotly.graph_objects as go
from XRR.plotly_layouts import plotly_layouts as layouts
from XRR.helpers import scan_numbers_arr
layout = layouts(transpPaper=False, transpPlot=False, unit_mode='slash')
from XRR.read_data.DELTA_BL9 import einlesenDELTA as DELTA
from XRR.utilities.conversion_methods import kev2angst, th2q
from XRR.utilities.math import shift_with_power

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg



def make_window(theme):

    # Change these parameters depending on your current run to have default paths and experimental details by GUI startup
    ##################################################################
    fiopath = '/home/mike/Dokumente/Uni/Promotion/Messzeit_DELTA_0521/Daten/FIO'
    tifpath = '/home/mike/Dokumente/Uni/Promotion/Messzeit_DELTA_0521/Daten/TIF'
    savepath = '/home/mike/Dokumente/Uni/Promotion/Arbeit/files/plots/XRR_datatreatment'

    roi_vertical_min_pix = str(76)
    roi_vertical_max_pix = str(112)
    roi_horizontal_min_pix = str(228)
    roi_horizontal_max_pix = str(233)

    run_prefix = 'run05_21_'

    beam_en = '27'
    beam_height = '100'

    ###################################################################
    header_font = ('Arial', 16)
    text_font = ('Arial', 12)
    menu_def = [
    ['&Help', ['&About']]
    ]
    
    roi_lcol = [[sg.Text('\tmin y-pixel:'), sg.InputText(roi_vertical_min_pix, key = 'roi_vertmin', expand_x = False, size = (15,1), do_not_clear=True)],
                [sg.Text('\tmin x-pixel:'), sg.InputText(roi_horizontal_min_pix, key = 'roi_horzmin', expand_x = False, size = (15,1), do_not_clear=True)],        
    ]
    roi_rcol = [[sg.Text('\tmax y-pixel:'), sg.InputText(roi_vertical_max_pix, key = 'roi_vertmax', expand_x = False, size = (15,1), do_not_clear=True)],
                [sg.Text('\tmax x-pixel:'), sg.InputText(roi_horizontal_max_pix, key = 'roi_horzmax', expand_x = False, size = (15,1), do_not_clear=True)],        
    ]
    roi_layout = [sg.Column(roi_lcol, element_justification='l'), sg.Column(roi_rcol, element_justification='l')]
    roi_lcol_dect_img = [[sg.Text('\tmin y-pixel:'), sg.InputText(roi_vertical_min_pix, key = 'roi_vertmin_dect_img', expand_x = False, size = (15,1), do_not_clear=True)],
                [sg.Text('\tmin x-pixel:'), sg.InputText(roi_horizontal_min_pix, key = 'roi_horzmin_dect_img', expand_x = False, size = (15,1), do_not_clear=True)],        
    ]
    roi_rcol_dect_img = [[sg.Text('\tmax y-pixel:'), sg.InputText(roi_vertical_max_pix, key = 'roi_vertmax_dect_img', expand_x = False, size = (15,1), do_not_clear=True)],
                [sg.Text('\tmax x-pixel:'), sg.InputText(roi_horizontal_max_pix, key = 'roi_horzmax_dect_img', expand_x = False, size = (15,1), do_not_clear=True)],        
    ]
    roi_layout_dect_img = [sg.Column(roi_lcol_dect_img, element_justification='l'), sg.Column(roi_rcol_dect_img, element_justification='l')]
    # tab layouts
    # layout for input parameters


    input_layout = [
        [sg.Text('Enter your filepaths:', justification='left')],
        [sg.Text('\tPath to FIO-files:', justification='right'),
        sg.InputText(fiopath, key = 'fiopath', expand_x = True, do_not_clear=True), sg.FolderBrowse()],
        [sg.Text('\tPath to TIF-files:', justification='right'),
        sg.InputText(tifpath, key = 'tifpath', expand_x = True, do_not_clear=True), sg.FolderBrowse()],
        [sg.Text('\tSave data in:', justification='right', expand_x = True),
        sg.InputText(savepath, key = 'savepath', do_not_clear=True), sg.FolderBrowse()],
        [sg.Text('run_prefix'), sg.InputText(run_prefix, key = 'prefix', do_not_clear=True), sg.FolderBrowse()],
        # [sg.Button(button_text='Ok', key = 'confirm_paths')],
        [sg.Text('_' * 100)],
        [sg.Text('Experimental details:\n\tRoi:'),],
        roi_layout,
        # [sg.Text('\t'), sg.Text('min y-pixel:'), sg.InputText('0', key = 'roi_vertmin', expand_x = False, size = (15,1), do_not_clear=True),
        # sg.Text('max y-pixel:'), sg.InputText('0', key = 'roi_vertmax', expand_x = False, size = (15,1), do_not_clear=True)],
        # [sg.Text('\tmin x-pixel:'), sg.InputText('0', key = 'roi_horzmin', expand_x = False, size = (15,1), do_not_clear=True),
        # sg.Text('max x-pixel:'), sg.InputText('0', key = 'roi_horzmax', expand_x = False, size = (15,1), do_not_clear=True)],
        [sg.Text('\tBeam energy:'), sg.InputText(beam_en, key = 'energy', expand_x = False, size = (15,1), do_not_clear=True,), sg.Text('keV')],
        [sg.Text('\tBeam height in microns:'), sg.InputText(beam_height, key = 'beam_height', expand_x = False, size=(5,1)),sg.Text(u'\u03bc' + 'm')],
        # [sg.Button(button_text='Confirm inputs', key='confirm_inputs')]
    ]
    # another tab-layout
    plot_refl_layout = [
    [sg.Text('Plot recorded data:', font = ('Arial', 18))],
    [sg.Text('\tNumber of start scan:'), sg.InputText('1', expand_x = True, do_not_clear=True, key = 'start_scans')],
    [sg.Text('\tNumber of scans per measurement:',), sg.InputText('4', expand_x = True, do_not_clear=True, key = 'anz_scans')],
    [sg.Text('\t(Optional) Names:',), sg.InputText('', expand_x = True, do_not_clear=True, key = 'names')],
    [sg.Text('\tShift:',)],
    [sg.Text('\t'), sg.Checkbox(text = 'Shift:', key = 'shift_bool'), sg.Text('base:'),
    sg.Combo(values=('10', '20', '30', '40', '50', '100'), default_value='10', readonly=False, key='shift_base'),],
    [sg.Text('\tFigure properties:\n\twidth:'), sg.InputText('1600', key = 'fig_width'), sg.Text('\theight:'), sg.InputText('1200', key = 'fig_height')],
    [sg.Button(button_text='Plot', key = 'plot_XRR_scans')],
    [sg.Text('_' * 100)],
    [sg.Text('Plot running scan')],
    [sg.Button(button_text='Plot', key = 'plot_running_scan')],

    ]
    current_scan_layout = [
    [sg.Text('Plot running scan',)],

    ]
    dect_imgs_layout = [
    [sg.Text('Show detector images of a scan at an angle\nROI:',)],
    roi_layout_dect_img,
    [sg.Text('Scan number:'),sg.InputText('', key = 'scan_numb_dect_img', do_not_clear = True, expand_x = False, size = (15,1))],
    [sg.Text('Index of image number:'),sg.InputText('', key = 'th_ind_dect_img', do_not_clear = True, expand_x = False, size = (15,1))],
    [sg.Checkbox(text = 'Series of images', key = 'dect_series'), sg.Button(button_text = 'Show', key = 'plot_dect_img'), sg.Text('\t'),],

    ]
    dat_eval_layout = [
    [sg.Text('Evaluation of XRR data', font = ('Arial', 16))],

    ]
    # putting layouts togeter
    layout = [ [sg.MenubarCustom(menu_def, key='-MENU-', font='Courier 15', tearoff=True)],
    [sg.Text('XRR analysis', size=(38, 1), justification='center', font=("Helvetica", 16), relief=sg.RELIEF_RIDGE, k='-TEXT HEADING-', enable_events=True)]]

    layout +=[[sg.TabGroup([[  sg.Tab('Input Elements', input_layout),
                                   sg.Tab('Plot XRR', plot_refl_layout),
                                   # sg.Tab('Plot current scan', current_scan_layout),
                                   sg.Tab('Detector images', dect_imgs_layout),
                                   sg.Tab('Data evaluation', dat_eval_layout),
                                   # sg.Tab('Output', logging_layout)
                                   ]],
                                   key='-TAB GROUP-', expand_x=True, expand_y=True),

                   ]]
    window = sg.Window('DELTA BL9', layout, resizable=True, grab_anywhere=True, margins = (0,0), use_custom_titlebar=True, finalize=True,
        location = (0, 0),  size = (800, 600),
        keep_on_top=False,
        font = header_font,)
    # window.set_min_size(window.size)
    # window.Maximize()
    return window

def main():
    window = make_window(sg.theme())
    while True:
        event, values = window.read(timeout=10)
        
        fp, tp, sp = values['fiopath'], values['tifpath'], values['savepath']
        wavelength =kev2angst(float(values['energy']))
        roi = [int(values['roi_vertmin']), int(values['roi_vertmax']), int(values['roi_horzmin']), int(values['roi_horzmax']),]
        dat = DELTA(fp, tp, sp, prefix = values['prefix'],wavelength=wavelength)
        start_scans = values['start_scans']
        anz_scans = values['anz_scans']
        if event in 'plot_dect_img':
            series_bool = values['dect_series']
            roi_dect_img = [int(values['roi_vertmin_dect_img']), int(values['roi_vertmax_dect_img']), int(values['roi_horzmin_dect_img']), int(values['roi_horzmax_dect_img']),]
            dat.dectimage(int(values['scan_numb_dect_img']), int(values['th_ind_dect_img']), roi = roi_dect_img, series = series_bool)
        if event in 'plot_running_scan':
            dat .live_plot(roi)
        if event in 'plot_XRR_scans':
            names = list(values['names'].split(','))
            viridis = cm.get_cmap('Pastel1', len(start_scans) +2)
            viridis = [rgb2hex(viridis(c)) for c in range(0, viridis.N)]
            width, height = int(values['fig_width']), int(values['fig_height'])

            if isinstance(start_scans, str):
                try:
                    if not "[" in start_scans and not "]" in start_scans: 
                        start_scans  = list(start_scans.split(','))
                    else:
                        start_scans = list(start_scans.replace('[', '').replace(']','').split(','))
                except Exception as e:
                    print(e)
            start_scans = [int(elem) for elem in start_scans]
            
            if isinstance(anz_scans, str):
                if not "[" in anz_scans and not "]" in anz_scans:
                    anz_scans = np.repeat(anz_scans, len(start_scans))
                else:
                    anz_scans = list(anz_scans.replace('[', '').replace(']','').split(','))
            anz_scans = [int(elem) for elem in anz_scans]
            if values['shift_bool']:
                shift = shift_with_power(len(start_scans), spacing = int(values['shift_base']))
            else:
                shift = np.repeat(1, len(start_scans))
            
            if names == '':
                names = [f'scans: {start_scans[i]}-{start_scans[i]+anz_scans[i]-1}' for i in range(len(start_scans))]
            if not len(start_scans) == len(names):
                # sg.Popup(f'Length of arrays do not match.\nlen(start_scans) = {len(start_scans)}\tlen(names) = {len(names)}.\nName-array will be created automatically', title = 'Warning', )
                names = [f'scans: {start_scans[i]}-{start_scans[i]+anz_scans[i]-1}' for i in range(len(start_scans))]
            XRR_data, traces = dict(), list()
            layout_progressbar = [[sg.Text('Calculating data....')],
            [sg.ProgressBar(len(start_scans), orientation='h', size=(20, 20), key='progressbar')],
            [sg.Cancel()]]
            window_pg = sg.Window('Loading...',layout_progressbar)
            progress_bar = window_pg['progressbar']
            for i in range(len(start_scans)):
                if i == 0:
                    XRR_data[str(start_scans[i])] = dat.read_XRR(start_scans[i], anz_scans[i], roi = roi, plot_seperate_loops = True, fit_absorber_correction = True) 
                    shift_fit_text = XRR_data[str(start_scans[i])]['absorber']
                    shift_fit = [float(s) for s in shift_fit_text]
                    print(shift_fit)
                else:
                    XRR_data[str(start_scans[i])] = dat.read_XRR(start_scans[i], anz_scans[i], absorber_correc = shift_fit, roi = roi, plot_seperate_loops = True, fit_absorber_correction = True) 
                    
                data = XRR_data[str(start_scans[i])]['data']
                traces.append(go.Scattergl(x = data.q, y = data.counts2consider * shift[i], customdata=data.theta, mode='lines+markers',
                    line = dict(width = 4, color = viridis[i]),
                    marker = dict(size = 8, symbol = 'circle', color = viridis[i],
                        line=dict(color='rgb(255,255,255)', width=.5),),
                    name = names[i])
                    )
                pg_event, pg_value = window_pg.read(timeout = 10)
                if pg_event in ('Cancel', 'sg.WIN_CLOSED'):
                    window_pg.close()
                    break
                progress_bar.UpdateBar(i+1)
            window_pg.close()
            fig = go.Figure(data = traces, layout = layout.refl())
            fig.update_layout(width = width, height = height)
            fig.show()
        if event in (None, 'Exit', sg.WIN_CLOSED):
            print("[LOG] Clicked Exit!")
            break

    window.close()
    exit(0)

if __name__ == '__main__':
    # sg.theme('black')
    # sg.theme('dark red')
    # sg.theme('dark green 7')
    sg.theme('DefaultNoMoreNagging')
    main()
