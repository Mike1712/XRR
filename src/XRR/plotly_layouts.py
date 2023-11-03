import sys, os
import numpy as np
# from chart_studio import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, rgb2hex

def isDivisibleBy2(num):
    if (num % 2) == 0:
        return True
    else:
        return False
def get_decimal_places(f:float):
	return str(f)[::-1].find('.')

def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

def make_logtick_array(x0:float, x1:float, ticksbetween = 10, every_nth_val = 1):
	mainTicks= np.logspace(x0, x1, abs(x0) + abs(x1) +1)
	mainTicks = mainTicks[0::every_nth_val]
	tickvalsDummy, tickvals, realText = [], [], []

	for mT in mainTicks:
		tickvalsDummy.append(np.arange(mT, mT * ticksbetween, (mT)).tolist())
	tickvals = sum(tickvalsDummy, [])
	tickText = [str("{:.0e}".format(float(val))) if val in mainTicks else '' for val in tickvals]	
	for tick in tickText:
		if not tick == '':
			if not '00' in tick:
				dummy = tick.split('1e')[-1].lstrip('+ |0').replace('-0', '-')
				realText.append('10' + '<sup>' + dummy + '</sup>')
				# realText.append('10' + '<sup>' + y.split('1e')[-1] + '</sup>')
			else:
				# one = '1'
				one = '10<sup>0</sup>'
				realText.append(one)
		else:
			realText.append('')
	return tickvals, realText

def print_colors_from_plotly_figure_traces(traces, return_color_line=False):
    for i, prof in enumerate(traces):
        color_marker_line = prof.marker.line.color
        color_line = prof.line.color
        name = prof.name
        if not return_color_line:
        	if not color_marker_line==None and not color_line == 'black' and not color_marker_line == 'black':
        		print(color_marker_line,name)
        	else:
	        	print(color_line, name)


def _anz_elems_per_col(anz_items, cols):
    elems_per_col = anz_items // cols
    elems_in_last_col = anz_items // cols + anz_items % cols
    return elems_per_col, elems_in_last_col

def col_row_list2(anz_items ,cols, rows, prepend_item=True):
    elems_per_col, elems_in_last_col = _anz_elems_per_col(anz_items, cols)
    items = np.repeat(rows, cols)
    lst = []
    for ind, item in enumerate(items):
        lst_item = np.repeat(ind + 1, item).tolist()
        lst.append(lst_item)
    lst = sum(lst,[])
    if not elems_per_col == elems_in_last_col:
        if prepend_item:
            lst.insert(0,1)
        else:
            lst.append(ind + 1)
    return lst

def col_row_list(anz_items ,cols, prepend_item=True):
	elems_per_col, elems_in_last_col = _anz_elems_per_col(anz_items, cols)
	rows = anz_items / cols
	items = np.repeat(elems_per_col, cols)
	lst = []
	for ind, item in enumerate(items):
	    lst_item = np.repeat(ind + 1, item).tolist()
	    lst.append(lst_item)
	lst = sum(lst,[])
	if not elems_per_col == elems_in_last_col:
	    if prepend_item:
	    	to_prepend = np.repeat(1, elems_in_last_col - elems_per_col).tolist()
	    	if len(to_prepend) > 1:
	    		lst = [*to_prepend, *lst]
	    	else:
	    		lst.insert(0,*to_prepend)
	    else:
	    	to_append = np.repeat(ind + 1, elems_in_last_col - elems_per_col).tolist()
	    	if len(to_append) > 1:
	    		lst.extend(to_append)
	    	else:
	    		lst.append(ind + 1)
	return lst

def create_colormap(length:int, cmap:str, convert2hex = True, extendRangeWith = 0):
	all_maps = list(cm._colormaps._cmaps.keys())
	if not cmap in all_maps:
		print('Not a valid color map' )
		return
		
	else:
		col = cm.get_cmap(cmap, length + extendRangeWith)
		if convert2hex:
			return [rgb2hex(c) for c in col.colors]
		else:
			return colors



xticks = np.arange(0,2,0.05)


def extract_subplots_traces(figure, row, col):
    # Stellen Sie sicher, dass es sich um eine make_subplots-Figur handelt
    # if not isinstance(figure, plotly.subplots.make_subplots):
    #     raise ValueError("Die Eingabe-Figur sollte mit make_subplots erstellt worden sein.")

    # Überprüfen Sie, ob row und col innerhalb der zugewiesenen Subplots liegen
    if row > figure.rows or col > figure.cols:
        raise ValueError("Ungültige Zeilen- oder Spaltennummer.")

    # Index des Subplots berechnen
    subplot_index = (row - 1) * figure.cols + col

    # Extrahieren Sie alle Traces aus dem gewünschten Subplot
    subplot_traces = [trace for trace in figure.data if trace.xaxis == 'x{}'.format(subplot_index) and trace.yaxis == 'y{}'.format(subplot_index)]
    subplot_layout = figure.layout

    return subplot_traces, subplot_layout



class plotly_layouts:

	def __init__(self, bigTitles = True, bigTicks = True, bigTickFont = True,  transpPaper = True, transpPlot = True, transpLegend = True, FontColors = 'black', unit_mode = 'slash',
		locale_settings='en_US.utf8', FontFamily='latin modern roman', showgrid = False):

		self.showgrid = showgrid
		self.FontColors = FontColors
		self.unit_mode = unit_mode
		self.FontFamily = FontFamily
		self.locale_settings = locale_settings

		if 'DE' in self.locale_settings:
			self.separator = ',.'
		else:
			self.separator = '.,'
		# self.locale_settings = locale_settings
		# locale.setlocale(locale.LC_ALL, self.locale_settings)
		if bigTitles:
			self.titleFS, self.titleFS_x, self.titleFS_y, self.titleFS_legend, self.traceFS_legend = 30, 30, 30, 24, 20
			# 38, 38, 38, 30, 24
			self.title_x, self.title_y, self.title_xanchor, self.title_yanchor = 0.1, 0.93, 'left', 'top'
			self.title_standoff_y, self.title_standoff_x, self.margin = 65, 25, dict(l = 100, b = 100)
		else:
			self.titleFS, self.titleFS_x, self.titleFS_y, self.titleFS_legend, self.traceFS_legend =20, 20, 20, 20, 16
			self.title_x, self.title_y, self.title_xanchor, self.title_yanchor = 0.07, 0.91, 'left', 'top'
			self.title_standoff_y, self.title_standoff_x, self.margin = 0, 0, dict(l = 0)
		if bigTickFont:
			self.tickFS = 24
		else:
			self.tickFS=20

		if bigTicks:
			self.ticklen, self.tickwidth = 8, 4
		else:
			self.ticklen, self.tickwidth= 6, 2
		if transpPaper:
			self.paperbgcolor = 'rgba(0,0,0,0)'
		else:
			self.paperbgcolor = 'rgb(255,255,255)'
		if transpPlot:
			self.plotbgcolor = 'rgba(0,0,0,0)'
		else:
			self.plotbgcolor = 'rgb(255,255,255)'
		if transpLegend:
			self.legendbgcolor = 'rgba(0,0,0,0)'
		else:
			self.legendbgcolor = 'rgb(255,255,255)'

		self.standard_title = dict(text = None, font = dict(size = self.titleFS, color = self.FontColors, family = self.FontFamily),
	    	xanchor = self.title_xanchor, x = self.title_x, y = self.title_y)
		
		self.standard_linear_axis = dict(
	        title = dict(text = None, standoff= 0,
	        	font = dict(size = self.titleFS_x, color = 'black', family = self.FontFamily)
	        	),
	        type='linear', visible=True, linecolor='black', showgrid=self.showgrid, automargin=True, gridwidth=1, gridcolor='lightgrey', tickmode='auto',
	        zeroline=True, showline=True, ticks='inside', showticklabels=True,
	        nticks = 10, tickfont = dict(size=self.tickFS, color = self.FontColors), ticklen = self.ticklen, tickwidth = self.tickwidth,
	        mirror = 'all',)
		self.standard_log_axis = dict(
	        title = dict(text = None, standoff= 0,
	        	font = dict(size=self.titleFS_y, color = self.FontColors, family = self.FontFamily)
	        	),
	        type='log', exponentformat='power', showexponent='first', visible=True, linecolor='black', zeroline=True, showline=True, showgrid=self.showgrid, gridwidth=1,
		        gridcolor='lightgrey', ticks = 'inside', nticks = 8,tickfont = dict(size=self.tickFS, color = self.FontColors), ticklen = self.ticklen, tickwidth = self.tickwidth,
		        minor =  dict(ticks='inside', ticklen=self.ticklen -2, tickwidth = self.tickwidth -2, tickmode = 'auto', nticks = 10),
		        mirror = 'all',
	        )
		self.standard_legend = dict(
	    	title=dict(
	    		text=None, font = dict(
	    			size = self.titleFS_legend, color = self.FontColors, family = self.FontFamily),
	    	),
	    font = dict(size = self.traceFS_legend, color = self.FontColors, family = self.FontFamily),
	    	yanchor='top', xanchor='right', y=0.99, x=0.98, borderwidth=0, bgcolor = self.legendbgcolor, groupclick = 'toggleitem', itemsizing = 'trace',
	    	tracegroupgap = 0,
	    )



	def make_tick_array(self,x0:float, x1:float, dtickMain:float, ticksbetween=4):
	    mainTicks = np.arange(x0, x1 + dtickMain, dtickMain).tolist()
	    spacingMinor =np.linalg.norm(mainTicks[0] - mainTicks[1])/(ticksbetween+1)
	    allTicks = np.arange(x0, x1 + dtickMain, spacingMinor).tolist()
	    emptyList = np.repeat('', ticksbetween).tolist()
	    tickvalsDummy, tickvals, realText = [], [], []
	    if not ticksbetween == 0:
	    	for mT in mainTicks:
	    		tickvalsDummy.append([mT] + emptyList)
	    	tickvals = sum(tickvalsDummy,[])
	    else:
	    	tickvals = allTicks

	    if not (spacingMinor).is_integer():
	    	dec_places = get_decimal_places(dtickMain)
	    	realText = [f'{float(t):.{dec_places}f}' if not t == '' else '' for t in tickvals]	
	    else:
	    	# print(tickvals)
	    	realText = tickvals
	    if self.locale_settings == 'DE' and not (spacingMinor).is_integer():
    		if float(realText[0]) == 0.0:
    			realText[0] = '0'
	    	realText = [rt.replace('.',',') for rt in realText]

	    return allTicks, realText

	def format_e(self, n):
	    a = '%E' % n
	    numb = a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]
	    if self.locale_settings == 'DE':
	    	numb = numb.replace('.', ',')
	    return numb		    

	def refl(self):
		if self.unit_mode == 'slash':
			ttext_x = '<i>q</i><sub>z</sub>/&#8491;<sup>-1</sup>'
			if not 'DE' in self.locale_settings:
				ttext_y = '<i>I</i>/arb. u.'
			else:
				ttext_y = '<i>I</i>/w. E.'
		elif self.unit_mode == 'bracket':
			if not 'DE' in self.locale_settings:
				ttext_x = '<i>q</i><sub>z</sub><span>&#8201;[<span>&#8491;</span><sup>-1</sup></span>]'
				ttext_y = '<i>I</i>&#8201;[arb. u.]'
			else:
				ttext_x = '<i>q</i><sub>z</sub><span>&#8201;[<span>&#8491;</span><sup>-1</sup></span>]'
				ttext_y = '<i>I</i>&#8201;[w. E.]'
		refl = go.Layout(
			separators = self.separator, 
	    title = dict(**self.standard_title),
	    yaxis = dict(**self.standard_log_axis, title_text =ttext_y) ,
	    xaxis = dict(**self.standard_linear_axis, title_text = ttext_x),
	    legend = dict(**self.standard_legend),
	    autosize=False, width=800, height=566, paper_bgcolor=self.paperbgcolor,plot_bgcolor=self.plotbgcolor, margin = self.margin,
	    hovermode = 'x'
		)
		return refl

	def eldens(self):
		if self.unit_mode == 'slash':
			ttext_x_html = "<i>z</i>/<span>&#8491;</span>"
			ttext_y_html = "&#961;<sub>e<sup>-</sup></sub>/e<sup>-</sup>&#183;&#8201;&#8491;<sup>-3</sup>"
		elif self.unit_mode == 'bracket':
			ttext_x_html = "<i>z</i>&#8201;[<span>&#8491;</span>]"
			ttext_y_html = "&#961;<sub>e<sup>-</sup></sub>&#8201;[e<sup>-</sup> / &#8491;<sup>3</sup>]"

		layout_eldens = go.Layout(
			separators= self.separator,
			yaxis = dict(**self.standard_linear_axis, title_text = ttext_y_html),
			xaxis = dict(**self.standard_linear_axis, title_text = ttext_x_html),
			legend = dict(**self.standard_legend),
			autosize=False, width=800, height=566, paper_bgcolor=self.paperbgcolor, plot_bgcolor=self.plotbgcolor,
			margin = self.margin
			)

		return layout_eldens

	def sbs(self):
		if self.unit_mode == 'slash':
			ttext_x = '<i>q</i><sub>z</sub>/&#8491;<sup>-1</sup>'
			if 'DE' in self.locale_settings:
				ttext_y = '<i>I</i>/w. E.'	
			else:
				ttext_y = '<i>I</i>/arb. u.'
			ttext_x_html_eldens = "<i>z</i>/<span>&#8491;</span>"
			ttext_y_html_eldens = "&#961;<sub>e<sup>-</sup></sub>/e<sup>-</sup>&#183;&#8201;&#8491;<sup>-3</sup>"
		elif self.unit_mode == 'bracket':
			if 'DE' in self.locale_settings:
				ttext_y = '<i>I</i>&#8201;[w. E.]'
			else:
				ttext_y = '<i>I</i>&#8201;[arb. u.]'
			ttext_x = '<i>q</i><sub>z</sub>&#8201;<span>[<span>&#8491;</span><sup>-1</sup></span>]'

			ttext_x_html_eldens = "<i>z</i>&#8201;[<span>&#8491;</span>]"
			ttext_y_html_eldens = "&#961;<sub>e<sup>-</sup></sub>&#8201;[e<sup>-</sup>/&#8491;<sup>3</sup>]"

		sbs = go.Layout(
			separators=self.separator,
	   		title = dict(**self.standard_title),
	    yaxis = dict(**self.standard_log_axis, title_text = ttext_y),
	    xaxis = dict(**self.standard_linear_axis, domain = [0, 0.48]), 
	    yaxis2 = dict(**self.standard_linear_axis, anchor = 'x2', side = 'right',title_text = ttext_y_html_eldens),
	    xaxis2 = dict(**self.standard_linear_axis, title_text = ttext_x_html_eldens, domain = [0.52, 1],),
	    legend = dict(**self.standard_legend),
	    autosize=False, width=1200, height=850, paper_bgcolor=self.paperbgcolor,plot_bgcolor=self.plotbgcolor, margin = self.margin
		)
		
		return sbs

	def VolumeFractionProfile(self):
		layout_VFP = self.eldens()
		if self.unit_mode == 'slash':
			ttext_x_html = "<i>z</i>/&#8491;"
			ttext_y_html = '(&#961;<sub>System</sub> - &#961;<sub>Substrat</sub>)/e<sup>-</sup>&#183;&#8201;&#8491;<sup>-3</sup>'
		elif self.unit_mode == 'bracket':
			ttext_x_html = "<i>z</i>&#8201;[&#8491;]"
			# ttext_y_html = "<span>&#961;</span><sub>e<sup>-</sup></sub> [e<sup>-</sup> / <span>&#8491;</span><sup>3</sup>]"
			ttext_y_html = '&#961;<sub>System</sub> - &#961;<sub>Substrat</sub>&#8201;[e<sup>-</sup>&#183;&#8201;&#8491;<sup>-3</sup>]'
		layout_VFP.xaxis.title.text, layout_VFP.yaxis.title.text = ttext_x_html, ttext_y_html 
		layout_VFP.xaxis.title.standoff=self.title_standoff_x- self.title_standoff_x
		layout_VFP.yaxis.title.standoff=self.title_standoff_y- self.title_standoff_y
		return layout_VFP



	def int_eldens(self, dtick_x = .1, dtick_y = 10, x0 = 0, x1=2, y0 = -10, y1= 200, numb_xaxis_minorticks=0, numb_yaxis_minorticks=0):
		ytickvals, yticktext = self.make_tick_array(x0 = y0, x1=y1, dtickMain = dtick_y, ticksbetween = numb_yaxis_minorticks)
		xtickvals, xticktext = self.make_tick_array(x0 = x0, x1 = x1, dtickMain = dtick_x, ticksbetween=numb_xaxis_minorticks)

		if self.unit_mode == 'slash':
			if not 'DE' in self.locale_settings:
				ttext_y_html = 'molecules/nm<sup>2</sup>'
				ttext_x_html = "<i>p</i>/bar"
			else:
				ttext_y_html = 'Moleküle/nm<sup>2</sup>'
				ttext_x_html = "<i>p</i>/bar"
		elif self.unit_mode == 'bracket':
			if not 'DE' in self.locale_settings:
				ttext_y_html = 'molecules/nm<sup>2</sup>'
				ttext_x_html = "<i>p</i>&#8201;[bar]"
			else:
				ttext_y_html = '&#961;<sub>A</sub>&#8201;[Moleküle&#183;nm<sup>-2</sup>]'
				# ttext_y_html = 'Moleküle/nm<sup>2</sup>'
				ttext_x_html = "<i>p</i>&#8201;[bar]"
		
		layout_int_eldens = go.Layout(
			separators= self.separator,
			title = dict(**self.standard_title),
			yaxis = dict(**self.standard_linear_axis, title_text = ttext_y_html,),
			xaxis = dict(**self.standard_linear_axis, title_text= ttext_x_html,),
			legend = dict(**self.standard_legend),
			autosize=False, width=800, height=566, paper_bgcolor=self.paperbgcolor, plot_bgcolor=self.plotbgcolor,
			margin = self.margin
		)		
		return layout_int_eldens

	def mondio(self, dtick = 0.1):
		if self.unit_mode == 'slash':
			ttext_x_html = '<i>q</i><sub>z</sub> / <span>&#8491;</span><sup>-1</sup>'
			ttext_y_html = '<i>I</i> / arb. u.'
		elif self.unit_mode == 'bracket':
			ttext_x_html = '<i>q</i><sub>z</sub> [<span>&#8491;</span><sup>-1</sup>]'
			ttext_y_html = '<i>I</i> [arb. u.]'

		layout_mondio = go.Layout(
			 title = dict(text = '', font= dict(size = self.titleFS, color = self.FontColors), xanchor = 'left', x = self.title_x, y = self.title_y), 
	    yaxis = dict(
	        title = dict(text = ttext_y_html,
	        	font = dict(size=self.titleFS_y, color = self.FontColors),
	        	standoff = self.title_standoff_y),
	        type='linear' , visible=True, linecolor='black',zeroline=True, showline=True, automargin = True,
	        showgrid=self.showgrid, gridwidth=1, gridcolor='lightgrey', ticks='outside', showticklabels=True, mirror = True,
	        tickfont = dict(size=self.tickFS, color = self.FontColors), ticklen=10, tickwidth=5,
	    ),
	    xaxis = dict(
	        title = dict(text = ttext_x_html, standoff= self.title_standoff_x,
	        	font = dict(size = self.titleFS_x, color = 'black'),
	        	),
	        tickfont = dict(size=self.tickFS, color = self.FontColors), ticklen=10, tickwidth=5,
	        visible=True, linecolor='black', showgrid=self.showgrid, automargin = True, mirror = True,
	        gridwidth=1, gridcolor='lightgrey', zeroline=True, showline=True, ticks='outside', tickmode='linear', dtick = dtick
	    ), 
		legend = dict(title=dict(text=None,
			    	font = dict(size = self.titleFS_legend, color = self.FontColors)
			    	),
			    font = dict(size = self.traceFS_legend, color = self.FontColors,),
			    	yanchor='bottom', xanchor='right', x=0.99, y=0.01, borderwidth=0, bgcolor = self.legendbgcolor
	    ),
	    autosize=False, width=1200, height=800, paper_bgcolor=self.paperbgcolor, plot_bgcolor=self.plotbgcolor,
	    margin = self.margin
	    )
		return layout_mondio



	def blank(self):
		layout_blank = go.Layout(
			paper_bgcolor = self.paperbgcolor, plot_bgcolor = self.plotbgcolor,
			yaxis = dict(type ='linear', title_font_size = 20, visible = True, linecolor = 'black', automargin = True, zeroline = True, showline = True, showgrid = self.showgrid,
				gridwidth = 1, gridcolor = 'lightgrey', ticks = 'outside', showticklabels = True, tick0 = 0, mirror = True),
			xaxis = dict(visible=True, title_font_size = 20, linecolor = 'black', showgrid = self.showgrid, automargin = True, gridwidth = 1, gridcolor = 'lightgrey', zeroline = True,
				showline = True, ticks = 'outside', showticklabels = True, mirror = True),
			legend = dict( font_size = 16, yanchor = 'top', xanchor = 'right', y=0.99, x=0.99, borderwidth=0)
			)

		return layout_blank

	def beam_current(self):
		beam_current_layout = go.Layout(
			paper_bgcolor = self.paperbgcolor,
			plot_bgcolor = self.plotbgcolor, title = '',
			xaxis = dict( visible = True, linecolor = 'black', zeroline = True, showline = True, showgrid = self.showgrid, gridwidth = 1, gridcolor = 'lightgrey',
				type ='category', title = 'time / s', title_font_size = 18, mirror = True),
			yaxis = dict(visible = True, linecolor = 'black', title = 'beam current / mA', title_font_size = 18, zeroline = True, showline = True,
				showgrid = self.showgrid, gridwidth = 1, gridcolor = 'lightgrey', mirror = True,),
			legend = dict(yanchor='top', xanchor='right', y=0.99, x=0.99, borderwidth=0, font_size = 14,),
			)

		return beam_current_layout

	def thickness(self, dtick = 5):
		if self.unit_mode == 'slash':
			ttext_yaxis = '<i>d</i> / <span>&#8491;</span>'
		elif self.unit_mode == 'bracket':
			ttext_yaxis = '<i>d</i> [<span>&#8491;</span>]'
		ttext_xaxis = '<i>p</i> / <i>p</i><sub>crit</sup> [<i>p</i> / <i>p</i><sub>sat</sup>]'
		layout_thickness = go.Layout(
			 title = dict(text = '', font= dict(size = self.titleFS, color = self.FontColors), xanchor = 'left', x = self.title_x, y = self.title_y), 
	    yaxis = dict(
	        title = dict(text = ttext_yaxis,
	        	font = dict(size=self.titleFS_y, color = self.FontColors),
	        	standoff = self.title_standoff_y),
	        	type='linear' , visible=True, linecolor='black',zeroline=True, showline=True, automargin = True,
	        	showgrid=self.showgrid, gridwidth=1, gridcolor='lightgrey', ticks='outside', showticklabels=True, mirror = True,
	        	tickfont = dict(size=self.tickFS, color = self.FontColors),
	    ),
	    xaxis = dict(
	        title = dict(text = ttext_xaxis, standoff= self.title_standoff_x,
	        	font = dict(size = self.titleFS_x, color = 'black'),
	        	),
	        tickfont = dict(size=self.tickFS, color = self.FontColors), ticklen=10, tickwidth=5,
	        visible=True, linecolor='black', showgrid=self.showgrid, automargin = True, mirror = True,
	        gridwidth=1, gridcolor='lightgrey', zeroline=True, showline=True, ticks='outside', tickmode='linear', dtick = dtick
	    ), 
		legend = dict(title=dict(text=None,
			    	font = dict(size = self.titleFS_legend, color = self.FontColors)
			    	),
			    font = dict(size = self.traceFS_legend, color = self.FontColors,),
			    	yanchor='top', xanchor='left', y=0.99, x=0.02, borderwidth=0, bgcolor = self.legendbgcolor
	    ),
	    autosize=False, width=1200, height=800, paper_bgcolor=self.paperbgcolor, plot_bgcolor=self.plotbgcolor,
	    margin = self.margin
	    )
		return layout_thickness

	def pT_diag(self, dtick_x=50, dtick_y = 20):

		xticksMain = np.arange(start = -20, stop = 1000, step = dtick_x, dtype = 'int')
		xtickValsDummy, xtickvals = [], []
		for xtM in xticksMain:
		    xtickValsDummy.append(np.arange(start = xtM, stop = xtM +dtick_x, step = 1, dtype = int).tolist())
		xtickvals = sum(xtickValsDummy, [])
		xtickText = [str(val) for val in xticksMain]

		
		yticksMain = np.arange(start = -20, stop = 1000, step = dtick_y, dtype = 'int')
		ytickValsDummy, ytickvals = [], []
		for ytM in yticksMain:
		    ytickValsDummy.append(np.arange(start = ytM, stop = ytM +dtick_y, step = 1, dtype = int).tolist())
		ytickvals = sum(ytickValsDummy, [])
		ytickText = [str(val) for val in yticksMain]
		# ytick_text = [locale.format_string('%.1f', yt) for yt in ytickvals]

		if self.unit_mode == 'slash':
			ttext_x_html = "<i>T</i>/<span>&#8451;</span>"
			ttext_y_html = "<i>p</i>/bar"
		elif self.unit_mode == 'bracket':
			ttext_x_html = "<i>T</i>&#8201;[<span>&#8451;</span>]"
			ttext_y_html = "<i>p</i>&#8201;[bar]"

		layout_pT = go.Layout(
	    title = dict(text = '', font= dict(size = self.titleFS, color = self.FontColors), xanchor = 'left', x = self.title_x + 0.03, y = self.title_y), 
	    yaxis = dict(
	        title = dict(text = ttext_y_html,
	        	font = dict(size=self.titleFS_y, color = self.FontColors, family = self.FontFamily),
	        	standoff = self.title_standoff_y),
	        	type='linear' , visible=True, linecolor='black',zeroline=True, showline=True, automargin = True,
	        	showgrid=self.showgrid, gridwidth=1, gridcolor='lightgrey', ticks='inside', dtick = dtick_y,
	        	showticklabels=True, mirror = 'all',
	        	tickfont = dict(size=self.tickFS, color = self.FontColors), tickwidth=3,
	    ),
	    xaxis = dict(
	        title = dict(text = ttext_x_html, standoff= self.title_standoff_x,
	        	font = dict(size = self.titleFS_x, color = 'black', family = self.FontFamily),
	        	),
	        tickfont = dict(size=self.tickFS, color = self.FontColors),
	        visible=True, linecolor='black', showgrid=self.showgrid, automargin = False, mirror = 'all',
	        gridwidth=1, gridcolor='lightgrey', zeroline=True, showline=True, ticks='inside', tickmode='linear', dtick = dtick_x,
	        tickwidth=3
	    ), 
		legend = dict(
			title=dict(
				text=None, font = dict(
					size = self.titleFS_legend, color = self.FontColors, family = self.FontFamily),
			    	),
			    font = dict(size = self.traceFS_legend, color = self.FontColors, family = self.FontFamily),
			    	yanchor='top', xanchor='left', y=0.99, x=0.03, borderwidth=0, bgcolor = self.legendbgcolor,
	    ),
	    autosize=False, width=1200, height=800, paper_bgcolor=self.paperbgcolor, plot_bgcolor=self.plotbgcolor,
	    margin = self.margin
		)
		return layout_pT

	def sbs_subplots(self):
		fig = make_subplots(rows = 1, cols = 2, subplot_titles = ('a)', 'b)'),)
		if self.unit_mode == 'slash':
			ttext_x = '<i>q</i><sub>z</sub>/&#8491;<sup>-1</sup>'
			if 'DE' in self.unit_mode:
				ttext_y = '<i>I</i>/w. E.'
			else:
				ttext_y = '<i>I</i>/arb. u.'
			ttext_x_html_eldens = "<i>z</i>/<span>&#8491;</span>"
			ttext_y_html_eldens = "&#961;<sub>e<sup>-</sup></sub>/e<sup>-</sup>&#183;&#8201;&#8491;<sup>-3</sup>"
		elif self.unit_mode == 'bracket':
			ttext_x = '<i>q</i><sub>z</sub>&#8201;<span>[<span>&#8491;</span><sup>-1</sup></span>]'
			if 'DE' in self.unit_mode:
				ttext_y = '<i>I</i&#8201;[w. E.]'
			else:
				ttext_y = '<i>I</i&#8201;[arb. u.]'
			ttext_x_html_eldens = "<i>z</i>&#8201;[<span>&#8491;</span>]"
			ttext_y_html_eldens = "&#961;<sub>e<sup>-</sup></sub>&#8201;[e<sup>-</sup> / &#8491;<sup>3</sup>]"

		sbs = go.Layout(
			separators=self.separator,
	    title = dict(text = None, font = dict(size = self.titleFS, color = self.FontColors, family = self.FontFamily),
	    	xanchor = self.title_xanchor, x = self.title_x, y = self.title_y), 
	    yaxis = dict(**self.standard_log_axis, title_text = ttext_y_html
	        ),
	    xaxis = dict(**self.standard_linear_axis, title_text = ttext_x, domain = [0, 0.48],
	    ), 
	    yaxis2 = dict(anchor = 'x2', side = 'right',
	        title = dict(
	        	text = ttext_y_html_eldens, **self.standard_linear_axis
	        	)
	        ),
	    xaxis2 = dict(**self.standard_linear_axis,
	        title = dict(
	        	text = ttext_x_html_eldens, domain = [0.52, 1],
	        	),
	        ),
	    legend = self.standard_legend,
	    autosize=False, width=800, height=566, paper_bgcolor=self.paperbgcolor,plot_bgcolor=self.plotbgcolor, margin = self.margin
		)
		return sbs
	def fm_trace(self):
		# layout_fm = self.eldens()
		if self.unit_mode == 'slash':
			ttext_x_html = "<i>T</i>/°C"
			ttext_y_html = '<i>q</i><sub>z</sub>/&#8491;<sup>-1</sup>'
		elif self.unit_mode == 'bracket':
			ttext_x_html = "<i>T</i>&#8201;[°C]"
			ttext_y_html = '<i>q</i><sub>z</sub>&#8201;[&#8491;<sup>-1</sup>]'

		layout_fm = go.Layout(
			separators= self.separator,
			title = dict(**self.standard_title),
			yaxis = dict(**self.standard_linear_axis, title_text = ttext_y_html,),
			xaxis = dict(**self.standard_linear_axis, title_text= ttext_x_html,),
			legend = dict(**self.standard_legend),
			autosize=False, width=800, height=566, paper_bgcolor=self.paperbgcolor, plot_bgcolor=self.plotbgcolor,
			margin = self.margin
		)		
		return layout_fm
	# def layout_sbs_4plots(self):
	# 	refl_layout = self.refl()
	# 	eldens_layout = self.eldens_new()
	
	def gamma_ex(self):
		if self.unit_mode == 'slash':
			ttext_x_html = "<i>p</i><sub>red</sub>"
			ttext_y_html = '&#915;<sub>ex</sub>/Moleküle&#183;nm<sup>-2</sup>'
		elif self.unit_mode == 'bracket':
			ttext_x_html = "<i>p</i><sub>red</sub>"
			ttext_y_html = '&#915;<sub>ex</sub>&#8201;[Moleküle&#183;nm<sup>-2</sup>]'
		layout_gamma = go.Layout(
			separators= self.separator,
			title = dict(**self.standard_title),
			yaxis = dict(**self.standard_linear_axis, title_text = ttext_y_html,),
			xaxis = dict(**self.standard_linear_axis, title_text= ttext_x_html,),
			legend = dict(**self.standard_legend),
			autosize=False, width=800, height=566, paper_bgcolor=self.paperbgcolor, plot_bgcolor=self.plotbgcolor,
			margin = self.margin)
		return layout_gamma