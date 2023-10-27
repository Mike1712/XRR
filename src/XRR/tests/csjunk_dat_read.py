import sys, os
import time
import numpy as np
from XRR.helpers import *
from XRR.utilities.file_read import *
from XRR.utilities.conversion_methods import th2q, kev2angst
from XRR.read_data.DELTA_BL9 import einlesenDELTA as DELTA
import matplotlib.pyplot as plt


fp = r'\\COMPTON\saw2user\inhouse\2023\run01_23_xrr_inhouse'
tp = r'Z:\2023\run01_23_xrr_inhouse'
prefix = 'run01_23_'
wl = kev2angst(27)
sp_vacuum_2 = r'C:\data\2023\run01_23_xrr_inhouse\data\vacuum_2'
roi = [70, 120, 241, 246]

data = DELTA(fiopath = fp, tifpath = tp, savepath = sp_vacuum_2, prefix = prefix, wavelength = wl)
# data.plot_running_scan(roi = roi, plot = True)
data.live_plot(roi)
