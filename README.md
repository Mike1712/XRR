# XRR
Python code to read raw XRR data recorded at beamlines ID31 (ESRF), P08 (DESY, PETRA III), BL9 (DELTA) and evaluate data.<br>
You can also simulate XRR profiles and electron density profiles by providing a layer stack of your sample system.<br>
This code is developed and tested on Fedora Linux. 
## Requirements
     "numpy >= 1.22.4",
     "scipy >= 1.8.1",
     "pandas >= 1.4.2",
     "Pillow >= 9.1.1",
     "matplotlib >= 3.5.2",
     "matplotlib-inline >= 0.1.3",
     "plotly >= 5.8.0",
      "IPython >= 8.4.0",
      "ipykernel >= 6.13.0",
      "ipywidgets >= 7.7.0",
      "jupyter_client >= 7.4.7",
      "jupyter_core >=  5.0.0",
      "jupyter_server >= 1.23.3",
      "jupyterlab >= 3.5.0",
      "nbclient >= 0.5.13",
      "nbconvert >= 7.2.5",
      "nbformat >= 5.4.0",
      "notebook >= 6.4.11",
      "qtconsole >= 5.3.0",
      "traitlets >= 5.2.2",
      "PySimpleGUI >= 4.60.4",
      "PyPDF4 >=1.27.0
## Installation
<b>Consider installation to a virtual environment (pip, conda), since the package uses specific version numbers of several packages.</b>
### Linux
`git clone https://github.com/Mike1712/XRR.git`<br>
`cd XRR`<br>
`pip install ./setup.py`
## Exemplary workflow with example data
There are a few [jupyter notebooks](https://github.com/Mike1712/XRR/tree/main/src/XRR/notebooks) (in progress) to show how the scripts work:<br>
$\quad$ 1. Read raw data: read_data.ipynb<br>
$\quad$ 2. Simulation of layer stacks: layer_stacks.ipynb<br>
$\quad$ 3. Evaluating and plotting experimental data: data_eval.ipynb<br>

