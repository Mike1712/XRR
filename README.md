# XRR
Python code to read raw XRR data recorded at beamlines ID31 (ESRF), P08 (DESY, PETRA III), BL9 (DELTA) and evaluate data.<br>
You can also simulate XRR profiles and electron density profiles by providing a layer stack of your sample system.<br>
This code is developed and tested on Fedora Linux. 
## Requirements
     "numpy >= 1.22.4",<br>
     "scipy >= 1.8.1",<br>
     "pandas >= 1.4.2",<br>
     "Pillow >= 9.1.1",<br>
     "matplotlib >= 3.5.2",<br>
     "matplotlib-inline >= 0.1.3",<br>
     "plotly >= 5.8.0",<br>
      "IPython >= 8.4.0",<br>
      "ipykernel >= 6.13.0",<br>
      "ipywidgets >= 7.7.0",<br>
      "jupyter_client >= 7.4.7",<br>
      "jupyter_core >=  5.0.0",<br>
      "jupyter_server >= 1.23.3",<br>
      "jupyterlab >= 3.5.0",<br>
      "nbclient >= 0.5.13",<br>
      "nbconvert >= 7.2.5",<br>
      "nbformat >= 5.4.0",<br>
      "notebook >= 6.4.11",<br>
      "qtconsole >= 5.3.0",<br>
      "traitlets >= 5.2.2",<br>
      "PySimpleGUI >= 4.60.4",<br>
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

