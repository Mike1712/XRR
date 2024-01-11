# XRR
Python code to read raw XRR data recorded at beamlines ID31 (ESRF), P08 (DESY, PETRA III), BL9 (DELTA) and evaluate data.<br>
You can also simulate XRR profiles and electron density profiles by providing a layer stack of your sample system.<br>
This code is developed and tested on Fedora Linux. 
## Requirements

## Installation
### Linux
`git clone https://github.com/Mike1712/XRR.git`<br>
`cd XRR`<br>
`pip install ./setup.py`
## Exemplary workflow with example data
There are a few [jupyter notebooks](src/XRR/notebooks) to show how the scripts work:<br>
$\quad$ 1. Read raw data: read_data.ipynb<br>
$\quad$ 2. Simulation of layer stacks: layer_stacks.ipynb<br>
$\quad$ 3. Evaluating and plotting experimental data: data_eval.ipynb<br>



