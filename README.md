Spatial-Spectral Analysis (SSA)
================================

Calibrating tumor growth and invasion parameters from tumor biopsy tissues using 2-point correlation function (2pCF) and power spectral density (PSD).


1. Overview
-----------
This repository contains Python code for:
  1) Compute the two-point correlation function (2pCF) from a set of 2D coordinates representing cancer cells in a biopsy image.
  2) Compute the power spectral density (PSD) from that 2pCF.

The methodology is fully described in the paper:
Pasetto S., Montejo M., Zahid M.U., Rosa M., Gatenby R., Schlicke P., Diaz R., and Enderling H.
"Calibrating tumor growth and invasion parameters with spectral-spatial analysis of cancer biopsy tissues."
npj Systems Biology and Applications (2024).
DOI: https://doi.org/10.1038/s41540-024-00439-0

No calibration of the reaction-diffusion (R–D) model parameters is provided by matching the PSD of a simulated R–D equation to the PSD of the real biopsy data. Eq. (7) in the paper can achieve the inference exercise.

Contact email for questions: stefano.pasetto.usa@gmail.com


2. Repository Structure
-----------------------
SSA/
  ssa/
    __init__.py             # Makes "ssa/" a Python package
    ssa.py                  # Main Python code (2pCF, PSD, and R–D pipeline)
  tests/
    test_ssa.py             # Unit tests for 2pCF & PSD
  requirements.txt          # Dependencies (to be created)
  README.md                 # This file
  LICENSE

- ssa/ssa.py includes:
    GPU detection. Falls back to CPU if CuPy not available.
    Functions:
      two_point_correlation_function(...)
      compute_psd(...)
    A main() function that demonstrates reading input coordinates, computing 2pCF & PSD, and plotting/saving results.
- tests/test_ssa.py: Basic tests checking that:
    2pCF ~ 1 for a uniform random distribution.
    No NaNs or errors for small synthetic datasets.
    PSD returns non-negative real values.


3. Installation
---------------
1) Clone or download this repository:
   git clone https://github.com/StefanoPasetto/Spatial-Spectral-Analysis.git   
   cd SSA

3) Install dependencies:
   pip install -r requirements.txt

4) (Optional) GPU Acceleration
   If your environment is configured, this code can run on GPGPUs (both NVIDIA and AMD) via CuPy. Otherwise, it automatically defaults to CPU (NumPy).

   - NVIDIA: A working CUDA toolkit is required.
   - AMD: Requires ROCm/HIP versions of CuPy.

   If you do not install CuPy, the code uses CPU-based NumPy.


4. Input Data Format
--------------------
No patient data are included in this repository. You must provide your coordinate data, for instance, from histology, multiplex IF scans, 10x Genomics Visium, Slide-seq, or MERFISH, etc.

The code expects a text file (default name: data.dat) with lines of the form {x1, y1, gene1_exp, gene2_exp, ...}, ...
This code version accounts only for the spatial distribution, and no spatial transcriptomics is considered:

{12.345, 67.890}
{13.250, 68.130}
{14.111, 62.220}
...

That is:
  - Each line has an X and Y coordinate inside braces { }, separated by a comma.
  - The code in ssa.py reads a file named data.dat by default, but you can adjust this as needed.


5. Usage
--------
1) Place your coordinate file (e.g., data.dat) in ssa/, or update the path in ssa.py.
2) Run the script:

   cd ssa
   python ssa.py

3) Outputs:
   - 2pCF.dat (distance vs. correlation)
   - PSD.dat (wavenumber vs. power)
   - Plots of 2pCF and PSD will be shown on screen.

4) Importing functions in your code:
   from ssa.ssa import two_point_correlation_function, compute_psd

   Suppose you have a NumPy array of shape (N, 2) called "points".
   r_vals, xi_vals = two_point_correlation_function(points, r_min=0, r_max=50, bins=20, box_size=(100,100))
   k_vals, psd_vals = compute_psd(xi_vals, klmin=0.001, klmax=1.0)


6. Testing
----------
We provide a small test suite in tests/test_ssa.py. From the SSA root folder:
   python -m unittest discover -s tests -p "test_*.py"

You should see something like:
  Ran 3 tests in 0.2s
  OK

Each test checks:
  - 2pCF is ~ 1 for a uniform random distribution.
  - We get valid results (no NaNs) for a tiny synthetic dataset.
  - PSD has the correct shape and no negative values.


7. References
-------------
Pasetto S. et al. (2024). "Calibrating tumor growth and invasion parameters with spectral-spatial analysis of cancer biopsy tissues." npj Systems Biology and Applications. https://doi.org/10.1038/s41540-024-00439-0


8. License
----------
MIT License

Copyright (c) 2025 Stefano Pasetto

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


9. Contact & Acknowledgments
----------------------------
- Primary contact/questions:
  stefano.pasetto.usa@gmail.com

- Please cite the above npj Systems Biology and Applications paper if you are using or building on this work.

- For more details on methodology, please look at the original publication.

