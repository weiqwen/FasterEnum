This folder contains all the implementation/codes used in this submission.


Installation
------------

1. We assume Python 3 because it allows for a nicer `begins <https://pypi.org/project/begins/>`__ based interfaces.
2. Run

   .. code-block:: shell

      pip3 install -r requirements

   preferably in a virtual environment


   
Programs
--------

   ** Main folder: contains codes for Figures in Section 2 and 5.

- `blck.py` simulation code for SVP with overshooting/procrastination. This script produces the data for `fig:enumeration-cost-obkz-simulation`
- `call.py` call SVP/HSVP-1.05/SVP with overshooting implementations to obtain timings. This scriptproduces the data for `fig:state-of-the-art-enumeration-cost-observation`
- `chal.py` simulation code for HSVP-1.05. This script produces the data for `fig:svp-challenge`
- `conv.py` convert file formats to each other and do curve fitting
- `cost.py` simulation code for SVP in FPLLL. This script produces the data for `fig:state-of-the-art-enumeration-cost-simulation`
- `impl.py` implementation of BKZ with overshooting/procrastination.
- `qual.py` compare basis qualtiy (rhf: δ, slope: ρ) of different BKZ variants
- `simu.py` BKZ (w and w/o overshooting) simulation of output qualtiy

  
   *** Sub-folder: "code_asymptotic" contains codes for Figures in Section 1 and 4.
- `simu_c_cost.py` interpolation of full enumeration cost for re-examining BKZ/SDBKZ-reduced basis
- `simu_asym.py` generates the Figures in Section 4, studying the behavior of the asymptotic algorithm
