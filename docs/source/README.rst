PromptX
=======

Prompt X-ray counterparts of neutron star mergers, for structured relativistic outflows viewed from any observer angle.

Project Description
--------------------

PromptX simulates emission from relativistic outflows (e.g., gamma-ray burst (GRB) jets and magnetar-powered X-ray winds) using phenomenological light curve and spectral models. The code handles Gaussian, power-law, and top-hat outflow profiles, and computes observed properties from arbitrary viewing angles.

The simulation pipeline includes:

- **Jet Emission Modeling**: Structured GRB jets with user-defined profiles for energy and Lorentz factor.
- **Wind Modeling**: X-ray winds powered by magnetar spin-down energy.
- **Normalization and Integration**: Emission is normalized to match a fixed isotropic-equivalent energy (E_iso), Doppler-boosted, and integrated across solid angles.
- **Visualization Tools**: Plotting functions for emission structure, light curves, and spectra.

Features
--------

- Jet and wind emission models with structured energy and Lorentz factor profiles
- Doppler-boosted calculations for on-axis and off-axis observers
- Generation of light curves and spectra for X-rays and gamma-rays

File Structure
--------------

- **`main.py`**: Main driver for setting up and running simulations.
- **`jet.py`**: GRB jet modeling class (`Jet`).
- **`wind.py`**: Magnetar-powered wind modeling class (`Wind`).
- **`helper.py`**: Utility functions for calculations.
- **`examples.py`**: Example scripts for typical usage.

Dependencies
------------

- Python 3.x
- NumPy
- Matplotlib
- SciPy

Installation
------------

Clone the repository:

.. code-block:: bash

    git clone https://github.com/cjules0609/promptx.git
    cd promptx
    pip install -r requirements.txt

Example Usage
-------------

Run default example simulations in scripts/:

.. code-block:: bash

   cd scripts/
   python examples.py
   
Outputs including plots and data files are saved to the `.scripts/output/` directory.

License
-------

MIT License

Acknowledgements
----------------

Chen, Wang, and Zhang (2025), in preparation.
