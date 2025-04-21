# PromptX
Prompt X-ray counterparts of neutron star mergers for any observer

## Project Description

This project simulates the emission from relativistic gamma-ray burst (GRB) jets and magnetar-central-engine-powereed X-ray winds, taking into account the structure of the jets and observer viewing angles. The goal is to model jet and wind emission viewed from arbitrary lines of sight, with a focus on calculating the observed light curves and spectra in X-rays. The code integrates emission profiles over solid angles, normalizes the energy distribution to match a fixed isotropic-equivalent energy (E_iso), and applies Doppler boosting to compute observer-frame quantities.

The main components of the simulation include:

- **Jet Emission Model**: A simulation of GRB jet structure using Gaussian, power-law, or other profiles to model energy distribution and Lorentz factor.
- **Wind Model**: A simulation of magnetar-powered wind emission.
- **Normalization and Integration**: Doppler-boosted emission is normalized for different viewing angles, and the results are integrated to generate light curves and spectra.
- **Visualization**: The project includes various plotting functions to visualize the energy distribution, emission profiles, and observed light curves.

## Features

- Jet and wind emission modeling with structured energy profiles.
- Doppler-boosted emission calculation for various viewing angles.
- Generation of GRB light curves and spectra in X-rays and gamma-rays for on-axis and off-axis observers.

## File Structure

- **`main.py`**: The main script for running simulations, including setting up the jet and wind models, running the observer calculations, and saving results.
- **`Jet.py`**: Contains the `Jet` class, which models the GRB jet emission, including normalization and Doppler boosting.
- **`Wind.py`**: Contains the `Wind` class, which models magnetar-powered wind emission.
- **`helper.py`**: Functions for handling generic calculations and tasks.
- **`examples.py`**: Example functions for intrductory use.

## Dependencies

- Python 3.x
- NumPy
- Matplotlib
- SciPy

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/cjules0609/promptx.git
   cd promptx
   ```


2. Install the required dependencies:
   ```bash
   pip install -e requirements.txt
   ```
## Example Usage

1. To run a selection of default examples contained in examples,py:

   ```bash
   python examples.py
   ```

3. The output, including plots and data files, will be saved in the ./output/ directory.

## License
This project is licensed under the MIT License.

## Acknowledgements
Chen, Wang, and Zhang (2025) in prep.
