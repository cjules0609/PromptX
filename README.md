# PromptX

<img align="left" src="https://github.com/cjules0609/PromptX/blob/main/assets/logo.svg" alt="PromptX Logo" width="350"/>

![Python Version](https://img.shields.io/badge/python-3.6%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
[![DOI](https://zenodo.org/badge/970365196.svg)](https://doi.org/10.5281/zenodo.16923796)

<div align="left">

**[Latest Release Notes](CHANGELOG.md#v030---2025-09-06)** | **[Full Changelog](CHANGELOG.md)** | **[Install Now](#installation)**
</div>

**PromptX** calculates prompt X-ray emission from neutron star mergers for arbitrarily structured relativistic outflows viewed from any viewing angle.

<br clear="left"/>





## Overview

PromptX simulates the emission from gamma-ray burst (GRB) jets and magnetar-powered winds using phenomenological models. It supports Gaussian, power-law, and top-hat outflows, calculating spectra and light curves observed from arbitrary angles.

## Features

- Structured jet and wind emission modeling
- Doppler-boosted observer-frame calculations
- Light curve and spectrum generation across X-ray and gamma-ray bands

## System Compatibility

The code has currently only been tested on macOS. It may work on other operating systems, but has not been extensively tested or officially supported. All documentation assumes a Unix-based system (e.g., macOS or Linux).

## Installation

```bash
git clone https://github.com/cjules0609/promptx.git
cd promptx
pip install -r requirements.txt
```

## Example Usage

To run default examples:

   ```bash
   cd scripts/
   python examples.py
   ```

Outputs are saved to ./scripts/out directory.

## Documentation
Full documentation is available locally by following these steps:

1. Install Sphinx and ReadTheDocs theme:
   ```bash 
   pip install sphinx sphinx-rtd-theme
   ```

2. Build the HTML
   ```bash
   sphinx-build -b html docs/source docs/build
   ```

3. Open the HTML documentation
   ```bash
   open docs/build/index.html
   ```

## License
MIT License

## Acknowledgements
[Chen, Wang, and Zhang (2025)](https://doi.org/10.48550/arXiv.2505.01606).
