# Cleanroom CFD Demo

A simplified CFD-style 2D cleanroom airflow and heat-transfer demo.

## Features

- SVG-based room geometry
- Obstacle mask generation
- Pressure-projection airflow update
- Temperature advection and diffusion
- Air-sock tracer visualization
- Notebook demo for testing and presentation

## Setup

This project can be installed with Conda using the provided environment file.

Create the environment with:

```bash
conda env create -f environment.yml
```
Activate it with 

```bash
conda activate cleanroom-cfd
```
## Run
Install requirements, then open `notebooks/demo.ipynb`.