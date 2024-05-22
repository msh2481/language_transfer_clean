# Transfer of Structural Knowledge from Synthetic Languages

## Installation
To install the required dependencies, please follow these steps:

1. Install Poetry, a dependency management tool for Python. You can install it by following the instructions on the [Poetry website](https://python-poetry.org/docs/#installation).

2. Once Poetry is installed, navigate to the project directory and run the following command to install the dependencies:
   ```bash
   poetry install
   ```

3. After the dependencies are installed, activate the virtual environment by running:
   ```bash
   poetry shell
   ```

## Reproducing results

To reproduce Table 2 (Results on Tiny-Cloze benchmark) and Figures 2-3 (Spectrum and Clustering), run the following command:
```bash
python generate_figures.py
```

It takes several minutes to run on Apple M1 processor.

TODO: Training and Figure 1
TODO: Figure 4
