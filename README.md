# Transfer of Structural Knowledge from Synthetic Languages

NB: For anonymity reasons, author-related information is contained in `SECRET.txt`.

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

To generate Parquet files for the synthetic languages, run:
```bash
python generate_synthetic_languages.py <language_name>
```
The process takes a couple of hours for each language. Then you can upload the files to HuggingFace to use them in further stages.

To pre-train the language model use `training.ipynb` notebook and to fine-tune use `finetuning.ipynb`.
The notebooks will output the values used for Table 1, together with some visualizations of per token losses.
Pre-training takes ~4h for each language and fine-tuning takes ~1h in each direction, on T4 GPU.

To reproduce Table 2 (Results on Tiny-Cloze benchmark) and Figures 2-4 (Spectrum, Clustering, Probes), first run:
```bash
python feature_engineering.py
```
It will produce `word_features.csv` in ~10 minutes on CPU.

Then run:
```bash
python generate_figures.py
```
It will take several more minutes.


