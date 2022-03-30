# Welcome

Welcome. This repository contains the relevant code for the paper ["Linguistically Context Aware Text Corpora Bias Measurement"](reports/paper.pdf). In it, we show that the traditional subtractive metric [(Caliskan et al., 2017)](https://www.science.org/doi/10.1126/science.aal4230) used to assess bias between two protected groups in GloVe word embeddings is nonzero by a statistically significant amount in multiple sociological domains. This holds important implications for creating both useful and socially fair language models.

# Environment Setup

To rerun this code, clone the repo into a local folder and create a conda environment [the instructions for creating an environment from a .yml file on the conda website](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

This code was created and tested with [Miniconda](https://docs.conda.io/en/latest/miniconda.html) version 4.9.1.

# Code structure
The structure of this repository loosely follows the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) template:

- The `data` folder contains all data that is read from or written to for this project. All scripts that load/save data do so from/to this folder. The exception to this is figures generated from the visualization notebooks. These figures are stored in `reports`.
- The `src` folder contains all `.py` files, and overall most of the code for processing the data:
  - The `src/data` folder contains the source code for manipulating data: loading the models, defining the experiments, etc.
  - The `src/models` folder contains the source code for the mathematical calculations of the project. This includes all bias calculations, generation of distributions,  and calculations of the statistical metrics of the distributions.
- The `notebooks` folder is the home of all Jupyter notebooks for visualization and late-stage data processing. 
- The `reports` folder contains the figures that are saved from the visualization notebooks in `notebooks`.

# Pipeline and Reproduction Procedures

This section is meant to be a full step-by-step tutorial. Experienced developers should be able to follow individual steps without excessive modification of source code.

## Model format

Caliskan et al., 2017 completed 2 experiments. The results published in the main paper were from an experiment using a [GloVe format](https://nlp.stanford.edu/projects/glove/) Common Crawl model. An additional experiment used a [word2vec model trained on Google News](https://code.google.com/archive/p/word2vec/). Because we focus on replicating the main experiment, the code here loads GloVe vectors. 

Note, however, that the load pipeline converts GloVe vectors to word2vec vectors, and the processing in `src/models` processes word2vec vectors. So, developers should find the source code clear enough to modify if running our algorithm on a word2vec model is desired. 

## Loading data

After [setting up your environment](#environment-setup), The first reproduction step is cloning this repository and clearing your `data` folder. Create a new folder in `data` called `external` and place your (GloVe) model in this folder. 

Navigate to the home directory and run:

```bash
cd src/data && python convert_glove.py
```

(Note: because of the way the file paths are coded in the source, it is not possible to simply run `python src/data/convert_glove.py` or the program will not be able to find your saved data.)

This script converts the GloVe vectors into the word2vec format necessary to run the rest of the processing. We recommend compressing the vectors (for details, see the links inside `convert_glove.py`). Without compression, the Common Crawl model takes about 10 minutes to load on a standard laptop (which occurs in all notebooks and `src/model` scripts). The compression process itself only takes a few minutes, but after compression, loading the compressed vectors is lightning-fast.

## Defining Experiments

The words that define each target sets and attribute sets must be manually coded and added to a file `src/data/<model_name>_experiment_definitions.py` For example,`src/data/glove_840B_experiment_definitions.py`.

Once you have a model and the experiment definitions defined in the format laid out in this file, run

```bash
cd src/data && python glove_840B_experiment_definitions.py
```
This script converts the hard-coded experiments into a format easily processed later in the pipeline.

## Bias Values for Individual Words

To calculate the bias values for each of the experiments, run

```bash
cd src/models && python run_singleword_experiments.py
```

**Approximate runtime for all 10 experiments on a personal machine with i7 processor/8GB RAM:** 5 minutes







