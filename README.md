# Welcome

This repository accompanies the paper <paper_name_here>. 

# Environment Setup

This code was created and tested with [Miniconda](https://docs.conda.io/en/latest/miniconda.html) version 4.9.1.

To rerun this code, clone the repo into a local folder. Then, follow [the instructions for creating an environment from a .yml file on the conda website](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

# Code structure
The structure of this repository loosely follows the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) template:

- The `data` folder contains all data that is read from or written to for this project. All scripts that load/save data do so from/to this folder. The exception to this is figures generated from the visualization notebooks. These figures are stored in `reports`.
- The `notebooks` folder is the home of all [Jupyter](https://jupyter.org/) notebooks. In this project, most of the code for processing the data is in the `src` folder. Notebooks contains the notebooks for visualization. And, if you wish to explore the data in notebooks yourself, the `notebooks` folder is a good place to keep this. You can find more detailed organizational suggestions in the Cookiecutter  documentation.
- The `reports` folder contains the figures that are saved from the visualization notebooks in `notebooks`.
- The `src` folder contains all `.py` files:
  - The `src/data` folder contains the source code for manipulating data: loading the models, defining the experiments, etc.
  - The `src/models` folder contains the source code for the mathematical calculations of the project. This includes all bias calculations, generation of distributions,  and calculations of the statistical metrics of the distributions.

*Note on visualization code: By the standard Cookiecutter template, visualization functions are stored in the `src/visualization` folder. We removed this folder because the visualizations for this project are non-standard (in particular the visualization for single-word biases), so we choose to keep all visualization code in notebooks to allow for easier modification of the visualizations.

# 