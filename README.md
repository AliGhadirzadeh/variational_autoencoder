Refactored code for training, optimizing and testing different machine learning models.

The file environment.yml is a template for creating a conda environment suitable for running the code

The analysis directory contains files needed to perform the above.
* models.py contains models
* test_environment.py is a script for single model evaluation
* hp_optimize.py is a script for automated hyperparameter optimization of models using gaussian processes
* Models are created using a Network interface (containing functionality specific to neural networks in PyTorch) and the Estimator interface (for compatibility with Sci-Kit-type libraries)


The scripts in the analysis directory requires a data directory containing the EEG-data. To do: add zip-file containing data directory on google docs, make source for data modifiable
