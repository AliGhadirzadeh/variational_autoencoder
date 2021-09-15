Refactored code for training, optimizing and testing different machine learning models.

The file environment.yml is a template for creating a conda environment suitable for running the code

The analysis directory contains files needed to perform the above.
* estimators.py contains estimator and constructor classes
* test_environment.py is a script for single model evaluation
* pipeline.py is a script for automated hyperparameter optimization of an estimator using BayesianCV from skopt


The scripts in the analysis directory requires a data directory containing the EEG-data. To do: add zip-file containing data directory on google docs, make source for data modifiable

To do:
* Add estimators (temporal convolutions, SNAIL)
* GPU compatibility
* Compatibility with different snippet length/overlap