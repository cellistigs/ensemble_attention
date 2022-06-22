# This data contains benchmarks for single models and ensembles that we will use to compare our new results against in the code `kl_weight.py`. 
# Of interest are the following folders:


- 'benchmark_ensemble_data': models trained using the interp_ensembles code, corresponding to models labeled "ResNet18", and "ResNet18.{7,8,9,10}" in the dictionary `all_dataindices` in the script `plot_metrics` with the key `cifar10.1`. This grounds our comparison in the same models we studied in the previous paper.  
  - we calculate the variance in ind and ood predictions across individual models here, and also form "bootstrap ensembles" of size 4 which leave out one ensemble member each. We calculate the variance across these bootstrap ensembles and plot them as well.

- `simultrained_ensemble_data`: models trained using this code, in particular the `ensemble` module in combination with script `run.py`. These differ from the ensembles shown above in that they do not differ in minibatch ordering, and therefore represent a direct baseline for ensemble performance, although it doesn't look as if ensemble performance varies too much between these ensembles and the others we already have. These ensembles have the following timestamps: 
  - ensemble 0: 2022-04-28/16-01-07
  - ensemble 1: 2022-04-28/16-53-14
  - ensemble 2: 2022-04-28/17-43-04
  - ensemble 3: 2022-04-28/18-33-03

- `test_kl_data`: models that were trained with gamma KL weightings. each directory indicates the weighting used. The following timestamps can be used to identify each hyperparameter setting with a model training instance. 
  - gamma_0: 04-29-22/20-02-14
  - gamma_0.001: 04-29-22/21-57-06 
  - gamma_0.01: 04-29-22/22-47-17 
  - gamma_0.1: 04-29-22/23-37-36 
  - gamma_0.2: 04-30-22/00-27-51 
  - gamma_0.5: 04-29-22/19-00-46 
  - gamma_0.8: 04-29-22/17-19-08 
  - gamma_1: 04-29-22/15-34-14 
  - gamma_1.2: 04-29-22/16-29-01 
  - gamma_1.5: 04-30-22/16-04-38 
  - gamma_1.8: 04-30-22/16-54-53 
  - gamma_2: 04-29-22/18-10-20 
  - gamma_3: 04-30-22/01-18-15 
  - gamma_5: 04-29-22/20-52-35 
