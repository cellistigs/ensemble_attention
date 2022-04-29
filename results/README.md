# This data contains benchmarks for single models and ensembles that we will use to compare our new results against in the code `kl_weight.py`. 
# Of interest are the following folders:


- 'benchmark_ensemble_data': models trained using the interp_ensembles code, corresponding to models labeled "ResNet18", and "ResNet18.{7,8,9,10}" in the dictionary `all_dataindices` in the script `plot_metrics` with the key `cifar10.1`. This grounds our comparison in the same models we studied in the previous paper.  
  - we calculate the variance in ind and ood predictions across individual models here, and also form "bootstrap ensembles" of size 4 which leave out one ensemble member each. We calculate the variance across these bootstrap ensembles and plot them as well.

- `simultrained_ensemble_data`: models trained using this code, in particular the `ensemble` module in combination with script `run.py`. These differ from the ensembles shown above in that they do not differ in minibatch ordering, and therefore represent a direct baseline for ensemble performance, although it doesn't look as if ensemble performance varies too much between these ensembles and the others we already have. These ensembles have the following timestamps: 
  - ensemble 0: 2022-04-28/16-01-07
  - ensemble 1: 2022-04-28/16-53-14
  - ensemble 2: 2022-04-28/17-43-04
  - ensemble 3: 2022-04-28/18-33-03
