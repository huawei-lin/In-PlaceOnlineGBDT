# In-Place Online GBDT

## Quick Start
### Installation guide
Run the following commands to build Online Boost from source:
```
mkdir build
cd build
cmake ..
make
cd ..
```

### Datasets 

Two datasets are provided under `data/` folder: [pendigits](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits) and [optdigits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits).

### Training
```
./onlineGBDT_train -method robustlogit -data ./data/optdigits.train.csv -v 0.1 -J 20 -iter 100 -feature_split_sample_rate 0.1
```
This command will generate `optdigits.train.csv_robustlogit_J20_v0.1.model` that used for the following unlearning or tuning.

### Incremental Learning
Here we would like to tune (add) a new dataset `./data/optdigits.tune.csv` to the `optdigits.train.csv_robustlogit_J20_v0.1.model`.
Please note that it need to load the original data of the model.
```
./onlineGBDT_tune -method robustlogit -data ./data/optdigits.train.csv -tuning_data_path ./data/optdigits.tune.csv -model optdigits.train.csv_robustlogit_J20_v0.1.model
```

### Decremental Learning 
Here we would like to unlearn (delect) the 9-th data sample from the `optdigits.train.csv_robustlogit_J20_v0.1.model`.
Please note that it need to load the original data of the model.
```
for i in {0..9}; do echo ${i}; done > unids.txt
./onlineGBDT_unlearn -data ./data/optdigits.train.csv -model optdigits.train.csv_robustlogit_J20_v0.1.model -unlearning_ids_path unids.txt
```


### Predicting
Here we would like to evaluate these three models in `./data/optdigits.test.csv`.
```
./onlineGBDT_predict -data ./data/optdigits.test.csv -model optdigits.train.csv_robustlogit_J20_v0.1.model
./onlineGBDT_predict -data ./data/optdigits.test.csv -model optdigits.train.csv_robustlogit_J20_v0.1_tune.model
./onlineGBDT_predict -data ./data/optdigits.test.csv -model optdigits.train.csv_robustlogit_J20_v0.1_unlearn.model
```

## More Configuration Options:
#### Data related:
* `-data_min_bin_size` minimum size of the bin
* `-data_max_n_bins` max number of bins (default 1000)
* `-data_path, -data` path to train/test data
#### Tree related:
* `-tree_max_n_leaves`, `-J` (default 20)
* `-tree_min_node_size` (default 10)
* `-tree_n_random_layers` (default 0)
* `-feature_split_sample_rate` (default 1.0)
#### Model related:
* `-model_data_sample_rate` (default 1.0)
* `-model_feature_sample_rate` (default 1.0)
* `-model_shrinkage`, `-shrinkage`, `-v`, the learning rate (default 0.1)
* `-model_n_iterations`, `-iter` (default 1000)
* `-model_n_classes` (default 0) the max number of classes allowed in this model (>= the number of classes on current dataset, 0 indicates do not set a specific class number)
* `-model_name`, `-method` regression/lambdarank/mart/abcmart/robustlogit/abcrobustlogit (default robustlogit)
#### Unlearning related:
* `-unlearning_ids_path` path to unlearning indices
* `-lazy_update_freq` (default 1)
#### Tuning related:
* `-tuning_data_path` path to tuning data
#### Parallelism:
* `-n_threads`, `-threads` (default 1)
#### Other:
* `-save_log`, 0/1 (default 0) whether save the runtime log to file
* `-save_model`, 0/1 (default 1)
* `-save_prob`, 0/1 (default 0) whether save the prediction probability for classification tasks
* `-save_importance`, 0/1 (default 0) whether save the feature importance in the training

