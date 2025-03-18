# ProvNinja [[1]](#references)

This folder contains the ProvNinja GNN-based IDS scripts and helper utilities.

```shell
$ python provninja.py -if 768 -hf 16 -lr 0.001 -e 30 -n 3 -bs 16 -dl 'C:\Users\prov-dp\GNN-based-IDS\Data\FiveDirections' --device cpu -bdst 0.99 -at 0.1
```

```shell
$ python provninja.py -if 768 -hf 16 -lr 0.001 -e 30 -n 3 -bs 16 -dl 'C:\Users\prov-dp\GNN-based-IDS\Data\FiveDirections-DP' --device cpu -bdst 0.99 -at 0.1
```

### Output

#### ProvNinja using Original Dataset
```shell
$ python provninja.py -if 768 -hf 16 -lr 0.001 -e 30 -n 3 -bs 16 -dl 'C:\Users\prov-dp\GNN-based-IDS\Data\FiveDirections' --device cpu -bdst 0.99 -at 0.1   
2025-03-17 22:54:52,959 | INFO  | using C:\Users\prov-dp\GNN-based-IDS\Data\FiveDirections as input data directory
2025-03-17 22:54:52,960 | INFO  | 3 Layer GAT. Input Feature Size: 768. Hidden Layer Size(s): 16. Loss Rate: 0.001. Batch Size: 16
2025-03-17 22:54:52,960 | INFO  | Input Device: cpu
2025-03-17 22:54:52,960 | INFO  | Benign Down Sampling: 0.99
2025-03-17 22:54:52,960 | INFO  | Variable Prediction Threshold for Anomalous graphs have been enabled & set to 0.1
2025-03-17 22:54:52,960 | INFO  | Training on 30 epochs...
Processed 350/350 benign graphs (100.00%)
Processed 14/14 anomaly graphs (100.00%)
Done saving data into cached files.
Processed 50/50 benign graphs (100.00%)
Processed 2/2 anomaly graphs (100.00%)
Done saving data into cached files.
Processed 100/100 benign graphs (100.00%)
Processed 4/4 anomaly graphs (100.00%)
Done saving data into cached files.
2025-03-17 22:55:26,481 | INFO  | Length of dataset: 520
2025-03-17 22:55:26,490 | INFO  | Training on Device: cpu
2025-03-17 22:55:26,491 | INFO  | Number benign in training dataset: 350
2025-03-17 22:55:26,492 | INFO  | Number anomaly in training dataset: 14
2025-03-17 22:55:26,492 | INFO  | Number benign in validation dataset: 50
2025-03-17 22:55:26,492 | INFO  | Number anomaly in validation dataset: 2
2025-03-17 22:55:26,493 | INFO  | Number benign in test dataset: 100
2025-03-17 22:55:26,493 | INFO  | Number anomaly in test dataset: 4
2025-03-17 22:55:26,493 | INFO  | # Parameters in model: 258897
2025-03-17 22:55:26,494 | INFO  | # Trainable parameters in model: 258897
2025-03-17 22:55:26,494 | INFO  | Stratified sampler enabled
2025-03-17 22:55:26,496 | INFO  | Computed weights for loss function: tensor([ 0.5200, 13.0000])
2025-03-17 22:56:24,181 | INFO  | Epoch 0: Training Accuracy: 0.42308, Average Training Loss: 0.97024, Validation Accuracy: 0.59615, Average Validation Loss: 0.10532
2025-03-17 22:57:21,891 | INFO  | Epoch 1: Training Accuracy: 0.55220, Average Training Loss: 0.21789, Validation Accuracy: 0.57692, Average Validation Loss: 0.09447
2025-03-17 22:58:19,901 | INFO  | Epoch 2: Training Accuracy: 0.60714, Average Training Loss: 0.09631, Validation Accuracy: 0.80769, Average Validation Loss: 0.08905
2025-03-17 22:59:17,352 | INFO  | Epoch 3: Training Accuracy: 0.73352, Average Training Loss: 0.08049, Validation Accuracy: 0.82692, Average Validation Loss: 0.08426
...
2025-03-17 23:23:12,182 | INFO  | Epoch 29: Training Accuracy: 0.87363, Average Training Loss: 0.34434, Validation Accuracy: 0.94231, Average Validation Loss: 0.08473
2025-03-17 23:23:26,494 | INFO  | === test stats ===
2025-03-17 23:23:26,494 | INFO  | Number Correct: 104
2025-03-17 23:23:26,494 | INFO  | Number Graphs in test Data: 104
2025-03-17 23:23:26,494 | INFO  | test accuracy: 1.00000
2025-03-17 23:23:26,495 | INFO  | [[100   0]
 [  0   4]]
2025-03-17 23:23:26,497 | INFO  |               recision    recall  f1-score   support

      Benign       1.00      1.00      1.00       100
     Anamoly       1.00      1.00      1.00         4

    accuracy                           1.00       104
   macro avg       1.00      1.00      1.00       104
weighted avg       1.00      1.00      1.00       104
```
#### ProvNinja using DP Dataset
```shell
$ python provninja.py -if 768 -hf 16 -lr 0.001 -e 30 -n 3 -bs 16 -dl 'C:\Users\prov-dp\GNN-based-IDS\Data\FiveDirections-DP\\' --device cpu -bdst 0.99 -at 0.1
2025-03-17 21:12:43,221 | INFO  | using C:\Users\prov-dp\GNN-based-IDS\Data\FiveDirections-DP\ as input data directory
2025-03-17 21:12:43,222 | INFO  | 3 Layer GAT. Input Feature Size: 768. Hidden Layer Size(s): 16. Loss Rate: 0.001. Batch Size: 16
2025-03-17 21:12:43,222 | INFO  | Input Device: cpu
2025-03-17 21:12:43,222 | INFO  | Benign Down Sampling: 0.99
2025-03-17 21:12:43,222 | INFO  | Variable Prediction Threshold for Anomalous graphs have been enabled & set to 0.1
2025-03-17 21:12:43,222 | INFO  | Training on 30 epochs...
Processed 350/350 benign graphs (100.00%)
Processed 14/14 anomaly graphs (100.00%)
Done saving data into cached files.
Processed 50/50 benign graphs (100.00%)
Processed 2/2 anomaly graphs (100.00%)
Done saving data into cached files.
Processed 100/100 benign graphs (100.00%)
Processed 4/4 anomaly graphs (100.00%)
Done saving data into cached files.
2025-03-17 21:13:19,020 | INFO  | Length of dataset: 520
2025-03-17 21:13:19,029 | INFO  | Training on Device: cpu
2025-03-17 21:13:19,030 | INFO  | Number benign in training dataset: 350
2025-03-17 21:13:19,031 | INFO  | Number anomaly in training dataset: 14
2025-03-17 21:13:19,031 | INFO  | Number benign in validation dataset: 50
2025-03-17 21:13:19,031 | INFO  | Number anomaly in validation dataset: 2
2025-03-17 21:13:19,032 | INFO  | Number benign in test dataset: 100
2025-03-17 21:13:19,032 | INFO  | Number anomaly in test dataset: 4
2025-03-17 21:13:19,032 | INFO  | # Parameters in model: 258897
2025-03-17 21:13:19,033 | INFO  | # Trainable parameters in model: 258897
2025-03-17 21:13:19,034 | INFO  | Stratified sampler enabled
2025-03-17 21:13:19,035 | INFO  | Computed weights for loss function: tensor([ 0.5200, 13.0000])
2025-03-17 21:14:29,681 | INFO  | Epoch 0: Training Accuracy: 0.78571, Average Training Loss: 0.50133, Validation Accuracy: 0.94231, Average Validation Loss: 0.37205
2025-03-17 21:15:40,425 | INFO  | Epoch 1: Training Accuracy: 0.90385, Average Training Loss: 0.18996, Validation Accuracy: 0.94231, Average Validation Loss: 0.05457
2025-03-17 21:16:50,928 | INFO  | Epoch 2: Training Accuracy: 0.92308, Average Training Loss: 0.06454, Validation Accuracy: 0.98077, Average Validation Loss: 0.04273
2025-03-17 21:18:00,140 | INFO  | Epoch 3: Training Accuracy: 0.92033, Average Training Loss: 0.04588, Validation Accuracy: 0.98077, Average Validation Loss: 0.02993
...
2025-03-17 21:49:40,330 | INFO  | Epoch 29: Training Accuracy: 0.93956, Average Training Loss: 0.00305, Validation Accuracy: 1.00000, Average Validation Loss: 0.16428
2025-03-17 21:50:00,061 | INFO  | === test stats ===
2025-03-17 21:50:00,061 | INFO  | Number Correct: 100
2025-03-17 21:50:00,062 | INFO  | Number Graphs in test Data: 104
2025-03-17 21:50:00,062 | INFO  | test accuracy: 0.96154
2025-03-17 21:50:00,065 | INFO  | [[97   3]
 [  1   3]]
2025-03-17 21:50:00,068 | INFO  |               precision    recall  f1-score   support

      Benign       0.99      0.97      0.98       100
     Anamoly       0.50      0.75      0.60         4

    accuracy                           0.96       104
   macro avg       0.74      0.86      0.79       104
weighted avg       0.97      0.96      0.97       104
```
## File Structure

* [dataloaders/](dataloaders/)
  * [BaseDataloader.py](dataloaders/BaseDataloader.py)
      * This is a [DGL Dataset](https://docs.dgl.ai/en/0.6.x/guide/data.html) loader. It will take in the `.csv` files
        generated by [jsonToCsv.py] and parse them into a format consumable by DGL (ie a DGL heterograph). This class 
        serves as the base class for all other dataloaders.
      * More specifically, it will go through all the different relations (edges) in the dataset and put these relations
        into a DGL heterograph. From there, we import user defined node/edge attributes from the .csv files into the DGL
        heterograph.
      * The constructor function has documentation on how to instantiate a ProvDataset object.
      * You probably do not need to worry about this file.
  * [AnomalyBenignDataset.py](dataloaders/AnomalyBenignDataset.py)
    * Loads data for binary classification (anomaly/benign)
* [nn_types/](nn_types/)

  * [gat.py](nn_types/gat.py)
    * Houses the code for relational graph attention network  [Graph Attention Layer](https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/9_gat.html)
    * We use `HeteroRGATLayer` as a building block for RGAT. It is a single layer in the RGAT.
    * We use `KLayerHeteroRGAT` as a K number of layers layer RGAT
    * See provninja.py to run GAT network

* [provninja.py](provninja.py)
    * Houses the code for running ProvNinja.
    * In order to run the gnn:
        * For binary classification, your data must be set up in a way such that benign data is is split into train, test, and validation folders
          * Each train, test, and validation folder must have an anomaly/ and benign/ subfolder containing the respective graphs
        * Usage: `python3 provninja.py (binary|multi) (gcn|gat|mlp) -dl <dataset_dir_path> -if <input feature size> -hf <hidden feature size> -lr <loss rate> -e <# epochs> -n <# layers> -bs <batch_size> --device <device> [-bdst <percentage to downsample>] [-at <anomaly threshold>] `          
          * You can specify the device you want to run the models on with `--device <device>`. By default, the model will use GPU if it's available. (`<device>` parameter can be cpu, cuda, cuda:1, etc..)
          * You can add percentage for benign downsampling for training data `-bdst <percentage to downsample [0.0-1.0]>` flag
          * You can add a prediction threshold for classification of anomaly graphs `-at <anomaly threshold [0.0-1.0]>` flag for binary classification

* [gnnUtils.py](gnnUtils.py)
  * Houses utility functions for provninja.py.

* [samplers.py](samplers.py)
  * Contains Samplers & Batch Samplers for the DGL's GraphDataLoader class.

## Outputs

* [run](runs/) - Stats for each run is stored in the `runs` folder and can be viewed using tensorboard by doing `tensorboard --logdir=runs`
* [models](models/) - Model for each run is stored in the `models` folder
* [logs](logs/) - Logs for each run is stored in the `logs` folder

## References 

[1] K. Mukherjee, et al., “_Evading Provenance-Based ML Detectors with Adversarial System Actions_,” in
USENIX Security Symposium (SEC), 2023. <br>