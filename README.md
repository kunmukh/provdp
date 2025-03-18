# ProvDP: Differential Privacy for System Provenance Dataset

Reproducibility artifacts for the paper ProvDP: Differential Privacy for System Provenance Dataset.

## Environment Setup
1. Create a new virtual environment: `pip -m venv venv` or `conda create --name provdp python=3.11`
2. Activate virtual environment: `source venv/bin/activate` or `conda activate provdp`
3. Install dependencies via `pip install -r requirements.txt`

Note: Scripts are ran using `python -m` to avoid having to manipulate the `PYTHONPATH` environment variable.

## Folder Structure

| Folder | Description|
| -------|-----------|
| `DP`        | Folder containing the code and data to execute the DP algorithm. |
| `GNN-based-IDS` | Folder containing the code and data files for IDS execution. |

## ProvDP

- To run the ProvDP pipeline, run the following command. More information on the arguments can be found in the
`parse_args()` function in [`perturb.py`](src/cli/perturb.py).
- 500 benign provenance graphs from FiveDirections dataset: [FiveDirections/benign-500](DP/FiveDirections/benign-500/) directory.
- Output of ProvDP: 500 benign DP provenance graphs from FiveDirections dataset: [FiveDirections/benign-500-dp_N=500_epsilon=1.0_delta=0.5_alpha=0.7_beta=0.1_gamma=0.1_eta=0.1_k=0.1](DP/FiveDirections/benign-500-dp_N=500_epsilon=1.0_delta=0.5_alpha=0.7_beta=0.1_gamma=0.1_eta=0.1_k=0.1/) directory.

```shell
$ python -m src.cli.perturb -i ../FiveDirections/benign-500 -o ../FiveDirections/benign-500-dp --epsilon 1 --alpha 0.7 --beta 0.1 --gamma 0.1 --eta 0.1
```

### ProvNinja [[1]](#references)

* Driver script for [ProvNinja](GNN-based-IDS/ProvNinja/provninja.py), which is an GAT based IDS that detects anomalous graphs.
* Pre-processed original benign and anomalous graphs available in [FiveDirections](GNN-based-IDS/Data/FiveDirections/) and benign processed by ProvDP and anomalous graphs available in [FiveDirections-DP](GNN-based-IDS/Data/FiveDirections-DP/)  directory.

```shell
$ python provninja.py binary gat -if 768 -hf 50 -lr 0.001 -e 5 -n 5 -bs 8 -bi -dl <xx> -tvcm -bdst 0.99 -at 0.2
```

```shell
$ python provninja.py -if 768 -hf 16 -lr 0.001 -e 30 -n 3 -bs 16 -dl 'C:\Users\prov-dp\GNN-based-IDS\Data\FiveDirections-DP\\' --device cpu -bdst 0.99 -at 0.1
```

## Citing us

```
@inproceedings{mukherjee2025provDP,
	title        = {ProvDP: Differential Privacy for System Provenance Dataset},
	author       = {Kunal Mukherjee and Jonathan Yu and Partha De and Dinil Mon Divakaran},
	year         = 2025,
	booktitle    = {23rd Conference on Applied Cryptography and Network Security (ACNS)},
	series       = {ACNS '25}
}
```

## References 

[1] K. Mukherjee, et al., “_Evading Provenance-Based ML Detectors with Adversarial System Actions_,” in
USENIX Security Symposium (SEC), 2023. <br>