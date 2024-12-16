# Hierarchical Network with Local-Global Awareness for Ethereum Account De-anonymization


Official implementation of Hierarchical Network with Local-Global Awareness for Ethereum Account De-anonymization.

## Requirements

```
numpy
scipy
einops
ogb
torch-geometric==2.0.2
scikit-learn==1.3.2
torch==1.9.1
```

## Data
Download data in PYG format from this [page](https://jjzhou.notion.site/Ethident-Data-861199675dc7454eb36157eeee09cf5b) and place it under the path 'datasets_lw_AIG/data/'.


Transform the datasets:
```
python data_transform.py
```

## Usage

```
python train.py
```


