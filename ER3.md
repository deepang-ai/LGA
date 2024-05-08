python train_lw_AIG.py --gnn-type gat --se khopgnn --dataset ico_wallets/Volume --num-layers 2  --dim-hidden 64 --k-hop 1


|          | Eth-ICO (Amount) | Eth-ICO (Times) | Eth-ICO (avgAmount) |
|----------|------------------|-----------------|---------------------| 
| Ethident | 94.05±0.034      | 93.36±0.032     | 94.38±0.028         |
| SAT      | 95.07±1.887      | 94.85±1.236     | 94.72±1.265         |


|          | Eth-Mining (Amount) | Eth-Mining (Times) | Eth-Mining (avgAmount) | 
|----------|---------------------|--------------------|------------------------| 
| Ethident | 86.38±0.049         | 87.00±0.040        | 85.30±0.057            |
| SAT      | 88.01±1.666         | 87.38±1.892        | 88.23±1.630            |



|          | Eth-Exchange (Amount) | Eth-Exchange (Times) | Eth-Exchange (avgAmount) | 
|----------|-----------------------|----------------------|--------------------------|
| Ethident | 93.16±0.021           | 93.55±0.027          | 93.34±0.022              |
| SAT      | 91.55±0.921           | 92.10±0.974          | 91.191.086               |


|          | Eth-Phish&Hack (Amount) | Eth-Phish&Hack (Times) | Eth-Phish&Hack (avgAmount) | 
|----------|-----------|------------------|---------------------|
| Ethident | 97.93     | 97.58            | 97.98               |
| Ours     | 97.95     | 97.73            | 98.00               |