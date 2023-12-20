|          | Eth-ICO (Amount) | Eth-ICO (Times) | Eth-ICO (avgAmount) |
|----------|------------------|-----------------|---------------------| 
| Ethident | 94.05±0.034      | 93.36±0.032     | 94.38±0.028         |
| SAT      | 94.86±1.658 *    | 94.10±1.600     | 94.31±1.444         |


python train_lw_AIG.py --gnn-type gat --se khopgnn --dataset ico_wallets/Volume --num-layers 2  --dim-hidden 64 --k-hop 1 --aggr max


|          | Eth-Mining (Amount) | Eth-Mining (Times) | Eth-Mining (avgAmount) | 
|----------|---------------------|--------------------|------------------------| 
| Ethident | 86.38±0.049         | 87.00±0.040        | 85.30±0.057            |
| SAT      | 87.16±2.018         | 87.24±1.996        | 87.16±1.470            |



|          | Eth-Exchange (Amount) | Eth-Exchange (Times) | Eth-Exchange (avgAmount) | 
|----------|-----------------------|----------------------|--------------------------|
| Ethident | 93.16±0.021           | 93.55±0.027          | 93.34±0.022              |
| SAT      | 91.55±0.921           | 92.10±0.974          | 91.191.086               |


|          | Eth-Phish&Hack (Amount) | Eth-Phish&Hack (Times) | Eth-Phish&Hack (avgAmount) | 
|----------|-------------------------|-------------------|----------------------|
| Ethident | 97.93±0.002             | 97.58±0.004       | 97.98±0.003          |
| SAT      | 94.45±1.849             |                   |                      |