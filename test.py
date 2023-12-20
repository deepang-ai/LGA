import torch
import numpy as np

print(torch.__version__)
print(torch.cuda.is_available())



c = np.array(([1,2],[3,4],[4,5]))
f = np.repeat(c,3)
print(f)
print('c形状：',c.shape)
