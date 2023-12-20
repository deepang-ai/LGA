import torch
import os
import sys

from sat.models import GraphTransformer
print(torch.__version__)
print(torch.cuda.is_available())

print(type(GraphTransformer))

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# print(sys.path)

