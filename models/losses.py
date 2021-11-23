import torch.nn as nn


def gather():
  pass

def accuracy():
  pass

class NTXent(nn.Module):
  """ Normalized Temperature-scaled Cross Entropy loss (contrastive) that has support for distributed training with data parallelism """

  def __init__(self, batch_size, temperature=0.5) -> None:
    super(NTXent, self).__init__()
    self.batch_size = batch_size
    self.temp = temperature

  def forward(self, e_i, e_j):
    pass