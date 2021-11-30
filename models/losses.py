import torch
import torch.nn as nn
import torch.nn.functional as F

def get_pos_and_neg_mask(batch_size, sim_ij, sim_ji):
    pos_mask = torch.cat([sim_ij, sim_ji], dim=0)
    neg_mask = (~torch.eye(batch_size * 2.0, batch_size * 2.0, dtype=bool)).float()
    return pos_mask, neg_mask


class NTXent(nn.Module):
  """ Normalized Temperature-scaled Cross Entropy loss (contrastive) """

  def __init__(self, batch_size, tau=1.0) -> None:
    super(NTXent, self).__init__()
    self.batch_size = batch_size
    self.temperature = torch.tensor(tau)

  def forward(self, e_i, e_j):
    """ forward pass of embedding batch pairs with corresponding indices"""
    z_i, z_j = F.normalize(e_i, dim=1), F.normalize(e_j, dim=1)

    representations = torch.cat([z_i, z_j], dim=0)
    sim_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    sim_ij = torch.diag(sim_matrix, self.batch_size)
    sim_ji = torch.diag(sim_matrix, -self.batch_size)
    positives, negatives = get_pos_and_neg_mask(self.batch_size, sim_ij, sim_ji)

    nominator = torch.exp(positives / self.temperature)
    denominator = negatives * torch.exp(sim_matrix / self.temperature)

    loss = torch.sum(-torch.log(nominator / torch.sum(denominator, dim=1))) / (2.0 * self.batch_size)

    return loss