import math
import pytorch_lightning as pl
from esm import Alphabet, FastaBatchedDataset, pretrained
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import pkg_resources
from .prodeeploc_model import DeepLocModel

# TODO rewrite to ESM2
# TODO set checkpoint names in subcel_clfs
# TODO make sure DeepLocModel return tensor shapes match the aggregation here
# TODO replace sigmoid with softmax
class ESM1bE2E(pl.LightningModule):
    def __init__(self):
        super().__init__()
        model, alphabet = pretrained.load_model_and_alphabet("esm1b_t33_650M_UR50S")
        self.embedding_func = model.eval()
        # subcel_cfs needs to be ModuleList of DeepLocModel
        self.subcel_clfs = nn.ModuleList([ESM1bFrozen.load_from_checkpoint(pkg_resources.resource_filename(__name__,f"models/models_esm1b/{i}_1Layer.ckpt"), map_location="cpu").eval() for i in range(5)])
             

    def forward(self, toks, lens, non_mask):#, dct_mat, idct_mat):
        # in lightning, forward defines the prediction/inference actions
        device = self.device
        x = self.embedding_func(toks.to(self.device), repr_layers=[33])["representations"][33][:, 1:-1].float()
        x_loc_preds, x_attnss = [], [], [], []
        for i in range(5):
          x_pred, x_pool, x_attns = self.subcel_clfs[i].predict(x, lens.to(self.device), non_mask[:, 1:-1].to(self.device))
          x_loc_preds.append(torch.sigmoid(x_pred))
          x_attnss.append(x_attns)

        return torch.stack(x_loc_preds).mean(0).cpu().numpy(), torch.stack(x_attnss).mean(0).cpu().numpy()
