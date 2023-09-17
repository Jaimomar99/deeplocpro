from esm import pretrained
import torch
import torch.nn as nn
import pkg_resources
from .prodeeploc_model import DeepLocModel
from typing import List

# TODO make sure DeepLocModel return tensor shapes match the aggregation here
# TODO rewrite forward(), replace sigmoid with softmax
# TODO complete embed_batch
class EnsembleModel(nn.Module):
    '''An ensemble model of ESM2 and all checkpoints.'''
    def __init__(self):
        super().__init__()
        self.esm_model, self.esm_alphabet = pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
        self.esm_model.eval()
        subcel_clfs = []
        for i in range(5):
            for j in range(5):
                if i == j:
                    continue
                model = DeepLocModel(1280, 256, 6)
                model.load_state_dict(torch.load(pkg_resources.resource_filename(__name__,f"models/checkpoints/model_{i}_{j}"), map_location="cpu"))
                model.eval()
                subcel_clfs.append(model)

        self.subcel_clfs = nn.ModuleList(subcel_clfs)
        # self.subcel_clfs = nn.ModuleList([DeepLocModel.load_from_checkpoint(pkg_resources.resource_filename(__name__,f"models/checkpoints/model_{i}_1Layer.ckpt"), map_location="cpu").eval() for i in range(5)])
             

    def forward(self, embeddings, masks):#, dct_mat, idct_mat):
        '''Embed, get all predictions and aggregate.'''
        device = self.device
        # x = self.embedding_func(toks.to(self.device), repr_layers=[33])["representations"][33][:, 1:-1].float()
        x_loc_preds, x_attnss = [], [], []
        for clf in self.subcel_clfs):
          x_pred, x_pool, x_attns = clf(embeddings.to(device), masks.to(self.device))
          x_loc_preds.append(torch.sigmoid(x_pred))
          x_attnss.append(x_attns)

        return torch.stack(x_loc_preds).mean(0).cpu().numpy(), torch.stack(x_attnss).mean(0).cpu().numpy()
    


    def embed_batch(self, sequences: List[str], repr_layers=[33]):
        '''Embed a list of sequences using ESM2. Return padded tensors and a mask.'''

        embeddings = []
        for s in sequences:
          #TODO @Jaime add embedding code here. use self.esm_model and self.esm_alphabet

          embeddings.append(embedding)


        embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
        mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True)
      
        return embeddings, mask