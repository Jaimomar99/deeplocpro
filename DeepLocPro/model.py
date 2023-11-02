from esm import pretrained
import torch
import torch.nn as nn
import pkg_resources
from .prodeeploc_model import ProDeepLocModel
from typing import List

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
                model = ProDeepLocModel(1280, 256, 6)
                model.load_state_dict(torch.load(pkg_resources.resource_filename(__name__,f"models/checkpoints/model_{i}_{j}"), map_location="cpu"))
                model.eval()
                subcel_clfs.append(model)

        self.subcel_clfs = nn.ModuleList(subcel_clfs)

    def forward(self, embeddings, masks):
        '''Embed, get all predictions and aggregate.'''
        x_loc_preds, x_attnss = [], []
        for clf in self.subcel_clfs: 
            x_pred, x_attns = clf(embeddings, masks)
            x_loc_preds.append(torch.softmax(x_pred, dim=-1))
            x_attnss.append(x_attns)
        return torch.stack(x_loc_preds).mean(0).cpu().numpy(), torch.stack(x_attnss).mean(0).cpu().numpy()
    

    def embed_batch(self, sequences: List[str], repr_layers=[33]):
        '''Embed a list of sequences using ESM2. Return padded tensors and a mask.'''

        batch_converter = self.esm_alphabet.get_batch_converter()
        embeddings, masks= [], []
        for seq in sequences:
            seqs = list([("seq", s) for s in [seq]])
            labels, strs, toks = batch_converter(seqs)

            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)
            out = self.esm_model(toks, repr_layers=[33], return_contacts=False)["representations"][33]
            out = out.cpu()
            out[out!=out] = 0.0 # set nan to zeros

            res = out.transpose(0,1)[1:-1] 
            seq_embedding = res[:,0]
            mask = torch.ones(seq_embedding.shape[0])
            embeddings.append(seq_embedding)
            masks.append(mask)

        embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
        masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True)
        return embeddings, masks