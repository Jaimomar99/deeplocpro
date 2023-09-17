import pickle
import torch
from Bio import SeqIO
import re

class BatchedSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, data_df):
        self.data_df = data_df

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        return self.data_df["Sequence"][idx], self.data_df["ACC"][idx]

    def get_batch_indices(self, toks_per_batch, extra_toks_per_seq=0):
        sizes = [(len(s), i) for i, s in enumerate(self.data_df["Sequence"])]
        sizes.sort(reverse=True)
        batches = []
        buf = []
        max_len = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            max_len = 0
        start = 0
        #start = random.randint(0, len(sizes))
        for j in range(len(sizes)):
            i = (start + j) % len(sizes)
            sz = sizes[i][0]
            idx = sizes[i][1]    
            sz += extra_toks_per_seq
            if (max(sz, max_len) * (len(buf) + 1) > toks_per_batch):
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(idx)

        _flush_current_buf()
        return batches

    @staticmethod
    def collate_fn(batches):
        '''expects batch of (sequence, name) tuples. Returns list of sequences and list of names'''
        seqs, names = [], []
        for batch in batches:
            seqs.append(batch[0])
            names.append(batch[1])

        return seqs, names



def read_fasta(fastafile):
    """Parse a file with sequences in FASTA format and store in a dict"""
    proteins = list(SeqIO.parse(fastafile, "fasta"))
    res = {}
    for prot in proteins:
        res[str(prot.id)] = str(prot.seq)
    return res
