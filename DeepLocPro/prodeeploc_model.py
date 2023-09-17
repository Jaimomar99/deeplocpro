import torch
import torch.nn as nn


class DeepLocModel(nn.Module):
    '''Single-label localization prediction model'''
    def __init__(self, embedding_dim: int=1280, mlp_dim: int=256, num_classes: int=9, drop: float=0.1):
        
        super().__init__()

        self.initial_ln = nn.LayerNorm(embedding_dim)

        self.attention_net = nn.Linear(embedding_dim, 1)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, num_classes)
        )

        self.drop_layer = nn.Dropout(p=drop)

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor):
        """Predict the location from the precomputed embeddings.
        Args:
            embeddings (torch.Tensor): precomputed embeddings. (batch_size, seq_len, embedding_dim)
            mask (torch.Tensor): Padding mask for embeddings. 0 where padded.
        """
        # embeddings = embeddings.to(torch.float16)
        embeddings = self.initial_ln(embeddings.float())

        embeddings = self.drop_layer(embeddings)

        # compute the attention weight of each sequence position
        attention_weights = self.attention_net(embeddings.float()) # (batch_size, seq_len, 1) // float16 to float 32

        # set masked positions to a large negative number so that their softmax will be 0
        #unsqueeze(-1) adds a new dimension after the last index
        attention_weights = attention_weights.masked_fill(mask.unsqueeze(-1) == 0, -1e9)

        # softmax, make weights sum to 1 for each sequence
        attention_weights = nn.functional.softmax(attention_weights, dim=1) # (batch_size, seq_len, 1) #dim=1 rows sum 1

        # weighted sum
        sequence_embeddings = (embeddings * attention_weights).sum(dim=1) # (batch_size, embedding_dim)

        # predict from sequence embeddings
        logits = self.mlp(sequence_embeddings)

        return logits, attention_weights