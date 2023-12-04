import warnings
warnings.filterwarnings("ignore")
import sys
from .data import read_fasta, BatchedSequenceDataset
import os
import argparse
import pandas as pd
import numpy as np
import torch
import time
from .model import EnsembleModel
from .utils import convert_label2string, generate_attention_plot_files, slugify, remap_probabilities


def run_model(embed_dataloader, args, test_df):
    prob_dict = {}
    attn_dict = {}
    with torch.no_grad():
        model = EnsembleModel().to(args.device)
        for i, (sequences, names) in enumerate(embed_dataloader):
            embeddings, masks = model.embed_batch(sequences)
            ml_out, attn_out = model(embeddings.to(args.device), masks.to(args.device))

            prob_dict[names[0]] = ml_out
            attn_dict[names[0]] = attn_out
    
    prob_df = pd.DataFrame(prob_dict.items(), columns=['ACC', 'label'])
    attn_df = pd.DataFrame(attn_dict.items(), columns=['ACC', 'Attention'])
    pred_df = test_df.merge(prob_df).merge(attn_df)
    return pred_df


def main(args):
    fasta_dict = read_fasta(args.fasta)
    test_df = pd.DataFrame(fasta_dict.items(), columns=['ACC', 'Sequence'])
    labels = ["Cell wall & surface",
                "Extracellular",
                "Cytoplasmic",
                "Cytoplasmic Membrane",
                "Outer Membrane",
                "Periplasmic"]
    
    #get_batch_indices(0,) means that each batch will only be a single sequence.
    embed_dataset = BatchedSequenceDataset(test_df)
    embed_batches = embed_dataset.get_batch_indices(0, extra_toks_per_seq=1)
    embed_dataloader = torch.utils.data.DataLoader(embed_dataset, batch_sampler=embed_batches)
    pred_df = run_model(embed_dataloader, args, test_df)
    pred_df["label"] = pred_df["label"].apply(lambda x: x[0])

    if args.group in ['archaea', 'positive']:
        # periplasm = 5, outer = 4
        pred_df["label"] = pred_df["label"].apply(remap_probabilities)
    else:
        pass # no remapping for gram neg

    #make it beautiful
    pred_df['Class_int'] = pred_df['label'].apply(lambda x: np.argmax(x))
    pred_df['Localization'] = pred_df['Class_int'].apply(lambda x: convert_label2string(x))
        

    if args.plot:
        generate_attention_plot_files(pred_df, args.output)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    csv_out = f'{args.output}/results_{timestr}.csv'

    #extend output, create csv to
    prob_df = pred_df["label"].apply(pd.Series)
    prob_df.columns = labels
    prob_df = prob_df.round(4)
    pred_df = pred_df.join(prob_df)
    pred_df.drop(['Sequence','Attention', 'Class_int', 'label'], axis=1).to_csv(csv_out, float_format='%.4f')

    #plot by sequence
    for prot_ind,prot in pred_df.iterrows():
        seq_id = prot['ACC']
        seq_aa = prot['Sequence']
        if args.plot:
            attention_path = os.path.join(args.output, 'alpha_{}'.format(slugify(seq_id)))
            alpha_out = "{}.csv".format(attention_path)
            alpha_values = pred_df["Attention"][prot_ind][0, :]
            with open(alpha_out, 'w') as alpha_f:
                alpha_f.write("AA,Alpha\n")
                for aa_index,aa in enumerate(seq_aa):
                    alpha_f.write("{},{}\n".format(aa,str(alpha_values[aa_index])))


def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f","--fasta", type=str, required=True, help="Input protein sequences in the FASTA format"
    )
    parser.add_argument(
        "-o","--output", type=str, default="./outputs/", help="Output directory"
    )
    parser.add_argument(
        "-p","--plot", default=False, action='store_true', help="Plot attention values"
    )
    parser.add_argument(
        "-d","--device", type=str, default="cpu", choices=['cpu', 'cuda', 'mps'], help="One of cpu, cuda, mps"
    )
    parser.add_argument(
        "-g","--group", type=str, default="any", choices=['any', 'archaea', 'positive', "negative"], help="Prevent outer membrane & periplasm prediction when Archaea/positive"
    )
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    main(args)
