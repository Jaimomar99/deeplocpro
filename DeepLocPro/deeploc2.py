
import warnings
warnings.filterwarnings("ignore")
import onnxruntime
import sys
from .data import read_fasta, BatchedSequenceDataset
import os
import pickle
import argparse
import pandas as pd
import numpy as np
import re
import torch
import time
import pkg_resources
from .model import EnsembleModel
import logging
logging.set_verbosity_error()
from utils import convert_label2string, generate_attention_plot_files


def run_model(embed_dataloader, args, test_df):
    multilabel_dict = {}
    attn_dict = {}
    with torch.no_grad():
        model = EnsembleModel().to(args.device)
        for i, (sequences, names) in enumerate(embed_dataloader):
              
              embeddings, masks = model.embed_batch(sequences)
              ml_out, attn_out = model(embeddings, masks)

            #   multilabel_dict[names] = ml_out
            #   attn_dict[labels[0]] = attn_out
    
    # TODO clean up multilabel stuff. We just want probabilities for each class.

    multilabel_df = pd.DataFrame(multilabel_dict.items(), columns=['ACC', 'multilabel'])
    attn_df = pd.DataFrame(attn_dict.items(), columns=['ACC', 'Attention'])
    pred_df = test_df.merge(multilabel_df)
    
    return pred_df




def main(args):
    fasta_dict = read_fasta(args.fasta)
    test_df = pd.DataFrame(fasta_dict.items(), columns=['ACC', 'Sequence'])
    #TODO @Jaime adapt in correct order
    labels = ["Cytoplasm","Nucleus","Extracellular","Cell membrane","Mitochondrion","Plastid","Endoplasmic reticulum","Lysosome/Vacuole","Golgi apparatus","Peroxisome"]

    
    # TODO don't really see the point of all of this.
    # I think get_batch_indices(0,) means that each batch will only be a single sequence.
    embed_dataset = BatchedSequenceDataset(test_df)
    embed_batches = embed_dataset.get_batch_indices(0, extra_toks_per_seq=1)
    embed_dataloader = torch.utils.data.DataLoader(embed_dataset, batch_sampler=embed_batches)
    pred_df = run_model(embed_dataloader, args, test_df)

    # TODO adapt convert_label2string to work in multiclass
    # 1. get argmax of probs 2. replace argmax integer with name from labels
    # pred_df["Class_MultiLabel"] = pred_df["multilabel"].apply(lambda x: convert_label2string(x, label_threshold))
    pred_df['Class'] = #TODO
    pred_df["multilabel"] = pred_df["multilabel"].apply(lambda x: x[0, 1:])


    if args.plot:
        #TODO ensure this works with new pred_df format
        generate_attention_plot_files(pred_df, args.output)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    csv_out = f'{args.output}/results_{timestr}.csv'
    out_file = open(csv_out,"w")
    out_file.write(f"Protein_ID,Localization,{','.join(labels)}\n")
    

    # TODO adapt all of this to work with multiclass. But we want a format that 
    # is very similar to deeploc, for consistency.
    # but maybe all the below stuff is a waste of time and we can just restructure
    # pred_df a bit and then to_csv()
    for prot_ind,prot in pred_df.iterrows():
        #idd = str(ids_test[prot]).split("/")
        pred_labels = prot['Class']

        order_pred = np.argsort(prot['multilabel'])
        
        if pred_labels == "":
            pred_labels = labels[order_pred[-1]]


        pred_prob = np.around(prot['multilabel'], decimals=4)
        thres_prob = pred_prob-label_threshold[1:]
        thres_prob[thres_prob < 0.0] = 0.0
        thres_max = 1.0 - label_threshold[1:]
        thres_prob = thres_prob / thres_max
        csv_prob = np.around(prot['multilabel'], decimals=4)
        likelihood = [ '%.4f' % elem for elem in pred_prob.tolist()]
        thres_diff = [ '%.4f' % elem for elem in thres_prob.tolist()]
        csv_likelihood = csv_prob.tolist()
        

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
        out_line = ','.join([seq_id,pred_labels.replace(", ","|"),pred_signals.replace(", ","|"),pred_memtypes.replace(", ","|")]+list(map(str,csv_likelihood))+list(map(str,csv_memtype_likelihood[0])))
        out_file.write(out_line+"\n")
    out_file.close()


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
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    main(args)

predict()
