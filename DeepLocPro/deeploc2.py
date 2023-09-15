
import warnings
warnings.filterwarnings("ignore")
import onnxruntime
import sys
from DeepLoc2.data import *
import os
import pickle
import argparse
import pandas as pd
import numpy as np
import re
import torch
import time
import pkg_resources
from DeepLoc2.model import *
from DeepLoc2.utils import *
from transformers import T5EncoderModel, T5Tokenizer, logging
logging.set_verbosity_error()


def run_model_esm1b(embed_dataloader, args, test_df):
    multilabel_dict = {}
    signaltype_dict = {}
    attn_dict = {}
    memtype_dict = {}
    with torch.no_grad():
        model = ESM1bE2E().to(args.device)
        for i, (toks, lengths, np_mask, labels) in enumerate(embed_dataloader):
              ml_out, attn_out, st_out, mt_out = model(toks, lengths, np_mask)
              multilabel_dict[labels[0]] = ml_out
              signaltype_dict[labels[0]] = st_out
              attn_dict[labels[0]] = attn_out
              memtype_dict[labels[0]] = mt_out

    multilabel_df = pd.DataFrame(multilabel_dict.items(), columns=['ACC', 'multilabel'])
    signaltype_df = pd.DataFrame(signaltype_dict.items(), columns=['ACC', 'signaltype'])
    attn_df = pd.DataFrame(attn_dict.items(), columns=['ACC', 'Attention'])
    memtype_df = pd.DataFrame(memtype_dict.items(), columns=['ACC', 'memtype'])

    pred_df = test_df.merge(multilabel_df).merge(signaltype_df).merge(attn_df).merge(memtype_df)
    
    return pred_df

def run_model_prott5(embed_dataloader, args, test_df):
    multilabel_dict = {}
    signaltype_dict = {}
    attn_dict = {}
    memtype_dict = {}
    with torch.no_grad():
        model = ProtT5E2E().to(args.device)
        for i, (toks, lengths, np_mask, labels) in enumerate(embed_dataloader):
              ml_out, attn_out, st_out, mt_out = model(toks, lengths, np_mask)
              multilabel_dict[labels[0]] = ml_out
              signaltype_dict[labels[0]] = st_out
              attn_dict[labels[0]] = attn_out
              memtype_dict[labels[0]] = mt_out

    multilabel_df = pd.DataFrame(multilabel_dict.items(), columns=['ACC', 'multilabel'])
    signaltype_df = pd.DataFrame(signaltype_dict.items(), columns=['ACC', 'signaltype'])
    attn_df = pd.DataFrame(attn_dict.items(), columns=['ACC', 'Attention'])
    memtype_df = pd.DataFrame(memtype_dict.items(), columns=['ACC', 'memtype'])

    pred_df = test_df.merge(multilabel_df).merge(signaltype_df).merge(attn_df).merge(memtype_df)
    
    return pred_df




def main(args):
    fasta_dict = read_fasta(args.fasta)
    test_df = pd.DataFrame(fasta_dict.items(), columns=['ACC', 'Sequence'])
    labels = ["Cytoplasm","Nucleus","Extracellular","Cell membrane","Mitochondrion","Plastid","Endoplasmic reticulum","Lysosome/Vacuole","Golgi apparatus","Peroxisome"]
    signals = ["Signal peptide", "Transmembrane domain", "Mitochondrial transit peptide", "Chloroplast transit peptide", "Thylakoid luminal transit peptide", "Nuclear localization signal", "Nuclear export signal", "Peroxisomal targeting signal"]
    memtypes = ["Peripheral", "Transmembrane", "Lipid anchor", "Soluble"]

    
    if args.model == "Fast":
        def clip_middle(x):
            if len(x)>1022:
                x = x[:511] + x[-511:]
            return x
        test_df["Sequence"] = test_df["Sequence"].apply(lambda x: clip_middle(x))
        alphabet_path = pkg_resources.resource_filename('DeepLoc2',"models/ESM1b_alphabet.pkl")

        with open(alphabet_path, "rb") as f:
            alphabet = pickle.load(f)
        #alphabet = Alphabet(proteinseq_toks)
        embed_dataset = FastaBatchedDatasetTorch(test_df)
        embed_batches = embed_dataset.get_batch_indices(0, extra_toks_per_seq=1)
        embed_dataloader = torch.utils.data.DataLoader(embed_dataset, collate_fn=BatchConverter(alphabet), batch_sampler=embed_batches)
        pred_df = run_model_esm1b(embed_dataloader, args, test_df)
        label_threshold = np.array([0.45380859, 0.46953125, 0.52753906, 0.64638672, 0.52368164, 0.63730469, 0.65859375, 0.62783203, 0.56484375, 0.66777344, 0.71679688])
        signal_threshold = np.array([0.32466422, 0.39748752, 0.47921867, 0.67772838, 0.71795298, 0.48740039, 0.63968924, 0.40770178, 0.61593741])
        memtype_threshold = np.array([0.62, 0.56, 0.65, 0.47])

    else:
        def clip_middle(x):
            if len(x)>4000:
                x = x[:2000] + x[-2000:]
            return x
        test_df["Sequence"] = test_df["Sequence"].apply(lambda x: clip_middle(x))

        alphabet = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
        #alphabet = Alphabet(proteinseq_toks)
        embed_dataset = FastaBatchedDatasetTorch(test_df)
        embed_batches = embed_dataset.get_batch_indices(0, extra_toks_per_seq=1)
        embed_dataloader = torch.utils.data.DataLoader(embed_dataset, collate_fn=BatchConverterProtT5(alphabet), batch_sampler=embed_batches)
        pred_df = run_model_prott5(embed_dataloader, args, test_df)
        label_threshold = np.array([0.45717773, 0.47612305, 0.50136719, 0.61728516, 0.56464844, 0.62197266, 0.63945312, 0.60898438, 0.58476562, 0.64941406, 0.73642578])
        signal_threshold = np.array([0.30484808, 0.47878058, 0.55917172, 0.74695907, 0.79056934, 0.53644955, 0.61476384, 0.38718303, 0.62338418])
        memtype_threshold = np.array([0.60, 0.51, 0.82, 0.50])

    pred_df["Class_MultiLabel"] = pred_df["multilabel"].apply(lambda x: convert_label2string(x, label_threshold))
    pred_df["Class_SignalType"] = pred_df["signaltype"].apply(lambda x: convert_signal2string(x, signal_threshold))
    pred_df["Class_Memtype"] = pred_df["memtype"].apply(lambda x: convert_memtype2string(x, memtype_threshold))
    pred_df["multilabel"] = pred_df["multilabel"].apply(lambda x: x[0, 1:])

    if args.plot:
        generate_attention_plot_files(pred_df, args.output)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    csv_out = '{}/results_{}.csv'.format(args.output,timestr)
    out_file = open(csv_out,"w")
    out_file.write("Protein_ID,Localizations,Signals,{},{}\n".format(",".join(labels),",".join(memtypes)))

    for prot_ind,prot in pred_df.iterrows():
        #idd = str(ids_test[prot]).split("/")
        pred_labels = prot['Class_MultiLabel']
        pred_signals = prot['Class_SignalType']
        pred_memtypes = prot['Class_Memtype']
        order_pred = np.argsort(prot['multilabel'])
        order_memtype_pred = np.argsort(prot['memtype'])
        
        if pred_labels == "":
            pred_labels = labels[order_pred[-1]]
        if pred_memtypes == "":
            pred_memtypes = memtypes[order_memtype_pred[-1]]

        pred_prob = np.around(prot['multilabel'], decimals=4)
        thres_prob = pred_prob-label_threshold[1:]
        thres_prob[thres_prob < 0.0] = 0.0
        thres_max = 1.0 - label_threshold[1:]
        thres_prob = thres_prob / thres_max
        csv_prob = np.around(prot['multilabel'], decimals=4)
        likelihood = [ '%.4f' % elem for elem in pred_prob.tolist()]
        thres_diff = [ '%.4f' % elem for elem in thres_prob.tolist()]
        csv_likelihood = csv_prob.tolist()
        
        pred_memtype_prob = np.around(prot['memtype'], decimals=4)[0]
        thres_memtype_prob = pred_memtype_prob-memtype_threshold
        thres_memtype_prob[thres_memtype_prob < 0.0] = 0.0
        thres_memtype_max = 1.0 - memtype_threshold
        thres_memtype_prob = thres_memtype_prob / thres_memtype_max
        csv_memtype_prob = np.around(prot['memtype'], decimals=4)
        likelihood_memtype = [ '%.4f' % elem for elem in pred_memtype_prob.tolist()]
        thres_memtype_diff = [ '%.4f' % elem for elem in thres_memtype_prob.tolist()]
        csv_memtype_likelihood = csv_memtype_prob.tolist()

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
        "-m","--model",
        default="Fast",
        choices=['Accurate', 'Fast'],
        type=str,
        help="Model to use."
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
