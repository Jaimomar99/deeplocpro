"""
Alternative entry point to work better with DTU Health Tech way of passing arguments.
This follows what is already online for DeepLoc 2.0
"""
import argparse
import DeepLocPro
import json
import pandas as pd
import os
import unicodedata
import re


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')

class FakeArgs():
    def __init__(self, **kwargs):
        # store all kwargs as attributes.
        self.__dict__.update(kwargs)

    def __repr__(self):
        return str(self.__dict__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--FASTA_PATH", type=str, help="Input protein sequences in the FASTA format"
    )
    parser.add_argument(
        "--OUTPUT_PATH", type=str, default="./outputs/", help="Output directory"
    )
    parser.add_argument(
        "--NUM_THREADS",
        default=os.cpu_count(),
        type=int,
        help="Number of threads to use."
    )
    parser.add_argument(
        "--GROUP", 
        default="any",
        choices=['any', 'archaea', 'positive', 'negative'],
        type=str,
        help="Group of input."
    )
    parser.add_argument(
        "--JOBID", type=str, help="Job id from webface"
    )
    parser.add_argument(
        "--FORMAT", default='long', choices=['long','short'],type=str, help="Output format"
    )
    args = parser.parse_args()


    input_args = FakeArgs(
        fasta=args.FASTA_PATH,
        output=os.path.join(args.OUTPUT_PATH, args.JOBID),
        device="cpu",
        plot=False if args.FORMAT == 'short' else True,
        group=args.GROUP,
    )

    if not os.path.exists(input_args.output):
        os.mkdir(input_args.output)
    DeepLocPro.deeplocpro.main(input_args)

    #make json - this is based on the biolib postprocessing code

    # find the file that starts with results_ and ends with .csv in args.output
    csv_file = None
    for file in os.listdir(input_args.output):
        if file.startswith('results_') and file.endswith('.csv'):
            csv_file = file
            break
    if csv_file is None:
        print("No results file found in output directory")

    pred_df = pd.read_csv(os.path.join(input_args.output, csv_file))

    # csv_out = '{}/results_{}.csv'.format(args.OUTPUT_PATH,args.JOBID)
    json_data = dict()
    json_data['sequences'] = dict()
    json_data['info'] = dict()
    json_data['csv_file'] = f"/services/DeepLocPro-1.0/tmp/{args.JOBID}/{csv_file}"
    json_data['Localization'] = ["Cell wall & surface",
                "Extracellular",
                "Cytoplasmic",
                "Cytoplasmic Membrane",
                "Outer Membrane",
                "Periplasmic"]



    json_data['info']['size'] = len(pred_df)
    json_data['info']['failedjobs'] = 0
    # json_data['Localization'] = labels
    json_data['format'] = args.FORMAT

    for prot_ind,prot in pred_df.iterrows():
      #idd = str(ids_test[prot]).split("/")
        pred_label = prot['Localization']

        # probability table
        probs = prot[["Cell wall & surface","Extracellular","Cytoplasmic","Cytoplasmic Membrane","Outer Membrane","Periplasmic"]]
 
        # output_md += tabulate(prob_df, tablefmt='github', headers='keys')
        json_data['sequences'][prot['ACC']] = {
            'Prediction':pred_label,
            'Probability': probs.tolist(),
            'Name':prot['ACC']
            }
        #     'Attention': [attention_png, attention_eps]

        if args.FORMAT == 'long':
            attention_path = f"/services/DeepLocPro-1.0/tmp/{args.JOBID}/"+ 'alpha_{}'.format(slugify(prot['ACC']))
            json_data['sequences'][prot['ACC']]['Attention'] = [attention_path+'.png', attention_path+'.csv']


    #   pred_prob = np.around(prot['multilabel'], decimals=4)
    #   csv_prob = np.around(prot['multilabel'], decimals=4)
    #   likelihood = [ '%.4f' % elem for elem in pred_prob.tolist()]
    #   thres_diff = [ '%.4f' % elem for elem in thres_prob.tolist()]
    #   csv_likelihood = csv_prob.tolist()
    #   seq_id = prot['ACC']
    #   seq_aa = prot['Sequence']
    #   attention_path = os.path.join(args.OUTPUT_PATH, 'alpha_{}'.format(slugify(seq_id)))
    #   attention_plt = '/services/DeepLoc-2.0/tmp/{}/alpha_{}'.format(args.JOBID, slugify(seq_id))
    #   json_data['sequences'][seq_id] = {'Predictions':pred_labels,'Signals':pred_signals,'Likelihood':likelihood,'Threshold_difference':thres_diff ,'Attention':["{}.png".format(attention_plt),"{}.eps".format(attention_plt),"{}.csv".format(attention_plt)],'Name':seq_id}

    with open(os.path.join(input_args.output, 'results.json'), 'w') as f:
        json.dump(json_data, f, indent=4)

