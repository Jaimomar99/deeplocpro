'''
Process DeepLocPro output to a nice markdown file.
'''
import argparse
from tabulate import tabulate
import os
import pandas as pd
import re
import unicodedata


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



def write_fancy_output():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o","--output", type=str, default="./outputs/", help="Output directory"
    )
    parser.add_argument(
        "-p","--plot", default=False, action='store_true', help="Plot attention values"
    )
    args, extra = parser.parse_known_args()

    # find the file that starts with results_ and ends with .csv in args.output
    csv_file = None
    for file in os.listdir(args.output):
        if file.startswith('results_') and file.endswith('.csv'):
            csv_file = file
            break
    if csv_file is None:
        print("No results file found in output directory")
        return

    # read the csv file
    pred_df = pd.read_csv(os.path.join(args.output, csv_file))

    output_md = ""
    output_md += f'## DeepLocPro - Results\n'
    # output_md += f'### Summary of {results["INFO"]["size"]} predicted sequences\n'
    # output_md += f'Predictions list. Use the instruction page for more detailed description of the output page.\n\n'
    output_md += f'Download:\n\n'
    output_md += f'[CSV Summary]({csv_file})\n\n'
    output_md += f'### Predicted Proteins\n'

    for idx, row in pred_df.iterrows():
        output_md += f'#### {row["ACC"]}\n\n'
        output_md += f'**Prediction:** {row["Localization"]}\n\n'

        # probability table
        prob_df = pd.DataFrame(row[["Cell wall & surface","Extracellular","Cytoplasmic","Cytoplasmic Membrane","Outer Membrane","Periplasmic"]]).T
        prob_df.index=['Probability']
        output_md += tabulate(prob_df, tablefmt='github', headers='keys')


        # plot file name: made from ACC using slugify
        plot_file =  'alpha_' + slugify(row["ACC"]) + ".png"

        if args.plot:
            output_md += f'\n\n ![plot]({plot_file})'
            output_md += f'Download: [PNG]({plot_file}) [CSV]({plot_file.replace(".png",".csv")})'
        output_md += f' \n\n ***\n\n'


        output_md += f' \n\n_________________\n\n'

    
    open(os.path.join(args.output,"output.md"), "w").write(output_md)


# def write_fancy_output(results, out_file: str = 'output.md'):

#     output_md = ""

#     output_md += f'## DeepLocPro - Results\n'
#     # output_md += f'### Summary of {results["INFO"]["size"]} predicted sequences\n'
#     # output_md += f'Predictions list. Use the instruction page for more detailed description of the output page.\n\n'
#     output_md += f'Download:\n\n'
#     # output_md += f'[JSON Summary](output.json)\n\n'
#     output_md += f'### Predicted Proteins\n'

#     for name, preds in results["PREDICTIONS"].items():
#         output_md += f'#### {name}\n\n'
#         #output_md += f'**Prediction:** {sequence["Prediction"]}\n\n'
#         #output_md += f'{sequence["CS_pos"]}\n\n'
#         if preds['figure']:
#             output_md += f'\n\n ![plot]({preds["figure"].split("/")[-1]})'

#         output_md += f'\n\n'
#         output_md += tabulate(preds['peptides'], tablefmt='github', headers='keys')
#         output_md += f'\n\n'

#         output_md += f'**Download:**'
#         if preds['figure']:
#             output_md += f' [PNG]({preds["figure"].split("/")[-1]}) '
#         # if sequence['Plot_eps']:
#         #     output_md += f' [EPS]({sequence["Plot_eps"].split("/")[-1]}) / '
#         # if sequence['Plot_txt']:
#         #     output_md += f' [Tabular]({sequence["Plot_txt"].split("/")[-1]})'


    #     # output_md += f' \n\n ***\n\n'
    #     output_md += f' \n\n_________________\n\n'

    # open("output.md", "w").write(output_md)

if __name__ == "__main__":
    write_fancy_output()