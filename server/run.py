'''
Run deeplocpro and make fancy output.
'''

import json
from tabulate import tabulate

def write_fancy_output(results, out_file: str = 'output.md'):

    output_md = ""

    output_md += f'## DeepLocPro - Results\n'
    # output_md += f'### Summary of {results["INFO"]["size"]} predicted sequences\n'
    # output_md += f'Predictions list. Use the instruction page for more detailed description of the output page.\n\n'
    output_md += f'Download:\n\n'
    # output_md += f'[JSON Summary](output.json)\n\n'
    output_md += f'### Predicted Proteins\n'

    for name, preds in results["PREDICTIONS"].items():
        output_md += f'#### {name}\n\n'
        #output_md += f'**Prediction:** {sequence["Prediction"]}\n\n'
        #output_md += f'{sequence["CS_pos"]}\n\n'
        if preds['figure']:
            output_md += f'\n\n ![plot]({preds["figure"].split("/")[-1]})'

        output_md += f'\n\n'
        output_md += tabulate(preds['peptides'], tablefmt='github', headers='keys')
        output_md += f'\n\n'

        output_md += f'**Download:**'
        if preds['figure']:
            output_md += f' [PNG]({preds["figure"].split("/")[-1]}) '
        # if sequence['Plot_eps']:
        #     output_md += f' [EPS]({sequence["Plot_eps"].split("/")[-1]}) / '
        # if sequence['Plot_txt']:
        #     output_md += f' [Tabular]({sequence["Plot_txt"].split("/")[-1]})'


        # output_md += f' \n\n ***\n\n'
        output_md += f' \n\n_________________\n\n'

    open("output.md", "w").write(output_md)