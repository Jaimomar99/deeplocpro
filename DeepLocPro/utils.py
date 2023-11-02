import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
import unicodedata
import re
import numpy as np

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


def letter_at(letter, x, y, yscale=1, ax=None):
    '''Draws a letter at a given location in data coordinates.'''
    fp = FontProperties(family= "Arial", weight="bold")
    globscale = 1.35
    LETTERS = { "A" : TextPath((-0.305, 0), "A", size=1, prop=fp),
            "R" : TextPath((-0.384, 0), "R", size=1, prop=fp),
            "N" : TextPath((-0.35, 0), "N", size=1, prop=fp),
            "D" : TextPath((-0.366, 0), "D", size=1, prop=fp),
            "B" : TextPath((-0.366, 0), "B", size=1, prop=fp),
            "C" : TextPath((-0.366, 0), "C", size=1, prop=fp),
            "Q" : TextPath((-0.366, 0), "Q", size=1, prop=fp),
            "E" : TextPath((-0.366, 0), "E", size=1, prop=fp),
            "Z" : TextPath((-0.366, 0), "Z", size=1, prop=fp),
            "G" : TextPath((-0.366, 0), "G", size=1, prop=fp),
            "H" : TextPath((-0.366, 0), "H", size=1, prop=fp),
            "I" : TextPath((-0.2, 0), "I", size=1, prop=fp),
            "L" : TextPath((-0.366, 0), "L", size=1, prop=fp),
            "K" : TextPath((-0.366, 0), "K", size=1, prop=fp),
            "M" : TextPath((-0.366, 0), "M", size=1, prop=fp),
            "F" : TextPath((-0.366, 0), "F", size=1, prop=fp),
            "P" : TextPath((-0.366, 0), "P", size=1, prop=fp),
            "S" : TextPath((-0.366, 0), "S", size=1, prop=fp),
            "T" : TextPath((-0.366, 0), "T", size=1, prop=fp),
            "W" : TextPath((-0.42, 0), "W", size=1, prop=fp),
            "Y" : TextPath((-0.366, 0), "Y", size=1, prop=fp),
            "V" : TextPath((-0.366, 0), "V", size=1, prop=fp),
            "X" : TextPath((-0.3, 0), "X", size=1, prop=fp)
            }
    COLOR_SCHEME = { "A" : "#d7191c",
                "R" : "#2c7bb6",
                "N" : "#abd9e9",
                "D" : "#abd9e9",
                "B" : "black",
                "C" : "#abd9e9",
                "Q" : "#abd9e9",
                "E" : "#abd9e9",
                "Z" : "black",
                "G" : "#fdae61",
                "H" : "#abd9e9",
                "I" : "#fdae61",
                "L" :"#fdae61",
                "K" : "#abd9e9",
                "M" : "#fdae61",
                "F" : "#fdae61",
                "P" : "#fdae61",
                "S" : "#abd9e9",
                "T" : "#abd9e9",
                "W" : "#fdae61",
                "Y" : "#abd9e9",
                "V" : "#fdae61",
                "X" : "black"
                }
    if letter not in COLOR_SCHEME:
        letter="X"
    text = LETTERS[letter]
    t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
        mpl.transforms.Affine2D().translate(x,y) + ax.transData
    p = PathPatch(text, lw=0, fill=True,fc=COLOR_SCHEME[letter],  transform=t)
    if ax is not None:
        ax.add_artist(p)

    return p

def create_logo(alphas, seqs, acc, acc_file, out_path, max_len, offset):
    aa_dict = ["A","R","N","D","B","C","Q","E","Z","G","H","I","L","K","M","F","P","S","T","W","Y","V","X"]
    list_positions = np.arange(0,len(seqs)+max_len,max_len)
    alphas_norm = (alphas - alphas.min()) / (alphas.max() - alphas.min())+offset
    n_chunck = len(list_positions)-1
    font = {'family' : 'serif',
        'size'   : 40}
    mpl.rc('font', **font)
    mpl.rcParams['axes.linewidth'] = 4
    fig, axs = plt.subplots(n_chunck,1, figsize=(60, (n_chunck*7)+3), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = 0.3-(n_chunck*0.01), wspace=.001, top=0.83+(n_chunck*0.012))
    for index, pos in enumerate(list_positions[:-1]):
        seq = seqs[list_positions[index]:list_positions[index+1]]
        curr_alpha = alphas_norm[list_positions[index]:list_positions[index+1]]
        x = 1
        if n_chunck == 1:
            axs_chunck = axs
        else:
            axs_chunck = axs[index]

        for ii, aa in enumerate(seq):
            if aa not in aa_dict:
                aa = "X"
            letter_at(aa, x,0, curr_alpha[ii], axs_chunck)
            x += 1

        axs_chunck.axhline(y=offset, color='black', linestyle='--')
        axs_chunck.set_xticks(np.arange(0,max_len+10, 10))
        axs_chunck.set_xlim((0,max_len+1))
        axs_chunck.set_xticklabels(np.arange(max_len*index,max_len*index+max_len+10, 10))
        axs_chunck.set_ylim((0,1+offset))
        axs_chunck.set_yticks(np.array([0.0,0.25,0.50,0.75,1.0])+offset)
        axs_chunck.set_yticklabels(["0.00","0.25","0.50","0.75","1.00"])
        axs_chunck.tick_params('both', length=20, width=4, which='major')

    plt.xlabel('Sequence position')
    fig.text(0.085, 0.5, 'Sorting signal importance', va='center', rotation='vertical')
    plt.xlabel('Sequence position')
    fig.text(0.085, 0.5, 'Sorting signal importance', va='center', rotation='vertical')
    fig.savefig(os.path.join(out_path, f"alpha_{acc_file}.png"), bbox_inches='tight')



def convert_label2string(x):
    labels = ["Cell wall & surface",
                "Extracellular",
                "Cytoplasmic",
                "Cytoplasmic Membrane",
                "Outer Membrane",
                "Periplasmic"]
    return labels[x]


def generate_attention_plot_files(output_df, out_path):
    for i in range(len(output_df)):
        acc = output_df["ACC"][i]
        acc_file = slugify(acc)
        y = output_df["Attention"][i][0]
        seqs = output_df["Sequence"][i]
        create_logo(y, seqs, acc, acc_file, out_path, 100, 0.15)


def remap_probabilities(probs):
    '''Sum outer membrane and periplasm into extracellular'''
    probs_fixed = probs.copy()
    probs_fixed[1] = probs_fixed[1] + probs_fixed[4] + probs_fixed[5]
    probs_fixed[4] = 0
    probs_fixed[5] = 0 
    return probs_fixed
