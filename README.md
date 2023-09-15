DeepLoc 2.1
===========

DeepLoc 2.1 predicts the subcellular localization(s) and membrane association of eukaryotic proteins. DeepLoc 2.1 is a multi-label predictor, which means that is able to predict one or more localizations for any given protein. It can differentiate between 10 different localizations: Nucleus, Cytoplasm, Extracellular, Mitochondrion, Cell membrane, Endoplasmic reticulum, Chloroplast, Golgi apparatus, Lysosome/Vacuole and Peroxisome. Furthermore DeepLoc 2.1 also predicts multi-labels for protein membrane types given by the four labels: Peripheral membrane protein, Transmembrane protein, Lipid anchored protein and Soluble protein (non membrane-bound). Additionally, DeepLoc 2.1 can predict the presence of the sorting signal(s) that had an influence on the prediction of the subcellular localization(s).

The DeepLoc 2.1 tool can be run using two different models.

The 'Accurate' model utilizes the ProtT5-XL-Uniref50 transformer (ProtT5). This model provides a more accurate prediction at the expense of longer computation time due to the size of the model (3 billion parameters). Use case: high-quality prediction for a small number of proteins.
The 'Fast' model utilizes the 33-layer ESM transformer (ESM1b). This smaller model (650 million parameters) has the advantage of a faster computation time with a slight decrease in accuracy compared to the ProtT5 model. Use case: high-throughput prediction for a larger number of proteins.

The DeepLoc 2.1 server requires protein sequence(s) in fasta format, and can not handle nucleic acid sequences.

Publication
------------

Vineet Thumuluri, José Juan Almagro Armenteros, Alexander Rosenberg Johansen, Henrik Nielsen, Ole Winther, DeepLoc 2.0: multi-label subcellular localization prediction using protein language models, Nucleic Acids Research, 2022;, gkac278, https://doi.org/10.1093/nar/gkac278

More information about the method can be found at:

	https://services.healthtech.dtu.dk/service.php?DeepLoc-2.0

Pre-installation
----------------

DeepLoc 2.1 will run and has been tested under Linux and OS X. The only prerequisite is to have python3.6 or above installed.


Installation
------------

The installation procedure is:


  1. Install DeepLoc 2.1 package:
        pip install deeploc2_1.tar.gz
     or within the deeploc2_package directory:
         pip install .

  2. Test DeepLoc 2.1 by running:
     deeploc2 -f test.fasta
     
     the result should look like the file in the 'output' directory

This will download only the 'Fast' model (ESM1b). The 'Accurate' model (ProtT5) uses more memory (approx. 32GB), therefore, it is not recommended for personal computers with limited memory. The 'Accurate' model will be downloaded the first time that the user chooses it at run time.

Running
--------

DeepLoc will be installed under the name 'deeploc2'. It has 4 possible arguments:

 * -f, --fasta. Input in fasta format of the proteins.
 * -o, --output. Output folder name.
 * -m, --model. High-quality (Accurate) model or high-throughput (Fast) model. Default: Fast.
 * -p, --plot. Plot and save attention values for each individual protein. 

Output
-------

The output is a tabular file with the following format:

 * 1st column: Protein ID.
 * 2nd column: Predicted localization(s).
 * 3rd column: Predicted sorting signal(s).
 * 4th-13th column: Probability for each of the individual localizations. 

If --plot is defined, a plot and a text file with the sorting signal importance for each protein will be generated.

Problems and questions
----------------------

In case of technical problems (bugs etc.) please contact packages@cbs.dtu.dk.

Questions on the scientific aspects of the DeepLoc 2.0 method should go to Henrik
Nielsen, hennin@dtu.dk.
