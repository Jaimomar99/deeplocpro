
DeepLocPro 1.0
===========

DeepLocPro 1.0 is a multiclass subcellular localization prediction tool for prokaryotic proteins, trained on experimentally verified data curated from Uniprot and PSORTdb. DeepLocPro has been trained to work with prokaryotic proteins from a wide range of organisms covering Archaea, Gram-positive bacteria, and Gram-negative bacteria. It can differentiate between six different localizations: Cell wall & surface, Extracellular, Cytoplasmic, Cytoplasmic membrane, Outer membrane and Periplasmic.

The DeepLocPro 1.0 server requires protein sequence(s) in fasta format, and can not handle nucleic acid sequences.

Publication
------------

The preprint can be accessed here: https://lnkd.in/dgWsKHdN 

More information about the method can be found at:

	https://services.healthtech.dtu.dk/services/DeepLocPro-1.0/

Pre-installation
----------------

DeepLocPro 1.0 will run and has been tested under Linux and OS X. The only prerequisite is to have python3.6 or above installed.


Installation
------------

The installation procedure is:


  1. Install DeepLocPro 1.0 package:
    
    # Within the deeplocpro directory
    pip install .

  2. Test DeepLocPro 1.0 by running:
     
    deeplocpro -f test.fasta
     
The result should look like the file in the 'output' directory.

Running
--------

DeepLoc will be installed under the name 'deeplocpro'. It has 4 possible arguments:

 * -f, --fasta. Input protein sequences in the fasta format.
 * -o, --output. Output folder name.
 * -p, --plot. Plot and save attention values for each individual protein. 
 * -d, --device. One of cpu, cuda or mps. Default: cpu.
 * -g, --group. Prevent outer membrane & periplasm prediction when Archaea/positive. One of any, archaea, positive or negative. Default: any

Output
-------

The output is a comma separated file with the following format:

 * 1st column: Protein ID.
 * 2nd column: Predicted localization.
 * 3rd-8column: Probability for each of the individual localizations. 

If --plot is defined, a plot and a text file with the feature importance of the position for the prediction will be generated for each query protein.

Problems and questions
----------------------

In case of technical problems (bugs etc.) please contact packages@cbs.dtu.dk.

Questions on the scientific aspects of the DeepLocPro 1.0 method should go to Henrik
Nielsen, hennin@dtu.dk.
