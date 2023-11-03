FROM continuumio/miniconda3

WORKDIR /home/deeplocpro
COPY server .
COPY DeepLocPro DeepLocPro
COPY setup.py .
COPY server/README.md .
RUN pip install .

COPY server/sample.fasta .

# run once to download weights
RUN deeplocpro --fasta sample.fasta --output test
RUN rm -rf test
# COPY webserver/make_output.py .
# COPY webserver/run.sh .
# RUN chmod +x run.sh
