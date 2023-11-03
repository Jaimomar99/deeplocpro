FROM continuumio/miniconda3

WORKDIR /home/deeplocpro
COPY server .
RUN pip install .

COPY server/sample.fasta .
# COPY webserver/make_output.py .
# COPY webserver/run.sh .
# RUN chmod +x run.sh
