FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime


WORKDIR /home/deeplocpro
COPY server .
COPY DeepLocPro DeepLocPro
COPY setup.py .
COPY server/README.md .
RUN pip3 install --no-cache-dir --find-links https://download.pytorch.org/whl/torch_stable.html .
RUN pip3 install --no-cache-dir tabulate==0.9.0

COPY server/sample.fasta .
COPY server/make_markdown.py .

# run once to download weights
RUN deeplocpro --fasta sample.fasta --output test
RUN rm -rf test
# COPY webserver/make_output.py .
# COPY webserver/run.sh .
# RUN chmod +x run.sh
RUN chmod +x run.sh

