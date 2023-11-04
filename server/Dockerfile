FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime


WORKDIR /home/deeplocpro
COPY server .
COPY DeepLocPro DeepLocPro
COPY setup.py .
COPY server/README.md .
RUN pip3 install --no-cache-dir --find-links https://download.pytorch.org/whl/torch_stable.html .

COPY server/sample.fasta .

# run once to download weights
RUN deeplocpro --fasta sample.fasta --output test
RUN rm -rf test
# COPY webserver/make_output.py .
# COPY webserver/run.sh .
# RUN chmod +x run.sh
RUN chmod +x run.sh

