FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y wget \
    python3-pip

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

