FROM nvidia/cuda:11.1.1-runtime-ubuntu20.04

RUN rm /etc/apt/sources.list.d/cuda.list && \
    apt-key del 7fa2af80 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN apt update
RUN apt install python3 python3-pip -y

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

