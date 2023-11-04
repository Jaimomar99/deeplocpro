#!/bin/bash

# nvidia-smi
deeplocpro "$@"
python3 make_markdown.py "$@"
