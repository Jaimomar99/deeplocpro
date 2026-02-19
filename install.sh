#!/bin/bash
pip3 uninstall torch -y
pip3 install torch==1.13.1 --index-url https://download.pytorch.org/whl/cpu
pip3 install -r requirements.txt
