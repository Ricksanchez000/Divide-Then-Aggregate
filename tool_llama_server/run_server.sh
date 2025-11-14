#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
python llama2_server.py --llama_model_path=$2 \
                          --llama_template=tool-llama-parallel \
                          --llama_device=cuda:0 \
                          --port=8899
