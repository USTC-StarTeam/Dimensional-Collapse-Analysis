#!/bin/bash
model_name=FM
expid=FM_avazu

echo "Generating embeddings for analysis"

python analyze.py --model ${model_name} --expid ${expid} --gpu 0


echo "Plot figures in file analysis.ipynb"

