#!/bin/bash
echo "Training recommendation models"

# expid can be found in model_zoo/*/config/model_config.yaml

python model_zoo/FM/run_expid.py --expid FM_avazu --gpu 0
python model_zoo/FM/run_expid.py --expid NFM_avazu --gpu 0
python model_zoo/FM/run_expid.py --expid DeepFM_avazu --gpu 0

