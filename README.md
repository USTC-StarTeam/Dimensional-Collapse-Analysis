Code of [SIGIR'26 short] "Understanding DNNs in Feature Interaction Models: A Dimensional Collapse Perspective".
This repo is implemented based on [FuxiCTR](https://github.com/reczoo/FuxiCTR) and [GE4Rec](https://github.com/USTC-StarTeam/GE4Rec).

## Enviroments

```bash
conda create -n FuxiCTR_analysis python=3.10 -y
conda activate FuxiCTR_analysis
pip3 install torch torchvision torchaudio
pip3 install -r requirements.txt
```

## Prepare dataset and models

```bash
bash 1.prepare.sh
bash 2.train_model.sh
```

## Dimensional Collapse Analysis

Run the following script to generate a batch of samples for dimensional collapse analysis:
```bash

bash 3.analyze.sh
```
Once the embeddings are generated, you may use the code provided in **analysis.ipynb** to visualize and analyze the results.


Advanced usage & Code explanations can refer to [GE4Rec](https://github.com/USTC-StarTeam/GE4Rec).