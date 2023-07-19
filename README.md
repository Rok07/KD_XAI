# On the Impact of KD for Model Interpretability

This repository is the official pytorch implementation for ''[On the Impact of KD for Model Interpretability](https://openreview.net/pdf?id=XOTFW2BK6i)'' [ICML 2023]
We adopted the code of [NetDissect-Lite](https://github.com/CSAILVision/NetDissect-Lite). 
We revised the loader/model_loader.py for compatible with torchdistill model, and visualize/feature_operation.py for visualize the activation map of concept detectors.
Copyright of this code belongs to the authors of [Network Dissection: Quantifying Interpretability of Deep Visual Representations](http://netdissect.csail.mit.edu). 
Please see the original repository for detailed issues in implemantation.

# Training KD models

If you already have a pre-trained KD model, you can skip the following steps.

If not, please refer to the README.md file in the `torchdistill` folder for the code related to KD training.


# Obtaining the concept detectors

First, clone the codes from our repository 

```
git clone https://github.com/Rok07/KD_XAI.git
```

Then, download the Broden dataset to obtain concept detectors

```
bash script/dlbroden.sh
```

Adjust the settings in `settings.py` to either load your pre-trained model or modify the default parameters.

Finally, run NetDissect.

```
python main.py
```

# Reference

If our work is helpful, please cite the following paper.

```
@inproceedings{Han2023,
  title={On the Impact of Knowledge Distillation for Model Interpretability},
  author={Hyeongrok, Han and Siwon, Kim and Hyun-SOO, Choi and Sungroh, Yoon},
  booktitle={International Conference on Machine Learning},
  year={2023}
}
```


