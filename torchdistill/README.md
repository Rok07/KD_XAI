# Training KD models

For the KD training, this repository heavily based on the code of [Torchdistill](https://github.com/yoshitomo-matsubara/torchdistill). 
We revised the examples/image_classification.py for trainig KD models from ls-trained teacher, and config files.
Please see the original repository for detailed issues in implemantation. 
Copyright of this code belongs to [Yoshitomo Matsubara](https://github.com/yoshitomo-matsubara/torchdistill) who write this code.


# Preparation

## Data preparation
You should download the [ImageNet](https://www.image-net.org/download.php) dataset to train KD models.

## Teacher model preparation
You can use own your pre-trained teacher model to distill the student model.
In our works, we used the pre-trained ResNet-34
model provided by [Torchvision](https://pytorch.org/vision/0.8/models.html).

# Run experiments

You can configure each `.yaml` file to either load your pre-trained teacher model or modify the default parameters.

## KD
If you want to train [vanilla KD](https://arxiv.org/pdf/1503.02531.pdf) model, run commands with the following.

```
python3 examples/image_classification.py --config configs/official/ilsvrc2012/yoshittomo-mattsubara/rrpr2020/kd-resnet18_from_resnet34.yaml --log log/kd-resnet18_from_resnet34.txt
```

## KD from LS trained teacher
If you want to train KD models from ls-trained teacher, run commands with the following.
```
python3 examples/image_classification.py --config configs/official/ilsvrc2012/yoshittomo-mattsubara/rrpr2020/kd-resnet18_from_LSresnet34.yaml --log log/kd-resnet18_from_LSresnet34.txt -ls_teacher
```

## AT
If you want to train [AT (Attention Transfer)](https://arxiv.org/pdf/1612.03928.pdf) model, run commands with the following.

```
python3 examples/image_classification.py --config configs/official/ilsvrc2012/yoshittomo-mattsubara/rrpr2020/at-resnet18_from_resnet34.yaml --log log/at-resnet18_from_resnet34.txt
```

## FT
If you want to train [FT (Factor Transfer)](https://arxiv.org/pdf/1802.04977.pdf) model, run commands with the following.

```
python3 examples/image_classification.py --config configs/official/ilsvrc2012/yoshittomo-mattsubara/rrpr2020/ft-resnet18_from_resnet34.yaml --log log/ft-resnet18_from_resnet34.txt
```

## CRD
If you want to train [CRD (Contrastive Representation Distillation)](https://arxiv.org/pdf/1910.10699.pdf) model, run commands with the following.

```
python3 examples/image_classification.py --config configs/official/ilsvrc2012/yoshittomo-mattsubara/rrpr2020/crd-resnet18_from_resnet34.yaml --log log/crd-resnet18_from_resnet34.txt
```

## SSKD
If you want to train [SSKD (Self-Supervision KD)](https://arxiv.org/pdf/2006.07114.pdf) model, run commands with the following.

```
python3 examples/image_classification.py --config configs/official/ilsvrc2012/yoshittomo-mattsubara/rrpr2020/sskd-resnet18_from_resnet34.yaml --log log/sskd-resnet18_from_resnet34.txt
```
