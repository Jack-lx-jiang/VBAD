# Black-box Adversarial Attacks on Video Recognition Models. (VBAD)

## Introduction

This is the code for the paper "Black-box Adversarial Attacks on Video Recognition Models". It utilizes transferred 
perturbations from ImageNet pre-trained model and reduce dimensionality of attack space by partition-based rectification
, to boost the black-box attack. More information can be found on the [paper](https://dl.acm.org/ft_gateway.cfm?id=3351088&ftid=2091857&dwn=1&CFID=167915373&CFTOKEN=ede3e02bf61282e4-B658F898-F2EF-7614-2A0400702E8B056B).

## Requirement
The code is tested on the python 3.6.7  pytorch 0.4.1
```
pip install -r requirements.txt  # install requirements
```

We use the pre-trained I3D model from https://github.com/piergiaj/pytorch-i3d.

## Usage

### Targeted attack
Run `sh ./targeted_attack.sh`

### Untargetd attack
Run `sh ./untargeted_attack.sh`

## Cite
If you find this work is useful, please cite the following:
```
@inproceedings{jiang2019black,
  author    = {Linxi Jiang and
               Xingjun Ma and
               Shaoxiang Chen and
               James Bailey and
               Yu{-}Gang Jiang},
  title     = {Black-box Adversarial Attacks on Video Recognition Models},
  booktitle = {Proceedings of the 27th {ACM} International Conference on Multimedia,
               {MM} 2019, Nice, France, October 21-25, 2019},
  pages     = {864--872},
  year      = {2019}
}
```

## Contact
For questions related to VBAD, please send an email to `lxjiang18@fudan.edu.cn`