#### Introduction

```
![License](https://img.shields.io/badge/License-MIT-brightgreen)
![Copyright](https://img.shields.io/badge/Copyright-CVTEAM-red)

<img src="https://github.com/iCVTEAM/PART/blob/master/figs/motivation.jpg" width = "600" height = "300" align=center />
```

This code provides an initial version for the implementation of the **SCIENTIA SINICA Informationis** paper "Generalized representation of local relationships for few-shot
incremental learning". The projects are still under construction.

#### 简介

《局部关系泛化表征的小样本增量学习》，中国科学，信息科学，2022，已接收。代码示例如下。

#### Requirements

```
PyTorch>=1.1, tqdm, torchvsion.
```

#### Data Preparation

```
1. Download the benchmark dataset and unzip them in your customized path.
    CUB-200-2011 [links](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
    For miniImageNet dataset from other sharing links in CEC [A] [links](https://drive.google.com/drive/folders/11LxZCQj2FRCs0JTsf_dafvTHqFn2yGSN?usp=sharing)
2. Modify the lines in train.py from 3~5
3. unzip or tar these datasets
```


#### How to run

##### For Pretraining Stage:

1. cd the /pretrain file 

2.1  For  mini_imagenet
```
$python train.py -project base -dataset mini_imagenet -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.1 -lr_new 0.1 -decay 0.0005 -epochs_base 100 -schedule Milestone -milestones 40 70 -gpu 0,1 -temperature 16
```
2.2  For  CUB dataset 
```
$python train.py -project base -dataset cub200 -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.002 -lr_new 0.1 -decay 0.0005 -epochs_base 100 -schedule Milestone -milestones 40 70 -gpu 0,1 -temperature 16
```
##### For Meta-Learning Stage:

3. cd the /meta-learning file 
4.1   For  mini_imagenet dataset
```
$python train.py -project frn -dataset mini_imagenet -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.001 -lr_new 0.0001 -decay 0.0005 -epochs_base 103 -epochs_new 10 -schedule Milestone -milestones 40 70  -temperature 16 -gpu '0,1'  -episode_way 20 -episode_shot 10 -model_dir "/yourpathhere.pth"
```
4.2   For  cub dataset
```
$python train.py -project frn -dataset cub200 -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.002 -lr_new 0.001 -decay 0.0005 -epochs_base 101 -schedule Milestone -milestones 40 60 80 -episode_way 20 -episode_shot 10 -gpu '0,1' -temperature 16  -model_dir "/yourpathhere.pth"
```


##### Pretraining models
| Type/Datasets | CUB-200-2011                                                 | mini-ImageNet                                                |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Pretrained    | [Links](https://drive.google.com/file/d/1aoCCe9mDJspHtrLEUp6fDIc2aTEst_At/view?usp=share_link) | [Links](https://drive.google.com/file/d/1DULS9Imimgo_06ni4oOJ0mGh2bUeNlWh/view?usp=share_link) |
| Meta-Learning | [61.81%](https://drive.google.com/file/d/1XoFEyOGEn_9H1Rkj8wa5jtvQSP675ecM/view?usp=share_link) | [49.02%](https://drive.google.com/file/d/1XoFEyOGEn_9H1Rkj8wa5jtvQSP675ecM/view?usp=share_link) |


#### Running Tips

1. The performance may be fluctuated in different GPUs and PyTorch platforms. Pytorch versions higher than 1.7.1 are tested. 
2.  Two K80 GPUs are used in our experiments. 


#### To do

1. The project is still ongoing, finding suitable platforms and GPU devices for complete stable results.

2. The project is re-constructed for better understanding, we release this version for a quick preview of our paper.

   
#### License

The code of the paper is freely available for non-commercial purposes. Permission is granted to use the code given that you agree:

1. That the code comes "AS IS", without express or implied warranty. The authors of the code do not accept any responsibility for errors or omissions.

2. That you include necessary references to the paper in any work that makes use of the code. 

3. That you may not use the code or any derivative work for commercial purposes as, for example, licensing or selling the code, or using the code with a purpose to procure a commercial gain.

4. That you do not distribute this code or modified versions. 

5. That all rights not expressly granted to you are reserved by the authors of the code.

#### Citations:

Please remember to cite us if u find this useful :)

@inproceedings{zhao2022local,
  title={局部关系泛化表征的小样本增量学习},
  author={赵一凡, 李甲，田永鸿},
  booktitle={中国科学：信息科学},
  year={2022},
}



#### Acknowledgment
Our project references the codes in the following repos.
Please refer to these codes for details.

- [CEC](https://github.com/icoz69/CEC-CVPR2021)
- [FRN](https://github.com/Tsingularity/FRN)


