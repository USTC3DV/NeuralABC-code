# Neural-ABC: Neural Parametric Models for Articulated Body with Clothes

PyTorch implementation of the paper "Neural-ABC: Neural Parametric Models for Articulated Body with Clothes". This repository contains the training and inference code, data.

**|[Project Page](https://ustc3dv.github.io/NeuralABC/)|**
![teaser](figs/teaser.png)
We proposed Neural-ABC, a novel parametric model based on neural implicit functions that can represent clothed human bodies with disentangled latent spaces for identity, clothing, shape, and pose. 

## Pipeline
Neural-ABC is a neural implicit parametric model with latent spaces of human identity, clothing, shape and pose. 
It can generate various human identities and different clothes. 
The clothed human body can deform into different body shapes and poses. 

![pipeline](figs/pipeline.png)


## Citation

If you find our paper useful for your work please cite:

```
@article{Chen2024NeuralABC,
  title={Neural-ABC: Neural Parametric Models for Articulated Body with Clothes},
  author={Honghu Chen, Yuxin Yao, and Juyong Zhang},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2024}，
  publisher={IEEE}
}
```

## Contact
For more questions, please contact honghuc@mail.ustc.edu.cn

## Acknowledgement

Our data is processed with the help of [StereoPIFu](https://github.com/CrisHY1995/StereoPIFu_Code), [DrapeNet](https://github.com/liren2515/DrapeNet) and [MeshUDF](https://github.com/cvlab-epfl/MeshUDF):
```
@inproceedings{yang2021stereopifu,
  author    = {Yang Hong and Juyong Zhang and Boyi Jiang and Yudong Guo and Ligang Liu and Hujun Bao},
  title     = {StereoPIFu: Depth Aware Clothed Human Digitization via Stereo Vision},
  booktitle = {{IEEE/CVF} Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2021}
}

@inproceedings{de2023drapenet,
  author = {De Luigi, Luca and Li, Ren and Guillard, Benoit and Salzmann, Mathieu and Fua, Pascal},
  title = {{DrapeNet: Garment Generation and Self-Supervised Draping}},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year = {2023}
}


@inproceedings{guillard2022udf,
  author = {Guillard, Benoit and Stella, Federico and Fua, Pascal},
  title = {MeshUDF: Fast and Differentiable Meshing of Unsigned Distance Field Networks},
  booktitle = {European Conference on Computer Vision},
  year = {2022}
}
```
