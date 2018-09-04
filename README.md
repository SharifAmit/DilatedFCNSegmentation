
# Efficient Yet Deep Convolutional Neural Networks for Semantic Segmentation

### Our paper has been accepted to 2018 International Symposium on Advanced Intelligent Informatics (SAIN). Will be availble in IEEE Xplore Digital Library soon.


Paper link 
Arxiv Pre-print
```
https://arxiv.org/abs/1707.08254
```

Please cite our paper if you use our codes or material in your work: 

# Youtube Demo 

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/cTd2NDoe3eI/0.jpg)](https://www.youtube.com/watch?v=cTd2NDoe3eI)


# Citation 
```
@article{kamranefficient,
  title={Efficient Yet Deep Convolutional Neural Networks for Semantic Segmentation},
  author={Kamran, Sharif Amit and Sabbir, Ali Shihab}
  journal={arXiv preprint arXiv:1707.08254},
  year={2017}
}
```
# Score and Leaderboard
- FCN2s-Dilated-VGG16 Mean Iou score 67.6 percent [FCN2s-Dilated-VGG16](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?cls=mean&challengeid=11&compid=6&submid=12146#KEY_FCN-2s_Dilated_VGG16)
- FCN2s-Dilated-VGG19 Mean Iou score 69 percent [FCN2s-Dilated-VGG19](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?cls=mean&challengeid=11&compid=6&submid=12146#KEY_FCN-2s_Dilated_VGG19)

# Installation
Make caffe with python wrapper. Detailed Instruction below

# Models

- FCN2s-Dilated-VGG16 download link - [FCN2s-Dilated-VGG16](https://drive.google.com/drive/folders/0ByGwXEdDYIN3SW55R3NZUVg0NHc?usp=sharing)
- FCN2s-Dilated-VGG19 download link - [FCN2s-Dilated-VGG19](https://drive.google.com/drive/folders/0ByGwXEdDYIN3SW55R3NZUVg0NHc?usp=sharing)

This models were only trained on SBD and VOC data and for 21 classes segmentation task for PASCAL VOC2012 Segmentation Challenge. 

Will be uploading the net trained on NYUDv2 dataset and Pascal-Context later on. Keep an eye on the page.
# Demo
Open demo.py and change line 29 for running demo with different images.
Run demo.py

# Tutorial

A tutorial with elaborated instructions for running the inference is provided at [ModelDepot.io](https://modeldepot.io/sharifamit/dilated-fcn-2s/overview)

# Surgery + Training on VOC2012 dataset

First read surgery-instructions.txt for details.

Then read training-instructions.txt for details.

# Training on SBD dataset

To recreate our result you have to first train on VOC2012 dataset and then SBD dataset.

Read the training-sbd-instructions for details.

# License
The code is released under the MIT License, you can read the license file included in the repository for details.

# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by Berkeley AI Research ([BAIR](http://bair.berkeley.edu))/The Berkeley Vision and Learning Center (BVLC) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BAIR reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

## Custom distributions

 - [Intel Caffe](https://github.com/BVLC/caffe/tree/intel) (Optimized for CPU and support for multi-node), in particular Xeon processors (HSW, BDW, Xeon Phi).
- [OpenCL Caffe](https://github.com/BVLC/caffe/tree/opencl) e.g. for AMD or Intel devices.
- [Windows Caffe](https://github.com/BVLC/caffe/tree/windows)

## Community

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BAIR/BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }




