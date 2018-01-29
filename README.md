
# Efficient Yet Deep Convolutional Neural Networks for Semantic Segmentation
Paper link 
Arxiv Pre-print
```
https://arxiv.org/abs/1707.08254
```

Please cite our paper if you use our codes or material in your work: 

# Citation 
```
@article{kamranefficient,
  title={Efficient Yet Deep Convolutional Neural Networks for Semantic Segmentation},
  author={Kamran, Sharif Amit and Hasan, Muhammad and Sabbir, Ali Shihab}
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
# Surgery
1. Download the pretrained models and the deploy.protxt of [VGGG16](https://gist.github.com/ksimonyan/211839e770f7b538e2d8) or [VGG19](3785162f95cd2d5fee77).
2. Open sovlesurgery.py and edit the path of proto and weights
For example: proto = '/home/XYZ/Desktop/DilatedFCNSegmentaiton/VGG_ILSVRC_16_layers_deploy.prototxt'
             weights = '/home/XYZ/Dekstop/DilatedFCNSegmentation/VGG_ILSVRC_16_layers.caffemodel'
3. Create a folder inside Dilated_FCN-2s_VGG16 called snapshot. Ex: DilatedFCNSegmentation/Dilated_FCN-2s_VGG16/snapshot
4. Open the solver.protoxt and edit snapshot_prefix. The path will be of the new created model with name.
             '/home/XYZ/Desktop/DilatedFCNSegmentation/Dilated_FCN-2s_VGG16/snapshot/vgg16surgery' 
5. Now run solvesurgery.py and this will create a file named vgg16surgery_iter_1.caffemodel inside the folder called
             Dilated_FCN-2s_VGG16/snapshot 
# Training on VOC2012 dataset
1. Open the solver.prototxt again and rename the snapshot to 
             '/home/XYZ/Desktop/DilatedFCNSegmentation/Dilated_FCN-2s_VGG16/snapshot/voctraining'
2. Open both train.prototxt and val.prototxt and inside the 'Dilated_FCN-2s_VGG16' folder.
3. Rewrite the path to your VOC2012 dataset. For example :
      For train &  val   : /home/XYZ/Desktop/DilatedFCNSegmentation/Dilated_FCN-2s_VGG16/data/VOC2012
4. Download the data from [VOC2012 site](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#data)
5. copy the VOC2012 directory to below folder.
             '/home/XYZ/Desktop/DilatedFCNSegmentation/Dilated_FCN-2s_VGG16/data/'
6. Run solvedilated.py for training
7. (Optional) you can resume training by the uncommenting line 13 and 16. And commenting line 12.


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




