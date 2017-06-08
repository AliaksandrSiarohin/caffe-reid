# A Discriminatively Learned CNN Embedding for Person Re-identification

A [caffe](https://github.com/BVLC/caffe)-based implementation of [this paper](https://arxiv.org/abs/1611.05666),
providing whole training, testing and evaluation codes.

The official code (written in matconvnet) is available [here](https://github.com/layumi/2016_person_re-ID).

![Structure](https://github.com/D-X-Y/caffe-reid/blob/master/figures/person-re-identification-struct.png)

## Preparation
- download Market-1501 dataset and put Market-1501 in $HOME/datasets/
- `cd examples/market1501/mat-codes` and `matlab -nodisplay -r  'generate_train(), exit()'` to generate train, test and qurey data lists.
- Build with NCLL / cuda-8.0

## Results on Market-1501

[Market-1501](http://liangzheng.com.cn/Project/state_of_the_art_market1501.html) is one of the most popular person re-identification datasets.

Models can be found in `models/market1501/model_name`

Many scripts (e.g initialization, testing, training, extract feature and evaluation) can be found in `examples/market1501/`

[iter_size * batch_size] = real batch_size

### caffenet
- `python models/market1501/generate_caffenet.py` for generate caffenet based person re-ID network and solver files.
- `sh models/market1501/caffenet/train.sh --gpu 0` for training models.
- `sh examples/market1501/extract/extract_prediction.sh 0 caffenet fc7` for extracting features of query and test data
- `cd examples/market1501/evaluation/` and `evaluation('caffenet')` to evaluate performance of the trained model on Market-1501
- final results are [1x128] : mAP = 0.402689, r1 precision = 0.639846 [Euclidean]

### googlenet
- GoogleNet-v1 model is already in `models/market1501/googlenet`
- `sh models/market1501/googlenet/train.sh --gpu 0`
- `sh examples/market1501/extract/extract_prediction.sh 0 googlenet pool5/7x7_s1`
- `cd examples/market1501/evaluation/` and `evaluation('googlenet')`
- final results are [2x 64] : mAP = 0.476404, r1 precision = 0.706948 [Euclidean]
- final results are [2x 64] : mAP = 0.489998, r1 precision = 0.710214 [Cos + Eucl]

### vgg16
- `python models/market1501/generate_vgg16.py` for generate caffenet based person re-ID network and solver files.
- `sh models/market1501/vgg16/train.sh --gpu 2,3` for training
- `sh examples/market1501/extract/extract_prediction.sh 0 vgg fc7` for extracting features
- `cd examples/market1501/evaluation/` and `evaluation('vgg16')` to evaluate performance of vgg16/fc7 on Market-1501
- final results are [2x 24] : mAP = 0.446430, r1 precision = 0.654394 [Failed]

### resnet-50
- `python models/market1501/generate_resnet50.py`
- `sh models/market1501/res50/train.sh --gpu 2,3`
- `sh examples/market1501/extract/extract_prediction.sh 0 res50 pool5`
- final results are [4x 16] : mAP = 0.593053, r1 precision = 0.801960 [Cos + Eucl]

# Citation
Please cite this paper in your publications if it helps your research:
```
@article{zheng2016discriminatively,
  title={A Discriminatively Learned CNN Embedding for Person Re-identification},
  author={Zheng, Zhedong and Zheng, Liang and Yang, Yi},
  journal={arXiv preprint arXiv:1611.05666},
  year={2016}
}
```

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
