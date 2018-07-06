# Faster R-CNN Implementation with MXNet Gluon API

This repo re-implements Faster R-CNN fully on MXNet [Gluon](http://gluon.mxnet.io) API,
which supports batch size larger than one and Multi-GPU training.
You can use the code to train/validate/test for object detection task.

## Features

- [x] RPN, Fast R-CNN, Faster R-CNN with VGG16 model
- [x] Inference and prediction in hybridize mode
- [x] Multi-GPU and lager batch size support
- [x] End to end training/validating/testing
- [ ] Alternate training/validating/testing

More functions are in developing...

## Requirements

Note: This repo depends on MXNet version 1.2.1+, due to [MXNet Symbol and Gluon Proposal API are 
inconsistent](https://github.com/apache/incubator-mxnet/pull/10242) in previous version.

This repo requires Python3 with the following packages: 
```
mxnet
tqdm
EasyDict
matplotlib
opencv-python
```

You may also need a GPU with at least 8GB memory for training.

## Installation

1. Install all required packages in python3.
2. Clone the Gluon Faster R-CNN repository.
   ```bash
   git clone https://github.com/WalterMa/gluon-faster-rcnn
   cd gluon-faster-rcnn
   ```

### Try the Demo

3. Download [pre-trained model parameters]() from release. Then extract it to ./model directory.
4. Run *demo_faster_rcnn.py*.
   ```bash
   python ./demo_faster_rcnn.py
   ```

### Train & Testing Faster R-CNN

Currently, this repo only support voc2007/2012 dataset. 
But you could easily modify or create your own dataset by reference 
[Gluon-CV dataset](https://github.com/dmlc/gluon-cv/tree/master/gluoncv/data) code, or generate and using 
[record dataset](./tools/README.md).

Note: Record Dataset is only available in num_workers=0, due to MXNet [issue](https://github.com/apache/incubator-mxnet/issues/9974).

3. We need the following three files from Pascal VOC:

   | Filename                                                                                                  | Size   | SHA-1                                    |
   | :-------------------------------------------------------------------------------------------------------- | :----- | :--------------------------------------- |
   | [VOCtrainval_06-Nov-2007.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) | 439 MB | 34ed68851bce2a36e2a223fa52c661d592c66b3c |
   | [VOCtest_06-Nov-2007.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)         | 430 MB | 41a8d6e12baa5ab18ee7f8f8029b9e11805b4ef1 |
   | [VOCtrainval_11-May-2012.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) | 1.9 GB | 4e443f8a2eca6b1dac8a6c57641b67dd40621a49 |

4. Download and extract voc dataset to ./data/VOCdevkit/, or you need to specify dataset path in .utils/config.py 
or related python scripts.

5. Start e2e training and validating:
   ```bash
   python ./train_faster_rcnn.py
   ```
   
## Experiments

| Method | Network | Training Data | Testing Data | Reference | Result |
| :----- | :------ | :------------ | :----------- | :-------: | :----: |
| Faster R-CNN end-to-end | VGG16 | VOC07+12 | VOC07test | 73.2 | - |

## Disclaimer

This is a re-implementation of original Faster R-CNN which is based on caffe. 
The arXiv paper is available [here](https://arxiv.org/abs/1506.01497).

This repository used code from [MXNet](https://github.com/dmlc/mxnet),
[Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn),
[MX R-CNN](https://github.com/ijkguo/mx-rcnn),
[MXNet SSD](https://github.com/zhreshold/mxnet-ssd),
[Gluon CV](https://github.com/dmlc/gluon-cv).