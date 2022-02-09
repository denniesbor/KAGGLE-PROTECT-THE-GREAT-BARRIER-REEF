<h2><center> <strong>Team Reefsave Help Protect the Great Barrier Reef</strong></center></h2>

This repo hosts notebooks, and helper shell and python files used in data preprocessing and training of `RCNN`, `YOLOv5`, and `YOLOv4` object detection neural net.
![](https://github.com/denniesbor/KAGGLE-PROTECT-THE-GREAT-BARRIER-REEF/blob/assets/reef.png?raw=True)
****
<h2><center> <strong>Literature</strong></center></h2>
The choice of a neural network is dependent on the available software and hardware resources, speed ,and the expected accuracy. Object detection networks are classified as multi-stage or single stage. 

Examples of single staged neural nets are the SSD, YOLO, etc. The multi staged approaches uses the region proposal networks in their architectures to extract feature maps from the backbone. Examples of multi stage networks are the RCNN and RFCN.

<h3><center> <strong>Architecture of a neural network</strong></center></h3>

Object detection nets consists of the input, backbone, neck and the head. The input takes in an image, and it outputs to a feature extractor consisting of dense convolution and max pooling layers. Residual Network(ResNet), ResNext,DenseNet, VGG16 etc. are the commonly used backbones. They are trained on standardized datasets such as [COCO](https://cocodataset.org/#home) or [ImageNet](https://image-net.org).
<br/>
The role of the neck is to extract feature maps e.g the Feature Pyramid Network. The head of a single stage network is dense prediction layer and sparse prediction for a two stage detector(i.e RCNN & RFCN)
<br />
![](https://github.com/denniesbor/KAGGLE-PROTECT-THE-GREAT-BARRIER-REEF/blob/e0c3e4253d8fd6a867252cab1feb3bab3d80f377/object_detection_arch.png?raw=true)

[**Figure 1.** Schematic representation of a single and multi stage neural network. Source: [arxiv]
<br />
<h3><center> <strong>The choice of a neural network.</strong></center></h3>

Computational resources determine the amount of time spent on training and inference. GPU and TPU runtime accelerate the training as well as the inference time. The computational resource demand differ from one model to another.

Speed is key in a real-time object detection system or video search engines.  A balance of speed and resource requirements  is considered to achieve optimal performance.

The implementation of the minimum viable product for the module one was based on the performance of the Faster R-CNN ResNet Inception, Yolov4 and Yolov5 on pre-processed TensorFlow-Protect the great barrier datasets.

<h3><center> <strong>Yolo(Single stage)</strong></center></h3>

Yolo is a single stage state of the art object detection algorithm. There are 4 documented versions of YOLO and the fifth version designed by Ultralytics team. [YOLO](https://github.com/ultralytics/yolov5) is described as a YOLOv4 implementation in Pytorch.
Compared with other algorithms, YOLO5 perfoms exceptionally well with a less GPU time.

According to [Huang,et al](https://arxiv.org/pdf/1611.10012.pdf) YOLO v4 attains a mean average precision of 43.5 running on a Tesla V100 GPUs while training on Common Objects in Context datasets. The neck of YOLO4 uses SPP and PAN.
<br />
![](https://github.com/denniesbor/KAGGLE-PROTECT-THE-GREAT-BARRIER-REEF/blob/assets/yolo4.png?raw=true)
<br />
[**Figure 2.** Yolo4 AP vs FPS against other object detectors: [arxiv](https://arxiv.org/pdf/1611.10012.pdf) ]
<br />
![yolo](https://github.com/denniesbor/KAGGLE-PROTECT-THE-GREAT-BARRIER-REEF/blob/assets/Yolov5_performance.png?raw=true)
<br/>
[**Figure 3.** Average precision vs GPU speed of *YOLO5* weights against *EfficientDet* on . on [COCO](https://cocodataset.org/#home) datasets. Source: [Ultralytics](https://github.com/ultralytics/yolov5) ]

### What are Bag of Freebies and Bag of Specials?

They define the inference - training trade-off of a model. The bag of freebies are the methods applied to the model and which does not interfere with inference. Some of these methods include the data augmentation, regularization techniques e.g., dropout, drop-connect and drop-block.

The bag of freebies are the methods which improve the accuracy of the model by at the expense of inference costs. These methods introduce attention mechanisms. SPP is an example of this feature and is applied in YOLOv4.

<h3><center> <strong>Faster R-CNN(Multi stage)</strong></center></h3>

R-CNN models is a multi layered conv neural network and consists of the feature extractor, a region proposal algorithm to generate bounding boxes, a regression and classification layer. R-CNNs tradeoff their speed for accuracy. 

In Faster R-CNN, Region Proposal Network generation is not CPU restricted compared to the previous flavours of region convolution neural network.
<br />
![](https://github.com/denniesbor/KAGGLE-PROTECT-THE-GREAT-BARRIER-REEF/blob/assets/feature_extractor_acc.png?raw=True)
<br />
[**Figure 4.** Mean average precision against backbone accuracy of Faster R-CNN, R-FCN and SSD]
<h2><center> <strong>MVP - Performance comparison of YOLOv4, YOLOv5 and R-CNN</strong></center></h2>
The TensorFlow- Save the Great Barrier mvp is implemented using Faster R-CNN, YOLO4 and YOLO5 default tuning parameters. Performance analysis of the three models is done using their mean average precision. Faster RCNN runs on Resnet Inception backbone, whereas YOLO4 is built on darknet.

### **Yolov5**
<br />
<img src='https://github.com/denniesbor/KAGGLE-PROTECT-THE-GREAT-BARRIER-REEF/blob/assets/yolo5.png?raw=true' height=500px width=720px></img>
<br />

[**Figure 5.** Yolov5 performance metrics on MVP]


### **Yolov4**
<br />
<img src='https://github.com/denniesbor/KAGGLE-PROTECT-THE-GREAT-BARRIER-REEF/blob/assets/yolov4_loss.png?raw=true' height=400px width=520px></img>
<br />

[**Figure 6.** Yolov4 performance metrics on MVP]

The loss metrics drops rapidly after training

### **FRCNN**

The Faster RCNN with ResNet Inception backbone is resource intensive and couldn't make any inference

<h2><center> <strong>Conclusion</strong></center></h2>
The yolov5 and yolov4 inference time is lower compared to F-RCNN.
Yolov5 has been shown to improve the speed and accuracy of detection, and therefore recommended for tackling The Help Protect Great Barrier Reef task.

<h2><center> <strong>Resources and References</strong></center></h2>

### Faster RCNN Resources

* `Colab Notebook` [colab](https://colab.research.google.com/drive/1-N2peQoX7WC85rFXgodoLbvduX86jY4s#scrollTo=Yo_jRkAhT2Xg)
* `object_detector` [model zoo](https://github.com/tensorflow/models.git)
* `model` [FRCNN](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8.tar.gz)
****
### YOLOv4 Resources

* `Colab Notebook` [colab](https://colab.research.google.com/drive/1PNvc1iJJGgSDXtXJ-0tEvLJlWjnPqY9z#scrollTo=a6DmZf6VBZnB)
* `model` [yolov4](https://github.com/ultralytics/yolov4)
* `object detector` [darknet](https://github.com/pjreddie/darknet.git)
* `yolov4.conv.137` [yolov4.conv.137](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/)
* `weights`       [yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)

****
### YOLOv5 Resources

* `Colab Notebook` [colab](https://colab.research.google.com/drive/1b5uWVfHZvK0qjuXzGeSi2mVmzfuHLkT2#scrollTo=yD-yfWt_Womh)

* `YOLO5 Model` [yolov5](https://github.com/ultralytics/yolov5.git)
****

### References <br />
[1]J. Huang et al., "Speed/accuracy trade-offs for modern convolutional object detectors", arXiv.org, 2022. [Online]. Available: https://arxiv.org/abs/1611.10012. [Accessed: 09- Feb- 2022].<br />
[2]A. Bochkovskiy, C. Wang and H. Liao, "YOLOv4: Optimal Speed and Accuracy of Object Detection", arXiv.org, 2022. [Online]. Available: https://arxiv.org/abs/2004.10934. [Accessed: 09- Feb- 2022].<br />
[3]S. Ren, K. He, R. Girshick and J. Sun, "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks", arXiv.org, 2022. [Online]. Available: https://arxiv.org/abs/1506.01497. [Accessed: 09- Feb- 2022].
