# object_detection_SSD
Python 3.6
Tensorflow 1.12
Implemention of Single Shot MultiBox Detector with MobileNetV2
Input size: 320x320
Dataset: DeepDrive
This model can be used to detect only "car", so that the input size is too small for this dataset. Many small objects are ignored in training
process. To solve this problem, we can use a large input or other kinds of feature extractor. MobileNet is a lightweight CNN, and I do not 
want to make this model to heavy, so a size of 300x300 is chosen. 

Threshold of nms is 0.6.
Threshold of score is 0.85

Reference:
1. SSD: https://arxiv.org/abs/1512.02325
2. MobileNetV2: https://arxiv.org/abs/1801.04381
3. https://github.com/tensorflow/models/tree/master/research/object_detection/models
4. https://github.com/tanakataiki/ssd_kerasV2

