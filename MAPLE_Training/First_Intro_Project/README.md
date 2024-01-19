ML enthusiasts can experiment with alternatives to the current best model (Mask-RCNN) for Ice-wedge Polygons (IWPs) to get exposed to the ML training pipeline for remote sensing. Existing training data for IWPs can be used for this exercise. Can start with a simple Mask RCNN model. This is a document to get you started.

First step would be to get access to MaskRCNN can do that on Google Colab notebook. [Good tutorial to follow](https://cloud.google.com/tpu/docs/tutorials/mask-rcnn-2.x)

MaskRCNN allows transfer learning so you can use the COCO dataset weights to lock part of the model and train only the heads that would allow you to customize it to the IWPs that we want to infer.
### Training
[Training data set for IWPs](https://drive.google.com/drive/folders/16NH5tOHI7ZLwDPE55wrif9Cacri-c-wN?usp=sharing) can be found (in the COCO fomat used by MaskRCNN)

### Inferencing
Once the model is trained you can try to do an inference on an some actual high resolution images. You may have to tile it to smaller chunks to send it through the ML inferencing pipeline.

Performance w.r.t accuracy is very important and we use the mAP, mAR, and mF1. In semantic segmentation we measure the metrics based on IoU. 

Once you are familiar with the data set and the task at hand you can try other ML models to experiment. One potential area is to explore transfer based SAM model. A simple [notebook on SAM](https://github.com/PermafrostDiscoveryGateway/MAPLE_v3/tree/main/MAPLE_Training/SAM) available to explore. Can explore with the [same data set shared here](https://drive.google.com/drive/folders/16NH5tOHI7ZLwDPE55wrif9Cacri-c-wN).
