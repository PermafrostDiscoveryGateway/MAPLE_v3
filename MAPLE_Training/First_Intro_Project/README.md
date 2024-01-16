ML enthusiasts can experiment with alternatives to the current best model (Mask-RCNN) for Ice-wedge Polygons (IWPs) to get exposed to the ML training pipeline for remote sensing. Existing training data for IWPs can be used for this exercise. Can start with a simple Mask RCNN model. This is a document to get you started.

First step would be to get access to MaskRCNN can do that on Google Colab notebook. [Good tutorial to follow](https://cloud.google.com/tpu/docs/tutorials/mask-rcnn-2.x)

MaskRCNN allows transfer learning so you can use the COCO dataset weights to lock part of the model and train only the heads that would allow you to customize it to the IWPs that we want to infer.
### Training
[Traing data set for IWPs](https://drive.google.com/drive/folders/18AxiK5tGTaOhbzWmV0NZAA5sEwprsWB-?usp=sharing) can be found (in the COCO fomat used by MaskRCNN)
NOTE: Access limited to PDG colaborators request access
### Inferencing
Once the model is trained you can try to do an inference on an [actual high resoultion satelite images](https://drive.google.com/drive/folders/1wr4jz6ZMa4mUYYlYfzCNv67V37OoPoZt?usp=sharing). There are two files one is a large file 5.6 GB Average size of files used is about 7GB and a smaller file 50 MB. You may have to tile it to smaller chunks.
NOTE: Access limited to PDG colaborators request access.
