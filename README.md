# Fine-tunable MobileNetSSD
This project makes the process of dataset making and fine-tuning easy and automatic. 
It consists of two parts:
  * OpenCV tracking for making images and .xml-files based on WebCam and fine-tuning
  * Object detector that loads the fine-tuned model
TESTED ONLY ON GPU AND WINDOWS 10!

## Table of contents

Setup:
  
  * <a href='https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#environment-setup'>Setup an environment with TensorFlow GPU 1.15 and object detection API</a><br>
  * Install <a href='requirements.txt'>requirements.txt</a><br>
  * Download <a href='http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz'>checkpoint</a> and put it in "models" folder<br>

Quick start:
