# Fine-tunable MobileNetSSD
This project makes the process of dataset making and fine-tuning easy and automatic. 
It consists of two parts:
  * OpenCV tracking for making images and .xml-files based on WebCam and fine-tuning<br>
  * Object detector that loads the fine-tuned model<br>
  
TESTED ONLY ON GPU AND WINDOWS 10!

## Instructions

### Setup
  
  * <a href='https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#environment-setup'>Setup an environment with TensorFlow GPU 1.15 and object detection API</a><br>
  * Install <a href='requirements.txt'>requirements.txt</a><br>
  * Download <a href='http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz'>checkpoint</a> and put it in "models" folder<br>
  * Create "train_annot", "train_images", "valid_annot", "valid_images" folders in the "dataset" folder<br>

### Quick start
  
  * Run <a href='opencv_tracking.py'>Dataset Maker</a> to start preparing your dataset <br>
  * Run <a href='detection_inference_test.py'>Detector</a> to start detecting objects from the created dataset <br>
  
### Dataset maker usage

  * Enter the name of the class you want the model to train on. Your webcam will be turned on<br>
  * Press "s" and draw a bounding box around the object with your mouse <br>
  * Press "Space" to start tracking the object<br>
  * Press "t" to make an image and an .xml-file and put those in your "train_images" and "train_annot" folders respectively<br>
  * Press "v" to make an image and an .xml-file and put those in your "valid_images" and "valid_annot" folders respectively<br>
  * Press "q" and run <a href='opencv_tracking.py'>Dataset Maker</a> once again if you want to make a new class<br>
  * You can also add ready-made images and .xml-files to "train_annot", "train_images", "valid_annot", "valid_images" folders<br>
  * Press "c" if you want the model to be fine-tuned on the dataset you made<br>