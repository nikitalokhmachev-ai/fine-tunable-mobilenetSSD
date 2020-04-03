import numpy as np
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import cv2
import os

from utils import label_map_util
from utils import visualization_utils as vis_util

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

os.chdir('training')
last_model = max([int(el[11:].replace('.index','')) for el in os.listdir() if el[-1] == 'x'])
os.chdir('..')
os.chdir('models/ssdmn_fine_tuned')

if 'ssdmn' in os.listdir():
    os.system('rmdir ssdmn /s /q')

os.chdir('../..')
infer_cmd = '''
python export_inference_graph.py --input_type image_tensor --pipeline_config_path models/ssdmn_fine_tuned/ssd_mobilenet_v1_coco.config --trained_checkpoint_prefix training/model.ckpt --output_directory models/ssdmn_fine_tuned/ssdmn
'''.replace('model.ckpt', 'model.ckpt-' + str(last_model))
os.system(infer_cmd)

os.chdir('dataset')
os.system('copy model_label_map.pbtxt \"../models/ssdmn_fine_tuned\" /Y')
os.chdir('..')

#PATH_TO_MODEL = 'models/original_model/'
PATH_TO_MODEL = 'models/ssdmn_fine_tuned/'

#MODEL_NAME = PATH_TO_MODEL + 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_NAME = PATH_TO_MODEL + 'ssdmn' 

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

#PATH_TO_LABELS = PATH_TO_MODEL +  'mscoco_label_map.pbtxt'
PATH_TO_LABELS = PATH_TO_MODEL + 'model_label_map.pbtxt'

with open(PATH_TO_LABELS, 'r') as f:
    data = f.read().split('item')[1:]
    
NUM_CLASSES = len(data)

cap = cv2.VideoCapture(0)
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

IMAGE_SIZE = (12, 8)


with detection_graph.as_default():
  with tf.compat.v1.Session(graph=detection_graph) as sess:
    while True:
      ret, image_np = cap.read()
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)

      cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
      if cv2.waitKey(25) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
