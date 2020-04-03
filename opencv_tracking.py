from data_utils import xml_to_csv, generate_pbtxt, make_config

import xmltodict
import imutils
import time
import cv2
import os

PATH_TO_DATA = 'dataset'
os.chdir(PATH_TO_DATA)

FULL_PATH = os.getcwd()

template_name = 'template.xml'

with open(template_name, "r", encoding='utf-8') as file:
    template_xml = file.read()


obj_class = str(input())

xml_d = xmltodict.parse(template_xml)

tr_ims_p = os.listdir(FULL_PATH + '\\' + 'train_images')
val_ims_p = os.listdir(FULL_PATH + '\\' + 'valid_images')

tr_cur_obj_lst = [int(el.replace(obj_class,'').replace('.jpg','')) for el in tr_ims_p if obj_class in el]
val_cur_obj_lst = [int(el.replace(obj_class,'').replace('.jpg','')) for el in val_ims_p if obj_class in el]

counter_t = max(tr_cur_obj_lst)+1 if len(tr_cur_obj_lst) != 0 else 0
counter_v = max(val_cur_obj_lst)+1 if len(val_cur_obj_lst) != 0 else 0

OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create
}

tracker = OPENCV_OBJECT_TRACKERS["csrt"]()
initBB = None

print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)
time.sleep(1.0)

while True:
    ret, frame = vs.read()
    frame = frame
    
    if ret is False:
        break

    frame = imutils.resize(frame, width=500)
    (H, W) = frame.shape[:2]
    frame_s = frame.copy()
    
    if initBB is not None:
        (success, box) = tracker.update(frame)
		
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("s"):
        initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        tracker.init(frame, initBB)
            
    elif key == ord("t"):
        if success:
            im_name = obj_class + str(counter_t) + '.jpg'
            im_full_name = FULL_PATH + '\\train_images\\' + im_name
            cv2.imwrite(im_full_name, frame_s)
            
            anot_dict = xml_d.copy()
            xml_d["annotation"]["folder"] = 'train_images'
            xml_d["annotation"]["filename"] = im_name
            xml_d["annotation"]["path"] = im_full_name
            xml_d["annotation"]["size"]["width"] = str(W)
            xml_d["annotation"]["size"]["height"] = str(H)
            xml_d["annotation"]["object"]["name"] = obj_class
            xml_d["annotation"]["object"]["bndbox"]["xmin"] = str(x)
            xml_d["annotation"]["object"]["bndbox"]["xmax"] = str(x + w)
            xml_d["annotation"]["object"]["bndbox"]["ymin"] = str(y)
            xml_d["annotation"]["object"]["bndbox"]["ymax"] = str(y + h)
            
            im_anot = xmltodict.unparse(anot_dict, pretty=True)
            im_anot = im_anot.split('<?xml version="1.0" encoding="utf-8"?>\n')[1] + '\n'
            
            xml_name = FULL_PATH + '\\train_annot\\' + im_name[:-4] + '.xml'
            with open(xml_name, "w") as file:
                file.write(im_anot)
            
            print('{} was saved to training folders!'.format(im_name[:-4]))
            counter_t += 1
    
    elif key == ord("v"):
        if success:
            im_name = obj_class + str(counter_v) + '.jpg'
            im_full_name = FULL_PATH + '\\valid_images\\' + im_name
            cv2.imwrite(im_full_name, frame_s)
            
            anot_dict = xml_d.copy()
            xml_d["annotation"]["folder"] = 'valid_images'
            xml_d["annotation"]["filename"] = im_name
            xml_d["annotation"]["path"] = im_full_name
            xml_d["annotation"]["size"]["width"] = str(W)
            xml_d["annotation"]["size"]["height"] = str(H)
            xml_d["annotation"]["object"]["name"] = obj_class
            xml_d["annotation"]["object"]["bndbox"]["xmin"] = str(x)
            xml_d["annotation"]["object"]["bndbox"]["xmax"] = str(x + w)
            xml_d["annotation"]["object"]["bndbox"]["ymin"] = str(y)
            xml_d["annotation"]["object"]["bndbox"]["ymax"] = str(y + h)
            
            im_anot = xmltodict.unparse(anot_dict, pretty=True)
            im_anot = im_anot.split('<?xml version="1.0" encoding="utf-8"?>\n')[1] + '\n'
            
            xml_name = FULL_PATH + '\\valid_annot\\' + im_name[:-4] + '.xml'
            with open(xml_name, "w") as file:
                file.write(im_anot)
            
            print('{} was saved to validation folders!'.format(im_name[:-4]))
            counter_v += 1
            
    elif key == ord('c'):
        
        for directory in ['valid_annot','train_annot']:
            image_path = os.path.join(os.getcwd(), directory )
            xml_df = xml_to_csv.xml_to_csv(image_path)
            xml_df.to_csv('{}_labels.csv'.format(directory), index=None)
            print('{} was successfully converted xml to csv.'.format(directory))
            
            if directory == 'train_annot':
                with open('model_label_map.pbtxt', "w") as file:
                    file.write(generate_pbtxt.generate_pbtxt(xml_df))
                    print('pbtxt-file was created!')
            
            
        os.system("python ..\\data_utils\\generate_tfrecord.py --csv_input=..\\dataset\\train_annot_labels.csv  --output_path=..\\dataset\\train.record --image_dir=..\\dataset\\train_images")
        os.system("python ..\\data_utils\\generate_tfrecord.py --csv_input=..\\dataset\\valid_annot_labels.csv  --output_path=..\\dataset\\valid.record --image_dir=..\\dataset\\valid_images")
        os.chdir('../models')
        if 'ssdmn_fine_tuned' in os.listdir():
            os.system('rmdir ssdmn_fine_tuned /s /q')
        os.system('mkdir ssdmn_fine_tuned')
        os.chdir('ssdmn_fine_tuned')
        os.system('tar xzvf ../ssd_mobilenet_v1_coco_11_06_2017.tar.gz')
        os.chdir('..')
        os.system('copy ssd_mobilenet_v1_coco.config ssdmn_fine_tuned')
        config_path = 'ssdmn_fine_tuned/ssd_mobilenet_v1_coco.config'
        config_data = make_config.make_config(config_path, len(xml_df['class'].unique()))
        with open(config_path, "w") as file:
            file.write(config_data)
        
        os.chdir('..')
        if 'training' in os.listdir():
            ret = os.system('rmdir training /s /q')
            ret = os.system('mkdir training')
        ret = os.system('python train.py --logtostderr --train_dir=training --pipeline_config_path=models/ssdmn_fine_tuned/ssd_mobilenet_v1_coco.config')
        
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
vs.release()
