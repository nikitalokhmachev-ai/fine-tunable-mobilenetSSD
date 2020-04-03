def make_config(path_to_config,
                num_classes, 
                batch_size=12,
                fine_tune_checkpoint='models/ssdmn_fine_tuned/ssd_mobilenet_v1_coco_11_06_2017/model.ckpt',
                input_path_train='dataset/train.record',
                label_map_path='dataset/model_label_map.pbtxt',
                input_path_test='dataset/valid.record'):
    
    with open(path_to_config, "r") as file:
        config_data = file.read()
    
    config_data = config_data.replace('num_classes: 90', 'num_classes: ' + str(num_classes))
    config_data = config_data.replace('batch_size: 24', 'batch_size: ' + str(batch_size))
    config_data = config_data.replace('PATH_TO_BE_CONFIGURED/model.ckpt', fine_tune_checkpoint)
    config_data = config_data.replace('PATH_TO_BE_CONFIGURED/mscoco_train.record-?????-of-00100', input_path_train)
    config_data = config_data.replace('PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt', label_map_path)
    config_data = config_data.replace('PATH_TO_BE_CONFIGURED/mscoco_val.record-?????-of-00010', input_path_test)
    
    return(config_data)
    
    