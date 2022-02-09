# From the tensorflow/models/research/ directory

PIPELINE_CONFIG_PATH='/content/drive/MyDrive/F-RCNN/DATA/PRETRAINEDMODEL/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8/pipeline.config'
MODEL_DIR='/content/drive/MyDrive/F-RCNN/TRAINIGDIR'
python object_detection/model_main_tf2.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --alsologtostderr

