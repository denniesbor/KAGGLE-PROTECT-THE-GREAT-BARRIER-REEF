#Run this file inside object detection

PYTHONSCRIPT=/content/drive/MyDrive/F-RCNN/models/research/object_detection/exporter_main_v2.py
PIPELINECONFIG=/content/drive/MyDrive/F-RCNN/DATA/PRETRAINEDMODEL/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8/pipeline.config
OUTPUTDIR=/content/drive/MyDrive/F-RCNN/INFERENCE/frozen_model
TRAINEDCHECKPOINTDIR=/content/drive/MyDrive/F-RCNN/TRAINIGDIR

python3 ${PYTHONSCRIPT}  \
	--input_type='image_tensor' \
	--pipeline_config_path=${PIPELINECONFIG}\
        --output_directory=${OUTPUTDIR} \
        --trained_checkpoint_dir=${TRAINEDCHECKPOINTDIR}
