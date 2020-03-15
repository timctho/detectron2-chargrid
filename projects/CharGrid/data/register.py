from detectron2.data.datasets import register_coco_instances
from data.data_reader import BIZCARD_LABEL_MAP
import logging
import yaml

with open('config/dataset_info.yaml', 'r') as f:
    dataset_info = yaml.load(f)

logging.info('Register Bizcard Train Set.')
register_coco_instances(
    dataset_info['DATASET']['BIZCARD']['TRAIN']['NAME'],
    {'thing_classes': list(BIZCARD_LABEL_MAP.keys())},
    dataset_info['DATASET']['BIZCARD']['TRAIN']['COCO_GT'],
    dataset_info['DATASET']['BIZCARD']['TRAIN']['IMAGE_DIR'])

logging.info('Register Bizcard Val Set.')
register_coco_instances(
    dataset_info['DATASET']['BIZCARD']['VAL']['NAME'],
    {'thing_classes': list(BIZCARD_LABEL_MAP.keys())},
    dataset_info['DATASET']['BIZCARD']['VAL']['COCO_GT'],
    dataset_info['DATASET']['BIZCARD']['VAL']['IMAGE_DIR'])
