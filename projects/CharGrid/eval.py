import os
import random
import cv2

import detectron2.utils.comm as comm
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader, get_detection_dataset_dicts, MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, verify_results
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from data.data_reader import BIZCARD_LABEL_MAP
import logging


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cfg = get_cfg()
    cfg.merge_from_file('config/mask_rcnn_R50_FPN_3x.yaml')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.freeze()

    register_coco_instances("bizcard_train", {}, "data/bizcard_coco_train.json",
                            "/data/training/business_card/input/source_images")
    register_coco_instances("bizcard_val", {'things_classes': list(BIZCARD_LABEL_MAP.keys())}, "data/bizcard_coco_val_sanity.json",
                            "/data/training/business_card/input/source_images")

    dataset_dicts = get_detection_dataset_dicts(['bizcard_val'])

    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("bizcard_val", cfg, False, cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "bizcard_val")
    inference_on_dataset(predictor.model, val_loader, evaluator)