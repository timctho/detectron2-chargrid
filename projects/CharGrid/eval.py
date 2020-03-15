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
import data


def setup(args):
    logging.basicConfig(level=logging.DEBUG)
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)

    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def main(args):
    cfg = setup(args)

    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("bizcard_val", cfg, False, cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "bizcard_val")
    inference_on_dataset(predictor.model, val_loader, evaluator)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
