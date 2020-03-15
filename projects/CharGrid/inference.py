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

if __name__ == '__main__':
    cfg = get_cfg()
    cfg.merge_from_file('config/mask_rcnn_R50_FPN_3x.yaml')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.freeze()

    predictor = DefaultPredictor(cfg)

    from detectron2.utils.visualizer import ColorMode

    register_coco_instances("bizcard_train", {}, "data/bizcard_coco_train.json",
                            "/data/training/business_card/input/source_images")
    register_coco_instances("bizcard_val", {}, "data/bizcard_coco_val.json",
                            "/data/training/business_card/input/source_images")

    dataset_dicts = get_detection_dataset_dicts(['bizcard_val'])
    for d in random.sample(dataset_dicts, 300):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=MetadataCatalog.get('bizcard_val'),
                       scale=0.8,
                       instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                       )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow('', cv2.resize((v.get_image()[:, :, ::-1]), None, fx=0.5, fy=0.5))
        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break