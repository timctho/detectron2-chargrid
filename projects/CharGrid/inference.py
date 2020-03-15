import os
import random
import cv2

import detectron2.utils.comm as comm
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader, get_detection_dataset_dicts, \
    MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, verify_results
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import logging
import glob
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

    if args.img != '':
        show_prediction(args.img, predictor, args.scale)
    elif args.dir != '':
        files = []
        for ext in ['/*.jpg', '/*.png']:
            files.extend(glob.glob(args.dir + ext))
        for file in files:
            key = show_prediction(file, predictor, args.scale)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
    else:
        dataset_dicts = get_detection_dataset_dicts(['bizcard_val'])
        for d in random.sample(dataset_dicts, 300):
            key = show_prediction(d['file_name'], predictor, args.scale)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break


def show_prediction(img_file, predictor, scale):
    im = cv2.imread(img_file)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=MetadataCatalog.get('bizcard_val'),
                   scale=scale,
                   instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                   )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow('', v.get_image()[:, :, ::-1])
    key = cv2.waitKey(0)
    return key


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--img', default='', type=str)
    parser.add_argument('--dir', default='', type=str)
    parser.add_argument('--scale', default=0.4, type=float)
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
