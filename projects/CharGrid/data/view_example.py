from util.viz import VizUtil
from data.data_reader import BizcardDataParser
from pathlib import Path
import numpy as np
import cv2
import argparse

import random
from detectron2.data.datasets import register_coco_instances
from detectron2.data import get_detection_dataset_dicts
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog


def view_gt_dir(image_dir, json_dir, id_list):
    with open(id_list) as fp:
        ids = fp.readlines()
    for id in ids:
        # processing hidden files
        id = Path(id.strip())
        print(id)

        if not (image_dir / id).with_suffix('.jpg').exists():
            image_path = str((image_dir / id).with_suffix('.jpeg'))
        else:
            image_path = str((image_dir / id).with_suffix('.jpg'))

        key = view_single(str((json_dir / id).with_suffix('.json')), image_path)
        if key == ord('q'):
            break


def view_single(json, image_path):
    gt = BizcardDataParser.parse_data(
        json, image_path)[0].resize_by_ratio(0.15)
    mask = gt.to_mask()
    print(mask.shape)
    view = np.concatenate([VizUtil.viz_boxes(gt.image, gt.words),
                           VizUtil.viz_mask(mask),
                           mask], axis=1)
    cv2.imshow('view', view.astype(np.uint8))
    key = cv2.waitKey(0)
    if key == ord('q'):
        cv2.destroyAllWindows()
    return key


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco', action='store_true')
    args = parser.parse_args()

    if args.coco:
        for d in random.sample(get_detection_dataset_dicts(['bizcard_train']), 300):
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get('bizcard_train'), scale=0.2)
            vis = visualizer.draw_dataset_dict(d)
            cv2.imshow('', vis.get_image()[:, :, ::-1])
            key = cv2.waitKey(0)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
    else:
        view_gt_dir(
            Path('/data/training/business_card/input/source_images'),
            Path('/data/training/business_card/input/ocr_and_ground_truth/OneOCR_GA-0.1.0/Bizcard'),
            '/data/training/business_card/input/id_lists/20200206/train.txt')
