from data.data_reader import BIZCARD_LABEL_MAP, BizcardDataParser
import argparse
from pathlib import Path
import os
import json
import cv2
import numpy as np


def convert_bizcard_to_coco_format(image_dir, json_dir, id_list, out_dir, out_name):
    coco_json = {}
    images = []
    annotations = []
    categories = []

    for _, key in enumerate(BIZCARD_LABEL_MAP.keys()):
        categories.append({
            'id': BIZCARD_LABEL_MAP[key],
            'name': key
        })

    with open(id_list) as fp:
        ids = fp.readlines()

    for idx, file_id in enumerate(ids):
        file_id = Path(file_id.strip())
        print(idx, file_id)

        if not (image_dir / file_id).with_suffix('.jpg').exists():
            file_id = file_id.with_suffix('.jpeg')
        else:
            file_id = file_id.with_suffix('.jpg')

        height, width = cv2.imread(str(image_dir / file_id)).shape[:2]
        images.append({
            'file_name': str(file_id),
            'id': idx,
            'height': height,
            'width': width
        })

        try:
            gt = BizcardDataParser.parse_data(str((json_dir / file_id).with_suffix('.json')), str(image_dir / file_id))[0]
            for word in gt.words:
                anno = {
                    'id': len(annotations),
                    'image_id': idx,
                    'bbox': [word.bbox.min_x, word.bbox.min_y, (word.bbox.max_x - word.bbox.min_x), (word.bbox.max_y - word.bbox.min_y)],
                    'segmentation': [word.bbox.val],
                    'category_id': word.label,
                    'iscrowd': 0,
                    'area': cv2.contourArea(np.reshape(word.bbox.val, [-1, 2]).astype(np.float32))
                }
                annotations.append(anno)
        except Exception as e:
            print(e)
            print(str(image_dir / file_id))

    coco_json['images'] = images
    coco_json['annotations'] = annotations
    coco_json['categories'] = categories
    with open(Path(out_dir, out_name), 'w') as f:
        json.dump(coco_json, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str)
    parser.add_argument('--gt_dir', type=str)
    parser.add_argument('--data_list', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--out_name', type=str)
    args = parser.parse_args()

    # convert_bizcard_to_coco_format(
    #     Path('/data/training/business_card/input/source_images'),
    #     Path('/data/training/business_card/input/ocr_and_ground_truth/OneOCR_GA-0.1.0/Bizcard'),
    #     '/data/training/business_card/input/id_lists/20191219/validation.txt',
    #     '',
    #     'bizcard_coco_val.json')

    if not Path(args.out_dir).exists():
        Path(args.out_dir).mkdir()

    convert_bizcard_to_coco_format(
        Path(args.img_dir),
        Path(args.gt_dir),
        args.data_list,
        args.out_dir,
        args.out_name)