import json
from jsonref import JsonRef
import numpy as np
from copy import deepcopy
import cv2
from pathlib import Path
from detectron2.structures import BoxMode

RECEIPT_LABEL_MAP = {
    'Other': 0,
    'MerchantName': 1,
    'MerchantLogo': 2,
    'MerchantAddress': 3,
    'MerchantPhoneNumber': 4,
    'TransactionDate': 5,
    'TransactionTime': 6,
    'Total': 7,
    'Subtotal': 8,
    'Tax': 9,
    'ItemName': 10,
    'ItemPrice': 11,
    'ItemTotalPrice': 12,
    'ItemQuantity': 13,
    'ItemDescription': 14
}

RECEIPT_ITEM_LABEL_MAP = {
    'Name': 'ItemName',
    'Price': 'ItemPrice',
    'Quantity': 'ItemQuantity',
    'TotalPrice': 'ItemTotalPrice',
    'Description': 'ItemDescription'
}

BIZCARD_LABEL_MAP = {
    'Other': 1,
    'PersonName': 2,
    'Company': 3,
    'CompanyLogo': 4,
    'JobTitle': 5,
    'WorkPhone': 6,
    'Fax': 7,
    'OtherPhone': 8,
    'Address': 9,
    'MobilePhone': 10,
    'Department': 11,
    'Email': 12,
    'WebSite': 13,
    'PartialPhone': 14
}


def find_label_text(label, label_dict):
    for key in label_dict.keys():
        if label == label_dict[key]:
            return key
    return ''


class Item(object):
    def __init__(self):
        self.words = []
        self._bbox = None

    def compute_bbox(self):
        x1 = min([_word.bbox.min_x for _word in self.words])
        y1 = min([_word.bbox.min_y for _word in self.words])
        x2 = max([_word.bbox.max_x for _word in self.words])
        y2 = max([_word.bbox.max_y for _word in self.words])
        return BoundingBox([x1, y1, x2, y2])

    @property
    def bbox(self):
        if self._bbox is None:
            self._bbox = self.compute_bbox()
        return self._bbox


class Word(object):
    def __init__(self, bbox, text, label=0, confidence=1.0, word_table={}):
        self.bbox = BoundingBox(bbox)
        self.text = text
        self.label = label
        self.confidence = confidence
        self.word_table = word_table

    def __str__(self):
        return '{:<20} | {:<30} | {}'.format(
            find_label_text(self.label, self.word_table), self.text, str(self.bbox))


class BoundingBox(object):
    def __init__(self, l):
        if not isinstance(l, list):
            raise ValueError('wrong bbox.')
        self._bbox = l

    def __str__(self):
        return ' '.join([str(val) for val in self._bbox])

    @property
    def val(self):
        return self._bbox

    @property
    def min_x(self):
        return min(self._bbox[::2])

    @property
    def max_x(self):
        return max(self._bbox[::2])

    @property
    def min_y(self):
        return min(self._bbox[1::2])

    @property
    def max_y(self):
        return max(self._bbox[1::2])


class OcrVerticalGT(object):
    LABEL_MAP = {}
    ITEM_LABEL_MAP = {}

    def __init__(self, words, item_list, image):
        self.words = words
        self.item_list = item_list
        self.LABEL_MAP = OcrVerticalGT.LABEL_MAP
        self.ITEM_LABEL_MAP = OcrVerticalGT.ITEM_LABEL_MAP
        self.image = image

    @classmethod
    def from_json(cls, data, image_path):
        if data.get('fields', None) is None:
            raise ValueError('Wrong schema.')

        item_list = []
        words = []
        data = data['fields']
        for key in data.keys():
            label_id = cls.LABEL_MAP.get(key, -1)

            if data[key] is None:
                continue

            value_type = data.get(key, {}).get('valueType', None)
            if value_type is None:
                continue

            if value_type == 'stringValue':
                cls.parse_string_value(data[key], label_id, words)
            elif value_type == 'arrayValue':
                cls.parse_array_value(data[key], label_id, words, item_list)

        image = cv2.imread(image_path)
        return cls(words, item_list, image)

    @classmethod
    def parse_string_value(cls, data, label, words, item=None):
        if not isinstance(data, dict):
            return

        if data.get('elements', None) is None:
            return

        for element in data['elements']:
            _word = Word(bbox=element.get('boundingBox', []),
                         text=element.get('text', ''),
                         confidence=element.get('confidence', 1.0),
                         label=label,
                         word_table=cls.LABEL_MAP)
            words.append(_word)
            if item is not None and isinstance(item, Item):
                item.words.append(deepcopy(_word))

    @classmethod
    def parse_array_value(cls, data, label, words, item_list):
        if not isinstance(data, dict) or data.get('value', None) is None:
            return

        for item in data['value']:
            cls._parse_item(item, label, words, item_list)

    @classmethod
    def _parse_item(cls, data, label, words, item_list):
        if not isinstance(data, dict):
            return

        if data['valueType'] == 'objectValue':
            data = data['value']
            _item = Item()
            for key in data.keys():
                if key in cls.ITEM_LABEL_MAP.keys():
                    cls.parse_string_value(data[key],
                                           cls.LABEL_MAP[cls.ITEM_LABEL_MAP[key]],
                                           words,
                                           _item)
            item_list.append(_item)
        else:
            _item = Item()
            cls.parse_string_value(data,
                                   label,
                                   words,
                                   _item)
            item_list.append(_item)

    def resize_by_value(self, h, w):
        scale_h = h / self.image.shape[0]
        scale_w = w / self.image.shape[1]

        def _resize_word_bbox(bbox, scale_h_, scale_w_):
            for i in range(len(bbox.val)):
                if i % 2 == 0:
                    bbox.val[i] = int(bbox.val[i] * scale_w_)
                else:
                    bbox.val[i] = int(bbox.val[i] * scale_h_)

        for _word in self.words:
            _resize_word_bbox(_word.bbox, scale_h, scale_w)

        for _item in self.item_list:
            for _word in _item.words:
                _resize_word_bbox(_word.bbox, scale_h, scale_w)

        self.image = cv2.resize(self.image, (w, h))
        return self

    def resize_by_ratio(self, ratio):
        def _resize_word_bbox(bbox, ratio_):
            for i in range(len(bbox.val)):
                bbox.val[i] = int(bbox.val[i] * ratio_)

        for _word in self.words:
            _resize_word_bbox(_word.bbox, ratio)

        for _item in self.item_list:
            for _word in _item.words:
                _resize_word_bbox(_word.bbox, ratio)

        self.image = cv2.resize(self.image, None, fx=ratio, fy=ratio)
        return self

    def to_mask(self):
        result = np.zeros(self.image.shape)
        for _word in self.words:
            pts = np.reshape(_word.bbox.val, (4, 2)).astype(np.int32)
            cv2.fillConvexPoly(result, points=pts, color=(_word.label, _word.label, _word.label))
        return result


class ReceiptGT(OcrVerticalGT):
    LABEL_MAP = RECEIPT_LABEL_MAP
    ITEM_LABEL_MAP = RECEIPT_ITEM_LABEL_MAP

    def __init__(self, words, item_list, image):
        super().__init__(words, item_list, image)


class BizcardGT(OcrVerticalGT):
    LABEL_MAP = BIZCARD_LABEL_MAP
    ITEM_LABEL_MAP = {}

    def __init__(self, words, item_list, image):
        super().__init__(words, item_list, image)


class DataParser(object):
    GTSchema = None

    def __init__(self):
        pass

    @classmethod
    def parse_data(cls, file, image_path):
        results = []
        with open(file, 'r') as f:
            data = JsonRef.replace_refs(json.load(f))
            data = data.get('understandingResults', None)
            if data is None or not isinstance(data, list):
                raise ValueError('Wrong schema.')

            for unit in data:
                results.append(cls.GTSchema.from_json(unit, image_path))
            return results


class ReceiptDataParser(DataParser):
    GTSchema = ReceiptGT

    def __init__(self):
        super().__init__()


class BizcardDataParser(DataParser):
    GTSchema = BizcardGT

    def __init__(self):
        super().__init__()


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


def get_chargrid_dicts(image_dir, json_dir, id_list):
    dataset_dicts = []
    with open(id_list) as fp:
        ids = fp.readlines()

    for idx, file_id in enumerate(ids):
        record = {}
        file_id = Path(file_id.strip())
        print(idx, file_id)

        if not (image_dir / file_id).with_suffix('.jpg').exists():
            image_path = str((image_dir / file_id).with_suffix('.jpeg'))
        else:
            image_path = str((image_dir / file_id).with_suffix('.jpg'))

        height, width = cv2.imread(image_path).shape[:2]
        record["file_name"] = image_path
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        gt = BizcardDataParser.parse_data(str((json_dir / file_id).with_suffix('.json')), image_path)[0]
        annotations = []
        for word in gt.words:
            anno = {
                'bbox': [word.bbox.min_x, word.bbox.min_y, word.bbox.max_x, word.bbox.max_y],
                'bbox_mode': BoxMode.XYXY_ABS,
                'segmentation': [word.bbox.val],
                'category_id': word.label,
                'is_crowd': False
            }
            annotations.append(anno)
        record['annotations'] = annotations
        dataset_dicts.append(record)

    return dataset_dicts


if __name__ == '__main__':
    from util.viz import VizUtil

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


    # from detectron2.data import DatasetCatalog, MetadataCatalog
    # from detectron2.data.datasets import register_coco_instances
    # from detectron2.utils.visualizer import Visualizer
    # import random
    #
    # for d in ["train", "val"]:
    #     DatasetCatalog.register("balloon_" + d, lambda d=d: get_chargrid_dicts(Path('/data/training/business_card/input/source_images'),
    #             Path('/data/training/business_card/input/ocr_and_ground_truth/OneOCR_GA-0.1.0/Bizcard'),
    #             '/data/training/business_card/input/id_lists/20200206/train.txt'))
    #     MetadataCatalog.get("balloon_" + d).set(thing_classes=list(BIZCARD_LABEL_MAP.keys()))
    # balloon_metadata = MetadataCatalog.get("balloon_train")
    # print(balloon_metadata)
    #
    # dataset_dicts = get_chargrid_dicts(Path('/data/training/business_card/input/source_images'),
    #             Path('/data/training/business_card/input/ocr_and_ground_truth/OneOCR_GA-0.1.0/Bizcard'),
    #             '/data/training/business_card/input/id_lists/20200206/train.txt')
    # for d in random.sample(dataset_dicts, 3):
    #     img = cv2.imread(d["file_name"])
    #     visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.2)
    #     vis = visualizer.draw_dataset_dict(d)
    #     cv2.imshow('', vis.get_image()[:, :, ::-1])
    #     cv2.waitKey(0)

    convert_bizcard_to_coco_format(
        Path('/data/training/business_card/input/source_images'),
        Path('/data/training/business_card/input/ocr_and_ground_truth/OneOCR_GA-0.1.0/Bizcard'),
        '/data/training/business_card/input/id_lists/20191219/validation.txt',
        '',
        'bizcard_coco_val.json')

    # view_gt_dir(Path('/data/training/business_card/input/source_images'),
    #             Path('/data/training/business_card/input/ocr_and_ground_truth/OneOCR_GA-0.1.0/Bizcard'),
    #             '/data/training/business_card/input/id_lists/20200206/train.txt')

    # import random
    # from detectron2.data.datasets import register_coco_instances
    # from detectron2.data import get_detection_dataset_dicts
    # from detectron2.utils.visualizer import Visualizer
    # from detectron2.data import DatasetCatalog, MetadataCatalog
    #
    # register_coco_instances("my_dataset_train", {}, "bizcard_coco_train.json", "/data/training/business_card/input/source_images")
    # register_coco_instances("my_dataset_val", {}, "bizcard_coco_train.json", "/data/training/business_card/input/source_images")
    # for d in random.sample(get_detection_dataset_dicts(['my_dataset_train']), 3):
    #     img = cv2.imread(d["file_name"])
    #     visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get('my_dataset_train'), scale=0.2)
    #     vis = visualizer.draw_dataset_dict(d)
    #     cv2.imshow('', vis.get_image()[:, :, ::-1])
    #     cv2.waitKey(0)

    # from PIL import Image, ImageOps
    # image = Image.open('/data/training/business_card/input/source_images/000/Bizcard_000_273.jpg')
    # image = ImageOps.exif_transpose(image)
    # image = image.convert('RGB')
    # image = np.asarray(image)
    # print(image.shape)
    # image = cv2.imread('/data/training/business_card/input/source_images/000/Bizcard_000_273.jpg')
    # print(image.shape)
