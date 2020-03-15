import cv2
from util.color_map import colormap
import numpy as np


class VizUtil(object):
    @staticmethod
    def viz_boxes(image, word_list):
        for word in word_list:
            color = colormap()[0]

            # pts = np.reshape(word.bbox.val, (4, 2)).astype(np.int32)
            # cv2.polylines(image, [pts], True, color=(int(color[0]), int(color[1]), int(color[2])))
        return image

    @staticmethod
    def viz_mask(orig_mask):
        mask = orig_mask.copy()
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)

        if mask.shape[2] == 1:
            mask = np.repeat(mask, 3, axis=2)
        color_map = colormap()
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                label = int(mask[i, j, 0])
                mask[i, j, 0] = int(color_map[label][0])
                mask[i, j, 1] = int(color_map[label][1])
                mask[i, j, 2] = int(color_map[label][2])
        return mask
