from torch.utils import data
import os
from PIL import Image, ImageDraw
from torchvision import transforms as T
import numpy as np
from get_anchors import get_anchors
import cv2
import torch as t
from collections import OrderedDict
from numpy import random as rd
import random
"""
img_dir
    img1.jpg
    img2.jpg
    ......
======================
label_dir
    gt_img1.txt
    gt_img2.txt
    ......
======================
gt_xxx.txt format(same with 'ICDAR 2015' dataset):
x1,y1,x2,y2,x3,y3,x4,y4,text1
x1,y1,x2,y2,x3,y3,x4,y4,text2
......
======================
every line of ground truth txt file is a bounding box of text,include 4 points, you should notice that the bounding box is not a standard rectangle:
top left: x1,y1
top right: x2,y2
bottom right: x3,y3
bottom left: x4,y4
"""


class MySet(data.Dataset):

    def __init__(self, anchor_batch_size, img_dir, label_dir, img_size, anchor_count, negative_anchor_iou_thresh, side_ref_dist_thresh, is_train, data_aug_level):
        """

        :param anchor_batch_size: batch size of anchors (postive+negtive)
        :param img_dir: image save dir
        :param label_dir: label save dir
        :param img_size: image size like (h, w), h and w must be multiple of 16
        :param anchor_count: number of anchor of every point in conv5 feature
        :param negative_anchor_iou_thresh: negative anchor iou threshold
        :param side_ref_dist_thresh: anchor x center to text line x center distance threshold, determin which anchor used for side refine
        :param is_train: True use data augmentation, False not use
        :param data_aug_level: 'h','m' or 'l', 'h' indicates high data augmentation, m indicates medium augmentation, l indicates low augmentation
        """
        self.data_aug_leve = data_aug_level
        self.negative_anchor_iou_thresh = negative_anchor_iou_thresh
        self.side_ref_dist_thresh = side_ref_dist_thresh
        self.batch_size = anchor_batch_size
        self.img_size = img_size
        self.is_train = is_train
        self.img_label_path = [[os.path.join(img_dir, name), os.path.join(label_dir, "gt_" + name.split(".")[0] + ".txt")] for name in os.listdir(img_dir)]
        if is_train:
            if self.data_aug_leve:
                self.transformer = T.Compose([
                    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
                    T.Resize(img_size),
                    T.ToTensor()
                ])
            else:
                self.transformer = T.Compose([
                    T.Resize(img_size),
                    T.ToTensor()
                ])
        else:
            self.transformer = T.Compose([
                T.Resize(img_size),
                T.ToTensor()
            ])
        self.to_pil = T.ToPILImage()
        self.x_start_end_of_every_anchor = list(zip(range(0, img_size[1], 16), range(16, img_size[1] + 1, 16)))
        self.anchor_heights = get_anchors(anchor_count)
        self.y_start_end_of_every_anchor = [[[y_start, y_start + anchor_h] for anchor_h in self.anchor_heights] for y_start in range(0, img_size[0], 16)]  # self.y_start_end_of_every_anchor[i] represent the start and end of anchor of line i of conv5 feature

    def __getitem__(self, index):
        img_pth, label_pth = self.img_label_path[index]
        img_pil = Image.open(img_pth)
        orig_w, orig_h = img_pil.size
        with open(label_pth, "r", encoding="utf-8-sig") as file:
            text_bboxes = np.array([[int(i) for i in line.split(",")[:8]] for line in file.readlines()]).astype(float)  # shape: [n, 8], every line of this ndarray is a bounding box of text of current image, so the 'n' indicates there are n bounding boxes in current image
        if self.is_train:
            img_pil, text_bboxes = self.data_aug(img_pil, text_bboxes, orig_h, orig_w)
        img_tensor = self.transformer(img_pil)
        h_ratio = orig_h / self.img_size[0]
        w_ratio = orig_w / self.img_size[1]
        # convert coordinates of original image to coordinates of resized image
        text_bboxes[:, ::2] = text_bboxes[:, ::2] / w_ratio
        text_bboxes[:, 1::2] = text_bboxes[:, 1::2] / h_ratio
        text_bboxes = text_bboxes.astype(int)
        # split every text line ground truth and get a sequence of ground truth of fine-scale text proposal whose width is 16 pixel
        text_proposals_of_every_text_line = []
        positive_anchors_of_every_text_line = []
        cls_label = OrderedDict()  # format of every item of cls_label is "(i, j, anchor_index): text_mark", i, j is the localtion of pixel of conv5 feature, anchor_index is the index of anchors of current pixel, text_mark is 1 if this anchor has text, otherwise 0
        reg_label = OrderedDict()  # format of every item of reg_label is "(i, j, anchor_index): [vc*, vh*]", the meaning of i, j and anchor index is same with cls_label, vc* and vh* is the y coordinage offset, same with it in paper
        side_ref_label = OrderedDict()  # format of every item of reg_label is "(i, j, anchor_index): o*", the meaning of i, j and anchor index is same with cls_label, o* is the x coordinage offset, same with it in paper
        total_posotive_anchor_count = 0
        for box in text_bboxes:
            mask = np.zeros(shape=self.img_size, dtype=np.uint8)
            cv2.fillPoly(mask, [np.array([box[:2], box[2:4], box[4:6], box[6:]])], color=255)
            results = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(results) == 3:
                contour = results[1][0].reshape((-1, 2))
            else:
                contour = results[0][0].reshape((-1, 2))
            proposals_of_current_text_line = []  # proposals_of_current_text_line[i] is a list of proposal from left to right of current text line, format like [proposal_1, proposal_2, ....], proposal_i format like [tl_x, tl_y, br_x, br_y]
            box_xs = box[::2]
            x_min = np.min(box_xs)
            x_max = np.max(box_xs)
            i_s = int(x_min // 16)
            i_e = int(x_max // 16)
            positive_anchors_of_current_text_line = []
            for i in range(i_s, i_e + 1):
                x_start = i * 16
                x_end = (i + 1) * 16
                contour_between_x_range = contour[np.logical_and(x_start <= contour[:, 0], contour[:, 0] < x_end), :]
                if contour_between_x_range.tolist() == []:
                    continue
                y_start = np.min(contour_between_x_range[:, 1])
                y_end = np.max(contour_between_x_range[:, 1])
                current_proposal = [x_start, y_start, x_end, y_end]
                proposals_of_current_text_line.append(current_proposal)
                # select best iou anchor with current text proposal

                if total_posotive_anchor_count < self.batch_size / 2:
                    max_iou_i = i
                    max_iou = -float("inf")
                    for j in range(self.img_size[0] // 16):
                        anchors_of_current_x_start_x_end = self.get_i_j_anchor(i, j)
                        for anchor_index, anchor in enumerate(anchors_of_current_x_start_x_end):
                            current_iou = self.bb_intersection_over_union(anchor, current_proposal)
                            if current_iou > max_iou and (max_iou_i, j, anchor_index) not in cls_label:
                                max_iou = current_iou
                                max_iou_j = j
                                max_iou_anchor = anchor
                                max_iou_anchor_index = anchor_index
                    if max_iou >= self.negative_anchor_iou_thresh:
                        cls_label[(max_iou_i, max_iou_j, max_iou_anchor_index)] = 1
                        reg_label[(max_iou_i, max_iou_j, max_iou_anchor_index)] = [((current_proposal[1] + current_proposal[3]) / 2 - (max_iou_anchor[1] + max_iou_anchor[3]) / 2) / (max_iou_anchor[3] - max_iou_anchor[1]), np.log((current_proposal[3] - current_proposal[1]) / (max_iou_anchor[3] - max_iou_anchor[1]))]
                        positive_anchors_of_current_text_line.append([max_iou_i, max_iou_j, max_iou_anchor_index, max_iou_anchor, current_proposal])
                        total_posotive_anchor_count += 1
            for positive_anchor in positive_anchors_of_current_text_line:
                text_line_center_x = (x_min + x_max) / 2
                anchor = positive_anchor[3]
                anchor_center_x = (anchor[0] + anchor[2]) / 2
                anchor_w = anchor[2] - anchor[0]
                if np.min(np.abs([anchor_center_x - x_min, anchor_center_x - x_max])) < self.side_ref_dist_thresh:
                    side_ref_label[(positive_anchor[0], positive_anchor[1], positive_anchor[2])] = (x_min - anchor_center_x) / anchor_w if anchor_center_x < text_line_center_x else (x_max - anchor_center_x) / anchor_w
            positive_anchors_of_every_text_line.append(positive_anchors_of_current_text_line)
            text_proposals_of_every_text_line.append(proposals_of_current_text_line)
        # negative_anchors = []
        total_negative_anchor_count = self.batch_size - total_posotive_anchor_count
        negative_anchor_count = 0
        stop_add_neg_anchor = False
        col_indexs = list(range(self.img_size[1] // 16))
        row_indexs = list(range(self.img_size[0] // 16))
        random.shuffle(col_indexs)
        random.shuffle(row_indexs)
        for i in col_indexs:
            for j in row_indexs:
                anchors = self.get_i_j_anchor(i, j)
                for anchor_index, anchor in enumerate(anchors):
                    if (i, j, anchor_index) in cls_label:
                        continue
                    negative_mark = True
                    for proposal_line in text_proposals_of_every_text_line:
                        for proposal in proposal_line:
                            iou = self.bb_intersection_over_union(anchor, proposal)
                            if iou >= self.negative_anchor_iou_thresh:
                                negative_mark = False
                                break
                        if not negative_mark:
                            break
                    if negative_mark:
                        # negative_anchors.append([i, j, anchor_index, anchor])
                        cls_label[(i, j, anchor_index)] = 0
                        negative_anchor_count += 1
                        stop_add_neg_anchor = negative_anchor_count == total_negative_anchor_count
                    if stop_add_neg_anchor:
                        break
                if stop_add_neg_anchor:
                    break
            if stop_add_neg_anchor:
                break
        return img_tensor.unsqueeze(0), cls_label, reg_label, side_ref_label

    def __len__(self):
        return len(self.img_label_path)

    def get_i_j_anchor(self, i, j):
        """

        :param i: pixel x index of conv5 feature
        :param j: pixel y index of conv5 feature
        :return: anchors localtion of point (x, y) in conv5 feature in resized image, format like [[tl_x_1, tl_y_1, br_x_1, br_y_1], [tl_x_2, tl_y_2, br_x_2, br_y_2], ......, [tl_x_anchor_count, tl_y_anchor_count, br_x_anchor_count, br_y_anchor_count]]
        """
        x_start_end = self.x_start_end_of_every_anchor[i]
        y_start_end = self.y_start_end_of_every_anchor[j]
        anchors = [[x_start_end[0], y_s_e[0], x_start_end[1], y_s_e[1]] for y_s_e in y_start_end]
        return anchors

    @staticmethod
    def bb_intersection_over_union(boxA, boxB):
        boxA = [int(x) for x in boxA]
        boxB = [int(x) for x in boxB]

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    def random_v_flip(self, img_pil, text_boxes, orig_h):
        text_boxes[..., 1::2] = orig_h - text_boxes[..., 1::2] - 1
        img_pil = img_pil.transpose(Image.FLIP_TOP_BOTTOM)
        return img_pil, text_boxes

    def random_h_flip(self, img_pil, text_boxes, orig_w):
        text_boxes[..., ::2] = orig_w - text_boxes[..., ::2] - 1
        img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
        return img_pil, text_boxes

    def random_v_shift(self, img_pil, text_boxes, orig_h):
        shift_pixels = rd.randint(-int(orig_h / 8), int(orig_h / 8))
        img_ndarray = np.array(img_pil)
        padding_zero = np.zeros(shape=[np.abs(shift_pixels), img_ndarray.shape[1], 3], dtype=img_ndarray.dtype)
        if shift_pixels == 0:
            return img_pil, text_boxes
        if shift_pixels > 0:
            img_pil = Image.fromarray(np.concatenate([padding_zero, img_ndarray[:orig_h - shift_pixels, :, :]], axis=0))
        else:
            img_pil = Image.fromarray(np.concatenate([img_ndarray[-shift_pixels:, :, :], padding_zero], axis=0))
        text_boxes[..., 1::2] = text_boxes[..., 1::2] + shift_pixels
        text_boxes = text_boxes[np.logical_or(np.logical_and(np.min(text_boxes[..., 1::2], axis=1) < (orig_h - 1), np.min(text_boxes[..., 1::2] >= 0, axis=1)), np.logical_and(np.max(text_boxes[..., 1::2], axis=1) <= (orig_h - 1), np.max(text_boxes[..., 1::2] > 0, axis=1))), :]
        text_boxes[..., 1::2] = np.where(text_boxes[..., 1::2] >= orig_h, orig_h - 1, text_boxes[..., 1::2])
        text_boxes[..., 1::2] = np.where(text_boxes[..., 1::2] < 0, 0, text_boxes[..., 1::2])
        return img_pil, text_boxes

    def random_h_shift(self, img_pil, text_boxes, orig_w):
        shift_pixels = rd.randint(-int(orig_w / 8), int(orig_w / 8))
        img_ndarray = np.array(img_pil)
        padding_zero = np.zeros(shape=[img_ndarray.shape[0], np.abs(shift_pixels), 3], dtype=img_ndarray.dtype)
        if shift_pixels == 0:
            return img_pil, text_boxes
        if shift_pixels > 0:
            img_pil = Image.fromarray(np.concatenate([padding_zero, img_ndarray[:, :orig_w - shift_pixels, :]], axis=1))
        else:
            img_pil = Image.fromarray(np.concatenate([img_ndarray[:, -shift_pixels:, :], padding_zero], axis=1))
        text_boxes[..., 0::2] = text_boxes[..., 0::2] + shift_pixels
        text_boxes = text_boxes[np.logical_or(np.logical_and(np.min(text_boxes[..., 0::2], axis=1) < (orig_w - 1),
                                                             np.min(text_boxes[..., 0::2] >= 0, axis=1)),
                                              np.logical_and(np.max(text_boxes[..., 0::2], axis=1) <= (orig_w - 1),
                                                             np.max(text_boxes[..., 0::2] > 0, axis=1))), :]
        text_boxes[..., 0::2] = np.where(text_boxes[..., 0::2] >= orig_w, orig_w - 1, text_boxes[..., 0::2])
        text_boxes[..., 0::2] = np.where(text_boxes[..., 0::2] < 0, 0, text_boxes[..., 0::2])
        return img_pil, text_boxes

    def random_noise(self, img_pil, mean, var):
        img_ndarray = np.array(img_pil)
        img_norm = img_ndarray / 255
        noise = rd.normal(mean, var ** 0.5, img_norm.shape)
        out = img_norm + noise
        if np.min(out) < 0:
            low_clip = -1
        else:
            low_clip = 0
        out = np.clip(out, low_clip, 1.0)
        out = Image.fromarray((out * 255).astype(np.uint8))
        return out

    def random_scale(self, img_pil, orig_h, orig_w, text_boxes):
        scale_factor = rd.uniform(0.6, 0.9)
        scaled_h = int(orig_h * scale_factor)
        scaled_w = int(orig_w * scale_factor)
        img_pil_scaled = img_pil.resize((scaled_w, scaled_h), Image.BICUBIC)
        text_boxes[..., ::2] = text_boxes[..., ::2] * scale_factor
        text_boxes[..., 1::2] = text_boxes[..., 1::2] * scale_factor
        out = np.zeros(shape=(orig_h, orig_w, 3), dtype=np.uint8)
        x_start = rd.randint(0, orig_w - scaled_w + 1)
        y_start = rd.randint(0, orig_h - scaled_h + 1)
        text_boxes[..., ::2] = text_boxes[..., ::2] + x_start
        text_boxes[..., 1::2] = text_boxes[..., 1::2] + y_start
        out[y_start:y_start + scaled_h, x_start:x_start + scaled_w, :] = np.array(img_pil_scaled)
        out = Image.fromarray(out)
        return out, text_boxes

    def data_aug(self, img_pil, text_boxes, orig_h, orig_w):
        if not self.data_aug_leve:
            return img_pil, text_boxes
        if rd.random() < 0.5:
            img_pil, text_boxes = self.random_v_flip(img_pil, text_boxes, orig_h)
        if rd.random() < 0.5:
            img_pil, text_boxes = self.random_h_flip(img_pil, text_boxes, orig_w)
        if self.data_aug_leve.lower() == "l":
            return img_pil, text_boxes
        if rd.random() < 0.5:
            img_pil, text_boxes = self.random_v_shift(img_pil, text_boxes, orig_h)
        if rd.random() < 0.5:
            img_pil, text_boxes = self.random_h_shift(img_pil, text_boxes, orig_w)
        if self.data_aug_leve.lower() == "m":
            return img_pil, text_boxes
        if rd.random() < 0.5:
            img_pil, text_boxes = self.random_scale(img_pil, orig_h, orig_w, text_boxes)
        if rd.random() < 0.5:
            img_pil = self.random_noise(img_pil, 0.2, 0.001)
        return img_pil, text_boxes


def collate_fn(batch):
    imgs = []
    cls_labels = []
    reg_labels = []
    side_ref_labels = []
    for sample in batch:
        imgs.append(sample[0])
        cls_labels.append(sample[1])
        reg_labels.append(sample[2])
        side_ref_labels.append(sample[3])
    imgs = t.cat(imgs, dim=0)
    return imgs, cls_labels, reg_labels, side_ref_labels


def make_loader(img_batch_size, anchor_batch_size, img_dir, label_dir, img_size, anchor_count, negative_anchor_iou_thresh, side_ref_dist_thresh, num_workers, is_train, data_aug_level):
    loader = iter(data.DataLoader(MySet(anchor_batch_size, img_dir, label_dir, img_size, anchor_count, negative_anchor_iou_thresh, side_ref_dist_thresh, is_train, data_aug_level), batch_size=img_batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn, num_workers=num_workers))
    return loader


if __name__ == "__main__":
    s = MySet(128, r"/home/yuyang/data/ICDAR_2015/train_image", r"/home/yuyang/data/ICDAR_2015/train_label", (256, 1024), 10, 0.5, 20, True, "l")
    loader = make_loader(4, 128, r"/home/yuyang/data/ICDAR_2015/train_image", r"/home/yuyang/data/ICDAR_2015/train_label", (256, 1024), 10, 0.5, 20, 4, True, "l")
    for imgs, cls_labels, reg_labels, side_ref_labels in loader:
        input()

