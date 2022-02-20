import math


def get_anchors(anchor_count):
    anchor_base_size = 11
    anchors = [math.ceil(anchor_base_size / 0.7 ** i) for i in range(anchor_count)]
    return anchors


def get_i_j_anchor(i, j, img_size, anchor_count):
    """

    :param i: pixel x index of conv5 feature
    :param j: pixel y index of conv5 feature
    :return: anchors localtion of point (x, y) in conv5 feature in resized image, format like [[tl_x_1, tl_y_1, br_x_1, br_y_1], [tl_x_2, tl_y_2, br_x_2, br_y_2], ......, [tl_x_anchor_count, tl_y_anchor_count, br_x_anchor_count, br_y_anchor_count]]
    """
    x_start_end_of_every_anchor = list(zip(range(0, img_size[1], 16), range(16, img_size[1] + 1, 16)))
    anchor_heights = get_anchors(anchor_count)
    y_start_end_of_every_anchor = [[[y_start, y_start + anchor_h] for anchor_h in anchor_heights] for y_start
                                        in range(0, img_size[0],
                                                 16)]  # self.y_start_end_of_every_anchor[i] represent the start and end of anchor of line i of conv5 feature
    x_start_end = x_start_end_of_every_anchor[i]
    y_start_end = y_start_end_of_every_anchor[j]
    anchors = [[x_start_end[0], y_s_e[0], x_start_end[1], y_s_e[1]] for y_s_e in y_start_end]
    return anchors