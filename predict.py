import torch as t
from torch import nn
import os
from model import CTPN
from get_anchors import get_anchors, get_i_j_anchor
from torchvision import transforms as T
from PIL import Image, ImageDraw
import numpy as np
from torchvision import ops
from text_line import TextProposalConnectorOriented
from copy import deepcopy
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
anchor_count = 10
backbone_type = "resnet34"
img_size = (640, 640)
use_best_model = True
cls_score_thresh = 0.2
nms_iou_thresh = 0.15
result_output_dir = r"result"
is_draw_proposals = False
is_draw_textline_bbox = True
image_pth = r"/home/yuyang/data/id_card/train_image"
transformer = T.Compose([
    T.Resize(img_size),
    T.ToTensor()
])
softmax_op = nn.Softmax(dim=3)
anchors = get_anchors(anchor_count)
textConn = TextProposalConnectorOriented()


def load_model():
    model = CTPN(anchor_count, backbone_type)
    model = nn.DataParallel(module=model, device_ids=[0])
    if use_best_model:
        model.load_state_dict(t.load("best.pth"))
    else:
        model.load_state_dict(t.load("epoch.pth"))
    model = model.cuda(0)
    model.eval()
    return model


model = load_model()


def load_one_image(img_pth):
    image_pil = Image.open(img_pth)
    image_tensor = transformer(image_pil).unsqueeze(0).cuda(0)
    return image_tensor, image_pil


def predict_one_image(image_tensor, image_pil, result_save_name):
    """

    :param image_tensor: image tensor
    :param image_pil: original PIL image
    :param result_save_name: result file name, if set None will not save result
    :return: text_on_orig_image: text line bounding box ndarray, format is as follow:
    [
    [x11,y11,x12,y12,x13,y13,x14,y14,score1],
    [x21,y21,x22,y22,x23,y23,x24,y24,score2],
    ....
    ]
    meaning of every element of the ndarray is [top left point, top right point, bottom left point, bottom right point, score]
    """
    with t.no_grad():
        rpn_cls, rpn_reg, side_ref = model(image_tensor)
    rpn_cls = rpn_cls.squeeze(0).view((rpn_cls.size()[1], rpn_cls.size()[2], -1, 2)).cpu()  # [H, W, anchor_count * 2]
    rpn_reg = rpn_reg.squeeze(0).cpu()  # [H, W, anchor_count * 2]
    side_ref = side_ref.squeeze(0).cpu()  # [H, W, anchor_count]
    rpn_cls_pos_prob = softmax_op(rpn_cls)[:, :, :, 1].view((rpn_cls.size()[0], rpn_cls.size()[1], -1)).numpy()
    j_s, i_s, anchor_indexs = np.where(rpn_cls_pos_prob > cls_score_thresh)
    all_predict_proposals = []
    for j, i, anchor_index in zip(j_s, i_s, anchor_indexs):
        pos_prob = rpn_cls_pos_prob[j, i, anchor_index]
        anchors = get_i_j_anchor(i, j, img_size, anchor_count)
        anchor = anchors[anchor_index]
        ha = anchor[-1] - anchor[1]
        cxa = (anchor[2] + anchor[0]) / 2
        cya = (anchor[-1] + anchor[1]) / 2
        vc, vh = rpn_reg[j, i, anchor_index * 2:anchor_index * 2 + 2].numpy()
        cy_pred = vc * ha + cya
        h_pred = np.exp(vh) * ha
        y_start = int(cy_pred - h_pred / 2)
        y_end = int(cy_pred + h_pred / 2)
        y_start = 0 if y_start < 0 else y_start
        y_end = img_size[0] if y_end >= img_size[0] else y_end
        x_start = i * 16
        x_end = (i + 1) * 16
        side_ref_of_current_proposal = side_ref[j, i, anchor_index] * 16 + cxa
        if side_ref_of_current_proposal < 0:
            side_ref_of_current_proposal = 0
        if side_ref_of_current_proposal >= img_size[1]:
            side_ref_of_current_proposal = img_size[1]
        predict_proposal = [x_start, y_start, x_end, y_end, pos_prob, side_ref_of_current_proposal]
        all_predict_proposals.append(predict_proposal)
    all_predict_proposals = np.array(all_predict_proposals)
    text_on_orig_image = np.array([])
    if all_predict_proposals.tolist():
        nms_keep_index = ops.nms(t.from_numpy(all_predict_proposals[:, :4]), t.from_numpy(all_predict_proposals[:, 4]), nms_iou_thresh)
        all_predict_proposals_after_nms = all_predict_proposals[nms_keep_index, :]
        if len(nms_keep_index) == 1:
            all_predict_proposals_after_nms = all_predict_proposals_after_nms.reshape((1, -1))
        all_predict_boxes_on_resized_image = all_predict_proposals_after_nms[:, :4]
        all_predict_pos_prob = all_predict_proposals_after_nms[:, 4]
        # print("text proposals probability:", all_predict_pos_prob)
        all_predict_side_ref_on_resized_image = all_predict_proposals_after_nms[:, 5]
        # connect proposals become a text line
        text_on_resized_image = textConn.get_text_lines(all_predict_boxes_on_resized_image, all_predict_pos_prob, all_predict_side_ref_on_resized_image, img_size)
        text_on_orig_image = deepcopy(text_on_resized_image)
        orig_w, orig_h = image_pil.size
        w_ratio = orig_w / img_size[1]
        h_ratio = orig_h / img_size[0]
        text_on_orig_image[:, 0:8:2] = text_on_orig_image[:, 0:8:2] * w_ratio
        text_on_orig_image[:, 1:8:2] = text_on_orig_image[:, 1:8:2] * h_ratio
        #####################################
        orig_img_draw = ImageDraw.Draw(image_pil)
        if is_draw_textline_bbox:
            for line_resize, line_orig in zip(text_on_resized_image, text_on_orig_image):
                orig_img_draw.line([(int(line_orig[0]), int(line_orig[1])), (int(line_orig[2]), int(line_orig[3]))], fill="blue", width=4)
                orig_img_draw.line([(int(line_orig[0]), int(line_orig[1])), (int(line_orig[4]), int(line_orig[5]))], fill="blue", width=4)
                orig_img_draw.line([(int(line_orig[6]), int(line_orig[7])), (int(line_orig[2]), int(line_orig[3]))], fill="blue", width=4)
                orig_img_draw.line([(int(line_orig[4]), int(line_orig[5])), (int(line_orig[6]), int(line_orig[7]))], fill="blue", width=4)
        if is_draw_proposals:
            for bbox, pos_prob_of_bbox in zip(all_predict_boxes_on_resized_image, all_predict_pos_prob):
                orig_img_draw.rectangle(((int(bbox[0] * w_ratio), int(bbox[1] * h_ratio)), (int(bbox[2] * w_ratio), int(bbox[3] * h_ratio))), outline=int(255 * pos_prob_of_bbox / 0.01), width=4, fill=None)
    if result_save_name:
        image_pil.save(os.path.join(result_output_dir, result_save_name))
        #####################################
    return text_on_orig_image


def main():
    if os.path.exists(result_output_dir):
        shutil.rmtree(result_output_dir)
    os.mkdir(result_output_dir)
    if os.path.isdir(image_pth):
        index_imgpths = [[i, os.path.join(image_pth, name)] for i, name in enumerate(os.listdir(image_pth))]
        for i, imgpth in index_imgpths:
            print(imgpth)
            result_save_name = "%d.png" % (i,)
            image_tensor, image_pil = load_one_image(imgpth)
            predict_one_image(image_tensor, image_pil, result_save_name)
    if os.path.isfile(image_pth):
        print(image_pth)
        image_tensor, image_pil = load_one_image(image_pth)
        predict_one_image(image_tensor, image_pil, "result.png")


if __name__ == "__main__":
    main()
