#-*- coding:utf-8 -*-
#'''
# Created on 18-12-11 上午10:05
#
# @Author: Greg Gao(laygin)
#'''
import numpy as np
import cv2
# from config import *


class TextLineCfg:
    TEXT_PROPOSALS_WIDTH = 16
    MAX_HORIZONTAL_GAP = 50
    MIN_V_OVERLAPS = 0.6
    MIN_SIZE_SIM = 0.6
    USE_SIDE_REFINE = True


# for predict
class Graph:
    def __init__(self, graph):
        self.graph = graph

    def sub_graphs_connected(self):
        sub_graphs = []
        for index in range(self.graph.shape[0]):
            if not self.graph[:, index].any() and self.graph[index, :].any():
                v = index
                sub_graphs.append([v])
                while self.graph[v, :].any():
                    v = np.where(self.graph[v, :])[0][0]
                    sub_graphs[-1].append(v)
        return sub_graphs


class TextProposalGraphBuilder:
    """
        Build Text proposals into a graph.
    """

    def get_successions(self, index):
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) + 1, min(int(box[0]) + TextLineCfg.MAX_HORIZONTAL_GAP + 1, self.im_size[1])):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    def get_precursors(self, index):
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) - 1, max(int(box[0] - TextLineCfg.MAX_HORIZONTAL_GAP), 0) - 1, -1):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    def is_succession_node(self, index, succession_index):
        precursors = self.get_precursors(succession_index)
        if self.scores[index] >= np.max(self.scores[precursors]):
            return True
        return False

    def meet_v_iou(self, index1, index2):
        def overlaps_v(index1, index2):
            # h1 = self.heights[index1]
            # h2 = self.heights[index2]
            # union = min(h1, h2)
            y0 = max(self.text_proposals[index2][1], self.text_proposals[index1][1])
            y1 = min(self.text_proposals[index2][3], self.text_proposals[index1][3])
            ################
            v_end = max(self.text_proposals[index2][3], self.text_proposals[index1][3])
            v_start = min(self.text_proposals[index2][1], self.text_proposals[index1][1])
            union = v_end - v_start
            ################
            return max(0, y1 - y0 + 1) / union

        def size_similarity(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            return min(h1, h2) / max(h1, h2)

        return overlaps_v(index1, index2) >= TextLineCfg.MIN_V_OVERLAPS and \
               size_similarity(index1, index2) >= TextLineCfg.MIN_SIZE_SIM

    def build_graph(self, text_proposals, scores, im_size):
        self.text_proposals = text_proposals
        self.scores = scores
        self.im_size = im_size
        self.heights = text_proposals[:, 3] - text_proposals[:, 1] + 1

        boxes_table = [[] for _ in range(self.im_size[1])]
        for index, box in enumerate(text_proposals):
            boxes_table[int(box[0])].append(index)
        self.boxes_table = boxes_table

        graph = np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)

        for index, box in enumerate(text_proposals):
            successions = self.get_successions(index)
            if len(successions) == 0:
                continue
            succession_index = successions[np.argmax(scores[successions])]
            if self.is_succession_node(index, succession_index):
                # NOTE: a box can have multiple successions(precursors) if multiple successions(precursors)
                # have equal scores.
                graph[index, succession_index] = True
        return Graph(graph)


class TextProposalConnectorOriented:
    """
        Connect text proposals into text lines
    """

    def __init__(self):
        self.graph_builder = TextProposalGraphBuilder()

    def group_text_proposals(self, text_proposals, scores, im_size):
        graph = self.graph_builder.build_graph(text_proposals, scores, im_size)
        return graph.sub_graphs_connected()

    def fit_y(self, X, Y, x1, x2):
        # len(X) != 0
        # if X only include one point, the function will get line y=Y[0]
        if np.sum(X == X[0]) == len(X):
            return Y[0], Y[0]
        p = np.poly1d(np.polyfit(X, Y, 1))
        return p(x1), p(x2)

    def get_text_lines(self, text_proposals, scores, side_refines, im_size):
        """
        text_proposals:boxes
        """
        # tp=text proposal
        tp_groups = self.group_text_proposals(text_proposals, scores, im_size)  # 首先还是建图，获取到文本行由哪几个小框构成
        text_lines = np.zeros((len(tp_groups), 8), np.float32)

        for index, tp_indices in enumerate(tp_groups):
            text_line_boxes = text_proposals[list(tp_indices)]  # 每个文本行的全部小框
            if TextLineCfg.USE_SIDE_REFINE:
                side_refine_value_left = side_refines[tp_indices[0]]
                side_refine_value_right = side_refines[tp_indices[-1]]
                text_line_boxes[0, 0] = side_refine_value_left if np.abs(text_line_boxes[0, 0] - side_refine_value_left) < TextLineCfg.TEXT_PROPOSALS_WIDTH else text_line_boxes[0, 0]
                text_line_boxes[-1, 2] = side_refine_value_right if np.abs(text_line_boxes[-1, 2] - side_refine_value_right) < TextLineCfg.TEXT_PROPOSALS_WIDTH else text_line_boxes[-1, 2]
            X = (text_line_boxes[:, 0] + text_line_boxes[:, 2]) / 2  # 求每一个小框的中心x，y坐标
            Y = (text_line_boxes[:, 1] + text_line_boxes[:, 3]) / 2
            z1 = np.polyfit(X, Y, 1)  # 多项式拟合，根据之前求的中心店拟合一条直线（最小二乘）

            x0 = np.min(text_line_boxes[:, 0])  # 文本行x坐标最小值
            x1 = np.max(text_line_boxes[:, 2])  # 文本行x坐标最大值

            offset = (text_line_boxes[0, 2] - text_line_boxes[0, 0]) * 0.5  # 小框宽度的一半

            # 以全部小框的左上角这个点去拟合一条直线，然后计算一下文本行x坐标的极左极右对应的y坐标
            lt_y, rt_y = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0 + offset, x1 - offset)
            # 以全部小框的左下角这个点去拟合一条直线，然后计算一下文本行x坐标的极左极右对应的y坐标
            lb_y, rb_y = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0 + offset, x1 - offset)

            score = scores[list(tp_indices)].sum() / float(len(tp_indices))  # 求全部小框得分的均值作为文本行的均值

            text_lines[index, 0] = x0
            text_lines[index, 1] = min(lt_y, rt_y)  # 文本行上端 线段 的y坐标的小值
            text_lines[index, 2] = x1
            text_lines[index, 3] = max(lb_y, rb_y)  # 文本行下端 线段 的y坐标的大值
            text_lines[index, 4] = score  # 文本行得分
            text_lines[index, 5] = z1[0]  # 根据中心点拟合的直线的k，b
            text_lines[index, 6] = z1[1]
            height = np.mean((text_line_boxes[:, 3] - text_line_boxes[:, 1]))  # 小框平均高度
            text_lines[index, 7] = height + 2.5

        text_recs = np.zeros((len(text_lines), 9), np.float)
        index = 0
        for line in text_lines:
            b1 = line[6] - line[7] / 2  # 根据高度和文本行中心线，求取文本行上下两条线的b值
            b2 = line[6] + line[7] / 2
            x1 = line[0]
            y1 = line[5] * line[0] + b1  # 左上
            x2 = line[2]
            y2 = line[5] * line[2] + b1  # 右上
            x3 = line[0]
            y3 = line[5] * line[0] + b2  # 左下
            x4 = line[2]
            y4 = line[5] * line[2] + b2  # 右下
            disX = x2 - x1
            disY = y2 - y1
            width = np.sqrt(disX * disX + disY * disY)  # 文本行宽度

            fTmp0 = y3 - y1  # 文本行高度
            fTmp1 = fTmp0 * disY / width
            x = np.fabs(fTmp1 * disX / width)  # 做补偿
            y = np.fabs(fTmp1 * disY / width)
            if line[5] < 0:
                x1 -= x
                y1 += y
                x4 += x
                y4 -= y
            else:
                x2 += x
                y2 += y
                x3 -= x
                y3 -= y
            text_recs[index, 0] = x1 if x1 > 0 else 0
            text_recs[index, 1] = y1 if y1 > 0 else 0
            text_recs[index, 2] = x2 if x2 < im_size[1] else im_size[1] - 1
            text_recs[index, 3] = y2 if y2 > 0 else 0
            text_recs[index, 4] = x3 if x3 > 0 else 0
            text_recs[index, 5] = y3 if y3 < im_size[0] else im_size[0] - 1
            text_recs[index, 6] = x4 if x4 < im_size[1] else im_size[1] - 1
            text_recs[index, 7] = y4 if y4 < im_size[0] else im_size[0] - 1
            text_recs[index, 8] = line[4]
            index = index + 1

        return text_recs


if __name__ == "__main__":
    textConn = TextProposalConnectorOriented()
    text = textConn.get_text_lines(np.random.randint(0, 10, (5, 4)), np.random.random(2), [512, 512])
    print(text)
