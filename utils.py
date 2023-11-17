import numpy as np
import torch
import random


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def compute_single_cls_ap(all_detections, all_annotations, iou_thres=0.5):
    """
    Inputs:
        all_detections: shape=[N, K, 5]，每张图片有K个bbox，K可能=0
        all_annotations: shape=[N, M, 4]，每张图片有M个bbox，M可能=0
    Returns:
        average precisions from 0.5 to 0.95
    """
    true_positives = []
    scores = []
    num_annotations = 0
    # 遍历batch张图片的标注
    for i in range(len(all_annotations)):
        detections = all_detections[i]
        annotations = all_annotations[i]
        # 全部正例数量
        num_annotations += annotations.shape[0]
        detected_annotations = []
        # 遍历图片中的每个bbox
        for *bbox, score in detections:
            scores.append(score)

            if annotations.shape[0] == 0:
                true_positives.append(0) # 当前box并非真正例
                continue

            overlaps = bbox_iou_numpy(np.array(bbox), annotations)
            assigned_annotation = np.argmax(overlaps) # 获取最大交并比的下标
            max_overlap = overlaps[assigned_annotation] # 获取最大交并比

            if max_overlap >= iou_thres and assigned_annotation not in detected_annotations:
                true_positives.append(1)
                detected_annotations.append(assigned_annotation)
            else:
                true_positives.append(0)

    # 如果没有物体出现在所有图片中, 在当前类的 AP 为 0
    if num_annotations == 0:
        AP = 0
    else:
        true_positives = np.array(true_positives) # 将列表转化成numpy数组
        false_positives = np.ones_like(true_positives) - true_positives

        # 按照socre进行排序
        indices = np.argsort(-np.array(scores))
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # 统计假正例和真正例
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # 计算召回率和准确率
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        AP = compute_ap(recall, precision)
    return AP


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def bbox_iou_numpy(rect1, rectangles, x1y1x2y2=True):
    """
    Arguments:
        rect1: pred bbox, shape=(4,)
        rectangles: target bboxes, shape=(B,4) or (4,)
    Returns:
        iou: iou between rect1 and each rect in rectangles.
    """
    if not x1y1x2y2:
        rect1 = xywh2xyxy(rect1)
        rectangles = xywh2xyxy(rectangles)

    # 计算交集区域的左上角坐标
    x_intersection = np.maximum(rect1[0], rectangles[:, 0])
    y_intersection = np.maximum(rect1[1], rectangles[:, 1])
    
    # 计算交集区域的右下角坐标
    x_intersection_end = np.minimum(rect1[2], rectangles[:, 2])
    y_intersection_end = np.minimum(rect1[3], rectangles[:, 3])
    
    # 计算交集区域的宽度和高度（可能为负数，表示没有重叠）
    intersection_width = np.maximum(0, x_intersection_end - x_intersection)
    intersection_height = np.maximum(0, y_intersection_end - y_intersection)
    
    # 计算交集区域的面积
    intersection_area = intersection_width * intersection_height
    
    # 计算矩形1的面积
    area_rect1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    
    # 计算其他矩形的面积
    area_rectangles = (rectangles[:, 2] - rectangles[:, 0]) * (rectangles[:, 3] - rectangles[:, 1])
    
    # 计算并集区域的面积
    iou = intersection_area / (area_rect1 + area_rectangles - intersection_area + 1e-16)
    return iou


def xywh2xyxy(pred):
    """
    Args:
        pred: (..., xywh) or (..., xywhc), 把最后一个维度从xywh变成xyxy
    Returns:
        output: (..., xyxy) or (..., xyxyc)
    """
    if type(pred) is np.ndarray:
        output = np.copy(pred).astype(np.float32)
    else:
        output = torch.clone(pred).float()
    output[..., :2] = pred[..., :2] - pred[..., 2:4] / 2
    output[..., 2:4] = pred[..., :2] + pred[..., 2:4] / 2
    return output
