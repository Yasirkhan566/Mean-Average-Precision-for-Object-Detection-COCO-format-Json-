import torch
from collections import Counter
def compute_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    x1_max = x1 + w1
    y1_max = y1 + h1
    x2_max = x2 + w2
    y2_max = y2 + h2

    intersect_x1 = max(x1, x2)
    intersect_y1 = max(y1, y2)
    intersect_x2 = min(x1_max, x2_max)
    intersect_y2 = min(y1_max, y2_max)

    intersect_area = max(0, intersect_x2 - intersect_x1) * max(0, intersect_y2 - intersect_y1)
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2

    iou = intersect_area / float(bbox1_area + bbox2_area - intersect_area)
    return iou

def mean_average_precision(pred_boxes, true_boxes, iou_threshold = 0.5, box_format = 'corners', num_classes = 2):
    average_precisions = []
    epsilon = 1e-6

    for c in range (num_classes):
        detections = []
        groundtruths = []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                groundtruths.append(true_box)

        amount_bboxes = Counter([gt[0] for gt in groundtruths])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        detections.sort(key = lambda x: x[2], reverse = True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(groundtruths)

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [bbox for bbox in groundtruths if bbox[0] == detection[0]]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = compute_iou(torch.tensor(detection[3:]), torch.tensor(gt[3:]))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes [detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes [detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))
    
    return sum(average_precisions) / len(average_precisions)


import json
def getPredBoxes(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    pred_boxes = []
    for item in data:
        image_id = item["image_id"]
        category_id = item["category_id"]
        score = item["score"]
        bbox = item["bbox"]
        pred_box = [image_id, category_id, score, bbox[0], bbox[1], bbox[2], bbox[3]]
        pred_boxes.append(pred_box)
    
    return pred_boxes

def getGtBoxes(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    annotations = data["annotations"]
    Gt_boxes = []
    for item in annotations:
        image_id = item["image_id"]
        category_id = item["category_id"]
        score = 1
        bbox = item["bbox"]
        Gt_box = [image_id, category_id, score, bbox[0], bbox[1], bbox[2], bbox[3]]
        Gt_boxes.append(Gt_box)
    
    return Gt_boxes


# main

gt_json_custom = "gt.json"
pred_json_custom = "pred.json"


prediction_boxes = getPredBoxes(pred_json_custom)
groundtruth_boxes = getGtBoxes(gt_json_custom)


print (mean_average_precision(prediction_boxes, groundtruth_boxes, 0.75))
                
