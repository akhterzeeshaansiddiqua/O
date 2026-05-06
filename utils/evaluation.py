"""
utils/evaluation.py
===================
Detection performance evaluation using Intersection over Union (IoU).
Computes per-class Precision, Recall, and Accuracy.
"""

from collections import defaultdict


def iou(box1: list, box2: list) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: [x1, y1, x2, y2] predicted box.
        box2: [x1, y1, x2, y2] ground truth box.

    Returns:
        IoU score between 0.0 and 1.0.
    """
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    inter_w = max(0, min(x2, x2g) - max(x1, x1g))
    inter_h = max(0, min(y2, y2g) - max(y1, y1g))
    inter = inter_w * inter_h

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2g - x1g) * (y2g - y1g)
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def evaluate_by_class(
    preds: list[tuple],
    gts: list[tuple],
    iou_thresh: float = 0.5,
) -> dict:
    """
    Evaluate detection performance per class using IoU matching.

    Args:
        preds:      List of (label, [x1,y1,x2,y2]) predicted detections.
        gts:        List of (label, [x1,y1,x2,y2]) ground truth boxes.
        iou_thresh: Minimum IoU to count a detection as a True Positive.

    Returns:
        Dict mapping class name -> {"Precision", "Recall", "Accuracy"}.
    """
    stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    matched_gt = set()

    # Match predictions to ground truths
    for p_label, p_box in preds:
        matched = False
        for i, (gt_label, gt_box) in enumerate(gts):
            if i in matched_gt:
                continue
            if p_label == gt_label and iou(p_box, gt_box) >= iou_thresh:
                stats[p_label]["TP"] += 1
                matched_gt.add(i)
                matched = True
                break
        if not matched:
            stats[p_label]["FP"] += 1

    # Count unmatched ground truths as False Negatives
    for i, (gt_label, _) in enumerate(gts):
        if i not in matched_gt:
            stats[gt_label]["FN"] += 1

    # Compute metrics per class
    results = {}
    for cls, m in stats.items():
        TP, FP, FN = m["TP"], m["FP"], m["FN"]
        results[cls] = {
            "Precision": round(TP / (TP + FP), 4) if TP + FP else 0.0,
            "Recall": round(TP / (TP + FN), 4) if TP + FN else 0.0,
            "Accuracy": round(TP / (TP + FP + FN), 4) if TP + FP + FN else 0.0,
            "TP": TP,
            "FP": FP,
            "FN": FN,
        }

    return results


def print_evaluation_report(results: dict) -> None:
    """
    Pretty-print evaluation metrics per class.

    Args:
        results: Output of evaluate_by_class().
    """
    print(f"\n{'='*60}")
    print(f"  DETECTION EVALUATION REPORT")
    print(f"{'='*60}")
    header = f"  {'Class':<12} {'Precision':>10} {'Recall':>8} {'Accuracy':>10} {'TP':>5} {'FP':>5} {'FN':>5}"
    print(header)
    print(f"  {'-'*56}")
    for cls, m in results.items():
        print(
            f"  {cls:<12} {m['Precision']:>10.4f} {m['Recall']:>8.4f} "
            f"{m['Accuracy']:>10.4f} {m['TP']:>5} {m['FP']:>5} {m['FN']:>5}"
        )
    print(f"{'='*60}\n")
