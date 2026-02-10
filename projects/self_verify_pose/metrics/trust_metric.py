from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from mmengine.evaluator import BaseMetric

from mmpose.registry import METRICS


def _binary_auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUROC without sklearn."""
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)

    pos = (y_true == 1)
    neg = (y_true == 0)
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')

    order = np.argsort(-y_score, kind='mergesort')
    y_true_sorted = y_true[order]

    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)

    tpr = tps / max(n_pos, 1)
    fpr = fps / max(n_neg, 1)

    # prepend (0,0)
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    return float(np.trapz(tpr, fpr))


def _binary_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute average precision (AUPR) without sklearn."""
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)

    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return float('nan')

    order = np.argsort(-y_score, kind='mergesort')
    y_true_sorted = y_true[order]

    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)

    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / n_pos

    # AP: sum over points where y_true==1 of precision at that point, / n_pos
    ap = float((precision[y_true_sorted == 1]).sum() / n_pos)
    return ap


def _ece(y_true: np.ndarray, y_score: np.ndarray, num_bins: int = 15) -> float:
    """Expected calibration error for binary probabilities."""
    y_true = y_true.astype(np.float64)
    y_score = np.clip(y_score.astype(np.float64), 0.0, 1.0)

    if y_true.size == 0:
        return float('nan')

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for i in range(num_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == num_bins - 1:
            mask = (y_score >= lo) & (y_score <= hi)
        else:
            mask = (y_score >= lo) & (y_score < hi)
        if not np.any(mask):
            continue
        conf = float(y_score[mask].mean())
        acc = float(y_true[mask].mean())
        ece += abs(conf - acc) * float(mask.mean())
    return float(ece)


@METRICS.register_module()
class JointTrustMetric(BaseMetric):
    """Evaluate joint-level trust scores against GT correctness labels.

    This metric expects the model to output `pred_instances.keypoint_trust`
    with shape (num_instances, K).

    We build a binary label per joint:
        correct = (L2(pred_xy - gt_xy) <= dist_thr * max(bbox_w, bbox_h))

    And report AUROC/AUPR/ECE and acc@0.5.

    Args:
        dist_thr (float): correctness threshold as a fraction of bbox size.
        num_bins (int): number of bins for ECE.
        prefix (str): metric name prefix.
    """

    def __init__(self,
                 dist_thr: float = 0.05,
                 num_bins: int = 15,
                 prefix: str = 'trust',
                 collect_device: str = 'cpu'):
        super().__init__(collect_device=collect_device)
        self.dist_thr = float(dist_thr)
        self.num_bins = int(num_bins)
        self.prefix = str(prefix)

    def process(self, data_batch: dict, data_samples: Sequence) -> None:
        for ds in data_samples:
            if not hasattr(ds, 'pred_instances') or not hasattr(ds, 'gt_instances'):
                continue
            pred = ds.pred_instances
            gt = ds.gt_instances

            if 'keypoint_trust' not in pred:
                continue
            if 'keypoints' not in pred or 'keypoints' not in gt:
                continue

            # In top-down pose, bbox information may be stored in pred_instances
            # (copied from gt_instances by the estimator). Some datasets/pipelines
            # may not keep `gt_instances.bboxes`.
            if 'bboxes' in gt:
                bbox_src = gt
            elif 'bboxes' in pred:
                bbox_src = pred
            else:
                continue

            # top-down: typically one instance per sample
            pred_kpts = pred.keypoints[0, :, :2].detach().cpu().numpy()
            gt_kpts = gt.keypoints[0, :, :2].detach().cpu().numpy()

            trust = pred.keypoint_trust[0].detach().cpu().numpy()

            bbox = bbox_src.bboxes[0].detach().cpu().numpy()
            bbox_w = float(max(bbox[2] - bbox[0], 1.0))
            bbox_h = float(max(bbox[3] - bbox[1], 1.0))
            norm = max(bbox_w, bbox_h)

            if 'keypoints_visible' in gt:
                vis = gt.keypoints_visible[0].detach().cpu().numpy().astype(np.float64)
                vis_mask = vis > 0
            else:
                vis_mask = np.ones((pred_kpts.shape[0],), dtype=bool)

            dist = np.sqrt(((pred_kpts - gt_kpts)**2).sum(axis=-1))
            y_true = (dist <= self.dist_thr * norm).astype(np.int64)
            y_score = np.clip(trust.astype(np.float64), 0.0, 1.0)

            y_true = y_true[vis_mask]
            y_score = y_score[vis_mask]

            if y_true.size == 0:
                continue

            self.results.append((y_true, y_score))

    def compute_metrics(self, results: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
        if len(results) == 0:
            return {}

        y_true = np.concatenate([r[0] for r in results], axis=0)
        y_score = np.concatenate([r[1] for r in results], axis=0)

        out = {
            f'{self.prefix}/AUROC': _binary_auc_roc(y_true, y_score),
            f'{self.prefix}/AUPR': _binary_average_precision(y_true, y_score),
            f'{self.prefix}/ECE': _ece(y_true, y_score, num_bins=self.num_bins),
            f'{self.prefix}/Acc@0.5': float(((y_score >= 0.5) == (y_true == 1)).mean()),
            f'{self.prefix}/PosRate': float((y_true == 1).mean()),
        }
        return out
