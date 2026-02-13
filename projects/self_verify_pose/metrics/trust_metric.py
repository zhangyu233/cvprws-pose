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

    default_prefix = 'trust'

    def __init__(self,
                 dist_thr: float = 0.05,
                 num_bins: int = 15,
                 prefix: Optional[str] = None,
                 collect_device: str = 'cpu'):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.dist_thr = float(dist_thr)
        self.num_bins = int(num_bins)

    def process(self, data_batch: dict, data_samples: Sequence) -> None:
        for ds in data_samples:
            # NOTE: In MMPose evaluation, PoseDataSample is converted to dict
            # (see MultiDatasetEvaluator), so `ds` is usually a dict.
            if isinstance(ds, dict):
                pred = ds.get('pred_instances', None)
                gt = ds.get('gt_instances', None)
                raw_ann_info = ds.get('raw_ann_info', None)
            else:
                if not hasattr(ds, 'pred_instances') or not hasattr(ds, 'gt_instances'):
                    continue
                pred = ds.pred_instances
                gt = ds.gt_instances
                raw_ann_info = None
                if hasattr(ds, 'metainfo') and isinstance(ds.metainfo, dict):
                    raw_ann_info = ds.metainfo.get('raw_ann_info', None)

            if pred is None or gt is None:
                continue

            if 'keypoint_trust' not in pred:
                continue

            if 'keypoints' not in pred:
                continue

            def _to_numpy(x):
                if x is None:
                    return None
                if hasattr(x, 'detach'):
                    return x.detach().cpu().numpy()
                return np.asarray(x)

            # Get GT keypoints.
            # In some test/val pipelines, GT keypoints may not be packed into
            # `gt_instances` (because COCO metric reads GT from ann_file).
            gt_kpts_xy = None
            gt_vis = None
            if 'keypoints' in gt:
                # May be numpy arrays (packed without tensor conversion).
                gt_kpts = _to_numpy(gt['keypoints'] if isinstance(gt, dict) else gt.keypoints)
                if gt_kpts is not None:
                    gt_kpts = np.asarray(gt_kpts)
                    if gt_kpts.ndim == 3:
                        gt_kpts_xy = gt_kpts[0, :, :2]
                    elif gt_kpts.ndim == 2:
                        gt_kpts_xy = gt_kpts[:, :2]

                if 'keypoints_visible' in gt:
                    vis_src = gt['keypoints_visible'] if isinstance(gt, dict) else gt.keypoints_visible
                    vis = _to_numpy(vis_src)
                    if vis is not None:
                        vis = np.asarray(vis)
                        gt_vis = vis[0] if vis.ndim == 2 else vis
            else:
                # Fallback: raw annotation in metainfo / dict root
                raw = raw_ann_info
                if isinstance(raw, (list, tuple)) and len(raw) > 0:
                    raw = raw[0]
                if isinstance(raw, dict) and 'keypoints' in raw:
                    kp = np.asarray(raw['keypoints'], dtype=np.float32).reshape(-1, 3)
                    gt_kpts_xy = kp[:, :2]
                    gt_vis = kp[:, 2]

            if gt_kpts_xy is None:
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
            pred_kpts_all = _to_numpy(pred['keypoints'] if isinstance(pred, dict) else pred.keypoints)
            if pred_kpts_all is None:
                continue
            pred_kpts = np.asarray(pred_kpts_all)[0, :, :2]

            trust_all = _to_numpy(pred['keypoint_trust'] if isinstance(pred, dict) else pred.keypoint_trust)
            if trust_all is None:
                continue
            trust = np.asarray(trust_all)
            if trust.ndim == 2:
                trust = trust[0]

            bbox_all = _to_numpy(bbox_src['bboxes'] if isinstance(bbox_src, dict) else bbox_src.bboxes)
            if bbox_all is None:
                continue
            bbox = np.asarray(bbox_all)[0]
            bbox_w = float(max(bbox[2] - bbox[0], 1.0))
            bbox_h = float(max(bbox[3] - bbox[1], 1.0))
            norm = max(bbox_w, bbox_h)

            if gt_vis is not None:
                vis_mask = np.asarray(gt_vis, dtype=np.float64) > 0
            else:
                vis_mask = np.ones((pred_kpts.shape[0],), dtype=bool)

            dist = np.sqrt(((pred_kpts - gt_kpts_xy)**2).sum(axis=-1))
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
            'AUROC': _binary_auc_roc(y_true, y_score),
            'AUPR': _binary_average_precision(y_true, y_score),
            'ECE': _ece(y_true, y_score, num_bins=self.num_bins),
            'Acc@0.5': float(((y_score >= 0.5) == (y_true == 1)).mean()),
            'PosRate': float((y_true == 1).mean()),
        }
        return out
