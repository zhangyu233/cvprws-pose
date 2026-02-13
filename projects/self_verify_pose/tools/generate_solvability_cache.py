from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint

from mmpose.utils.setup_env import register_all_modules

from projects.self_verify_pose.models.solvability_teacher import SolvabilityTeacher
try:
    from projects.self_verify_pose.models.solvability_teacher_fk import SolvabilityFKTeacher
except Exception:
    SolvabilityFKTeacher = None  # type: ignore


def _get_sample_id(data_sample: Any) -> int:
    # PoseDataSample exposes metainfo via `.metainfo` (dict)
    if hasattr(data_sample, 'metainfo') and isinstance(data_sample.metainfo, dict):
        if 'id' in data_sample.metainfo:
            return int(data_sample.metainfo['id'])
    # fallback to dict-like
    if isinstance(data_sample, dict) and 'id' in data_sample:
        return int(data_sample['id'])
    if hasattr(data_sample, 'get'):
        v = data_sample.get('id', None)
        if v is not None:
            return int(v)
    raise KeyError('Cannot find sample `id` in data_sample metainfo')


@torch.no_grad()
def _decode_simcc_argmax(pred_x: torch.Tensor, pred_y: torch.Tensor, split_ratio: float) -> torch.Tensor:
    # pred_x/pred_y: (B, K, Wx/Wy)
    x_locs = torch.argmax(pred_x, dim=-1).to(dtype=torch.float32)
    y_locs = torch.argmax(pred_y, dim=-1).to(dtype=torch.float32)
    kpts_xy = torch.stack([x_locs, y_locs], dim=-1) / float(split_ratio)
    return kpts_xy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate solvability pseudo-trust cache (.npz) for offline trust-head training.')
    parser.add_argument('config', help='Config file (e.g. baseline RTMPose config).')
    parser.add_argument('checkpoint', help='Trained pose checkpoint to load.')
    parser.add_argument('--out', required=True, help='Output .npz path.')
    parser.add_argument(
        '--split',
        choices=['train', 'val', 'test'],
        default='train',
        help='Which dataloader split to iterate.')
    parser.add_argument('--max-samples', type=int, default=-1, help='Limit number of samples (for smoke runs).')
    parser.add_argument('--device', default='cuda', help='Device, e.g. cuda or cpu.')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=50,
        help='Print progress every N processed samples (0 disables).')
    parser.add_argument(
        '--teacher',
        choices=['free', 'fk'],
        default='free',
        help='Teacher type: free=optimize 3D joints directly; fk=IK over kinematic chain (pytorch_kinematics).')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='Override config options, key=value.')

    # Teacher hyperparams (default match online cfg)
    parser.add_argument('--num-iters', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--w-reproj', type=float, default=1.0)
    parser.add_argument('--w-bone', type=float, default=10.0)
    parser.add_argument('--w-angle', type=float, default=1.0)
    parser.add_argument('--w-reg', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=10.0)
    parser.add_argument('--use-gt-visible', action='store_true', help='Mask reprojection with gt visibility.')

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    register_all_modules()

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)

    # Build dataloader via Runner helper so it respects cfg conventions
    if args.split == 'train':
        dataloader_cfg = cfg.train_dataloader
    elif args.split == 'val':
        dataloader_cfg = cfg.val_dataloader
    else:
        dataloader_cfg = cfg.test_dataloader

    dataloader = Runner.build_dataloader(dataloader_cfg)

    # Build model
    runner = Runner.from_cfg(cfg)
    model = runner.model

    t_ckpt0 = time.time()
    load_checkpoint(model, args.checkpoint, map_location='cpu', revise_keys=[(r'^module\\.', '')])
    t_ckpt1 = time.time()

    # Resolve device robustly.
    requested = str(args.device)
    if requested != 'cpu' and not torch.cuda.is_available():
        print('[WARN] torch.cuda.is_available() == False; falling back to CPU.')
        device = torch.device('cpu')
    else:
        device = torch.device(requested)

    # Diagnostics to quickly see why GPU is not used.
    print('[INFO] CacheGen starting')
    print(f'[INFO]  teacher={args.teacher} split={args.split} max_samples={args.max_samples} num_iters={args.num_iters}')
    print(f'[INFO]  requested_device={requested} resolved_device={device}')
    print(f'[INFO]  torch={torch.__version__} cuda_available={torch.cuda.is_available()} cuda_version={torch.version.cuda}')
    if torch.cuda.is_available():
        try:
            idx = device.index if device.type == 'cuda' else torch.cuda.current_device()
            print(f'[INFO]  cuda_device_index={idx} device_name={torch.cuda.get_device_name(idx)}')
        except Exception:
            # best-effort only
            pass
    print(f'[INFO]  checkpoint_load_sec={t_ckpt1 - t_ckpt0:.2f}')
    model.to(device)
    model.eval()

    if args.teacher == 'fk':
        if SolvabilityFKTeacher is None:
            raise SystemExit(
                'FK teacher requested but pytorch_kinematics is not available. '
                'Install via: pip install -U pytorch-kinematics')
        teacher = SolvabilityFKTeacher(
            num_iters=args.num_iters,
            lr=args.lr,
            w_reproj=args.w_reproj,
            w_limits=0.1,
            w_reg=args.w_reg,
            alpha=args.alpha,
            detach=True,
            device=device,
        )
    else:
        teacher = SolvabilityTeacher(
            num_iters=args.num_iters,
            lr=args.lr,
            w_reproj=args.w_reproj,
            w_bone=args.w_bone,
            w_angle=args.w_angle,
            w_reg=args.w_reg,
            alpha=args.alpha,
            detach=True,
            use_gt_visible=args.use_gt_visible,
            device=device,
        )

    ids: List[int] = []
    pseudo_trust: List[np.ndarray] = []
    reproj_err: List[np.ndarray] = []
    energy: List[np.ndarray] = []

    processed = 0
    t0 = time.time()
    t_last = t0
    last_processed = 0
    target_total = args.max_samples if args.max_samples and args.max_samples > 0 else None
    for data_batch in dataloader:
        # Let the model's preprocessor handle normalization/device transfer
        data = model.data_preprocessor(data_batch, training=False)
        inputs = data['inputs'].to(device)
        data_samples = data['data_samples']

        out = model(inputs, data_samples, mode='tensor')
        if not (isinstance(out, (tuple, list)) and len(out) == 2):
            raise RuntimeError('Expected SimCC head forward output (pred_x, pred_y).')
        pred_x, pred_y = out

        split_ratio = float(getattr(model.head, 'simcc_split_ratio', 1.0))
        kpts_xy = _decode_simcc_argmax(pred_x, pred_y, split_ratio).to(device=device, dtype=torch.float32)

        vis = None
        if args.use_gt_visible:
            vis_list = []
            for ds in data_samples:
                if hasattr(ds, 'gt_instances') and hasattr(ds.gt_instances, 'keypoints_visible'):
                    kv = ds.gt_instances.keypoints_visible
                    if isinstance(kv, np.ndarray):
                        kv = torch.from_numpy(kv)
                    # (1,K) -> (K,)
                    kv = kv[0].to(device=device, dtype=torch.float32)
                else:
                    kv = torch.ones((kpts_xy.size(1),), device=device, dtype=torch.float32)
                vis_list.append(kv)
            vis = torch.stack(vis_list, dim=0)

        # This call runs an inner optimization loop of `num_iters` steps.
        sol = teacher.compute(kpts_xy, visible=vis)

        b = kpts_xy.size(0)
        for i in range(b):
            ids.append(_get_sample_id(data_samples[i]))
            pseudo_trust.append(sol.pseudo_trust[i].detach().cpu().numpy().astype(np.float32))
            reproj_err.append(sol.reproj_err[i].detach().cpu().numpy().astype(np.float32))
            energy.append(sol.energy[i].detach().cpu().numpy().astype(np.float32))
            processed += 1

            if args.log_interval and args.log_interval > 0 and (processed % int(args.log_interval) == 0):
                now = time.time()
                dt = max(now - t_last, 1e-9)
                dn = processed - last_processed
                sps = dn / dt
                elapsed = now - t0
                msg = f'[PROGRESS] processed={processed}'
                if target_total is not None:
                    remain = max(target_total - processed, 0)
                    eta = remain / max(sps, 1e-9)
                    msg += f'/{target_total} ({processed/target_total*100:.1f}%)'
                    msg += f'  sps={sps:.2f}  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m'
                else:
                    msg += f'  sps={sps:.2f}  elapsed={elapsed/60:.1f}m'
                if device.type == 'cuda':
                    try:
                        mem = torch.cuda.memory_allocated(device) / (1024**2)
                        msg += f'  cuda_mem={mem:.0f}MB'
                    except Exception:
                        pass
                print(msg, flush=True)
                t_last = now
                last_processed = processed

            if args.max_samples > 0 and processed >= args.max_samples:
                break
        if args.max_samples > 0 and processed >= args.max_samples:
            break

    t1 = time.time()
    if processed > 0:
        print(f'[INFO] Done iterating. processed={processed} total_sec={t1 - t0:.2f} avg_sps={processed / max(t1 - t0, 1e-9):.2f}')

    ids_arr = np.asarray(ids, dtype=np.int64)
    pseudo_arr = np.stack(pseudo_trust, axis=0)
    reproj_arr = np.stack(reproj_err, axis=0)
    energy_arr = np.stack(energy, axis=0)

    # Basic sanity check
    if ids_arr.size != np.unique(ids_arr).size:
        raise RuntimeError('Duplicate ids found while generating cache. Check dataset/sample id behavior.')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, ids=ids_arr, pseudo_trust=pseudo_arr, reproj_err=reproj_arr, energy=energy_arr)

    print(f'[OK] Wrote cache: {args.out}')
    print(f'  num_samples={len(ids_arr)} pseudo_trust shape={tuple(pseudo_arr.shape)}')


if __name__ == '__main__':
    main()
