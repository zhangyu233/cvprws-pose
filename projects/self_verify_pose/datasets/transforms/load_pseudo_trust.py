from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from mmpose.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class LoadPseudoTrust(BaseTransform):
    """Load per-instance pseudo-trust vector from a cache file.

    The cache file should be an .npz produced by
    `projects/self_verify_pose/tools/generate_solvability_cache.py` with keys:
        - ids: (N,) int
        - pseudo_trust: (N, K) float
      Optional:
        - reproj_err: (N, K)
        - energy: (N,)

    This transform uses `results['id']` (the per-instance sample id) to lookup
    the pseudo labels, then writes:
        - results['pseudo_trust_sol'] = (K,) float32
        - results['instance_mapping_table'] updated so PackPoseInputs packs it
          into `data_sample.gt_instances.pseudo_trust_sol`.

    Args:
        cache_file (str): Path to .npz cache.
        field_name (str): Key to write into results. Defaults to
            'pseudo_trust_sol'.
        missing (str): What to do if id not found: 'error'|'skip'|'zeros'.
    """

    def __init__(
        self,
        cache_file: str,
        field_name: str = 'pseudo_trust_sol',
        missing: str = 'error',
    ):
        self.cache_file = str(cache_file)
        self.field_name = str(field_name)
        self.missing = str(missing)

        data = np.load(self.cache_file, allow_pickle=False)
        ids = data['ids'].astype(np.int64)
        pseudo = data['pseudo_trust'].astype(np.float32)

        self._id_to_index: Dict[int, int] = {int(i): int(idx) for idx, i in enumerate(ids.tolist())}
        self._pseudo = pseudo

    def transform(self, results: dict) -> Optional[dict]:
        if 'id' not in results:
            raise KeyError('LoadPseudoTrust requires `results[\'id\']`')

        sample_id = int(results['id'])
        idx = self._id_to_index.get(sample_id, None)

        if idx is None:
            if self.missing == 'skip':
                return None
            if self.missing == 'zeros':
                # infer K if possible
                k = int(self._pseudo.shape[1])
                vec = np.zeros((k,), dtype=np.float32)
            else:
                raise KeyError(f'Pseudo-trust not found for id={sample_id} in {self.cache_file}')
        else:
            vec = self._pseudo[idx]

        # In top-down COCO, each sample has exactly 1 person instance.
        # PackPoseInputs expects per-instance fields to have shape (N_inst, ...)
        results[self.field_name] = vec[None, :]

        # Tell PackPoseInputs to pack this field into gt_instances
        mapping = results.get('instance_mapping_table', None)
        if mapping is None:
            mapping = {}
        else:
            mapping = dict(mapping)
        mapping[self.field_name] = self.field_name
        results['instance_mapping_table'] = mapping

        return results
