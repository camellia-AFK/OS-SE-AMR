"""SimpleCIL baseline: frozen backbone with L2-normalized cosine prototype expansion."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from os_se_amr.continual.lwf import EvolutionReport
from os_se_amr.models.base import BaseBackbone


class SimpleCILLearner:
    """Gradient-free class-incremental baseline using frozen backbone prototypes."""

    def __init__(self, feature_layer: str = "penult", batch_size: int = 256) -> None:
        self.feature_layer = str(feature_layer)
        self.batch_size = int(batch_size)
        self.prototypes: dict[int, np.ndarray] = {}

    @torch.no_grad()
    def _extract_unit_features(
        self,
        backbone: BaseBackbone,
        x_iq: np.ndarray | torch.Tensor,
        device: torch.device,
    ) -> np.ndarray:
        backbone.eval()
        x_t = torch.as_tensor(x_iq, dtype=torch.float32)
        parts: list[np.ndarray] = []
        for start in range(0, len(x_t), self.batch_size):
            xb = x_t[start : start + self.batch_size].to(device, non_blocking=True)
            ms = backbone.extract_multiscale_features(xb)
            if self.feature_layer not in ms:
                raise KeyError(
                    f"Feature layer '{self.feature_layer}' missing from backbone outputs {list(ms.keys())}"
                )
            feats = F.normalize(ms[self.feature_layer], p=2, dim=-1)
            parts.append(feats.cpu().numpy())
        return np.concatenate(parts, axis=0) if parts else np.empty((0, backbone.feature_dim), dtype=np.float32)

    @staticmethod
    def _mean_unit_proto(feats: np.ndarray) -> np.ndarray:
        mean = feats.astype(np.float64).mean(axis=0)
        norm = max(float(np.linalg.norm(mean)), 1e-8)
        return (mean / norm).astype(np.float32)

    def fit_base(
        self,
        backbone: BaseBackbone,
        train_x: np.ndarray | torch.Tensor,
        train_y: np.ndarray | torch.Tensor,
        device: str | torch.device = "cuda",
    ) -> None:
        """Compute initial class prototypes strictly on Session 0 known training data."""
        dev = torch.device(device)
        backbone.to(dev)
        y_np = np.asarray(train_y, dtype=np.int64).reshape(-1)
        if len(y_np) == 0:
            raise ValueError("train_y must not be empty")
        unit_feats = self._extract_unit_features(backbone, train_x, device=dev)
        self.prototypes = {}
        for cid in sorted(np.unique(y_np).tolist()):
            self.prototypes[int(cid)] = self._mean_unit_proto(unit_feats[y_np == cid])

    def _build_head_weights(
        self,
        backbone: BaseBackbone,
        num_old: int,
        new_protos: np.ndarray,
    ) -> torch.Tensor | None:
        if self.feature_layer != "penult" or new_protos.shape[1] != backbone.feature_dim:
            return None
        if hasattr(backbone.classifier, "fc"):
            old_w = backbone.classifier.fc.weight.detach()[:num_old]
            mean_norm = float(old_w.norm(dim=1).mean().clamp_min(1e-4).item())
        elif hasattr(backbone.classifier, "weight"):
            old_w = backbone.classifier.weight.detach()[:num_old]
            mean_norm = float(old_w.norm(dim=1).mean().clamp_min(1e-4).item())
        else:
            mean_norm = 1.0
        return torch.from_numpy(new_protos * mean_norm)

    def evolve(
        self,
        backbone: BaseBackbone,
        new_samples: np.ndarray | torch.Tensor,
        new_pseudo_labels: np.ndarray | torch.Tensor,
        device: str | torch.device = "cuda",
    ) -> EvolutionReport:
        """Register new class centroids without updating any backbone feature weights."""
        t0 = time.time()
        dev = torch.device(device)
        backbone.to(dev)

        x_new = torch.as_tensor(new_samples, dtype=torch.float32)
        y_local = np.asarray(new_pseudo_labels, dtype=np.int64).reshape(-1)
        if len(x_new) == 0:
            raise ValueError("new_samples must not be empty")
        if len(x_new) != len(y_local):
            raise ValueError(f"Length mismatch: new_samples={len(x_new)}, new_pseudo_labels={len(y_local)}")

        min_lbl = int(y_local.min())
        max_lbl = int(y_local.max())
        if min_lbl < 0:
            raise ValueError(f"Noise labels (-1) must be filtered out before evolve(), got min={min_lbl}")

        num_old = int(backbone.num_classes)
        num_new = max_lbl + 1

        unit_feats = self._extract_unit_features(backbone, x_new, device=dev)
        new_proto_list: list[np.ndarray] = []
        for local_cid in range(num_new):
            mask = y_local == local_cid
            if not mask.any():
                raise ValueError(f"Missing samples for cluster id {local_cid} in SimpleCILLearner.evolve")
            proto = self._mean_unit_proto(unit_feats[mask])
            self.prototypes[num_old + local_cid] = proto
            new_proto_list.append(proto)

        proto_mat = np.stack(new_proto_list, axis=0)
        init_w = self._build_head_weights(backbone, num_old, proto_mat)
        backbone.expand_classifier(num_new, init_weights=init_w)
        backbone.to(dev)

        return EvolutionReport(
            num_old_classes=num_old,
            num_new_classes=num_new,
            epochs=0,
            final_loss=0.0,
            final_ce_loss=0.0,
            final_feat_kd=0.0,
            final_logit_kd=0.0,
            elapsed_sec=round(time.time() - t0, 4),
            history=[],
            bic_gammas=[],
        )

    @torch.no_grad()
    def predict_dataset(
        self,
        backbone: BaseBackbone,
        ds: Any,
        device: str | torch.device = "cuda",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Classify samples in `ds` via nearest cosine similarity to registered prototypes."""
        if not self.prototypes:
            raise RuntimeError("SimpleCILLearner must call fit_base() before predict_dataset()")

        dev = torch.device(device)
        backbone.to(dev)
        backbone.eval()

        class_ids = sorted(self.prototypes.keys())
        proto_mat = np.stack([self.prototypes[c] for c in class_ids], axis=0).astype(np.float32)
        proto_t = torch.from_numpy(proto_mat).to(dev)

        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=0)
        targets: list[np.ndarray] = []
        preds: list[np.ndarray] = []
        class_arr = np.asarray(class_ids, dtype=np.int64)

        for batch in loader:
            xb = batch[0].to(dev, non_blocking=True)
            yb = batch[1].numpy()
            ms = backbone.extract_multiscale_features(xb)
            z_unit = F.normalize(ms[self.feature_layer], p=2, dim=-1)
            sims = z_unit @ proto_t.T
            best_pos = sims.argmax(dim=1).cpu().numpy()
            preds.append(class_arr[best_pos])
            targets.append(yb)

        return np.concatenate(targets, axis=0), np.concatenate(preds, axis=0)
