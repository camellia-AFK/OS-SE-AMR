"""Naive incremental fine-tuning lower-bound learner (no replay, no KD, no BiC)."""

from __future__ import annotations

from os_se_amr.continual.lwf import DualLwFLearner


class NaiveFineTuneLearner(DualLwFLearner):
    """Naive fine-tuning baseline that updates only on novel-class samples."""

    def __init__(
        self,
        epochs: int = 35,
        batch_size_new: int = 128,
        lr: float = 8e-4,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
    ) -> None:
        super().__init__(
            epochs=epochs,
            batch_size_new=batch_size_new,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            beta_feat=0.0,
            beta_logit=0.0,
            use_replay=False,
            use_rf_aug=False,
            bic_max_steal=None,
        )
