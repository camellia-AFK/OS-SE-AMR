"""Continual incremental learning modules: replay buffer, Dual-LwF, BiC, and baselines."""

from os_se_amr.continual.baselines import DERLearner, DERWrapperBackbone, EWCLearner, WALearner
from os_se_amr.continual.finetune import NaiveFineTuneLearner
from os_se_amr.continual.icarl import iCaRLLearner
from os_se_amr.continual.lwf import (
    DualLwFLearner,
    EvolutionReport,
    apply_weight_alignment,
    calibrate_per_head_bias,
    rf_augment_batch,
    rf_phase_aligned_replay_ss,
)
from os_se_amr.continual.prototype_sync import sync_prototypes_and_mahalanobis
from os_se_amr.continual.replay import ReplayBuffer
from os_se_amr.continual.simplecil import SimpleCILLearner

__all__ = [
    "DERLearner",
    "DERWrapperBackbone",
    "DualLwFLearner",
    "EWCLearner",
    "EvolutionReport",
    "NaiveFineTuneLearner",
    "ReplayBuffer",
    "SimpleCILLearner",
    "WALearner",
    "apply_weight_alignment",
    "calibrate_per_head_bias",
    "iCaRLLearner",
    "rf_augment_batch",
    "rf_phase_aligned_replay_ss",
    "sync_prototypes_and_mahalanobis",
]
