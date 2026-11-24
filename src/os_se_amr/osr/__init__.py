"""Open-Set Recognition scorers, losses, and threshold calibration."""

from os_se_amr.osr.arpl import ARPLLoss, ARPLScorer
from os_se_amr.osr.base import BaseOSRScorer
from os_se_amr.osr.baselines import EnergyScorer, OpenMaxScorer
from os_se_amr.osr.mahalanobis import MahalanobisPrototypeScorer
from os_se_amr.osr.mls import MLSScorer
from os_se_amr.osr.prototype import DistancePrototypeLoss, DistancePrototypeScorer
from os_se_amr.osr.softmax import SoftmaxScorer
from os_se_amr.osr.ssd_amc import (
    SSDMahalanobisScorer,
    SSDProjectionHead,
    SupConLoss,
    ssd_rf_augment,
)
from os_se_amr.osr.threshold import calibrate_threshold

__all__ = [
    "ARPLLoss",
    "ARPLScorer",
    "BaseOSRScorer",
    "DistancePrototypeLoss",
    "DistancePrototypeScorer",
    "EnergyScorer",
    "MLSScorer",
    "MahalanobisPrototypeScorer",
    "OpenMaxScorer",
    "SSDMahalanobisScorer",
    "SSDProjectionHead",
    "SoftmaxScorer",
    "SupConLoss",
    "calibrate_threshold",
    "ssd_rf_augment",
]
