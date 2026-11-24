"""Unknown modulation discovery components (buffer, dual gate, HDBSCAN, and TLDNN-SS)."""

from os_se_amr.discovery.baselines import FINCHDiscoverer, KMeansDiscoverer
from os_se_amr.discovery.buffer import RejectionBuffer
from os_se_amr.discovery.gating import DualAwareGate, GatingReport
from os_se_amr.discovery.hdbscan_cluster import DiscoveryResult, HDBSCANDiscoverer
from os_se_amr.discovery.ss_augment import augment_core_clusters, ss_iq_pair, ss_phase_aligned_pair

__all__ = [
    "DiscoveryResult",
    "DualAwareGate",
    "FINCHDiscoverer",
    "GatingReport",
    "HDBSCANDiscoverer",
    "KMeansDiscoverer",
    "RejectionBuffer",
    "augment_core_clusters",
    "ss_iq_pair",
    "ss_phase_aligned_pair",
]
