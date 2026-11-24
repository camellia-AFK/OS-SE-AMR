"""Unit tests for Scheirer openness math and leakage-free RML2016.10a splitting."""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_se_amr.data.rml2016 import load_rml2016_dict
from os_se_amr.data.splitter import SplitBundle, compute_openness
from os_se_amr.utils.config import load_yaml


def test_openness_exact_values() -> None:
    expected_a = 1.0 - math.sqrt(14.0 / 18.0)  # ~0.11808 -> 11.81%
    expected_b = 1.0 - math.sqrt(10.0 / 16.0)  # ~0.20943 -> 20.94%
    assert compute_openness(7, 11) == pytest.approx(expected_a, rel=1e-7)
    assert compute_openness(5, 11) == pytest.approx(expected_b, rel=1e-7)
    assert round(compute_openness(7, 11) * 100.0, 2) == 11.81
    assert round(compute_openness(5, 11) * 100.0, 2) == 20.94
    assert compute_openness(7, 7) == pytest.approx(0.0)

    with pytest.raises(ValueError):
        compute_openness(0, 11)
    with pytest.raises(ValueError):
        compute_openness(8, 7)


def test_split_bundle_config_a_and_b_zero_leakage() -> None:
    cfg_a = load_yaml("configs/splits/config_a.yaml")
    cfg_b = load_yaml("configs/splits/config_b.yaml")
    raw_dict = load_rml2016_dict(cfg_a["dataset_path"])

    bundle_a = SplitBundle(raw_dict=raw_dict, split_cfg=cfg_a)
    assert bundle_a.verify_disjoint() is True
    assert bundle_a.openness == pytest.approx(compute_openness(7, 11))

    train_a, val_a, test_a = bundle_a.get_closed_set_datasets()
    assert len(train_a) == 42000  # 7 classes * 10 SNRs * 600
    assert len(val_a) == 14000    # 7 classes * 10 SNRs * 200
    assert len(test_a) == 14000   # 7 classes * 10 SNRs * 200

    known_test_a, unknown_test_a = bundle_a.get_open_set_test_datasets()
    assert len(known_test_a) == 14000
    assert len(unknown_test_a) == 8000  # 4 unknown classes * 10 SNRs * 200
    assert set(np.unique(known_test_a.y).tolist()) == set(range(7))
    assert set(np.unique(unknown_test_a.y).tolist()) == {7, 8, 9, 10}

    stress_a = bundle_a.get_stress_test_dataset()
    assert len(stress_a) == 7000  # 7 classes * 5 negative SNRs * 200
    assert set(np.unique(stress_a.snrs).tolist()) == {-10, -8, -6, -4, -2}

    # Verify locked-known label index alignment
    for idx, cls_name in enumerate(bundle_a.known_classes):
        assert bundle_a.known_to_idx[cls_name] == idx
        assert bundle_a.all_to_idx[cls_name] == idx

    # Determinism across repeated constructions
    bundle_a_dup = SplitBundle(raw_dict=raw_dict, split_cfg=cfg_a)
    train_a_dup, _, _ = bundle_a_dup.get_closed_set_datasets()
    assert np.array_equal(train_a.x, train_a_dup.x)
    assert np.array_equal(train_a.y, train_a_dup.y)

    # Config B checks
    bundle_b = SplitBundle(raw_dict=raw_dict, split_cfg=cfg_b)
    assert bundle_b.verify_disjoint() is True
    assert bundle_b.openness == pytest.approx(compute_openness(5, 11))
    train_b, val_b, test_b = bundle_b.get_closed_set_datasets()
    assert len(train_b) == 30000
    assert len(val_b) == 10000
    assert len(test_b) == 10000
    _, unknown_test_b = bundle_b.get_open_set_test_datasets()
    assert len(unknown_test_b) == 12000
