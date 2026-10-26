import numpy as np
import pytest

from ml.evaluate import (
    false_positive_rate_at_threshold,
    recall_at_threshold,
    threshold_for_fpr,
)


def test_fpr_threshold_uses_only_validation_benign_scores():
    y_validation = np.array([0, 0, 0, 0, 1])
    validation_scores = np.array([0.1, 0.2, 0.3, 0.4, 100.0])

    threshold = threshold_for_fpr(y_validation, validation_scores, fpr=0.25)

    assert threshold == pytest.approx(0.325)


def test_recall_uses_calibrated_threshold_without_recalibrating_on_test_benign():
    y_test = np.array([0, 0, 1, 1])
    test_scores = np.array([100.0, 200.0, 0.3, 0.5])

    assert recall_at_threshold(y_test, test_scores, threshold=0.4) == 0.5


def test_test_fpr_measures_benign_alerts_at_fixed_threshold():
    y_test = np.array([0, 0, 1])
    test_scores = np.array([0.3, 0.5, 0.7])

    assert false_positive_rate_at_threshold(
        y_test, test_scores, threshold=0.4
    ) == 0.5


@pytest.mark.parametrize("fpr", [-0.1, 1.0])
def test_fpr_threshold_rejects_invalid_fpr(fpr):
    with pytest.raises(ValueError, match="fpr"):
        threshold_for_fpr([0, 1], [0.1, 0.2], fpr=fpr)


def test_fpr_threshold_requires_validation_benign_examples():
    with pytest.raises(ValueError, match="benign"):
        threshold_for_fpr([1, 1], [0.1, 0.2])


def test_recall_requires_attack_examples():
    with pytest.raises(ValueError, match="attack"):
        recall_at_threshold([0, 0], [0.1, 0.2], threshold=0.15)


def test_test_fpr_requires_benign_examples():
    with pytest.raises(ValueError, match="benign"):
        false_positive_rate_at_threshold([1, 1], [0.1, 0.2], threshold=0.15)
