import numpy as np
import pytest

from hypothesaes.interpret_neurons import NeuronInterpreter


def test_compute_metrics_uses_precision_not_specificity():
    interpreter = NeuronInterpreter()
    annotations = np.array([1, 1, 1, 1])
    labels = np.array([1, 1, 0, 0])
    activations = np.array([2.0, 1.0, 0.0, 0.0])

    metrics = interpreter._compute_metrics(
        annotations=annotations,
        labels=labels,
        activations=activations,
    )

    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(2 / 3)
