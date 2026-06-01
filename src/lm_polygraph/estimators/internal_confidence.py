import numpy as np
from typing import Dict

from .estimator import Estimator
from lm_polygraph.stat_calculators.internal_confidence_probs import (
    compute_positional_attention,
)


class PYesTopRight(Estimator):
    """
    Estimates uncertainty using the binary-renormalized probability of 'Yes'
    at the last token and last layer of a self-evaluation prompt.
    """

    def __init__(self):
        super().__init__(["ic_pyes_matrix"], "sequence")

    def __str__(self):
        return "PYesTopRight"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        # ic_pyes_matrix: list of [K, L] arrays, one per sample
        return np.array([-m[-1, -1] for m in stats["ic_pyes_matrix"]])


class PYesNaiveAvg(Estimator):
    """
    Estimates uncertainty using the uniform average of binary-renormalized
    probability of 'Yes' over the last K tokens and all layers.
    """

    def __init__(self):
        super().__init__(["ic_pyes_matrix"], "sequence")

    def __str__(self):
        return "PYesNaiveAvg"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        return np.array([-m.mean() for m in stats["ic_pyes_matrix"]])


class InternalConfidence(Estimator):
    """
    Estimates uncertainty using an exponentially-decayed weighted average of
    binary-renormalized probability of 'Yes' centered at the last token and last layer.
    """

    def __init__(self, w: float = 1.0, s: float = 1.0):
        """
        Parameters:
            w (float): Locality parameter. Higher values concentrate weight more at the center.
            s (float): Symmetry parameter.
        """
        super().__init__(["ic_pyes_matrix"], "sequence")
        self.w = w
        self.s = s

    def __str__(self):
        return "InternalConfidence"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        scores = []
        for m in stats["ic_pyes_matrix"]:  # m: [K, L]
            K, L = m.shape
            layer_w = compute_positional_attention(L, L - 1, self.w, self.s)  # [L]
            token_w = compute_positional_attention(K, K - 1, self.w, self.s)  # [K]
            weighted = (m * layer_w) * token_w[:, np.newaxis]  # [K, L]
            scores.append(-weighted.sum())
        return np.array(scores)
