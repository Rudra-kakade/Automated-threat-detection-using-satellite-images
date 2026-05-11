"""
Gradient Underflow Monitor for FP16 / AMP Training
====================================================
Detects FP16 gradient underflow on minority-class outputs.

Blueprint v2, Section 5.
"""

import logging

logger = logging.getLogger(__name__)


class GradientUnderflowMonitor:
    """
    Detects FP16 gradient underflow on minority-class outputs.
    Prints a warning if any parameter's gradient norm drops below 1e-7.
    """

    def __init__(self, model, threshold: float = 1e-7, max_warnings: int = 5):
        self.model = model
        self.threshold = threshold
        self.max_warnings = max_warnings
        self._underflow_counts = {}

    def check(self, epoch: int) -> list:
        """Check all model parameters for gradient underflow."""
        underflows = []
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.abs().max().item()
                if grad_norm < self.threshold:
                    underflows.append((name, grad_norm))
                    self._underflow_counts[name] = self._underflow_counts.get(name, 0) + 1

        if underflows:
            msg = f"\n⚠ Epoch {epoch} — FP16 gradient underflow detected:"
            for name, norm in underflows[:self.max_warnings]:
                msg += f"\n   {name}: max_grad={norm:.2e}"
            msg += "\n   Consider class-weighted loss or oversampling minority class.\n"
            logger.warning(msg)
            print(msg)

        return underflows

    def summary(self) -> dict:
        """Return cumulative underflow counts per parameter."""
        return dict(self._underflow_counts)
