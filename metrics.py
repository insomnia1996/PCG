from typing import Optional
import torch

class SequenceAccuracy:
    
    def __init__(self) -> None:
        self.correct_count = 0.
        self.total_count = 0.

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        with torch.no_grad():
            predictions, gold_labels, mask = predictions.detach(), gold_labels.detach(), mask.detach()
            # Some sanity checks.
            num_classes = predictions.size(-1)

            predictions = predictions.view((-1, num_classes))
            gold_labels = gold_labels.view(-1).long()
            
            top_1 = predictions.max(-1)[1]
            correct = top_1.eq(gold_labels).float()

            if mask is not None:
                correct *= mask.view(-1).float()
                self.total_count += mask.sum()
            else:
                self.total_count += gold_labels.numel()
            self.correct_count += correct.sum()

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy.
        """
        if self.total_count > 1e-12:
            accuracy = float(self.correct_count) / float(self.total_count)
        else:
            accuracy = 0.0
        if reset:
            self.reset()
        return accuracy

    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0