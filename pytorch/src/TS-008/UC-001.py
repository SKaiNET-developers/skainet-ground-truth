import torch
from sklearn.metrics import accuracy_score

from gt.core import Executable

# UC-001: Accuracy (metric case, see TS-008/README.md)
#
# A metric case returns ([scores_0, targets_0, scores_1, targets_1, ...], expected):
# pairs of (predictions, targets), one pair per batch, and one expected value that
# the reference computes over all batches together.


def _batch(generator, samples, classes):
    scores = torch.randn(samples, classes, generator=generator)
    targets = torch.randint(0, classes, (samples,), generator=generator)
    return scores, targets


@Executable("Accuracy, three classes, one batch",
            op_type="metric.accuracy",
            op_params={"target_encoding": "index"})
def accuracy_one_batch():
    g = torch.Generator().manual_seed(8001)
    scores, y = _batch(g, 32, 3)
    expected = accuracy_score(y.numpy(), scores.argmax(dim=1).numpy())
    return [scores, y], torch.tensor([expected])


@Executable("Accuracy, three classes, two batches of different size",
            op_type="metric.accuracy",
            op_params={"target_encoding": "index"})
def accuracy_two_batches():
    g = torch.Generator().manual_seed(8002)
    scores_0, y_0 = _batch(g, 32, 3)
    scores_1, y_1 = _batch(g, 20, 3)
    y_true = torch.cat([y_0, y_1])
    y_pred = torch.cat([scores_0, scores_1]).argmax(dim=1)
    expected = accuracy_score(y_true.numpy(), y_pred.numpy())
    return [scores_0, y_0, scores_1, y_1], torch.tensor([expected])
