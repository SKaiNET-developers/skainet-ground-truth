import torch
from sklearn.metrics import f1_score

from gt.core import Executable

# UC-004: F1 (metric cases, see TS-008/README.md)
#
# Each case returns ([scores_0, targets_0, scores_1, targets_1, ...], expected). The reference
# value is computed once over all batches together, always with zero_division=0.


def _batch(generator, samples, classes):
    scores = torch.randn(samples, classes, generator=generator)
    targets = torch.randint(0, classes, (samples,), generator=generator)
    return scores, targets


def _reference(y_true, y_pred, average):
    value = f1_score(y_true.numpy(), y_pred.numpy(), average=average, zero_division=0)
    return torch.tensor([value])


def _two_batches(seed, average):
    g = torch.Generator().manual_seed(seed)
    scores_0, y_0 = _batch(g, 32, 3)
    scores_1, y_1 = _batch(g, 20, 3)
    y_pred = torch.cat([scores_0, scores_1]).argmax(dim=1)
    return [scores_0, y_0, scores_1, y_1], _reference(torch.cat([y_0, y_1]), y_pred, average)


@Executable("F1 macro, three classes, two batches",
            op_type="metric.f1",
            op_params={"average": "macro", "target_encoding": "index"})
def f1_macro_two_batches():
    return _two_batches(841, "macro")


@Executable("F1 micro, three classes, two batches",
            op_type="metric.f1",
            op_params={"average": "micro", "target_encoding": "index"})
def f1_micro_two_batches():
    return _two_batches(842, "micro")


@Executable("F1 macro, four classes, one-hot targets",
            op_type="metric.f1",
            op_params={"average": "macro", "target_encoding": "one_hot"})
def f1_macro_one_hot():
    g = torch.Generator().manual_seed(843)
    scores, y = _batch(g, 40, 4)
    one_hot = torch.nn.functional.one_hot(y, num_classes=4).float()
    return [scores, one_hot], _reference(y, scores.argmax(dim=1), "macro")


@Executable("F1 macro, class 3 has a score column but never occurs",
            op_type="metric.f1",
            op_params={"average": "macro", "target_encoding": "index"})
def f1_macro_class_never_occurs():
    # Class 3 is in neither the targets nor the predictions: the mean runs over three classes.
    g = torch.Generator().manual_seed(844)
    scores = torch.randn(40, 4, generator=g)
    scores[:, 3] = -10.0
    y = torch.randint(0, 3, (40,), generator=g)
    return [scores, y], _reference(y, scores.argmax(dim=1), "macro")


@Executable("F1 macro, class 2 is a target but never predicted",
            op_type="metric.f1",
            op_params={"average": "macro", "target_encoding": "index"})
def f1_macro_class_never_predicted():
    # Precision of class 2 is 0/0 and counts as 0.
    g = torch.Generator().manual_seed(845)
    scores, y = _batch(g, 40, 3)
    scores[:, 2] = -10.0
    return [scores, y], _reference(y, scores.argmax(dim=1), "macro")


@Executable("F1 macro, class 2 is predicted but never a target",
            op_type="metric.f1",
            op_params={"average": "macro", "target_encoding": "index"})
def f1_macro_class_never_a_target():
    # Recall of class 2 is 0/0 and counts as 0.
    g = torch.Generator().manual_seed(846)
    scores = torch.randn(40, 3, generator=g)
    y = torch.randint(0, 2, (40,), generator=g)
    return [scores, y], _reference(y, scores.argmax(dim=1), "macro")


@Executable("F1 binary, two score columns, argmax, two batches",
            op_type="metric.f1",
            op_params={"average": "binary", "target_encoding": "index"})
def f1_binary_argmax_two_batches():
    g = torch.Generator().manual_seed(847)
    scores_0, y_0 = _batch(g, 32, 2)
    scores_1, y_1 = _batch(g, 20, 2)
    y_pred = torch.cat([scores_0, scores_1]).argmax(dim=1)
    return [scores_0, y_0, scores_1, y_1], _reference(torch.cat([y_0, y_1]), y_pred, "binary")


# Fixed scores for the threshold cases, none of them near a threshold.
_PROBABILITIES = torch.tensor([0.05, 0.2, 0.35, 0.6, 0.8, 0.95])


@Executable("F1 binary, two probability columns, threshold 0.7",
            op_type="metric.f1",
            op_params={"average": "binary", "target_encoding": "index", "threshold": 0.7})
def f1_binary_threshold_two_columns():
    # Positive when the probability of class 1 is above 0.7; argmax would say above 0.5.
    g = torch.Generator().manual_seed(848)
    p1 = _PROBABILITIES[torch.randint(0, 6, (40,), generator=g)]
    scores = torch.stack([1.0 - p1, p1], dim=1)
    y = torch.randint(0, 2, (40,), generator=g)
    return [scores, y], _reference(y, (p1 > 0.7).long(), "binary")


@Executable("F1 binary, one probability column, threshold 0.5",
            op_type="metric.f1",
            op_params={"average": "binary", "target_encoding": "index", "threshold": 0.5})
def f1_binary_threshold_one_column():
    g = torch.Generator().manual_seed(849)
    p1 = _PROBABILITIES[torch.randint(0, 6, (40,), generator=g)]
    y = torch.randint(0, 2, (40,), generator=g)
    return [p1.reshape(40, 1), y], _reference(y, (p1 > 0.5).long(), "binary")


@Executable("F1 binary, no positive sample and no positive prediction",
            op_type="metric.f1",
            op_params={"average": "binary", "target_encoding": "index", "threshold": 0.5})
def f1_binary_no_positives():
    # Every denominator is zero: the value is 0, not NaN.
    scores = torch.full((16, 1), 0.2)
    y = torch.zeros(16, dtype=torch.long)
    return [scores, y], _reference(y, torch.zeros(16, dtype=torch.long), "binary")
