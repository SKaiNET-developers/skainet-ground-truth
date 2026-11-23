# TS-008: Classification metrics (scikit-learn reference)

This test suite holds *metric cases*. A metric case differs from the other suites: it
does not describe one tensor operation, but a metric that is fed batch by batch and
returns one number. The convention is defined in SKaiNET's SKEEP-009
("Ground-truth validation for stateful metrics").

How a case is written:
- `op_type="metric.<name>"`, for example `metric.accuracy`
- The inputs are pairs: predictions (class scores) and targets of batch 0, then of
  batch 1, and so on
- The result is one value, computed by the reference over all batches together
- `op_params={"target_encoding": "index"}` says the targets are class indices
  (`"one_hot"` for one-hot targets)

Reference library: scikit-learn (`sklearn.metrics`), version pinned in
`requirements.txt`.

- `op_params` also holds `"average"` (`"macro"`, `"micro"` or `"binary"`) for a metric that
  averages over classes, and `"threshold"` when a binary case uses one

Use Cases (UC files):
- UC-001: Accuracy, one batch and two batches
- UC-002: Precision
- UC-003: Recall
- UC-004: F1

UC-002 to UC-004 hold the same ten scenarios each: macro and micro over two batches, one-hot
targets, a class that never occurs, a class that is never predicted, a class that is never a
target, binary with argmax, binary with a threshold (two columns, one column), and a binary
case where every denominator is zero. The reference is always called with `zero_division=0`
and without `labels`, so macro averaging runs over the classes that occur in the targets or
the predictions.

How to run:
- From the `pytorch` directory with your environment activated:
  - Run this suite only: `gt src/TS-008 results`

Notes:
- The data is random with a fixed seed per case, so a rebuild gives the same files.
- The scores of the threshold cases are fixed values away from the threshold.
