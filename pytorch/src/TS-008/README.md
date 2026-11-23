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

Use Cases (UC files):
- UC-001: Accuracy, one batch and two batches

How to run:
- From the `pytorch` directory with your environment activated:
  - Run this suite only: `gt src/TS-008 results`

Notes:
- The data is random with a fixed seed per case, so a rebuild gives the same files.
- String values in `op_params` need a `gradienttracer` version that writes them.
