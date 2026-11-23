# GGUF fixture contract

This document is the stable, explicit contract between however these fixtures are
*produced* (today: `gradienttracer` + this repo's PyTorch test suites, run in Docker) and
however they're *consumed* (today: SKaiNET's `skainet-test/skainet-test-groundtruth`
Kotlin module). It exists so the two sides can agree on the file format without either
one reading the other's source — the actual precondition for this content moving to a
neutral, framework-agnostic home like [numcrux](https://github.com/numcrux) later without
breaking whoever consumes it.

Everything below is drawn directly from the current implementation
(`gradienttracer`'s `gt/pytorch/io/writer.py` and `gt/core.py`, and SKaiNET's
`GroundTruthLoader.kt`) — this is a description of what's actually produced and
consumed today, not an aspiration.

## File naming and layout

```
<output>/TS-<NNN>/TS_<NNN>_UC_<MMM>.<function_name>.gguf
```

`TS-<NNN>` is a test-suite directory (e.g. `TS-001`); each file is one `@Executable`
function's result. The directory name is also used as the `testSuite` identifier by
consumers.

## Tensor naming

| Tensor name | Meaning |
|---|---|
| `input_0`, `input_1`, ... | Positional input tensors, in the order returned by the `@Executable` function (`return [x, conv.weight, conv.bias], y` → `input_0`, `input_1`, `input_2`) |
| `result` (or `output`) | The operation's output tensor |
| `grad_<input_name>` | **Not currently produced.** `GroundTruthLoader` already recognizes this prefix and would populate `expectedGradients`, but `store_experiment_as_gguf` never writes a `grad_*` tensor today — gradient validation is unimplemented on the producer side, not just unused on the consumer side. |

All tensor data is written as F32, little-endian, row-major (converted from whatever the
source PyTorch tensor's dtype was via `__convert_to_f32__`/`__ensure_little_endian__`).

## Metadata keys

| Key | Type | Meaning |
|---|---|---|
| `general.description` | string | The `@Executable`'s human-readable description, e.g. `"Basic 2D Convolution"` |
| `general.name` | string | The operation name — `op_type` if the `@Executable` decorator specifies one, otherwise the Python function name |
| `op.<param_name>` | int32 / uint32 / float32 / string / array | Operation parameters passed via `@Executable(..., op_params={...})` — e.g. `op.padding`, `op.stride`, `op.groups`, `op.start_dim`. A plain Python `int`/`float` writes as a scalar; a `str` writes as a GGUF string; a `list`/`tuple` writes as an array; any other type is an error on the producer side. **Only written for params explicitly listed in `op_params`** — a param baked into the PyTorch call (e.g. `Conv2d(..., stride=2)`) but not also passed via `op_params={"stride": 2}` produces a numerically-correct `result` tensor with no way for a consumer to know `stride=2` was used. Every `@Executable` that sets a non-default PyTorch op argument must also pass it via `op_params`, or a consumer has no way to reproduce the call. |

Consumers should treat `op.*` values as the *only* source of truth for operation
parameters — not the `general.description` text. (SKaiNET's consumer used to
string-match on the description as a stopgap; that's gone as of the change that added
this document, precisely because it's unreliable — a description not containing the
word "strided" gives no signal at all, even when the underlying op genuinely used a
non-default stride.)

## Metric cases

A file whose `general.name` starts with `metric.` is a *metric case*. It does not describe
one tensor operation, but a metric that is fed batch by batch and returns one number
(accuracy, precision, recall, F1). Layout, naming and encoding are the same as above; only
the meaning of the tensors differs. The convention comes from SKaiNET's SKEEP-009.

| Item | Meaning |
|---|---|
| `general.name` | `metric.<name>`: `metric.accuracy`, `metric.precision`, `metric.recall`, `metric.f1` |
| `input_<2k>` | Predictions of batch `k`: class scores (F32), with the classes along the last dimension unless `op.dim` says otherwise |
| `input_<2k+1>` | Targets of batch `k` (F32 in the file like every tensor); read as `op.target_encoding` says |
| `result` | Exactly one element: the reference value over **all batches together** |
| `op.target_encoding` (string, required) | `index`: each value is a class index (a whole number ≥ 0), one per sample. `one_hot`: same shape as the predictions; the class is the position of the largest value |
| `op.average` (string) | `macro`, `micro` or `binary`. Required for precision, recall and F1; absent for accuracy |
| `op.threshold` (float32, optional) | For predictions with one or two score columns: a sample is predicted positive (class 1) when its class-1 score (the only column, or the second of two) is greater than the threshold. Absent: the predicted class is the argmax |
| `op.dim` (int, optional) | The dimension of the predictions that holds the classes. Absent: the last one |

Rules a consumer can rely on:

- The inputs come in pairs, so the input count is even, and `input_0 … input_n` has no gaps.
- The reference value is computed once over the concatenation of the batches. A consumer
  feeds the batches one at a time, in order; a case with more than one batch therefore also
  checks accumulation.
- Conventions of the reference (scikit-learn): a zero denominator gives `0`; macro averaging
  runs over the classes that occur in the targets or the predictions of the case; `binary`
  reports class 1.
- A consumer should fail on a metric name or an `op.*` key it does not know, instead of
  ignoring it.
- The compared value is F32 in the file. For a value in `[0, 1]` an absolute tolerance of
  `1e-6` is enough.

## What's not covered by this contract yet

- **Gradients** — `expectedGradients`/`grad_*` is a recognized shape on the consumer
  side with no producer implementation. Don't rely on it being populated.
- **Non-numeric metadata beyond description/name/op.\*** — nothing else is currently
  written or read.
- **Versioning** — there's no format version field. A breaking change to this contract
  (renaming a convention, changing encoding) needs a coordinated update on both sides;
  there's currently no automated way to detect a mismatch other than tests failing.
