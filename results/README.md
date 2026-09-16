# Results directory

Raw run outputs, per-function outcomes, logs, and temporary audits are not
committed because they are generated artifacts and may contain large code
fragments. Each new experiment should write to `results/<run-name>/` and keep:

- corpus, source-tree, dependency, model, adapter, and panel fingerprints;
- seed and complete hyperparameters;
- function and candidate kill rates;
- parse, invalid-candidate, timeout, and infrastructure-error rates;
- Wilson confidence intervals and per-function outcomes; and
- an explicit statement that the `test` split was or was not used.

The historical 67/100 validation smoke result is provisional under the
hardened evaluator. It must be reproduced before it is used as the current
headline result.

**The `test` split is consumed.** The sealed final evaluation was authorized
and attempted once, on 2026-09-16, and **failed after authorization** with
**zero sealed candidates and zero reportable metrics**. The one-time
authorization is spent and **the consumed split must not be rerun**. No result
directory here contains, or may ever contain, a final-test measurement.

The valid empirical evidence in this directory is **locked validation and
development evaluation only**. **No final-test Oneiros claim and no
Oneiros-versus-Atheris claim is supported.** See
[`../docs/SEALED_FINAL_INCIDENT.md`](../docs/SEALED_FINAL_INCIDENT.md) and
[`../docs/POST_INCIDENT_RESEARCH_STATUS.md`](../docs/POST_INCIDENT_RESEARCH_STATUS.md).
