# Results directory

Raw run outputs, per-function outcomes, logs, and temporary audits are not
committed because they are generated artifacts and may contain large code
fragments. Each new experiment should write to `results/<run-name>/` and keep:

- corpus, source-tree, dependency, model, adapter, and panel fingerprints;
- seed and complete hyperparameters;
- function and candidate kill rates;
- parse, invalid-candidate, timeout, and infrastructure-error rates;
- Wilson confidence intervals and per-function outcomes; and
- an explicit statement that the sealed test split was or was not used.

The historical 67/100 validation smoke result is provisional under the
hardened evaluator. It must be reproduced before it is used as the current
headline result. The final test split remains sealed until model selection is
complete.
