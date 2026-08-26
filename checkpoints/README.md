# Checkpoints directory

Model adapters, optimizer states, and frozen SFT references are generated
artifacts and are not committed to Git. They currently require multiple
gigabytes and must remain paired with their provenance metadata and checksums.

Place authorized run artifacts under `checkpoints/<run-name>/`. DPO must use
the frozen SFT reference recorded by that run; the code rejects a missing or
checksum-mismatched reference adapter.

If model artifacts need to be shared later, use controlled artifact storage
or a model registry and publish the recorded SHA-256 checksums. Do not commit
weights directly to this repository.
