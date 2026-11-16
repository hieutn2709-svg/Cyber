# E2 fixed-checkpoint train-mode snapshot

Continue E2 with a bounded measurement on real CTI inputs and close the prior
integration-test minor. Use both SHA-verified seed42 selected checkpoints, the
locked matched70 fold1 train partition (36 docs/49 windows per encoder), width6,
cap128, entity0.05, relation distance96, negative ratio6 and unchanged loss config.

A new full optimizer trajectory would be needed to measure collisions throughout
training. This continuation instead selects a fixed-checkpoint train-mode snapshot:
retain dropout and training candidate injection/sampling, hold all weights fixed,
run no backward/optimizer and evaluate no held-out inputs. This is an explicit
implementation choice, not a claim to reproduce historical training batches.

Reset RNG for each logging-on/off comparison. Compare full sampled batches/order,
loss tensors and metadata exactly; require matching post-forward Torch RNG,
unchanged Python/NumPy RNG, unchanged state-dict tensors and strict checkpoint load.
Record candidate inputs so pool/sampler counts can be independently reconstructed.
Use a separate output directory; all incomplete attempts remain excluded.

Use seed42 for snapshot stochastic state, the existing epoch1 shuffle schedule
(Random(42+1009)) and per-window sampling seeds42+100000+step. These schedule choices
do not turn the snapshot into an epoch of training. Report denominators separately
before/after sampling, per-window and unique-document incidence. No quality/F1,
historical prevalence, significance or runtime-speedup claim follows.

Regression addition: persist wrapper forwarding and diagnostics-aware recovery
history/next-epoch parity checks in tests/test_training_diagnostics_boundaries.py.
No production model or sampler change is intended.
