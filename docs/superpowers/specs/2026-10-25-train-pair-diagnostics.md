# E2 training-pair collision diagnostics

Continue the E2 readiness next step: observational, default-off diagnostics in
actual training candidate construction. No change to gold, supervision, sampler,
architecture, RNG consumption, optimizer or validation checkpoint selection.

Measure directed boundary groups with both existence labels in eligible lists
and actual sampled batches, including positive/negative denominators, self-edge
omissions, long positives, and a stable digest of sampled pair order and labels.
Window-local examples are bounded to five groups and ten type combinations per
label/group. The runner adds document/window, global offset, epoch, step and seed.

Approaches: direct observation of existing sampler lists is chosen; rebuilding
another pool risks policy drift. Changing collision policy or prioritizing hard
negatives is a separate experiment and is not part of this implementation.

Optional diagnostics dictionary flows from compute_training_window_loss to
build_training_pairs. --train-pair-diagnostics enables per-window records in
train history for overfit/smoke/dev. Both matched wrappers forward the flag.
Enabled observation is recorded in run configuration and recovery identity;
disabled output remains unchanged. No new full/test or pretrained training run.

Verification uses adversarial candidate pools and a small real PyTorch model:
exact pair/order/RNG parity, loss, gradients, optimizer parameters and epoch
history. Synthetic-model evidence does not measure historical baseline collision
prevalence. Corpus measurement needs a separately identified real training run.
