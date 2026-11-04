# Validation on 2026-11-04

RTX A6000 (SM86), Torch 2.12.0+cu130, Triton 3.7.0, Helion 1.5.2,
Python 3.13. Tests execute on CUDA rather than an interpreter-only reference.

- Full repository suite: **619 passed, 2 skipped**, including Helion GPU checks.
- Final CUDA inference/training regression suite after the additional routing,
  small-unit padding and indexing guards: **59 passed**.
- Ruff and `git diff --check`: passed.
- All twelve 784-pixel training timing cases completed: batches 128/256,
  units 32/256, each with softmax Adam, EM and Anemone.

The final targeted command was:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/kernels-pr-test-cache \
OMP_NUM_THREADS=4 python -m pytest \
    tests/test_cuda_dispatch.py tests/test_cuda_training.py -q
```

The tests cover all five stability modes, partial matrix tiles and two-unit
models, dynamic categorical routes/data/weights, autograd fallback, saved
forward routes and parameters, invalid-address masking, TF32/TF32x3 probability
floors, arbitrary upstream gradients, optimizer resume and CUDA graph replay.
The custom backward is first order only. Near-zero Adam gradients may amplify
float32 roundoff into individual logit differences; regression checks include
likelihoods and explicit gradient comparisons.

Historical prototype results are labeled `*-prior.csv`. Fresh tuning used
additional experimental variants that were discarded rather than installed in
the runtime. Training timing source hashes are preserved in the JSON report;
the final additional fixes do not change the timed training GPU kernel bodies.
