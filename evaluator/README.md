# OpenAI Guided Diffusion Evaluation

This folder contains evaluation code copied from OpenAI's guided-diffusion repository:

https://github.com/openai/guided-diffusion/tree/main/evaluations

Use `evaluator.py` to compare a reference image batch npz against a sample image batch npz.

```bash
python evaluator/evaluator.py reference.npz sample.npz
```

`evaluator_transplant.py` is a separate evaluator copy provided from the
Transplant/DCT workflow. It is kept independent from `evaluator.py`.
