# MCLMC burn-in and tuning task

Develop a burn-in and tuning procedure for [BlackJAX](https://github.com/blackjax-devs/blackjax) MCLMC. Your
`submission.py` returns, for each problem and chain, a production starting point and the kernel parameters (step size, decoherence length `L`,
diagonal inverse mass matrix). Production sampling always uses the fixed BlackJAX MCLMC kernel supplied here.
The objective is the **total number of gradient evaluations** (burn-in plus production) needed to reach a prescribed accuracy.

Start with `instructions/APPLICANT_BRIEF.md`. Repository layout:

```text
evaluator/      trusted runner, metrics, evaluation pipeline, compliance screening, sandbox
models/         public target models, initialization distributions, reference loader
references/     validated reference moments of the public problems (with diagnostics)
data/           the two data sets served to Inference Gym (German credit, radon)
baseline/       phased BlackJAX baseline and its per-problem calibration table
examples/       minimal valid submission
instructions/   applicant brief
tests/          interface, accounting, scoring and compliance checks
```

Setup (Python 3.12): `pip install -r requirements.txt`, then `python -m pytest tests -q`.
Evaluate a submission on the public problems: `python -m evaluator.run_public submission.py --out result.json`.
