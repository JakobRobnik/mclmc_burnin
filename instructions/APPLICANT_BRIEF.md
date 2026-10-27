# Applicant brief

## Setup (first thing to do)

Work inside the folder of your session. Use `python3` (3.12) and a virtual environment inside this folder; do not use conda or any other environment.

```bash
git clone https://github.com/JakobRobnik/mclmc_burnin.git repo/mclmc_burnin   # skip if repo/mclmc_burnin already exists
cd repo/mclmc_burnin
python3 -m venv ../venv && source ../venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python -m evaluator.run_public examples/submission.py --problems banana --bootstrap 20
```

Keep your own notes, scripts and results under `work/`. Read `README.md` for the repository layout.

## Task

For each problem you receive a numerical log-density (values and gradients only, never the function) of a vector `x` of dimension `d`.
Your submission runs a burn-in of at most **20000** gradient evaluations per chain and returns a production starting point and MCLMC parameters.
The trusted runner then samples with BlackJAX MCLMC (isokinetic McLachlan integrator, partial momentum refreshment) from your point with your parameters.

## Score

* Accuracy: after every production transition compute, for every coordinate `j` of the flattened constrained sample, the normalised squared
  error of the second moment, `(mean_k(x_j^2) - E[x_j^2])^2 / Var[x_j^2]`, and average it over coordinates (`reference` values: `references/`,
  analytic or validated NUTS runs). Burn-in samples are never used.
* Ten independent chains per problem. Take the **median over the ten chains of the error curve at each production prefix**, then find the first
  prefix at which this median curve is strictly below **0.01**. A brief crossing counts.
* `n_i` = scheduled burn-in evaluations + production evaluations at that prefix. Chains with invalid outputs or numerical failure have infinite error;
  if the curve never crosses within the production horizon, `n_i = inf`.
* Production horizon: `min(floor(3 n_i(base)), 500000)` gradient evaluations, extended to `min(floor(5 n_i(base)), 500000)` if the median curve has not
  crossed. `n_i(base)` is the baseline's total count (`baseline/calibration.json` for the public problems).
* Primary score `F = exp(mean_i(log n_i(base) - log n_i(candidate)))`; `F = 0` if any `n_i` is infinite. Supplementary: `F_finite` over the problems with finite counts.
  Exceeding the burn-in cap (an invalid phase schedule) disqualifies the submission (score 0).
* Cost of one MCLMC transition: 2 gradient evaluations; building the production state costs 1.

The private test suite contains problems that are not in this repository, including newly constructed variants of known models and problems from families
not represented here. The same evaluator and rules apply.

## Submission interface (`submission.py`, module-level functions)

```python
phase_fractions(config) -> tuple of floats        # config.dim, config.budget (=20000); sum <= 1, at most 8 phases, lengths = diff(floor(cumsum*budget))
init(x0, key, config) -> carry                    # dict with query_position, production_position, beta (evaluator.mclmc_stepper.Beta); other leaves free
step_phase(phase, carry, obs) -> (carry, diag)    # phase is a static Python int; one call per gradient evaluation
transition(carry, history, info) -> carry         # between phases; numerical only; history = PhaseHistory(position, logp, grad)
finalize(carry, histories, config) -> Result(position, beta)
```

`obs` (`evaluator.interface.Observation`) holds the evaluated point, `logp`, `grad`, the global and phase indices and a key. The runner evaluates the target at
`carry["query_position"]` once per active iteration. One iteration is one gradient evaluation, not necessarily a whole MCMC move; algorithms that need
several gradients per move must advance a state machine through several iterations (see `evaluator/mclmc_stepper.py`). An unfinished transition at the end of
the burn-in is discarded. `finalize` and `transition` make no target evaluations. The carry must keep its pytree structure, shapes and dtypes within a phase;
`transition` may construct the next phase's carry.

Valid output: finite position of shape `(d,)`, finite positive `L` and step size, finite positive diagonal inverse mass matrix of shape `(d,)`
(BlackJAX convention: `inverse_mass_matrix` rescales positions). Float64 throughout.

## Rules

* No access to the target, derivatives of it, or anything but the numerical observations: no file or network access, host callbacks, introspection of
  frames or closures, mutable global state, or import-time side effects; no higher-order target derivatives or finite-difference Hessian probes. Run
  `python -m evaluator.compliance submission.py` before submitting; non-compliant submissions are rejected, not repaired.
* Chains are independent: no pooling of parameters or state across chains.
* Permitted imports: `jax`, `numpy`, `scipy`, `blackjax` (numerical utilities), `evaluator.interface`, `evaluator.mclmc_stepper`, Python standard numerical helpers.
* Use the public problems for development (they are the examples, not the test). Create your own problems and validation splits as you like. The private suite
  is not an iterative debugging service.
* Final artifact: one `submission.py` plus a short method report (approach, public experiments, dependencies, limitations).
