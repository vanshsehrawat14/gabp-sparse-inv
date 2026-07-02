# Contributing to gabp-sparse-inv

Thanks for your interest in the project. This document is the entry point for
the three things JOSS calls "community guidelines": how to **contribute**, how to
**report issues**, and how to **get support**.

By participating you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Get support / ask a question

- **Usage questions** ("how do I call kernel X", "which kernel fits my matrix"):
  open a [GitHub issue](https://github.com/vanshsehrawat14/gabp-sparse-inv/issues)
  with the `question` label, or email the maintainer at
  <vanshdeep.sehrawat@gmail.com>.
- Before asking, skim the [README](README.md) quickstarts and
  [`docs/`](docs/), in particular [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
  (module map and conventions) and [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md)
  (what is implemented, tested, and explicitly out of scope).

## Report a bug or request a feature

Open a [GitHub issue](https://github.com/vanshsehrawat14/gabp-sparse-inv/issues).
A good bug report includes:

- what you ran (a **minimal** code snippet) and what you expected,
- the actual result or traceback,
- your OS, Python version, and `torch` / `numpy` versions,
- the matrix structure involved (chain / star / tree / junction / non-symmetric)
  and rough size (`n`, block size).

For scope questions, note that several regimes are **deliberately out of scope**
(iterative/loopy GaBP, pivoted non-symmetric LU, indefinite/complex-Hermitian
matrices); see the package docstring and README scope notes. Requests in
those areas are welcome as discussion, but may be declined to keep the package
focused.

## Contribute code

1. **Fork** the repository and create a topic branch.
2. **Set up a dev environment** and install in editable mode with the dev extra:

   ```bash
   python3.12 -m venv .venv && source .venv/bin/activate   # Windows: py -3.12 -m venv .venv; .\.venv\Scripts\Activate.ps1
   pip install torch numpy
   pip install -e ".[dev]"
   ```

3. **Make the change.** Please keep changes surgical and in the existing style:
   make the smallest change that solves the task, avoid speculative features,
   and match the surrounding code and documentation tone.
4. **Add or update tests.** Every numerical kernel is validated against a dense
   `O(n^3)` oracle and every differentiable kernel additionally with
   `torch.autograd.gradcheck`; new kernels are expected to follow the same bar.
5. **Run the suite** and make sure it is green:

   ```bash
   pytest -q
   ```

6. **Open a pull request** describing the change and how you verified it. CI
   (Ubuntu / Windows / macOS, Python 3.12 and 3.13) must pass.

### Adding a new kernel

[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) documents the kernel pattern
(layout dataclass -> factor/forward -> optional analytic backward -> exports ->
tests). Follow an existing kernel of similar structure as the template, export it
from `gabp_sparse_inv/__init__.py`, and add a test module mirroring the others in
`tests/`.

## Release process (maintainer)

Releases are cut manually; there is intentionally no automated semantic-release
machinery for a package of this size.

1. Update `__version__` in `gabp_sparse_inv/__init__.py` (the single source of
   truth; `pyproject.toml` reads it dynamically) and add a `CHANGELOG.md` entry.
2. Tag the release: `git tag -a vX.Y.Z -m "vX.Y.Z" && git push --tags`.
3. The tag triggers (or you manually run) the archive step; for the JOSS
   submission the tagged commit is archived on Zenodo and the resulting DOI is
   added to the paper. See `.zenodo.json` for the archive metadata.
