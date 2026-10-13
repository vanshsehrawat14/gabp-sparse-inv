# Contributing to gabp-sparse-inv

By participating you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Questions

Open a [GitHub issue](https://github.com/vanshsehrawat14/gabp-sparse-inv/issues) with the
`question` label, or email <vanshdeep.sehrawat@gmail.com>. The [README](README.md) lists
the supported scope and its limits; [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) maps
the code and [`docs/derivations.md`](docs/derivations.md) the mathematics.

## Reporting a bug or requesting a feature

Open a [GitHub issue](https://github.com/vanshsehrawat14/gabp-sparse-inv/issues). A useful
bug report includes a minimal snippet, the expected and actual results (or the traceback),
your OS and Python, `torch` and `gabp-sparse-inv` versions, and the matrix structure
(chain, star, tree, junction or non-symmetric) with its rough size (`n`, block size).

Some regimes are out of scope: iterative loopy belief propagation, pivoted non-symmetric
LU, and indefinite or complex-Hermitian matrices. Requests there are welcome as
discussion but may be declined to keep the package focused.

## Contributing code

1. Fork the repository and create a topic branch.
2. Install in editable mode with the development extra:

   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. Keep the change small and in the existing style.
4. Add or update tests. Every kernel is validated against a dense oracle, and every
   differentiable kernel also with `torch.autograd.gradcheck`;
   [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) describes the full bar and the pattern
   for adding a kernel.
5. Run `pytest` and make sure it passes.
6. Open a pull request describing the change and how you verified it. CI must pass.

## Releases (maintainer)

1. Set `__version__` in `gabp_sparse_inv/__init__.py` (the single source of truth) and move
   the changelog entries from *Unreleased* under the new version.
2. Build with `python -m build`, install the wheel into a clean environment, and run the
   test suite against it.
3. Tag the release commit `vX.Y.Z`, publish the tag, upload the built files to PyPI, and
   update `CITATION.cff`.
