## Summary

What does this change and why?

## Verification

- [ ] `pytest -q` passes locally
- [ ] New/changed numerical kernels are validated against a dense oracle (and
      differentiable kernels against `torch.autograd.gradcheck`)
- [ ] Docs/README updated if behavior, scope, or install steps changed
- [ ] Change is surgical and matches existing style
