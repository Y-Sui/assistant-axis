# Project Guidelines

This is a research codebase. Code should be clean, concise, and short.

- No need for production-level robustness, error handling, or abstractions
- Prefer simple, direct implementations over extensible ones
- No docstrings unless the logic is non-obvious
- No type annotations unless they clarify something ambiguous
- No unnecessary comments, logging, or validation

## Project Focus

This project is about **degeneration axes** only. Ignore the assistant axis.

The goal is to compute linear directions in a model's activation space that capture specific degeneration failure modes (repetition, verbosity, hallucination, refusal spirals, contradiction). The axis for each category:

```
axis = mean(clean) - mean(degen)
```

Higher projection = less degenerated. These axes can then be used with activation steering (addition or capping) at inference time to suppress degeneration without retraining.
