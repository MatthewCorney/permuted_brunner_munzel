# Permuted Brunner-Munzel Test

[![PyPI version](https://badge.fury.io/py/permuted_brunnermunzel.svg)](https://pypi.org/project/permuted_brunnermunzel/)
[![Tests](https://github.com/MatthewCorney/permuted_brunner_munzel/actions/workflows/tests.yml/badge.svg)](https://github.com/MatthewCorney/permuted_brunner_munzel/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A Python implementation of the **permuted Brunner-Munzel test**, a nonparametric test for comparing two independent samples.

## When to Use

The permuted Brunner-Munzel test is best suited for:
- Small sample sizes (7-10 observations per group)
- When standard parametric assumptions don't hold
- Comparing distributions that may differ in shape, not just location

For larger samples (>10 per group), consider using `scipy.stats.brunnermunzel`.

## Installation

```bash
pip install permuted_brunnermunzel
```

## Quick Start

```python
from permuted_brunnermunzel import permuted_brunnermunzel

# Sample data
x = [0, 0, 0, 1, 1, 1, 0]
y = [30, 20, 19, 18, 15, 10]

# Run the test
estimate, pvalue = permuted_brunnermunzel(x, y, alternative="less")

print(f"Effect size estimate: {estimate:.4f}")
print(f"P-value: {pvalue:.6f}")
```

**Output:**
```
Effect size estimate: 0.8571
P-value: 0.000583
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | list | required | First sample observations |
| `y` | list | required | Second sample observations |
| `alternative` | str | `"two_sided"` | `"two_sided"`, `"greater"`, or `"less"` |
| `nan_policy` | str | `"propagate"` | `"propagate"`, `"raise"`, or `"omit"` |
| `est` | str | `"original"` | `"original"` or `"difference"` |
| `force` | bool | `False` | Force test even for large samples |

## Returns

| Value | Description |
|-------|-------------|
| `estimate` | Effect size: P(X < Y) + 0.5*P(X = Y) for `est="original"`, or P(X < Y) - P(X > Y) for `est="difference"` |
| `pvalue` | The p-value for the test |

## Interpreting Results

- **estimate = 0.5**: No difference between groups
- **estimate > 0.5**: Values in Y tend to be larger than X
- **estimate < 0.5**: Values in X tend to be larger than Y
- **pvalue < 0.05**: Statistically significant difference (at alpha=0.05)

## Dependencies

- numpy >=1.20
- scipy >=1.7

## References

This is a Python reimplementation of the R package [brunnermunzel](https://cran.r-project.org/web/packages/brunnermunzel/).

Brunner, E. and Munzel, U. (2000). *The nonparametric Behrens-Fisher problem: Asymptotic theory and a small-sample approximation.*

## License

MIT License - see [LICENSE](LICENSE) for details.
