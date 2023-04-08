#Introduction
This package is an implementation of the permuted version of the Brunner-Munzel test is a nonparametric test.
The Permuted Brunner-Munzel is best suited to cases where the number of observations in one of the groups is below 10 and
ideally above 7

This is a reimplementation of the R/Fortran implementation found here https://cran.r-project.org/web/packages/brunnermunzel/index.html . This is for cases where is it not possible or complicated to call R from python

#Dependencies
The function requires the following Python packages:

- numpy
- scipy
- math

# Usage
The function takes five input arguments, x, y, alternative, nan_policy, est, and force.

- x and y are lists of observations for the two samples to be tested.
- alternative specifies the alternative hypothesis to be tested, and can take one of the following values: "two_sided", "greater", or "less". Default is "two_sided".
- nan_policy specifies how to handle missing values in the input data, and can take one of the following values: "propagate", "raise", or "omit". Default is "propagate".
- est specifies the estimator of the difference of the location shifts of the two distributions, and can take one of the following values: "original" or "difference". Default is "original".
- force specifies whether to force the use of the permuted Brunner-Munzel test even in cases where the normal Brunner-Munzel test would be more appropriate. Default is False.

The function returns a tuple of two float values:

- The first value is the estimated location shift between the two distributions (i.e., the P-value).
- The second value is the P-value of the test.

# Example
```
pip install permuted_brunnermunzel
```

```
from permuted_brunnermunzel import permuted_brunnermunzel

x = [0, 0, 0, 1, 1, 1, 0]
y = [30, 20, 19, 18, 15, 10, ]

result = permuted_brunnermunzel(x, y, alternative="less", nan_policy="propagate", est="original", force=False)
print(result)
```

```
(0.8571428571428571, 0.0005827505827505828)
```