#Introduction
The permuted_brunnermunzel Python function can be used to perform a permutation-based test for the problem of independent two-sample test. The function is an implementation of the permuted Brunner-Munzel test, and can compute two-sided, greater, or less tests, depending on the input argument. The function also allows the user to specify the estimator of the difference of the location shifts of the two distributions, as well as the policy for handling missing values (NaNs) in the input data.

This is a copy of the R/Fortran implementation found here https://cran.r-project.org/web/packages/brunnermunzel/index.html

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
from permuted_brunnermunzel import permuted_brunnermunzel

x = [1, 2, 3, 4, 5]
y = [3, 4, 5, 6, 7]

result = permuted_brunnermunzel(x, y, alternative="two_sided", nan_policy="propagate", est="original", force=False)
print(result)
```
```
(0.4444444444444444, 0.42000000000000004)
```