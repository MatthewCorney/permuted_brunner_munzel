import logging
import math
from typing import Literal

import numpy as np
from scipy.stats import brunnermunzel

# Constants
FLOAT_TOLERANCE = 1e-15
MIN_OBSERVATIONS = 2
RECOMMENDED_MIN_SAMPLE = 7
RECOMMENDED_MAX_SAMPLE = 10


def ge(a: float, b: float) -> bool:
    """
    Check if a >= b with floating-point tolerance.

    Parameters
    ----------
    a : float
        First value to compare.
    b : float
        Second value to compare.

    Returns
    -------
    bool
        True if a >= b (within FLOAT_TOLERANCE), False otherwise.
    """
    return abs(a - b) < FLOAT_TOLERANCE or a >= b


def divide_groups(length_x: int, length_y: int, all_data: list[float], idx: list[int]):
    """
    Split data into two groups based on index positions.

    Parameters
    ----------
    length_x : int
        Number of elements in the first group.
    length_y : int
        Number of elements in the second group.
    all_data : list of float
        The combined data to be split.
    idx : list of int
        1-based indices indicating which positions belong to group x.

    Returns
    -------
    x : list of float
        Elements belonging to the first group.
    y : list of float
        Elements belonging to the second group.
    xy : list of float
        Concatenation of x and y.
    """
    new_x = [0.0] * length_x
    new_y = [0.0] * length_y
    new_x_y = [0.0] * (length_x + length_y)
    temporary_index_list = [0] * (length_x + 1)

    temporary_index_list[0:length_x] = idx
    temporary_index_list[length_x] = -1

    ix = 0
    iy = 0
    for i in range(length_x + length_y):
        if i + 1 == temporary_index_list[ix]:
            new_x[ix] = all_data[i]
            ix += 1
        else:
            new_y[iy] = all_data[i]
            iy += 1

    new_x_y[0:length_x] = new_x[:]
    new_x_y[length_x:(length_x + length_y)] = new_y[:]
    return new_x, new_y, new_x_y


def calc_statistics(length_x: int, length_y: int, all_data: list[int | float], const: list, idx: list) -> float:
    """
    Calculate the Brunner-Munzel test statistic for a given permutation.

    Parameters
    ----------
    length_x : int
        Number of elements in the first group.
    length_y : int
        Number of elements in the second group.
    all_data : list of float
        Combined data from both groups.
    const : list of float
        Precomputed constants: [0.5*(nx+1), 0.5*(ny+1), nx/(nx-1), ny/(ny-1)].
    idx : list of int
        Indices of elements belonging to the first group.

    Returns
    -------
    float
        The test statistic for this permutation.
    """
    new_x, new_y, new_x_y = divide_groups(length_x, length_y, all_data, idx)
    rank_x = rank(new_x)
    rank_y = rank(new_y)
    rank_x_y = rank(new_x + new_y)
    mean_rank_x = np.mean(rank_x_y[:length_x])
    mean_rank_y = np.mean(rank_x_y[length_x:length_x + length_y])

    x_squared_deviation = (rank_x_y[:length_x] - rank_x - mean_rank_x + const[0]) ** 2
    y_squared_deviation = (rank_x_y[length_x:length_x + length_y] - rank_y - mean_rank_y + const[1]) ** 2
    vx: np.floating = np.sum(x_squared_deviation)
    vy: np.floating = np.sum(y_squared_deviation)

    v = const[2] * vx + const[3] * vy
    stat = (mean_rank_y - mean_rank_x) / np.sqrt(v)
    return stat


def bm_permutation_stat(length_total: int, length_x: int, combinations_of_x: int, all_data: list[float | int]):
    """
    Compute test statistics for all possible permutations.

    Generates all C(n, nx) combinations and calculates the Brunner-Munzel
    test statistic for each permutation.

    Parameters
    ----------
    length_total : int
        Total number of observations (nx + ny).
    length_x : int
        Number of elements in the first group.
    combinations_of_x : int
        Total number of combinations C(n, nx).
    all_data : list of float
        Combined data from both groups.

    Returns
    -------
    list of float
        Test statistics for each permutation.
    """
    statistics = []
    length_y = length_total - length_x
    const = [0.5 * (length_x + 1), 0.5 * (length_y + 1), length_x / (length_x - 1), length_y / (length_y - 1)]
    ini = np.arange(1, length_x + 1)
    idx = ini.tolist().copy()
    for _ in range(combinations_of_x):
        stat = calc_statistics(length_x, length_y, all_data, const, idx)
        statistics.append(stat)
        idx = combination(length_total, length_x, ini.tolist(), idx)
    return statistics


def combination(length_total: int, length_x: int, ini: list[int], arr: list[int | float]):
    """
    Generate the next combination in lexicographic order.

    Advances from the current combination to the next one. Modifies arr in place.

    Parameters
    ----------
    length_total : int
        Total number of elements to choose from.
    length_x : int
        Number of elements to select.
    ini : list of int
        Initial index values [1, 2, ..., length_x].
    arr : list of int
        Current combination indices (modified in place).

    Returns
    -------
    list of int
        The next combination indices.
    """
    bef = [arr[i] - ini[i] for i in range(length_x)]
    arr[:] = [0] * length_x
    key = [0] * length_x
    numx = length_total - length_x
    for i in range(length_x):
        if bef[i] == numx:
            key[i] = 1
    for i in range(length_x - 1):
        if key[i + 1] == 1:
            if key[i] == 1:
                if i != 0:
                    arr[i] = arr[i - 1]
            else:
                arr[i] = bef[i] + 1
        else:
            arr[i] = bef[i]
    if key[length_x - 1] == 1:
        arr[length_x - 1] = arr[length_x - 2]
    else:
        arr[length_x - 1] = bef[length_x - 1] + 1
    arr[:] = [arr[i] + ini[i] for i in range(length_x)]
    return arr


def rank(x: list) -> np.ndarray:
    """
    Return indices that would sort the input array.

    Parameters
    ----------
    x : list of float
        Input values.

    Returns
    -------
    ndarray
        Indices that would sort the array. For [5, 3, 2, 4, 1],
        returns [4, 2, 1, 3, 0] (index 4 has smallest value, etc.).
    """
    return np.argsort(x)


def permuted_brunnermunzel(x: list,
                           y: list,
                           alternative: Literal["two_sided", "greater", "less"] = "two_sided",
                           nan_policy: Literal["propagate", "raise", "omit"] = "propagate",
                           est: Literal["original", "difference"] = "original", force: bool = False) -> tuple[float, float]:
    """
    Compute the permuted Brunner-Munzel test for two independent samples.

    This test is designed for small sample sizes (7-10 observations per group)
    where the standard Brunner-Munzel test may not be reliable.

    Parameters
    ----------
    x : array_like
        First sample observations.
    y : array_like
        Second sample observations.
    alternative : {'two_sided', 'greater', 'less'}, default='two_sided'
        The alternative hypothesis:
        - 'two_sided': P(X < Y) != 0.5
        - 'greater': P(X < Y) > 0.5
        - 'less': P(X < Y) < 0.5
    nan_policy : {'propagate', 'raise', 'omit'}, default='propagate'
        How to handle NaN values:
        - 'propagate': Return NaN
        - 'raise': Raise ValueError
        - 'omit': Exclude NaN values
    est : {'original', 'difference'}, default='original'
        Estimator type:
        - 'original': P(X < Y) + 0.5*P(X = Y)
        - 'difference': P(X < Y) - P(X > Y)
    force : bool, default=False
        Force permuted test even when sample sizes exceed recommended range.

    Returns
    -------
    estimate : float
        The effect size estimate.
    pvalue : float
        The p-value for the test.

    Examples
    --------
    >>> from permuted_brunnermunzel import permuted_brunnermunzel
    >>> x = [0, 0, 0, 1, 1, 1, 0]
    >>> y = [30, 20, 19, 18, 15, 10]
    >>> estimate, pvalue = permuted_brunnermunzel(x, y, alternative='less')
    >>> print(f"Estimate: {estimate:.4f}, p-value: {pvalue:.6f}")
    Estimate: 0.8571, p-value: 0.000583

    Notes
    -----
    This is a Python reimplementation of the R package 'brunnermunzel'.
    For samples larger than 10 observations, consider using scipy.stats.brunnermunzel.

    References
    ----------
    .. [1] Brunner, E. and Munzel, U. (2000). The nonparametric Behrens-Fisher
           problem: Asymptotic theory and a small-sample approximation.
    """
    # Handling of NaN in the samples
    if nan_policy == 'raise':
        if any(math.isnan(item) for item in x) or any(math.isnan(item) for item in y):
            logging.error(f'{nan_policy=} and array contains NaN values')
            raise ValueError("Input contains NaN values")
    elif nan_policy == 'omit':
        x = [item for item in x if not math.isnan(item)]
        y = [item for item in y if not math.isnan(item)]
    elif nan_policy == 'propagate':
        if any(math.isnan(item) for item in x) or any(math.isnan(item) for item in y):
            logging.warning(f'{nan_policy=} and array contains NaN values')
            return np.nan, np.nan
    # deals with the Observation sizes
    length_x = len(x)
    length_y = len(y)
    length_total = int(length_x + length_y)
    if (length_x < MIN_OBSERVATIONS) or (length_y < MIN_OBSERVATIONS):
        logging.error('Not Enough Observations')
        raise ValueError
    elif (length_x < RECOMMENDED_MIN_SAMPLE) or (length_y < RECOMMENDED_MIN_SAMPLE):
        logging.warning(
            f'The permuted brunner-munzel test loses power with samples sizes with fewer than {RECOMMENDED_MIN_SAMPLE} observations')
    elif (length_x > RECOMMENDED_MAX_SAMPLE) or (length_y > RECOMMENDED_MAX_SAMPLE):
        logging.warning(f'The normal brunner-munzel test is more suitable for samples of {RECOMMENDED_MAX_SAMPLE} or greater observations')
    combinations_of_x = math.comb(length_x + length_y, length_x)
    all_data = x + y
    if alternative == "two_sided":
        alter = 1
    elif alternative == "greater":
        alter = 2
    elif alternative == "less":
        alter = 3
    else:
        logging.error(f"{alternative=} passed, alternative must be one of {Literal['two_sided', 'greater', 'less']}")
        raise ValueError
    if not force and (length_x > RECOMMENDED_MAX_SAMPLE and length_y > RECOMMENDED_MAX_SAMPLE):
        logging.error("Number of observations in both samples is over 10, defaulting to scipy brunner-munzel "
                      "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.brunnermunzel.html"
                      f"To stop this behaviour use the change {force=} to {True}")
        pst, p_value = brunnermunzel(x, y, alternative=alternative)
        return pst, p_value
    else:
        rank_all_data = rank(all_data)
        mean_rank_y = np.mean(rank_all_data[length_x:length_total])
        pst = (mean_rank_y - (length_total - length_x + 1) * 0.5) / length_x
        statistics = bm_permutation_stat(length_total, length_x, combinations_of_x, all_data)
        if alter == 1:  # "two sided"
            statistics = [abs(x) for x in statistics[:combinations_of_x]]
        elif alter == 2:  # "greater"
            statistics = [-x for x in statistics[:combinations_of_x]]
        else:  # "less"
            statistics = statistics
        reference = statistics[0]
        statistics_array = np.array(statistics)
        count: np.integer = np.sum(statistics_array >= reference - FLOAT_TOLERANCE)
        pval = count / combinations_of_x
        if est == "original":
            estimate = pst
        else:
            estimate = pst * 2 - 1
        return float(estimate), float(pval)
