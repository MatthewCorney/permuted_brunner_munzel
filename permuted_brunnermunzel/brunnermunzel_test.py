from typing import Literal
from typing import List
from typing import Union
import logging
import math
import numpy as np
from scipy.stats import brunnermunzel


def ge(a: float, b: float) -> bool:
    """
    Check if a is greater than or equal to b, accounting for rounding issues.

    Sometimes when working with floating point numbers, very small differences
    can occur due to rounding errors. This function checks if the difference
    between the two numbers is smaller than 1e-15, which is a commonly used
    threshold for such cases. If the difference is smaller than 1e-15, the
    function returns True, otherwise it returns the result of a >= b.

    :param a: A floating point number to compare.
    :param b: A floating point number to compare.
    :return: True if a is greater than or equal to b (accounting for rounding
             errors), False otherwise.
    """
    res = abs(a - b) < 1e-15 or a >= b
    return res


def divide_groups(length_x: int, length_y: int, all_data: List[float], idx: List[int]):
    """
    Divide data into two groups by a list of indexes. The function loops over the input data `dat`, and checks the
    corresponding index in `idx`. If the index belongs to the first group, the value is added to the `x` list. If the
    index belongs to the second group, the value is added to the `y` list. The resulting `x` and `y` lists are
    ordered based on the order of the indices in `idx`. Finally, the function concatenates the `x` and `y` lists to
    form the `xy` list.

    :param length_x: (int): The length of the first group.
    :param length_y: (int): The length of the second group.
    :param all_data: (List[float]): The data to be divided.
    :param idx: (List[int]): A list of indexes indicating which group each element in `dat` belongs to.

    :return: Tuple[List[float], List[float], List[float]]: A tuple containing three lists:
            - `x`: The elements of `dat` that belong to the first group, ordered according to their order in `idx`.
            - `y`: The elements of `dat` that belong to the second group, ordered according to their order in `idx`.
            - `xy`: A concatenation of `x` and `y` in the order they appear in `dat`.

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


def calc_statistics(length_x: int, length_y: int, all_data: List[Union[int, float]], const: list, idx: list) -> float:
    """
    Calculates the test statistic for the Brunner-Munzel permutation test.

    :param length_x: Number of elements in the first group.
    :param length_y: Number of elements in the second group.
    :param all_data: A list of length n with the data to be tested.
    :param const: A list of four constants used in the calculation of the statistic.
                  The four constants should be [0.5 * (nx + 1), 0.5 * (ny + 1), nx / (nx - 1), ny / (ny - 1)].
    :param idx: A list of length r with the current indices of the items in the first group.
    :return: The test statistic for the Brunner-Munzel permutation test.
    """
    new_x, new_y, new_x_y = divide_groups(length_x, length_y, all_data, idx)
    rank_x = rank(new_x)
    rank_y = rank(new_y)
    rank_x_y = rank(new_x + new_y)
    mean_rank_x = np.mean(rank_x_y[:length_x])
    mean_rank_y = np.mean(rank_x_y[length_x:length_x + length_y])

    x_squared_deviation = (rank_x_y[:length_x] - rank_x - mean_rank_x + const[0]) ** 2
    y_squared_deviation = (rank_x_y[length_x:length_x + length_y] - rank_y - mean_rank_y + const[1]) ** 2
    vx = 0
    for i in range(length_x):
        vx += x_squared_deviation[i]
    vy = 0
    for i in range(length_y):
        vy += y_squared_deviation[i]

    v = const[2] * vx + const[3] * vy
    stat = (mean_rank_y - mean_rank_x) / np.sqrt(v)
    return stat


def bm_permutation_stat(length_total: int, length_x: int, combinations_of_x: int, all_data: List[Union[float,int]]):
    """
    Returns a list of permutation statistics for the given data by performing all possible
    combinations of size r on the data and calculating the statistic for each combination.

    :param length_total: The total number of elements in the data.
    :param length_x: The size of each combination of elements to be considered.
    :param combinations_of_x: The total number of possible combinations to be considered.
    :param all_data: A list of numbers representing the data.
    :return: A list of statistics calculated for each combination.
    """
    statistics = []
    length_x = length_x
    length_y = length_total - length_x
    const = [0.5 * (length_x + 1), 0.5 * (length_y + 1), length_x / (length_x - 1), length_y / (length_y - 1)]
    ini = np.arange(1, length_x + 1)
    idx = ini.copy()
    for i in range(combinations_of_x):
        stat = calc_statistics(length_x, length_y, all_data, const, idx)
        statistics.append(stat)
        idx = combination(length_total, length_x, list(ini), idx)
    return statistics


def combination(length_total: int, length_x: int, ini: List[int], arr: List[Union[int,float]]):
    """
    Computes the combination of `length_x` elements from a set of `length_total` elements, using the initial values
    in `ini` and returning the results in `arr`.The function works by computing the difference between the initial
    values in `ini` and the sequence of indices that correspond to the selected elements in the combination. This
    sequence is determined by the variables `bef`, `key` and `numx`. The resulting combination is stored in `arr`.

    Note that the function modifies the `arr` parameter in place and returns it as well.

    :param length_total: (int): The total number of elements.
    :param length_x: (int): The number of elements to select in each combination.
    :param ini: (List[int]): The initial values for the combination. Must have length `r`.
    :param arr: (List[int]): A list to store the combination. Must have length `r`.
    :return: List[int]: A list with the selected combination of elements.

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


def rank(x: list) -> np.ndarray[int]:
    """
    Returns the ranks of each element in the input list. The rank of an element
    is defined as its index in the sorted list of elements, with ties assigned
    the average rank.

    :param x: A list of numerical values.
    :return: A list of the same length as `x`, containing the ranks of the input
        elements.
    """
    ranks = np.argsort(x)
    return ranks


def permuted_brunnermunzel(x: list,
                           y: list,
                           alternative: Literal["two_sided", "greater", "less"] = "two_sided",
                           nan_policy=Literal["propagate", "raise", "omit"],
                           est: Literal["original", "difference"] = "original",force:bool=False):
    """
    Computes the permuted Brunner-Munzel test for independent two-sample problem.

    :param x: list A list of observations for sample 1.
    :param y: list A list of observations for sample 2.
    :param alternative: {"two.sided", "greater", "less"}, optional
        The alternative hypothesis to be tested. It can be one of the following:
            - 'two.sided': The alternative hypothesis is that the location shift of the two distributions
              is not equal to zero (i.e., they differ).
            - 'greater': The alternative hypothesis is that the location shift of sample 2 is greater
              than that of sample 1.
            - 'less': The alternative hypothesis is that the location shift of sample 2 is less than that
              of sample 1.
        Default is 'two.sided'.
    :param nan_policy: {"propagate", "raise", "omit"}, optional
        Defines how to handle missing values (NaNs) in the input data. It can be one of the following:
            - 'propagate': propagate the NaNs and return NaNs as output.
            - 'raise': raise a ValueError exception when NaNs are encountered in the input data.
            - 'omit': ignore the NaNs and perform the test using only the non-missing values.
        Default is 'propagate'.
    :param est: {"original", "difference"}, optional
        The estimator of the difference of the location shifts of the two distributions. It can be one of
        the following:
            - 'original': The estimator is P(X<Y)+0.5*P(X=Y), where X and Y are the rank sums of the two
              samples.
            - 'difference': The estimator is P(X<Y)-P(X>Y).
        Default is 'original'.
    :param force: bool
     If the test should be forced to occur in cases where the normal brunner-munzel would be more appropriate
        Default is False
    :return: A tuple of two float values:
        - The first value is the estimated location shift between the two distributions (i.e., the P-value).
        - The second value is the P-value of the test.
    """
    # Handeling of NaN in the samples
    if nan_policy == 'raise':
        if any([math.isnan(item) for item in x]):
            logging.error(f'{nan_policy=} and array contains NaN values')
            raise ValueError
    elif nan_policy == 'omit':
        x = [item for item in x if not math.isnan(item)]
        y = [item for item in y if not math.isnan(item)]
    elif nan_policy == 'propagate':
        if any([math.isnan(item) for item in x]):
            logging.warning(f'{nan_policy=} and array contains NaN values')
            return np.nan, np.nan
    # deals with the Observation sizes
    length_x = len(x)
    length_y = len(y)
    length_total = int(length_x + length_y)
    if (length_x == 1) or (length_y == 1):
        logging.error('Not Enough Observations')
        raise ValueError
    elif (length_x < 7) or (length_x < 7):
        logging.warning('The permuted brunner-munzel test loses power with samples sizes with fewer than 7 observations')
        pass
    elif (length_x > 10) or (length_x > 10):
        logging.warning('The normal brunner-munzel test is more suitable for samples of 10 or greater observations')
        pass
    else:
        pass
    combinations_of_x = np.math.comb(length_x + length_y, length_x)
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
    if not force and ():
        logging.error("Number of observations in both samples is over 10, defaulting to scipy brunner-munzel "
                      "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.brunnermunzel.html"
                      f"To stop this behaviour use the change {force=} to {True}")
        pst, p_value = brunnermunzel(x, y, alternative=alternative)
        return pst, p_value
    else:
        rank_all_data = rank(all_data)
        mean_rank_x = np.mean(rank_all_data[:length_x])
        mean_rank_y = np.mean(rank_all_data[length_x:length_total])
        pst = (mean_rank_y - (length_total - length_x + 1) * 0.5) / length_x
        statistics = bm_permutation_stat(length_total, length_x, combinations_of_x, all_data)
        if alter == 1:  # "two sided"
            statistics = [abs(x) for x in statistics[:combinations_of_x]]
        elif alter == 2:  # "greater"
            statistics = [-x for x in statistics[:combinations_of_x]]
        else:  # "less"
            statistics = statistics
        count = 0
        for i in range(combinations_of_x):
            if ge(statistics[i], statistics[0]):
                count += 1
        pval = float(count) / combinations_of_x
        res = {}
        if est == "original":
            estimate = pst
            estimate_name = "P(X<Y)+.5*P(X=Y)"
        else:
            estimate = res['pst'] * 2 - 1
            estimate_name = "P(X<Y)-P(X>Y))"
        return pst, pval