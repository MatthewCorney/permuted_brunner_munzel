import numpy as np
import pytest


from permuted_brunnermunzel.brunnermunzel_test import ge
from permuted_brunnermunzel.brunnermunzel_test import divide_groups
from permuted_brunnermunzel.brunnermunzel_test import calc_statistics
from permuted_brunnermunzel.brunnermunzel_test import bm_permutation_stat
from permuted_brunnermunzel.brunnermunzel_test import combination
from permuted_brunnermunzel.brunnermunzel_test import rank
from permuted_brunnermunzel.brunnermunzel_test import permuted_brunnermunzel


@pytest.mark.parametrize("a, b, expected", [
    (1.0, 1.0, True),
    (1.0, 2.0, False),
    (1.0, 1.00000000000000000001, True),
    (1.0, 1.0000000000000000001, True),
    (1.0, 1.0000000000000001, True),
    (1.0, 1.000000000000001, False),
    (1.0, 1.1, False),
    (1.0, 2.1, False),
])
def test_ge(a, b, expected):
    assert ge(a, b) == expected


@pytest.mark.parametrize("length_x, length_y, all_data, idx, expected", [
    (3, 4, [1, 2, 1, 3, 3, 4, 3], [1, 2, 3],
     ([1, 2, 1], [3, 3, 4, 3], [1, 2, 1, 3, 3, 4, 3])),
    (3, 4, [1, 2, 1, 3, 3, 4, 3], [2, 5, 7],
     ([2, 3, 3], [1, 1, 3, 4], [2, 3, 3, 1, 1, 3, 4]))
])
def test_divide_groups(length_x, length_y, all_data, idx,
                       expected):
    x, y, xy = divide_groups(length_x, length_y, all_data, idx)
    assert x == expected[0]
    assert y == expected[1]
    assert xy == expected[2]


@pytest.mark.parametrize("length_x, length_y, all_data, const, idx, expected", [
    (3, 4, [1, 2, 1, 3, 3, 4, 3], [2.0, 2.5, 1.5, 1.3333333333333333], [1, 2, 3],
     1.1161374072507413),
    (3, 4, [1, 2, 1, 3, 3, 4, 3], [2.0, 2.5, 1.5, 1.3333333333333333], [2, 5, 7],
     0.18408308906337592
     )
])
def test_calc_statistics(length_x, length_y, all_data, const, idx, expected):
    actual_statistic = calc_statistics(length_x, length_y, all_data, const, idx)
    np.testing.assert_allclose(actual_statistic, expected)


@pytest.mark.parametrize("length_total, length_x, combinations_of_x, all_data, expected", [
    (7, 3, 35, [1, 2, 1, 3, 3, 4, 3],
     [1.1161374072507413,
      0.6906699082183761,
      0.6906699082183761,
      0.4372268185809192,
      0.6906699082183761,
      0.8478777704451413,
      0.8478777704451413,
      0.4700633956814719,
      0.8478777704451413,
      0.23411502173172788,
      0.15475519089988365,
      0.23411502173172788,
      0.15475519089988365,
      0.23411502173172788,
      0.1516729865061038,
      0.5339934229851844,
      0.5339934229851844,
      0.38802756138868133,
      0.5339934229851844,
      0.18408308906337592,
      0.1487678997147789,
      0.18408308906337592,
      0.1487678997147789,
      0.18408308906337592,
      0.15385544379235028,
      0.26648544566940835,
      0.18332398030762342,
      0.26648544566940835,
      0.18332398030762342,
      0.26648544566940835,
      0.1625710366680441,
      -0.30697030675746023,
      -0.3745225968783714,
      -0.2965108097305752,
      -0.2965108097305752,])
])
def test_bm_permutation_stat(length_total, length_x, combinations_of_x, all_data, expected):
    res = bm_permutation_stat(length_total, length_x, combinations_of_x, all_data)
    for i, item in enumerate(res):
        np.testing.assert_allclose(item, expected[i])


@pytest.mark.parametrize("length_total, length_x, ini, arr, expected_output", [
    (7, 3, [1, 2, 3], [1, 2, 3], [1, 2, 4]),
    (7, 3, [1, 3, 5], [1, 2, 3], [1, 3, 6]),
])
def test_combination(length_total, length_x, ini, arr, expected_output):
    arr = ini.copy()
    combination(length_total, length_x, ini, arr)
    assert arr == expected_output



@pytest.mark.parametrize("input_list, expected_output", [
    ([5, 3, 2, 4, 1], [4, 2, 1, 3, 0]),
    ([2, 2, 3, 3, 1], [4, 0, 1, 2, 3]),
    ([0, 0, 0, 0], [0, 1, 2, 3]),
    ([1.5, 2.7, -3, 4.2, 0], [2, 4, 0, 1, 3]),
])
def test_rank(input_list, expected_output):
    assert list(rank(input_list)) == expected_output


@pytest.mark.parametrize("x, y, alternative, nan_policy, est, expected_output", [
    ([0, 0, 0, 1, 1, 1, 0],
     [30, 20, 19, 18, 15, 10, ],
     'greater',
     'propagate',
     'original',
     (0.8571428571428571, 1)),
    ([0, 0, 0, 1, 1, 1, 0],
     [30, 20, 19, 18, 15, 10, ],
     'less',
     'propagate',
     'original',
     (0.8571428571428571, 0.0005827505827505828)),
    ([0, 0, 0, 1, 1, 1, 0],
     [30, 20, 19, 18, 15, 10, ],
     'two_sided',
     'propagate',
     'original',
     (0.8571428571428571, 0.0005827505827505828)),
    ([0, 0, 0, 1, 1, np.nan, 0],
     [30, 20, 19, 18, 15, 10, ],
     'two_sided',
     'propagate',
     'original',
     (np.nan, np.nan)),
    ([0, 0, 0, 1, 1, np.nan, 0],
     [30, 20, 19, 18, 15, 10, ],
     'two_sided',
     'raise',
     'original',
     ValueError),
    ([0, 0, 0, 1, np.nan, 1, 0],
     [30, 20, 19, 18, 15, 10, ],
     'less',
     'omit',
     'original',
     (0.8333333333333334, 0.0010822510822510823)),
    # Test est="difference" parameter
    ([0, 0, 0, 1, 1, 1, 0],
     [30, 20, 19, 18, 15, 10, ],
     'less',
     'propagate',
     'difference',
     (0.7142857142857142, 0.0005827505827505828)),  # 0.8571... * 2 - 1
    # Test NaN in y array with propagate
    ([0, 0, 0, 1, 1, 1, 0],
     [30, 20, np.nan, 18, 15, 10, ],
     'two_sided',
     'propagate',
     'original',
     (np.nan, np.nan)),
    # Test NaN in y array with raise
    ([0, 0, 0, 1, 1, 1, 0],
     [30, 20, np.nan, 18, 15, 10, ],
     'two_sided',
     'raise',
     'original',
     ValueError),
])
def test_permuted_brunnermunzel(x,
                                y,
                                alternative,
                                nan_policy,
                                est,
                                expected_output):
    try:
        pst, pval = permuted_brunnermunzel(x=x,
                                           y=y,
                                           alternative=alternative,
                                           nan_policy=nan_policy,
                                           est=est)
        np.testing.assert_allclose(pst, expected_output[0])
        np.testing.assert_allclose(pval, expected_output[1])
    except ValueError:
        assert expected_output==ValueError


def test_package_import():
    """Test that the package can be imported correctly."""
    from permuted_brunnermunzel import permuted_brunnermunzel
    assert callable(permuted_brunnermunzel)
