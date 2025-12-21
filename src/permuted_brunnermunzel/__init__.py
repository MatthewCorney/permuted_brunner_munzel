"""
Permuted Brunner-Munzel Test

A Python implementation of the permuted Brunner-Munzel test for
comparing two independent samples with small sample sizes (7-10 observations).
"""

from .brunnermunzel_test import permuted_brunnermunzel

__version__ = "0.1.0"
__all__ = ['permuted_brunnermunzel', '__version__']
