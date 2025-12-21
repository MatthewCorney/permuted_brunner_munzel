# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-12-21

### Added
- Type hints throughout the codebase
- `py.typed` marker for type checker support
- Comprehensive test suite with 29 tests
- CI/CD with GitHub Actions
- NumPy-style docstrings
- Named constants for configuration values

### Fixed
- Broken package import (missing relative import)
- `nan_policy` parameter default value
- Sample size validation (now checks both x and y)
- NaN handling for y array (was only checking x)
- Dead code in scipy fallback condition
- Type annotation for `rank()` function

### Changed
- Modernized packaging (pyproject.toml only)
- Vectorized loops for better performance
- Updated README with correct return value descriptions
- Minimum Python version increased to 3.8

### Removed
- Deprecated `setup.py`
- Redundant code

## [0.0.2] - Previous release

### Added
- Initial public release
