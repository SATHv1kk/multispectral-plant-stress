"""Multispectral analysis of plant physiological stress.

Predicts leaf temperature (Tleaf) and stomatal conductance (gsw) in oats and
barley from gantry-captured multispectral imagery.

Project page and live demo: https://sathvik.info
"""

__version__ = "1.0.0"

from . import config, indices  # noqa: F401

__all__ = ["config", "indices", "__version__"]
