"""Vendored Tennessee Eastman Process (TEP) simulator.

Source: https://huggingface.co/spaces/jkitchin/tennessee-eastman-process
License: third_party/tep/LICENSE

This submission package defaults to the pure-Python backend for portability.
"""

from .simulator import TEPSimulator, ControlMode
from .controllers import PIController, DecentralizedController, ManualController
from .python_backend import PythonTEProcess
from .constants import (
    NUM_STATES,
    NUM_MEASUREMENTS,
    NUM_MANIPULATED_VARS,
    NUM_DISTURBANCES,
    COMPONENT_NAMES,
    MEASUREMENT_NAMES,
    MANIPULATED_VAR_NAMES,
    DISTURBANCE_NAMES,
    OPERATING_MODES,
    DEFAULT_OPERATING_MODE,
    DEFAULT_RANDOM_SEED,
)

__version__ = "vendored"


def get_available_backends():
    return ["python"]


def get_default_backend():
    return "python"


def is_fortran_available():
    return False


__all__ = [
    "TEPSimulator",
    "ControlMode",
    "PythonTEProcess",
    "PIController",
    "DecentralizedController",
    "ManualController",
    "NUM_STATES",
    "NUM_MEASUREMENTS",
    "NUM_MANIPULATED_VARS",
    "NUM_DISTURBANCES",
    "COMPONENT_NAMES",
    "MEASUREMENT_NAMES",
    "MANIPULATED_VAR_NAMES",
    "DISTURBANCE_NAMES",
    "OPERATING_MODES",
    "DEFAULT_OPERATING_MODE",
    "DEFAULT_RANDOM_SEED",
    "get_available_backends",
    "get_default_backend",
    "is_fortran_available",
]
