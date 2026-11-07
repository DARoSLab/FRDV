"""frdv: fast, dependency-light kinematic motion retargeting.

Runtime dependencies: numpy and mujoco only.
"""

from .kinematics import Kinematics
from .parallel import ParallelRetargeter, StitchReport, chunk_bounds, handoff
from .qp import QPResult, solve_qp
from .retarget import IKConfig, IKProblem, Retargeter
from .sqp import SQPOptions
from .utils import continuous_quaternions, mj_to_pin, pin_to_mj

__version__ = "2.0.0"

__all__ = [
    "IKConfig", "IKProblem", "Retargeter", "ParallelRetargeter", "StitchReport",
    "SQPOptions", "Kinematics", "solve_qp", "QPResult", "chunk_bounds", "handoff",
    "mj_to_pin", "pin_to_mj", "continuous_quaternions",
]
