import os
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = os.path.join(ROOT, "models", "franka_emika_panda", "mjx_panda.xml")

HOME = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
ALT_HOME = np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, -np.pi/4])

HOME_POS = np.array([ 0.307, -0., 0.49])
HOME_ROT = np.diag(np.array([1., -1., -1.]))

from .reach_pose import ReachPose
from .reach_joints import ReachJoints

from .jacobian import \
    expand_qpos, \
    fk_point, \
    fk_rotmat, \
    jac_point, \
    jac_angular

from .asynchronous import \
    run_interactive