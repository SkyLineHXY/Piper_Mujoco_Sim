from typing import List, Optional, Tuple, Union
import numpy as np
import spatialmath as sm
import spatialmath.base as smb
import mujoco as mj
from .rtb import make_tf


def set_body_pose(model: mj.MjModel, data: mj.MjData, body_name: Union[int, str], xpos: np.ndarray) -> None:
    body_id = (
        body_name if isinstance(body_name, int) else mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    )
    data.body(body_id).xpos = xpos[:3]

def get_body_pose(model: mj.MjModel, data: mj.MjData, body_name: Union[int, str]) -> sm.SE3:
    body_id = (
        body_name if isinstance(body_name, int) else mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    )
    t = data.body(body_id).xpos
    q = data.body(body_id).xquat
    return make_tf(pos=t, ori=q)


def set_joint_q(model: mj.MjModel, data: mj.MjData, joint_name: Union[int, str], q: Union[np.ndarray, float],
                unit: str = "rad") -> None:
    joint_id = (
        joint_name if isinstance(joint_name, int) else mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
    )

    if unit == 'deg':
        q = np.deg2rad(q)

    q_inds = get_joint_qpos_inds(model, data, joint_id)
    data.qpos[q_inds] = q


def get_joint_q(model: mj.MjModel, data: mj.MjData, joint_name: Union[int, str]) -> np.ndarray:
    joint_id = (
        joint_name if isinstance(joint_name, int) else mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
    )
    q_inds = get_joint_qpos_inds(model, data, joint_id)
    return data.qpos[q_inds]


def get_joint_qpos_inds(model: mj.MjModel, data: mj.MjData, joint_name: Union[int, str]) -> np.ndarray:
    joint_id = (
        joint_name if isinstance(joint_name, int) else mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
    )
    addr = get_joint_qpos_addr(model, joint_id)
    joint_dim = get_joint_dim(model, data, joint_id)
    return np.array(range(addr, addr + joint_dim))


def get_joint_qpos_addr(model: mj.MjModel, joint_name: Union[int, str]) -> int:
    joint_id = (
        joint_name if isinstance(joint_name, int) else mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
    )
    return model.jnt_qposadr[joint_id]


def get_joint_dim(model: mj.MjModel, data: mj.MjData, joint_name: Union[str, int]) -> int:
    joint_id = (
        joint_name if isinstance(joint_name, int) else mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
    )
    return len(data.joint(joint_id).qpos)


def set_free_joint_pose(model: mj.MjModel, data: mj.MjData, joint_name: Union[int, str], T: sm.SE3) -> None:
    t = T.t
    q = sm.base.r2q(T.R).data
    T_new = np.append(t, q)
    set_joint_q(model, data, joint_name, T_new)


def attach(model: mj.MjModel, data: mj.MjData, equality_name: str, free_joint_name: str, T: sm.SE3,
           eq_data=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
           eq_solimp: np.ndarray = np.array([[0.99, 0.99, 0.001, 0.5, 1]]),
           eq_solref: np.ndarray = np.array([0.0001, 1])
           ) -> None:
    eq_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_EQUALITY, equality_name)

    if eq_id is None:
        raise ValueError(
            f"Equality constraint with name '{equality_name}' not found in the model."
        )

    data.eq_active[eq_id] = 0

    set_free_joint_pose(model, data, free_joint_name, T)

    model.equality(equality_name).data = eq_data

    model.equality(equality_name).solimp = eq_solimp
    model.equality(equality_name).solref = eq_solref

    data.eq_active[eq_id] = 1
