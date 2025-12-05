import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

from fr3_mpc.reach_joints import ReachJoints


# Forward Kinematics

def fk_point(mjx_model, mjx_data0, qpos, body_id, point_local):
    data = mjx_data0.replace(qpos=qpos)
    data = mjx.forward(mjx_model, data)
    p_body = data.xpos[body_id]
    R_body = data.xmat[body_id]  # (3, 3)
    return p_body + R_body @ point_local


def fk_rotmat(mjx_model, mjx_data0, qpos, body_id):
    data = mjx_data0.replace(qpos=qpos)
    data = mjx.forward(mjx_model, data)
    return data.xmat[body_id]  # (3, 3)

# Jacobians

@jax.jit
def jac_point(mjx_model, mjx_data0, qpos, body_id, point_local):
    f = lambda q: fk_point(mjx_model, mjx_data0, q, body_id, point_local)
    return jax.jacobian(f)(qpos)  # (3, nq)


def _mat_to_omega(M):
    return jnp.array([M[2, 1], M[0, 2], M[1, 0]])  # vee operator

@jax.jit
def jac_angular(mjx_model, mjx_data0, qpos, body_id):
    f = lambda q: fk_rotmat(mjx_model, mjx_data0, q, body_id)
    dR_dq = jax.jacobian(f)(qpos)  # (3, 3, nq)

    data = mjx_data0.replace(qpos=qpos)
    data = mjx.forward(mjx_model, data)
    R = data.xmat[body_id]  # (3, 3)

    def col_to_omega(dR_col):
        skew = dR_col @ R.T
        return _mat_to_omega(skew)

    return jax.vmap(col_to_omega, in_axes=2, out_axes=1)(dR_dq)  # (3, nq)

@jax.jit
def jac_spatial(mjx_model, mjx_data0, qpos, body_id, point_local):
    Jp = jac_point(mjx_model, mjx_data0, qpos, body_id, point_local)  # (3, nq)
    Jw = jac_angular(mjx_model, mjx_data0, qpos, body_id)             # (3, nq)
    return jnp.vstack([Jp, Jw])  # (6, nq)


# 7DoF -> 9DoF

def expand_qpos(mj_model, arm_qpos7):
    full = jnp.zeros((mj_model.nq,))
    full = full.at[:7].set(arm_qpos7)
    return full


if __name__ == "__main__":
    pi = np.pi
    HOME7 = jnp.array([0, -pi/4, 0, -3*pi/4, 0, pi/2, pi/4])

    task = ReachJoints(target_q=np.asarray(HOME7))
    mj_model = task.mj_model
    mj_data  = mujoco.MjData(mj_model)

    mjx_model = mjx.put_model(mj_model)
    mjx_data0 = mjx.make_data(mjx_model)

    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripper")

    point_local = jnp.array(mj_model.site_pos[site_id])
    HOME = expand_qpos(mj_model, HOME7)

    import time

    print("spatial")
    for _ in range(5):
        t0 = time.time()
        Jp = jac_point(mjx_model, mjx_data0, HOME, body_id, point_local)
        print(time.time() - t0)

    print("rotational")
    for _ in range(5):
        t0 = time.time()
        Jw = jac_angular(mjx_model, mjx_data0, HOME, body_id)
        print(time.time() - t0)

    print("joint")
    for _ in range(5):
        t0 = time.time()
        Js = jac_spatial(mjx_model, mjx_data0, HOME, body_id, point_local)
        print(time.time() - t0)

    assert(Js.shape == (6, 9))

    # Add this to your main block:
    mj_data.qpos[:] = np.asarray(HOME)
    mujoco.mj_forward(mj_model, mj_data)

    # MuJoCo's Jacobians
    jacp = np.zeros((3, mj_model.nv))
    jacr = np.zeros((3, mj_model.nv))
    mujoco.mj_jacSite(mj_model, mj_data, jacp, jacr, site_id)

    assert np.max(np.abs(jacp - np.asarray(Jp))) < 1e-6
    assert np.max(np.abs(jacr - np.asarray(Jw))) < 1e-6
    print("OK")