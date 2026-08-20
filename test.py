import mujoco
import mujoco.viewer
import numpy as np
import time



def quintic_s(tau):
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def quintic_ds(tau):
    return 30 * tau**2 - 60 * tau**3 + 30 * tau**4


def quintic_dds(tau):
    return 60 * tau - 180 * tau**2 + 120 * tau**3


class TrajectoryPlanner:
    def __init__(self, start_pos, end_pos, duration, mass, gravity, target_roll):
        self.p0 = np.array(start_pos, dtype=float)
        self.p1 = np.array(end_pos, dtype=float)
        self.T = duration
        self.m = mass
        self.g = gravity
        self.target_roll = np.radians(target_roll)
        self.t = 0.0
        self.forward = self.p1 - self.p0
        self.forward[2] = 0.0
        n = np.linalg.norm(self.forward)
        self.forward = self.forward / n if n > 1e-8 else np.array([1.0, 0.0, 0.0])

    def update(self, dt):
        self.t = min(self.t + dt, self.T)
        tau = self.t / self.T
        s = quintic_s(tau)
        ds = quintic_ds(tau)
        dds = quintic_dds(tau)
        delta = self.p1 - self.p0
        pos_des = self.p0 + delta * s
        vel_des = delta * ds / self.T
        acc_des = delta * dds / self.T**2
        roll_des = self.target_roll * s
        g = np.array([0.0, 0.0, -self.g])
        force_world = self.m * (acc_des - g)
        thrust = np.linalg.norm(force_world)
        normal_world = force_world / thrust if thrust > 1e-8 else np.array([0.0, 0.0, 1.0])
        return pos_des, vel_des, acc_des, thrust, normal_world, roll_des


start_pos = [0.0, 0.0, 2.0]
end_pos = [0.0, 0.0, 2.0]
duration = 5.0
target_roll = 180.0

mass = 1.0
gravity = 9.81

planner = TrajectoryPlanner(start_pos, end_pos, duration, mass, gravity, target_roll)

model = mujoco.MjModel.from_xml_path("model.xml")
data = mujoco.MjData(model)

body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone")

data.qpos[:3] = start_pos
mujoco.mj_forward(model, data)

dt = model.opt.timestep

Kp_pos = 2.0
Kd_pos = 1.0

Kp_att = 1.0
Kd_att = 0.5
max_torque = 0.3

next_print = 0.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        pos_des, vel_des, acc_des, thrust, normal_world, roll_des = planner.update(dt)

        pos_error = pos_des - data.qpos[:3]
        vel_error = vel_des - data.qvel[:3]
        acc_cmd = acc_des + Kp_pos * pos_error + Kd_pos * vel_error

        force_world = mass * (acc_cmd + np.array([0.0, 0.0, gravity]))
        thrust = np.linalg.norm(force_world)
        normal_world = force_world / thrust if thrust > 1e-8 else np.array([0.0, 0.0, 1.0])

        z = normal_world
        x = planner.forward - z * np.dot(planner.forward, z)
        x /= max(np.linalg.norm(x), 1e-8)
        y = np.cross(z, x)

        R_base = np.column_stack((x, y, z))

        c = np.cos(roll_des)
        s = np.sin(roll_des)
        Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

        R_des = R_base @ Rx

        q_des = np.zeros(4)
        mujoco.mju_mat2Quat(q_des, R_des.flatten())

        q = data.qpos[3:7]
        q_inv = np.array([q[0], -q[1], -q[2], -q[3]])

        q_error = np.zeros(4)
        mujoco.mju_mulQuat(q_error, q_des, q_inv)

        if q_error[0] < 0:
            q_error *= -1

        omega = data.cvel[body_id][0:3]
        torque = Kp_att * q_error[1:4] - Kd_att * omega
        torque = np.clip(torque, -max_torque, max_torque)

        data.xfrc_applied[body_id, :3] = normal_world * thrust
        data.xfrc_applied[body_id, 3:6] = torque

        mujoco.mj_step(model, data)
        

        viewer.sync()


        time.sleep(dt)
