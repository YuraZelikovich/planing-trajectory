import mujoco
import mujoco.viewer

import numpy as np
import time


# ============================================================
# НАСТРОЙКИ
# ============================================================

MODEL_FILE = "model.xml"

# ------------------------------------------------------------
# Конечное положение
# ------------------------------------------------------------

GOAL_POSITION = np.array([
    5.0,
    0.0,
    5.0
])


# ------------------------------------------------------------
# Конечная ориентация
#
# roll  = вращение вокруг X
# pitch = вращение вокруг Y
# yaw   = вращение вокруг Z
#
# Например:
#
# [0, 0, 0]     - обычное положение
# [180, 0, 0]   - вверх ногами
# [0, 0, 90]    - поворот на 90 градусов
# ------------------------------------------------------------

GOAL_ROLL = 0.0
GOAL_PITCH = 0.0
GOAL_YAW = 0.0


# ------------------------------------------------------------
# Продолжительность траектории
# ------------------------------------------------------------

TRAJECTORY_TIME = 5.0


# ============================================================
# ПАРАМЕТРЫ КОНТРОЛЛЕРА
# ============================================================

Kp_position = 5.0
Kv_position = 5.0

Kp_rotation = 10.0
Kv_rotation = 2.0


# ============================================================
# MUJOCO
# ============================================================

model = mujoco.MjModel.from_xml_path(
    MODEL_FILE
)

data = mujoco.MjData(model)

drone_id = model.body(
    "drone"
).id

mass = model.body_mass[
    drone_id
]


# ============================================================
# ГРАВИТАЦИЯ
# ============================================================

gravity = np.array([
    0.0,
    0.0,
    -9.81
])


# ============================================================
# QUATERNION FUNCTIONS
# ============================================================

def quaternion_from_euler(
        roll,
        pitch,
        yaw
):

    """
    Euler angles -> quaternion.

    Углы задаются в радианах.
    """

    cr = np.cos(
        roll / 2
    )

    sr = np.sin(
        roll / 2
    )

    cp = np.cos(
        pitch / 2
    )

    sp = np.sin(
        pitch / 2
    )

    cy = np.cos(
        yaw / 2
    )

    sy = np.sin(
        yaw / 2
    )

    w = (
        cr * cp * cy
        +
        sr * sp * sy
    )

    x = (
        sr * cp * cy
        -
        cr * sp * sy
    )

    y = (
        cr * sp * cy
        +
        sr * cp * sy
    )

    z = (
        cr * cp * sy
        -
        sr * sp * cy
    )

    return np.array([
        w,
        x,
        y,
        z
    ])


def quaternion_normalize(q):

    return q / np.linalg.norm(q)


def quaternion_slerp(q0, q1, t):

    """
    Spherical Linear Interpolation.

    Позволяет плавно интерполировать
    ориентацию между двумя quaternion.
    """

    q0 = quaternion_normalize(q0)
    q1 = quaternion_normalize(q1)

    dot = np.dot(
        q0,
        q1
    )

    # Выбираем кратчайший путь вращения
    if dot < 0.0:

        q1 = -q1
        dot = -dot

    dot = np.clip(
        dot,
        -1.0,
        1.0
    )

    # Если quaternion почти одинаковые,
    # используем обычную линейную интерполяцию
    if dot > 0.9995:

        result = (
            q0
            +
            t * (q1 - q0)
        )

        return quaternion_normalize(
            result
        )

    theta_0 = np.arccos(
        dot
    )

    sin_theta_0 = np.sin(
        theta_0
    )

    theta = (
        theta_0 * t
    )

    sin_theta = np.sin(
        theta
    )

    coefficient_0 = (
        np.cos(theta)
        -
        dot
        *
        sin_theta
        /
        sin_theta_0
    )

    coefficient_1 = (
        sin_theta
        /
        sin_theta_0
    )

    return (
        coefficient_0 * q0
        +
        coefficient_1 * q1
    )


# ============================================================
# QUINTIC TRAJECTORY
# ============================================================

def quintic_position(
        start,
        goal,
        t,
        T
):

    """
    Пятая степень.

    Начальное состояние:

        position = start
        velocity = 0
        acceleration = 0

    Конечное состояние:

        position = goal
        velocity = 0
        acceleration = 0
    """

    if t <= 0.0:

        return (
            start,
            np.zeros(3),
            np.zeros(3)
        )

    if t >= T:

        return (
            goal,
            np.zeros(3),
            np.zeros(3)
        )

    tau = t / T

    # Позиционная функция
    s = (
        10 * tau**3
        -
        15 * tau**4
        +
        6 * tau**5
    )

    # Первая производная
    ds = (
        30 * tau**2
        -
        60 * tau**3
        +
        30 * tau**4
    )

    # Вторая производная
    dds = (
        60 * tau
        -
        180 * tau**2
        +
        120 * tau**3
    )

    delta = (
        goal
        -
        start
    )

    position = (
        start
        +
        delta * s
    )

    velocity = (
        delta
        *
        ds
        /
        T
    )

    acceleration = (
        delta
        *
        dds
        /
        T**2
    )

    return (
        position,
        velocity,
        acceleration
    )


# ============================================================
# TRAJECTORY PLANNER
# ============================================================

class TrajectoryPlanner:

    def __init__(
            self,
            start_position,
            goal_position,
            start_orientation,
            goal_orientation,
            duration
    ):

        self.start_position = (
            start_position.copy()
        )

        self.goal_position = (
            goal_position.copy()
        )

        self.start_orientation = (
            start_orientation.copy()
        )

        self.goal_orientation = (
            goal_orientation.copy()
        )

        self.duration = duration


    def evaluate(self, t):

        """
        Получить полное желаемое состояние
        в момент времени t.
        """

        # ----------------------------------------------------
        # POSITION
        # ----------------------------------------------------

        position, velocity, acceleration = (
            quintic_position(
                self.start_position,
                self.goal_position,
                t,
                self.duration
            )
        )


        # ----------------------------------------------------
        # ORIENTATION
        # ----------------------------------------------------

        if t <= 0.0:

            tau = 0.0

        elif t >= self.duration:

            tau = 1.0

        else:

            tau = (
                t
                /
                self.duration
            )


        # Используем ту же quintic-функцию
        # для ориентации

        s = (
            10 * tau**3
            -
            15 * tau**4
            +
            6 * tau**5
        )

        ds = (
            30 * tau**2
            -
            60 * tau**3
            +
            30 * tau**4
        )

        dds = (
            60 * tau
            -
            180 * tau**2
            +
            120 * tau**3
        )


        # ----------------------------------------------------
        # QUATERNION
        # ----------------------------------------------------

        orientation = quaternion_slerp(
            self.start_orientation,
            self.goal_orientation,
            s
        )


        # ----------------------------------------------------
        # ANGULAR MOTION
        #
        # Для учебной версии считаем,
        # что изменение ориентации происходит
        # вокруг фиксированной оси.
        # ----------------------------------------------------

        q0 = self.start_orientation
        q1 = self.goal_orientation

        if np.dot(q0, q1) < 0:

            q1 = -q1

        # Относительное вращение
        relative = np.zeros(4)

        mujoco.mju_negQuat(
            relative,
            q0
        )

        relative = np.zeros(4)

        mujoco.mju_mulQuat(
            relative,
            q1,
            q0
        )

        # Извлекаем axis-angle
        angle = 2.0 * np.arctan2(
            np.linalg.norm(
                relative[1:]
            ),
            abs(relative[0])
        )

        axis_norm = np.linalg.norm(
            relative[1:]
        )

        if axis_norm > 1e-8:

            axis = (
                relative[1:]
                /
                axis_norm
            )

        else:

            axis = np.array([
                0.0,
                0.0,
                0.0
            ])


        # Угловая скорость
        angular_velocity = (
            axis
            *
            angle
            *
            ds
            /
            self.duration
        )


        # Угловое ускорение
        angular_acceleration = (
            axis
            *
            angle
            *
            dds
            /
            self.duration**2
        )


        return {
            "position": position,
            "velocity": velocity,
            "acceleration": acceleration,

            "orientation": orientation,

            "angular_velocity":
                angular_velocity,

            "angular_acceleration":
                angular_acceleration
        }


# ============================================================
# ПОЛУЧАЕМ НАЧАЛЬНОЕ СОСТОЯНИЕ
# ============================================================

start_position = (
    data.body("drone")
    .xpos.copy()
)


# MuJoCo freejoint:
#
# qpos[0:3] = position
# qpos[3:7] = quaternion
#
# quaternion:
#
# [w, x, y, z]

start_orientation = (
    data.qpos[3:7].copy()
)


# ============================================================
# КОНЕЧНАЯ ОРИЕНТАЦИЯ
# ============================================================

goal_orientation = quaternion_from_euler(

    np.deg2rad(
        GOAL_ROLL
    ),

    np.deg2rad(
        GOAL_PITCH
    ),

    np.deg2rad(
        GOAL_YAW
    )
)


# ============================================================
# СОЗДАЕМ ПЛАНИРОВЩИК
# ============================================================

planner = TrajectoryPlanner(

    start_position=start_position,

    goal_position=GOAL_POSITION,

    start_orientation=start_orientation,

    goal_orientation=goal_orientation,

    duration=TRAJECTORY_TIME
)


# ============================================================
# ИНФОРМАЦИЯ
# ============================================================

print()
print("========================================")
print("       TRAJECTORY PLANNER")
print("========================================")

print(
    "Start position:",
    np.round(
        start_position,
        3
    )
)

print(
    "Goal position:",
    np.round(
        GOAL_POSITION,
        3
    )
)

print(
    "Trajectory time:",
    TRAJECTORY_TIME,
    "seconds"
)

print(
    "Goal roll:",
    GOAL_ROLL,
    "degrees"
)

print(
    "Goal pitch:",
    GOAL_PITCH,
    "degrees"
)

print(
    "Goal yaw:",
    GOAL_YAW,
    "degrees"
)


# ============================================================
# ЗАПУСК MUJOCO
# ============================================================

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    simulation_start = time.time()

    last_print = -1.0

    while viewer.is_running():

        # ====================================================
        # ВРЕМЯ
        # ====================================================

        t = (
            time.time()
            -
            simulation_start
        )


        # ====================================================
        # ПОЛУЧАЕМ ЖЕЛАЕМОЕ СОСТОЯНИЕ
        # ====================================================

        desired = planner.evaluate(
            t
        )


        desired_position = (
            desired["position"]
        )

        desired_velocity = (
            desired["velocity"]
        )

        desired_acceleration = (
            desired["acceleration"]
        )

        desired_orientation = (
            desired["orientation"]
        )

        desired_angular_velocity = (
            desired["angular_velocity"]
        )


        # ====================================================
        # ТЕКУЩЕЕ СОСТОЯНИЕ
        # ====================================================

        actual_position = (
            data.body("drone")
            .xpos.copy()
        )

        actual_velocity = (
            data.qvel[0:3].copy()
        )

        actual_orientation = (
            data.qpos[3:7].copy()
        )

        actual_angular_velocity = (
            data.qvel[3:6].copy()
        )


        # ====================================================
        # POSITION CONTROLLER
        # ====================================================

        position_error = (
            desired_position
            -
            actual_position
        )

        velocity_error = (
            desired_velocity
            -
            actual_velocity
        )


        acceleration_command = (

            desired_acceleration

            +

            Kp_position
            *
            position_error

            +

            Kv_position
            *
            velocity_error
        )


        # ====================================================
        # FORCE
        # ====================================================

        force = mass * (
            acceleration_command
            -
            gravity
        )


        # ====================================================
        # ROTATION ERROR
        # ====================================================

        q_error = np.zeros(4)

        q_actual_inverse = np.zeros(4)

        mujoco.mju_negQuat(
            q_actual_inverse,
            actual_orientation
        )

        mujoco.mju_mulQuat(
            q_error,
            desired_orientation,
            q_actual_inverse
        )


        # ----------------------------------------------------
        # Quaternion error:
        #
        # q = [w, x, y, z]
        #
        # Векторная часть показывает направление
        # необходимого вращения.
        # ----------------------------------------------------

        rotation_error = (
            q_error[1:]
        )


        # ====================================================
        # ANGULAR VELOCITY ERROR
        # ====================================================

        angular_velocity_error = (
            desired_angular_velocity
            -
            actual_angular_velocity
        )


        # ====================================================
        # TORQUE CONTROLLER
        # ====================================================

        torque = (

            Kp_rotation
            *
            rotation_error

            +

            Kv_rotation
            *
            angular_velocity_error
        )


        # ====================================================
        # FORCE LIMIT
        # ====================================================

        MAX_FORCE = 40.0

        force_norm = np.linalg.norm(
            force
        )

        if force_norm > MAX_FORCE:

            force = (
                force
                /
                force_norm
                *
                MAX_FORCE
            )


        # ====================================================
        # TORQUE LIMIT
        # ====================================================

        MAX_TORQUE = 10.0

        torque_norm = np.linalg.norm(
            torque
        )

        if torque_norm > MAX_TORQUE:

            torque = (
                torque
                /
                torque_norm
                *
                MAX_TORQUE
            )


        # ====================================================
        # ПРИКЛАДЫВАЕМ FORCE + TORQUE
        # ====================================================

        data.xfrc_applied[
            drone_id
        ] = np.concatenate([
            force,
            torque
        ])


        # ====================================================
        # ВЫВОД СОСТОЯНИЯ
        # ====================================================

        if (
            t - last_print
            >= 0.5
        ):

            print()

            print(
                "t =",
                round(t, 2)
            )

            print(
                "desired position =",
                np.round(
                    desired_position,
                    3
                )
            )

            print(
                "actual position  =",
                np.round(
                    actual_position,
                    3
                )
            )

            print(
                "desired velocity =",
                np.round(
                    desired_velocity,
                    3
                )
            )

            print(
                "actual velocity  =",
                np.round(
                    actual_velocity,
                    3
                )
            )

            print(
                "force =",
                np.round(
                    force,
                    3
                )
            )

            print(
                "torque =",
                np.round(
                    torque,
                    3
                )
            )

            last_print = t


        # ====================================================
        # MUJOCO
        # ====================================================

        mujoco.mj_step(
            model,
            data
        )

        viewer.sync()

        time.sleep(0.002)