import mujoco
import mujoco.viewer

import numpy as np
import heapq
import time


# ============================================================
# MODEL
# ============================================================

MODEL_FILE = "model.xml"

model = mujoco.MjModel.from_xml_path(MODEL_FILE)
data = mujoco.MjData(model)

drone_id = model.body("drone").id

mass = model.body_mass[drone_id]


# ============================================================
# START
# ============================================================

START_POSITION = np.array([
    0.0,
    0.0,
    2.0
], dtype=float)

START_ROLL = 0.0
START_PITCH = 0.0
START_YAW = 0.0


# ============================================================
# GOAL
# ============================================================

GOAL_POSITION = np.array([
    0.0,
    0.0,
    2.0
])

GOAL_ROLL = 0.0
GOAL_PITCH = 180.0
GOAL_YAW = 0.0


# ============================================================
# WORKING AREA
# ============================================================

GRID_SIZE = 0.5

X_MIN = -2.0
X_MAX = 40.0

Y_MIN = -12.0
Y_MAX = 12.0

Z_MIN = 0.5
Z_MAX = 3

# ============================================================
# DRONE SIZE
# ============================================================

# Радиус горизонтальной проекции дрона.
DRONE_RADIUS = 0.45

# Запас сверху/снизу.
DRONE_HEIGHT = 0.20

# Дополнительный запас безопасности.
SAFETY_MARGIN = 0.15


# ============================================================
# CONTROLLER
# ============================================================

Kp_position = 2.0
Kd_position = 2.5


Kp_attitude = 8.0
Kd_attitude = 2.0


# ============================================================
# LIMITS
# ============================================================

MAX_TILT = np.radians(15.0)

MAX_HORIZONTAL_ACCEL = 2.5

MAX_VERTICAL_ACCEL = 3.0

MAX_TOTAL_THRUST = 35.0

MAX_TORQUE = 0.8


# ============================================================
# MOTORS
# ============================================================

ARM = 0.28

MIN_THRUST_PER_MOTOR = 0.0

MAX_THRUST_PER_MOTOR = 10.0


# ============================================================
# QUATERNION
# ============================================================

def quaternion_from_euler(
    roll,
    pitch,
    yaw
):

    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy
    ])


q_start = quaternion_from_euler(
    START_ROLL,
    START_PITCH,
    START_YAW
)

q_goal = quaternion_from_euler(
    GOAL_ROLL,
    GOAL_PITCH,
    GOAL_YAW
)


# ============================================================
# OBSTACLES
# ============================================================

def get_obstacles():

    obstacles = []

    # Обновляем состояние MuJoCo,
    # чтобы geom_xpos и geom_xmat содержали
    # актуальные мировые координаты.
    mujoco.mj_forward(model, data)

    for geom_id in range(model.ngeom):

        # Нас интересуют только BOX.
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
            continue

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_id
        )

        # Пол не является препятствием.
        if name == "ground":
            continue

        body_id = model.geom_bodyid[geom_id]

        # Геометрия самого дрона не является препятствием.
        if body_id == drone_id:
            continue

        # МИРОВАЯ координата центра.
        center = data.geom_xpos[geom_id].copy()

        # Размер BOX в локальной системе координат.
        half_local = model.geom_size[geom_id].copy()

        # Мировая ориентация BOX.
        rotation = data.geom_xmat[geom_id].reshape(3, 3)

        # Получаем консервативный AABB.
        half_world = np.abs(rotation) @ half_local

        obstacles.append(
            (
                center,
                half_world
            )
        )

    return obstacles


OBSTACLES = get_obstacles()


# ============================================================
# COLLISION
# ============================================================

def collision(position):

    x, y, z = position

    # --------------------------------------------
    # WORKING AREA
    # --------------------------------------------

    if x < X_MIN or x > X_MAX:
        return True

    if y < Y_MIN or y > Y_MAX:
        return True

    if z < Z_MIN or z > Z_MAX:
        return True

    # --------------------------------------------
    # OBSTACLES
    # --------------------------------------------

    total_horizontal_margin = (
        DRONE_RADIUS
        +
        SAFETY_MARGIN
    )

    total_vertical_margin = (
        DRONE_HEIGHT
        +
        SAFETY_MARGIN
    )

    for center, half in OBSTACLES:

        min_x = (
            center[0]
            -
            half[0]
            -
            total_horizontal_margin
        )

        max_x = (
            center[0]
            +
            half[0]
            +
            total_horizontal_margin
        )

        min_y = (
            center[1]
            -
            half[1]
            -
            total_horizontal_margin
        )

        max_y = (
            center[1]
            +
            half[1]
            +
            total_horizontal_margin
        )

        min_z = (
            center[2]
            -
            half[2]
            -
            total_vertical_margin
        )

        max_z = (
            center[2]
            +
            half[2]
            +
            total_vertical_margin
        )

        if (
            min_x <= x <= max_x
            and
            min_y <= y <= max_y
            and
            min_z <= z <= max_z
        ):
            return True

    return False


# ============================================================
# SEGMENT COLLISION
# ============================================================

def segment_collision(
    start,
    end
):

    start = np.asarray(
        start,
        dtype=float
    )

    end = np.asarray(
        end,
        dtype=float
    )

    distance = np.linalg.norm(
        end - start
    )

    if distance < 1e-9:

        return collision(start)

    # Проверяем отрезок с достаточно маленьким шагом.
    step = GRID_SIZE * 0.25

    number_of_checks = max(
        2,
        int(
            np.ceil(
                distance / step
            )
        )
    )

    for i in range(
        number_of_checks + 1
    ):

        alpha = (
            i
            /
            number_of_checks
        )

        point = (
            start
            +
            alpha
            *
            (end - start)
        )

        if collision(point):
            return True

    return False


# ============================================================
# WORLD -> GRID
# ============================================================

def world_to_grid(position):

    return (
        int(round(
            (position[0] - X_MIN)
            /
            GRID_SIZE
        )),

        int(round(
            (position[1] - Y_MIN)
            /
            GRID_SIZE
        )),

        int(round(
            (position[2] - Z_MIN)
            /
            GRID_SIZE
        ))
    )


# ============================================================
# GRID -> WORLD
# ============================================================

def grid_to_world(node):

    return np.array([
        X_MIN + node[0] * GRID_SIZE,
        Y_MIN + node[1] * GRID_SIZE,
        Z_MIN + node[2] * GRID_SIZE
    ])


# ============================================================
# HEURISTIC
# ============================================================

def heuristic(a, b):

    return np.linalg.norm(
        np.array(a, dtype=float)
        -
        np.array(b, dtype=float)
    )


# ============================================================
# A*
# ============================================================

def astar(
    start,
    goal
):

    # --------------------------------------------
    # CHECK START / GOAL
    # --------------------------------------------

    if collision(start):

        raise RuntimeError(
            "Старт находится внутри препятствия "
            "или вне рабочей зоны."
        )

    if collision(goal):

        raise RuntimeError(
            "Цель находится внутри препятствия "
            "или вне рабочей зоны."
        )

    start_node = world_to_grid(start)

    goal_node = world_to_grid(goal)

    # --------------------------------------------
    # GRID LIMITS
    # --------------------------------------------

    max_x = int(
        round(
            (X_MAX - X_MIN)
            /
            GRID_SIZE
        )
    )

    max_y = int(
        round(
            (Y_MAX - Y_MIN)
            /
            GRID_SIZE
        )
    )

    max_z = int(
        round(
            (Z_MAX - Z_MIN)
            /
            GRID_SIZE
        )
    )

    # --------------------------------------------
    # 26 NEIGHBOURS
    # --------------------------------------------

    directions = []

    for dx in [-1, 0, 1]:

        for dy in [-1, 0, 1]:

            for dz in [-1, 0, 1]:

                if (
                    dx == 0
                    and
                    dy == 0
                    and
                    dz == 0
                ):
                    continue

                directions.append(
                    (
                        dx,
                        dy,
                        dz
                    )
                )

    # --------------------------------------------
    # OPEN SET
    # --------------------------------------------

    open_set = []

    heapq.heappush(
        open_set,
        (
            heuristic(
                start_node,
                goal_node
            ),
            0.0,
            start_node
        )
    )

    came_from = {}

    g_score = {
        start_node: 0.0
    }

    closed = set()

    # --------------------------------------------
    # SEARCH
    # --------------------------------------------

    while open_set:

        (
            _,
            current_g,
            current
        ) = heapq.heappop(
            open_set
        )

        if current in closed:
            continue

        if current == goal_node:

            path = []

            node = current

            while node in came_from:

                path.append(
                    grid_to_world(node)
                )

                node = came_from[node]

            path.append(
                grid_to_world(
                    start_node
                )
            )

            path.reverse()

            return path

        closed.add(current)

        # ----------------------------------------
        # NEIGHBOURS
        # ----------------------------------------

        for dx, dy, dz in directions:

            neighbor = (
                current[0] + dx,
                current[1] + dy,
                current[2] + dz
            )

            # Grid bounds.
            if (
                neighbor[0] < 0
                or
                neighbor[0] > max_x
            ):
                continue

            if (
                neighbor[1] < 0
                or
                neighbor[1] > max_y
            ):
                continue

            if (
                neighbor[2] < 0
                or
                neighbor[2] > max_z
            ):
                continue

            current_position = grid_to_world(
                current
            )

            neighbor_position = grid_to_world(
                neighbor
            )

            # Проверяем конечную точку.
            if collision(
                neighbor_position
            ):
                continue

            # ВАЖНО:
            # проверяем весь переход,
            # а не только соседний узел.
            if segment_collision(
                current_position,
                neighbor_position
            ):
                continue

            movement_cost = np.sqrt(
                dx * dx
                +
                dy * dy
                +
                dz * dz
            )

            new_g = (
                current_g
                +
                movement_cost
            )

            if new_g < g_score.get(
                neighbor,
                np.inf
            ):

                came_from[neighbor] = current

                g_score[neighbor] = new_g

                f = (
                    new_g
                    +
                    heuristic(
                        neighbor,
                        goal_node
                    )
                )

                heapq.heappush(
                    open_set,
                    (
                        f,
                        new_g,
                        neighbor
                    )
                )

    return None


# ============================================================
# SAFE PATH SMOOTHING
# ============================================================

def simplify_path(path):

    if len(path) <= 2:
        return path

    result = [path[0]]

    current_index = 0

    while current_index < len(path) - 1:

        best_index = current_index + 1

        # Ищем самый дальний узел,
        # до которого можно провести
        # прямой безопасный участок.
        for candidate in range(
            current_index + 1,
            len(path)
        ):

            if not segment_collision(
                path[current_index],
                path[candidate]
            ):

                best_index = candidate

        result.append(
            path[best_index]
        )

        current_index = best_index

    return result


# ============================================================
# QUATERNION ERROR
# ============================================================

def quaternion_error(
    current,
    target
):

    current = (
        current
        /
        np.linalg.norm(current)
    )

    target = (
        target
        /
        np.linalg.norm(target)
    )

    inverse = np.array([
        current[0],
        -current[1],
        -current[2],
        -current[3]
    ])

    error = np.zeros(4)

    mujoco.mju_mulQuat(
        error,
        target,
        inverse
    )

    if error[0] < 0:

        error = -error

    w = np.clip(
        error[0],
        -1.0,
        1.0
    )

    vector = error[1:]

    vector_norm = np.linalg.norm(
        vector
    )

    if vector_norm < 1e-8:

        return np.zeros(3)

    angle = (
        2.0
        *
        np.arctan2(
            vector_norm,
            w
        )
    )

    axis = (
        vector
        /
        vector_norm
    )

    return axis * angle


# ============================================================
# POSITION CONTROLLER
# ============================================================

def position_controller(
    target
):

    position = data.qpos[0:3]

    velocity = data.qvel[0:3]

    error = (
        target
        -
        position
    )

    acceleration = (
        Kp_position * error
        -
        Kd_position * velocity
    )

    # --------------------------------------------
    # HORIZONTAL LIMIT
    # --------------------------------------------

    horizontal = acceleration[:2]

    horizontal_norm = np.linalg.norm(
        horizontal
    )

    if (
        horizontal_norm
        >
        MAX_HORIZONTAL_ACCEL
    ):

        horizontal *= (
            MAX_HORIZONTAL_ACCEL
            /
            horizontal_norm
        )

    acceleration[:2] = horizontal

    # --------------------------------------------
    # VERTICAL LIMIT
    # --------------------------------------------

    acceleration[2] = np.clip(
        acceleration[2],
        -MAX_VERTICAL_ACCEL,
        MAX_VERTICAL_ACCEL
    )

    # --------------------------------------------
    # GRAVITY COMPENSATION
    # --------------------------------------------

    force = mass * (
        acceleration
        +
        np.array([
            0.0,
            0.0,
            9.81
        ])
    )

    return force


# ============================================================
# FORCE LIMIT
# ============================================================

def limit_force(force):

    force = np.asarray(
        force,
        dtype=float
    )

    length = np.linalg.norm(
        force
    )

    if length > MAX_TOTAL_THRUST:

        force *= (
            MAX_TOTAL_THRUST
            /
            length
        )

    return force


# ============================================================
# DESIRED ORIENTATION
# ============================================================

def desired_orientation(
    force
):

    force = np.asarray(
        force,
        dtype=float
    )

    force_norm = np.linalg.norm(
        force
    )

    if force_norm < 1e-8:

        return q_start.copy()

    # Не допускаем отрицательной вертикальной тяги.
    fz = max(
        force[2],
        0.1
    )

    horizontal = np.linalg.norm(
        force[:2]
    )

    maximum_horizontal = (
        fz
        *
        np.tan(
            MAX_TILT
        )
    )

    if horizontal > maximum_horizontal:

        force = force.copy()

        force[:2] *= (
            maximum_horizontal
            /
            horizontal
        )

    z_axis = (
        force
        /
        np.linalg.norm(force)
    )

    yaw = GOAL_YAW

    x_reference = np.array([
        np.cos(yaw),
        np.sin(yaw),
        0.0
    ])

    y_axis = np.cross(
        z_axis,
        x_reference
    )

    y_norm = np.linalg.norm(
        y_axis
    )

    if y_norm < 1e-8:

        return q_start.copy()

    y_axis /= y_norm

    x_axis = np.cross(
        y_axis,
        z_axis
    )

    x_axis /= np.linalg.norm(
        x_axis
    )

    rotation = np.column_stack([
        x_axis,
        y_axis,
        z_axis
    ])

    return rotation_matrix_to_quaternion(
        rotation
    )


# ============================================================
# ROTATION MATRIX -> QUATERNION
# ============================================================

def rotation_matrix_to_quaternion(R):

    trace = np.trace(R)

    if trace > 0:

        s = 2.0 * np.sqrt(
            trace + 1.0
        )

        qw = 0.25 * s

        qx = (
            R[2, 1]
            -
            R[1, 2]
        ) / s

        qy = (
            R[0, 2]
            -
            R[2, 0]
        ) / s

        qz = (
            R[1, 0]
            -
            R[0, 1]
        ) / s

    else:

        if (
            R[0, 0] > R[1, 1]
            and
            R[0, 0] > R[2, 2]
        ):

            s = 2.0 * np.sqrt(
                1.0
                +
                R[0, 0]
                -
                R[1, 1]
                -
                R[2, 2]
            )

            qw = (
                R[2, 1]
                -
                R[1, 2]
            ) / s

            qx = 0.25 * s

            qy = (
                R[0, 1]
                +
                R[1, 0]
            ) / s

            qz = (
                R[0, 2]
                +
                R[2, 0]
            ) / s

        elif R[1, 1] > R[2, 2]:

            s = 2.0 * np.sqrt(
                1.0
                +
                R[1, 1]
                -
                R[0, 0]
                -
                R[2, 2]
            )

            qw = (
                R[0, 2]
                -
                R[2, 0]
            ) / s

            qx = (
                R[0, 1]
                +
                R[1, 0]
            ) / s

            qy = 0.25 * s

            qz = (
                R[1, 2]
                +
                R[2, 1]
            ) / s

        else:

            s = 2.0 * np.sqrt(
                1.0
                +
                R[2, 2]
                -
                R[0, 0]
                -
                R[1, 1]
            )

            qw = (
                R[1, 0]
                -
                R[0, 1]
            ) / s

            qx = (
                R[0, 2]
                +
                R[2, 0]
            ) / s

            qy = (
                R[1, 2]
                +
                R[2, 1]
            ) / s

            qz = 0.25 * s

    q = np.array([
        qw,
        qx,
        qy,
        qz
    ])

    return (
        q
        /
        np.linalg.norm(q)
    )


# ============================================================
# ATTITUDE CONTROLLER
# ============================================================

def attitude_controller(
    target
):

    current = data.qpos[3:7]

    error = quaternion_error(
        current,
        target
    )

    angular_velocity = data.qvel[3:6]

    torque = (
        Kp_attitude * error
        -
        Kd_attitude * angular_velocity
    )

    torque_norm = np.linalg.norm(
        torque
    )

    if torque_norm > MAX_TORQUE:

        torque *= (
            MAX_TORQUE
            /
            torque_norm
        )

    return torque


# ============================================================
# MOTOR MIXER
# ============================================================

def motor_mixer(
    thrust,
    torque
):

    roll_torque = torque[0]
    pitch_torque = torque[1]

    base = thrust / 4.0

    # ------------------------------------------------
    # Преобразование требуемого момента
    # в разницу тяги моторов.
    #
    # Моторы:
    #
    # M2 -------- M1
    #  |          |
    #  |  DRONE   |
    #  |          |
    # M3 -------- M4
    #
    # ------------------------------------------------

    roll = (
        roll_torque
        /
        (4.0 * ARM)
    )

    pitch = (
        pitch_torque
        /
        (4.0 * ARM)
    )

    motors = np.array([

        # M1 (+X, +Y)
        base + roll - pitch,

        # M2 (-X, +Y)
        base + roll + pitch,

        # M3 (-X, -Y)
        base - roll + pitch,

        # M4 (+X, -Y)
        base - roll - pitch

    ])

    return np.clip(
        motors,
        MIN_THRUST_PER_MOTOR,
        MAX_THRUST_PER_MOTOR
    )


# ============================================================
# INITIAL STATE
# ============================================================

data.qpos[0:3] = START_POSITION

data.qpos[3:7] = q_start

data.qvel[:] = 0.0

data.ctrl[:] = 0.0

mujoco.mj_forward(
    model,
    data
)


# ============================================================
# GET OBSTACLES AGAIN
# ============================================================

OBSTACLES = get_obstacles()


# ============================================================
# PLAN PATH
# ============================================================

PATH = astar(
    START_POSITION,
    GOAL_POSITION
)

if PATH is None:

    raise RuntimeError(
        "A* не смог найти путь через препятствия."
    )


print()
print("A* первоначальный путь:")
print(
    "Количество точек:",
    len(PATH)
)


# ============================================================
# SAFE PATH SIMPLIFICATION
# ============================================================

PATH = simplify_path(
    PATH
)

print(
    "После безопасного упрощения:",
    len(PATH),
    "точек"
)


# ============================================================
# FINAL PATH VALIDATION
# ============================================================

for i in range(
    len(PATH) - 1
):

    if segment_collision(
        PATH[i],
        PATH[i + 1]
    ):

        raise RuntimeError(
            "Ошибка: итоговая траектория "
            "пересекает препятствие."
        )


# ============================================================
# PATH SPEED
# ============================================================

PATH_SPEED = 0.5

path_length = 0.0

for i in range(
    len(PATH) - 1
):

    path_length += np.linalg.norm(
        PATH[i + 1]
        -
        PATH[i]
    )


FLIGHT_TIME = (
    path_length
    /
    PATH_SPEED
)


# ============================================================
# POSITION AT TIME
# ============================================================

def position_at_time(t):

    if t >= FLIGHT_TIME:

        return GOAL_POSITION.copy()

    distance = (
        PATH_SPEED
        *
        t
    )

    accumulated = 0.0

    for i in range(
        len(PATH) - 1
    ):

        segment = (
            PATH[i + 1]
            -
            PATH[i]
        )

        length = np.linalg.norm(
            segment
        )

        if (
            accumulated
            +
            length
            >=
            distance
        ):

            alpha = (
                distance
                -
                accumulated
            ) / length

            return (
                PATH[i]
                +
                alpha
                *
                segment
            )

        accumulated += length

    return GOAL_POSITION.copy()


# ============================================================
# INFORMATION
# ============================================================

print()
print("==============================")
print("3D QUADROTOR TRAJECTORY PLANNER")
print("==============================")

print(
    "Mass:",
    mass
)

print(
    "Actuators:",
    model.nu
)

print(
    "Start:",
    START_POSITION
)

print(
    "Goal:",
    GOAL_POSITION
)

print(
    "Obstacles:",
    len(OBSTACLES)
)

print(
    "Path length:",
    round(
        path_length,
        2
    )
)

print(
    "Flight time:",
    round(
        FLIGHT_TIME,
        2
    )
)

print()

if model.nu != 4:

    raise RuntimeError(
        f"Ожидалось 4 actuator, "
        f"но MuJoCo обнаружил {model.nu}."
    )


print("FINAL A* PATH:")

for i, point in enumerate(PATH):

    print(
        i,
        np.round(
            point,
            2
        )
    )

print()


# ============================================================
# SIMULATION
# ============================================================

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    last_print = 0.0

    while viewer.is_running():

        t = data.time

        # --------------------------------------------
        # TARGET POSITION
        # --------------------------------------------

        target_position = position_at_time(
            t
        )

        # --------------------------------------------
        # POSITION CONTROL
        # --------------------------------------------

        desired_force = position_controller(
            target_position
        )

        desired_force = limit_force(
            desired_force
        )

        # --------------------------------------------
        # ATTITUDE TARGET
        # --------------------------------------------

        target_orientation = desired_orientation(
            desired_force
        )

        # --------------------------------------------
        # ATTITUDE CONTROL
        # --------------------------------------------

        desired_torque = attitude_controller(
            target_orientation
        )

        # --------------------------------------------
        # TOTAL THRUST
        # --------------------------------------------

        thrust = np.linalg.norm(
            desired_force
        )

        thrust = np.clip(
            thrust,
            0.0,
            MAX_TOTAL_THRUST
        )

        # --------------------------------------------
        # MOTOR MIXING
        # --------------------------------------------

        motors = motor_mixer(
            thrust,
            desired_torque
        )

        # --------------------------------------------
        # APPLY
        # --------------------------------------------

        data.ctrl[:] = motors

        mujoco.mj_step(
            model,
            data
        )

        viewer.sync()

        # --------------------------------------------
        # DEBUG
        # --------------------------------------------

        if (
            t - last_print
            >
            0.25
        ):

            position = data.qpos[0:3]

            velocity = data.qvel[0:3]

            print(
                f"t={t:6.2f} "
                f"| target={np.round(target_position, 2)} "
                f"| actual={np.round(position, 2)} "
                f"| vel={np.round(velocity, 2)} "
                f"| motors={np.round(motors, 2)}"
            )

            last_print = t

        time.sleep(0.001)
