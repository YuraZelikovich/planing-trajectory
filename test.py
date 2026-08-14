import mujoco
import mujoco.viewer
import numpy as np
import heapq
import time

model = mujoco.MjModel.from_xml_path("model.xml")
data = mujoco.MjData(model)

drone_id = model.body("drone").id
mass = model.body_mass[drone_id]

START_POSITION = np.array([0.0, 0.0, 2.0])

START_ROLL = 0.0
START_PITCH = np.deg2rad(0.0)
START_YAW = 0.0

GOAL_POSITION = np.array([18.0, 6.0, 2.0])

GOAL_ROLL = 0.0
GOAL_PITCH = np.deg2rad(0.0)
GOAL_YAW = 0.0

GRID_SIZE = 0.5

X_MIN = -1.0
X_MAX = 20.0

Y_MIN = -10.0
Y_MAX = 10.0

DRONE_RADIUS = 0.5

Kp_position = 8.0
Kd_position = 5.0

MAX_FORCE = 100.0

Kp_rotation = 8.0
Kd_rotation = 4.0

MAX_TORQUE = 3.0

PATH_SPEED = 1.0
ROTATION_TIME = 5.0

POSITION_TOLERANCE = 0.05
ORIENTATION_TOLERANCE = np.deg2rad(1.0)


def get_obstacles():
    obstacles = []

    for geom_id in range(model.ngeom):
        geom_type = model.geom_type[geom_id]

        if geom_type != mujoco.mjtGeom.mjGEOM_BOX:
            continue

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_id
        )

        if name == "ground":
            continue

        body_id = model.geom_bodyid[geom_id]

        if body_id == drone_id:
            continue

        pos = model.geom_pos[geom_id].copy()
        size = model.geom_size[geom_id].copy()

        obstacles.append({
            "name": name,
            "center": pos,
            "half_size": size
        })

    return obstacles


OBSTACLES = get_obstacles()


print()
print("OBSTACLES FROM MUJOCO")
print("Количество препятствий:", len(OBSTACLES))

for obstacle in OBSTACLES:
    print(
        obstacle["name"],
        "center =",
        np.round(obstacle["center"], 2),
        "half_size =",
        np.round(obstacle["half_size"], 2)
    )

print()


def collision_xy(x, y):
    for obstacle in OBSTACLES:
        center = obstacle["center"]
        half_size = obstacle["half_size"]

        min_x = center[0] - half_size[0] - DRONE_RADIUS
        max_x = center[0] + half_size[0] + DRONE_RADIUS

        min_y = center[1] - half_size[1] - DRONE_RADIUS
        max_y = center[1] + half_size[1] + DRONE_RADIUS

        if (
            min_x <= x <= max_x
            and
            min_y <= y <= max_y
        ):
            return True

    return False


def world_to_grid(position):
    gx = int(round((position[0] - X_MIN) / GRID_SIZE))
    gy = int(round((position[1] - Y_MIN) / GRID_SIZE))

    return gx, gy


def grid_to_world(node):
    x = X_MIN + node[0] * GRID_SIZE
    y = Y_MIN + node[1] * GRID_SIZE

    return np.array([x, y])


def heuristic(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]

    return np.sqrt(dx * dx + dy * dy)



def astar(start, goal):
    start_node = world_to_grid(start)
    goal_node = world_to_grid(goal)

    print("START GRID:", start_node)
    print("GOAL GRID:", goal_node)

    if collision_xy(start[0], start[1]):
        raise RuntimeError(
            "Старт находится внутри препятствия."
        )

    if collision_xy(goal[0], goal[1]):
        raise RuntimeError(
            "Цель находится внутри препятствия."
        )

    open_set = []

    heapq.heappush(
        open_set,
        (0.0, start_node)
    )

    came_from = {}

    g_score = {
        start_node: 0.0
    }

    closed = set()

    directions = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1)
    ]

    max_x = int(
        (X_MAX - X_MIN) / GRID_SIZE
    )

    max_y = int(
        (Y_MAX - Y_MIN) / GRID_SIZE
    )

    while open_set:
        _, current = heapq.heappop(open_set)

        if current in closed:
            continue

        closed.add(current)

        if current == goal_node:
            path = []
            node = current

            while node in came_from:
                path.append(
                    grid_to_world(node)
                )

                node = came_from[node]

            path.append(
                grid_to_world(start_node)
            )

            path.reverse()

            return path

        for dx, dy in directions:
            neighbor = (
                current[0] + dx,
                current[1] + dy
            )

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

            position = grid_to_world(neighbor)

            if collision_xy(
                position[0],
                position[1]
            ):
                continue

            if dx != 0 and dy != 0:
                side_1 = grid_to_world(
                    (
                        current[0] + dx,
                        current[1]
                    )
                )

                side_2 = grid_to_world(
                    (
                        current[0],
                        current[1] + dy
                    )
                )

                if collision_xy(
                    side_1[0],
                    side_1[1]
                ):
                    continue

                if collision_xy(
                    side_2[0],
                    side_2[1]
                ):
                    continue

            if dx != 0 and dy != 0:
                movement_cost = np.sqrt(2.0)
            else:
                movement_cost = 1.0

            tentative_g = (
                g_score[current]
                +
                movement_cost
            )

            if (
                neighbor not in g_score
                or
                tentative_g < g_score[neighbor]
            ):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                f_score = (
                    tentative_g
                    +
                    heuristic(
                        neighbor,
                        goal_node
                    )
                )

                heapq.heappush(
                    open_set,
                    (f_score, neighbor)
                )

    return None


# ==================== SIMPLIFY PATH ====================

def simplify_path(path):
    if len(path) <= 2:
        return path

    result = [path[0]]
    previous_direction = None

    for i in range(1, len(path) - 1):
        direction = (
            path[i + 1]
            -
            path[i]
        )

        direction_norm = np.linalg.norm(direction)

        if direction_norm < 1e-9:
            continue

        direction /= direction_norm

        if previous_direction is None:
            previous_direction = direction
            continue

        change = np.linalg.norm(
            direction - previous_direction
        )

        if change > 0.01:
            result.append(path[i])

        previous_direction = direction

    result.append(path[-1])

    return result


# ==================== QUATERNION ====================

def quaternion_from_euler(
    roll,
    pitch,
    yaw
):
    cr = np.cos(roll / 2.0)
    sr = np.sin(roll / 2.0)

    cp = np.cos(pitch / 2.0)
    sp = np.sin(pitch / 2.0)

    cy = np.cos(yaw / 2.0)
    sy = np.sin(yaw / 2.0)

    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy
    ])


# ==================== SLERP ====================

def slerp(q1, q2, t):
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    dot = np.dot(q1, q2)

    if dot < 0.0:
        q2 = -q2
        dot = -dot

    dot = np.clip(
        dot,
        -1.0,
        1.0
    )

    if dot > 0.9995:
        result = (
            q1
            +
            t * (q2 - q1)
        )

        return result / np.linalg.norm(result)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    a = (
        np.sin((1.0 - t) * theta)
        /
        sin_theta
    )

    b = (
        np.sin(t * theta)
        /
        sin_theta
    )

    return a * q1 + b * q2


# ==================== QUATERNION ERROR ====================

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

    current_inverse = np.array([
        current[0],
        -current[1],
        -current[2],
        -current[3]
    ])

    error = np.zeros(4)

    mujoco.mju_mulQuat(
        error,
        target,
        current_inverse
    )

    error /= np.linalg.norm(error)

    if error[0] < 0.0:
        error = -error

    w = np.clip(
        error[0],
        -1.0,
        1.0
    )

    vector = error[1:]
    vector_norm = np.linalg.norm(vector)

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

    axis = vector / vector_norm

    return axis * angle


# ==================== POSITION TRAJECTORY ====================

def make_position_trajectory():
    position_difference = (
        GOAL_POSITION
        -
        START_POSITION
    )

    distance = np.linalg.norm(
        position_difference
    )

    if distance < POSITION_TOLERANCE:
        return [START_POSITION.copy()], 0.0

    path = astar(
        START_POSITION,
        GOAL_POSITION
    )

    if path is None:
        raise RuntimeError(
            "A* не смог найти путь."
        )

    path = simplify_path(path)

    length = 0.0

    for i in range(len(path) - 1):
        length += np.linalg.norm(
            path[i + 1]
            -
            path[i]
        )

    flight_time = length / PATH_SPEED

    return path, flight_time


PATH, FLIGHT_TIME = make_position_trajectory()


# ==================== POSITION AT TIME ====================

def position_at_time(t):
    if FLIGHT_TIME <= 0.0:
        return GOAL_POSITION.copy()

    if t >= FLIGHT_TIME:
        return GOAL_POSITION.copy()

    distance = PATH_SPEED * t
    accumulated = 0.0

    for i in range(len(PATH) - 1):
        segment_vector = (
            PATH[i + 1]
            -
            PATH[i]
        )

        segment = np.linalg.norm(
            segment_vector
        )

        if accumulated + segment >= distance:
            local = (
                distance - accumulated
            ) / segment

            xy = (
                PATH[i]
                +
                local * segment_vector
            )

            return np.array([
                xy[0],
                xy[1],
                START_POSITION[2]
            ])

        accumulated += segment

    return GOAL_POSITION.copy()


# ==================== ORIENTATION ====================

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


def orientation_at_time(t):
    if t <= FLIGHT_TIME:
        return q_start.copy()

    rotation_t = t - FLIGHT_TIME

    if rotation_t >= ROTATION_TIME:
        return q_goal.copy()

    s = rotation_t / ROTATION_TIME

    s = np.clip(
        s,
        0.0,
        1.0
    )

    s = (
        10.0 * s**3
        -
        15.0 * s**4
        +
        6.0 * s**5
    )

    return slerp(
        q_start,
        q_goal,
        s
    )


# ==================== POSITION CONTROLLER ====================

def position_controller(
    target_position
):
    position = data.qpos[0:3].copy()
    velocity = data.qvel[0:3].copy()

    error = (
        target_position
        -
        position
    )

    acceleration = (
        Kp_position * error
        -
        Kd_position * velocity
    )

    gravity = np.array([
        0.0,
        0.0,
        -9.81
    ])

    force = mass * (
        acceleration
        -
        gravity
    )

    force_norm = np.linalg.norm(force)

    if force_norm > MAX_FORCE:
        force *= (
            MAX_FORCE
            /
            force_norm
        )

    return force


# ==================== ORIENTATION CONTROLLER ====================

def orientation_controller(
    target_quaternion
):
    current_quaternion = (
        data.qpos[3:7].copy()
    )

    current_quaternion /= np.linalg.norm(
        current_quaternion
    )

    rotation_error = quaternion_error(
        current_quaternion,
        target_quaternion
    )

    angular_velocity = (
        data.qvel[3:6].copy()
    )

    torque = (
        Kp_rotation * rotation_error
        -
        Kd_rotation * angular_velocity
    )

    torque_norm = np.linalg.norm(torque)

    if torque_norm > MAX_TORQUE:
        torque *= (
            MAX_TORQUE
            /
            torque_norm
        )

    return torque


# ==================== INITIAL STATE ====================

data.qpos[0:3] = START_POSITION
data.qpos[3:7] = q_start
data.qvel[:] = 0.0

mujoco.mj_forward(
    model,
    data
)


# ==================== INFORMATION ====================

print()
print("TRAJECTORY PLANNER")

print(
    "START POSITION:",
    START_POSITION
)

print(
    "GOAL POSITION:",
    GOAL_POSITION
)

print(
    "START ORIENTATION:",
    np.rad2deg([
        START_ROLL,
        START_PITCH,
        START_YAW
    ])
)

print(
    "GOAL ORIENTATION:",
    np.rad2deg([
        GOAL_ROLL,
        GOAL_PITCH,
        GOAL_YAW
    ])
)

if FLIGHT_TIME <= 0.0:
    print("Режим: ВРАЩЕНИЕ НА МЕСТЕ")
else:
    print("Режим: A* + ПОЛЁТ + ВРАЩЕНИЕ")

print(
    "Flight time:",
    round(FLIGHT_TIME, 2)
)

print(
    "Rotation time:",
    ROTATION_TIME
)

print()


# ==================== PATH ====================

if len(PATH) > 1:
    print("PATH:")

    for i, point in enumerate(PATH):
        print(
            f"{i:3d}:",
            np.round(point, 2)
        )

    print()


# ==================== VIEWER ====================

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    last_print = 0.0

    while viewer.is_running():

        t = data.time

        target_position = position_at_time(t)

        target_orientation = orientation_at_time(t)

        force = position_controller(
            target_position
        )

        torque = orientation_controller(
            target_orientation
        )

        data.xfrc_applied[drone_id] = np.array([
            force[0],
            force[1],
            force[2],
            torque[0],
            torque[1],
            torque[2]
        ])

        mujoco.mj_step(
            model,
            data
        )

        viewer.sync()

        if t - last_print > 0.25:

            current_position = (
                data.qpos[0:3]
            )

            error_position = np.linalg.norm(
                GOAL_POSITION
                -
                current_position
            )

            error_orientation = np.linalg.norm(
                quaternion_error(
                    data.qpos[3:7],
                    q_goal
                )
            )

            print(
                "t =",
                round(t, 2),
                "| target =",
                np.round(target_position, 2),
                "| actual =",
                np.round(current_position, 2),
                "| pos_error =",
                round(error_position, 3),
                "| rot_error =",
                round(
                    np.rad2deg(
                        error_orientation
                    ),
                    2
                ),
                "| torque =",
                np.round(torque, 2)
            )

            last_print = t

        time.sleep(0.002)
