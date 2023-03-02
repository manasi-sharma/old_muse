import numpy as np

from muse.experiments import logger
from muse.models.model import Model
from muse.utils.input_utils import UserInput, KeyInput as KI
from attrdict import AttrDict

from muse.utils.transform_utils import convert_quat_to_rpt, convert_rpt


def bullet_keys_teleop_fn(env, user_input: UserInput):
    raise NotImplementedError
    keys_actions = {
        'a': np.array([-0.01, 0, 0]),
        'd': np.array([0.01, 0, 0]),
        'w': np.array([0, 0.01, 0]),
        's': np.array([0, -0.01, 0]),
        'i': np.array([0, 0, 0.01]),  # up
        'k': np.array([0, 0, -0.01])  # down
    }

    keys_orient_actions = {  # in rpt space
        '=': np.array([0, 0, 0.01]),
        '-': np.array([0, 0, -0.01]),
        '[': np.array([0, 0.02, 0]),
        ']': np.array([0, -0.02, 0]),
        ';': np.array([0.01, 0, 0]),
        '\'': np.array([-0.01, 0, 0]),
    }

    keys_grab_actions = {
        'g': np.array([15]),  # grab acceleration max
    }

    # SCALE = 0.1  # noise is 10% of velocity norm
    # SLOW_DOWN_ALPHA = 0.75  # how much to slow down

    all_keys = list(keys_actions.keys()) + list(keys_orient_actions.keys()) + list(keys_grab_actions.keys())
    for key in all_keys:
        user_input.register_callback(KI(key, KI.ON.pressed), lambda ui, ki: None)

    target_position = None
    target_orientation = None
    target_rpt_orientation = None
    grip_state = 0

    def model_forward_fn(model: Model, obs: AttrDict, goal: AttrDict, user_input_state=None):
        nonlocal target_position, target_orientation, target_rpt_orientation, grip_state
        vel = np.array([0., 0.])
        grab_acc = np.array([0.])

        if user_input_state is None:
            user_input_state = user_input.read_input()

        for key, on_states in user_input_state.items():
            if key in keys_actions.keys() and KI.ON.pressed in on_states:
                vel += keys_actions[key]

            if key in keys_grab_actions.keys() and KI.ON.pressed in on_states:
                grab_acc += keys_grab_actions[key]

        # noise mode
        vel += np.linalg.norm(vel) * SCALE * np.random.randn(2)
        vel = np.where(np.abs(vel) < np.abs(last_vel), SLOW_DOWN_ALPHA * vel + (1 - SLOW_DOWN_ALPHA) * last_vel,
                       vel)

        last_vel[:] = vel
        last_grab_acc[:] = grab_acc
        return AttrDict(
            action=np.concatenate([vel, grab_acc])[None],
            target=AttrDict(
                position=np.zeros((1, 2)),  # don't use this field
                grab_binary=(grab_acc > 0)[None].astype(float),
            ),
            policy_type=np.array([[255]], dtype=np.uint8),
            policy_name=np.array([["mouse_teleop"]]),
        )

    return model_forward_fn


def bullet_teleop_keys_rollout(env, max_steps=10000):
    import pybullet as p

    keys_actions = {
        p.B3G_LEFT_ARROW: np.array([-0.01, 0, 0]),
        p.B3G_RIGHT_ARROW: np.array([0.01, 0, 0]),
        p.B3G_UP_ARROW: np.array([0, 0.01, 0]),
        p.B3G_DOWN_ARROW: np.array([0, -0.01, 0]),
        ord('i'): np.array([0, 0, 0.01]),
        ord('k'): np.array([0, 0, -0.01])
    }

    keys_orient_actions = {  # in rpt space
        ord('='): np.array([0, 0, 0.01]),
        ord('-'): np.array([0, 0, -0.01]),
        ord('['): np.array([0, 0.02, 0]),
        ord(']'): np.array([0, -0.02, 0]),
        ord(';'): np.array([0.04, 0, 0]),
        ord('\''): np.array([-0.04, 0, 0]),
    }

    observation, _ = env.reset()
    # Get the position and orientation of the end effector
    target_position, target_orientation = env.robot.get_end_effector_pos(), env.robot.get_end_effector_orn()
    target_rpt_orientation = convert_quat_to_rpt(target_orientation)  # rpt rotation about default

    this_object = (observation["objects"]).leaf_apply(lambda arr: arr[0, 0])

    grip_state = 0
    done = False

    i = 0

    while True:
        keys = p.getKeyboardEvents(physicsClientId=env.id)
        if done or i >= max_steps or ord('r') in keys and keys[ord('r')] & p.KEY_IS_DOWN:
            logger.debug(
                "Resetting (after %d iters)! done = %s" % (i, done))
            i = 0
            observation, _ = env.reset()
            target_position, target_orientation = env.robot.get_end_effector_pos(), env.robot.get_end_effector_orn()
            target_rpt_orientation = convert_quat_to_rpt(target_orientation)

        for key, action in keys_actions.items():
            if key in keys and keys[key] & p.KEY_IS_DOWN:
                target_position += action

        for key, action in keys_orient_actions.items():
            if key in keys and keys[key] & p.KEY_IS_DOWN:
                # orientation = p.getQuaternionFromEuler(np.array(p.getEulerFromQuaternion(orientation) + action) % (2 * np.pi))
                target_rpt_orientation = (target_rpt_orientation + action) % (2 * np.pi)

        # open w/ >
        if ord('.') in keys and keys[ord('.')] & p.KEY_IS_DOWN:
            grip_state = max(grip_state - 0.05, 0)
        # close w/ <
        if ord(',') in keys and keys[ord(',')] & p.KEY_IS_DOWN:
            grip_state = min(grip_state + 0.05, 1.)

        curr_pos, curr_orn = env.robot.get_end_effector_pos(), env.robot.get_end_effector_orn()
        curr_rpt = convert_quat_to_rpt(curr_orn)

        # decaying target position
        target_position = target_position * 0.9 + curr_pos * 0.1
        # target_rpt_orientation = target_rpt_orientation * 0.9 + curr_rpt * 0.1
        target_orientation, target_orientation_eul = convert_rpt(*target_rpt_orientation)

        # target end effector state
        # targ_frame = CoordinateFrame(world_frame_3D, R.from_quat(orientation).inv(), np.asarray(position))
        act = np.concatenate([np.asarray(target_position), np.asarray(target_orientation_eul), [grip_state * 255.]])

        observation, _, done = env.step(act)
        # queue.put(np.concatenate([observation.contact_force[0], np.zeros(3)]))
        # print(observation['joint_positions'])

        # aabb = (observation["objects/aabb"]).reshape(2, 3)
        # print(aabb[1] - aabb[0])

        i += 1
