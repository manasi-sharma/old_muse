import numpy as np
import pygame.mouse
import torch

from muse.models.model import Model
from muse.utils.input_utils import UserInput, KeyInput as KI
from muse.utils.torch_utils import to_torch
from attrdict import AttrDict


def get_posact_to_velact_fn(Kv_P=1.5, GRAB_THRESH=0.5, MAX_VEL=100):
    def posact_to_velact_fn(obs, posact: AttrDict):
        # inputs passed in ... x D
        pos_targ = posact["target/position"]
        grab_targ = posact["target/grab_binary"]
        pos = obs["position"]
        if len(pos.shape) == 2 and len(pos_targ.shape) == 3:
            pos = pos[:, None]

        ac_vel = Kv_P * (pos_targ - pos)
        if isinstance(grab_targ, torch.Tensor):
            ac_vel = torch.clip(ac_vel, -MAX_VEL, MAX_VEL)
            ac_grab = torch.max(grab_targ > GRAB_THRESH, dim=-1, keepdim=True)[0]  # max over blocks
            ac = torch.cat([ac_vel, ac_grab.to(dtype=torch.float32)], dim=-1)
        else:
            ac_vel = np.clip(ac_vel, -MAX_VEL, MAX_VEL)
            ac_grab = np.max(grab_targ > GRAB_THRESH, axis=-1, keepdims=True)[0]  # max over blocks
            ac = np.concatenate([ac_vel, ac_grab.astype(float)], axis=-1)

        # print(ac_grab.shape, ac_vel.shape, norm_out.leaf_apply(lambda arr: arr.shape))
        return ac

    return posact_to_velact_fn


def pygame_keys_teleop_fn(env, user_input: UserInput):
    keys_actions = {
        'j': np.array([-env._default_teleop_speed, 0]),  # left
        'l': np.array([env._default_teleop_speed, 0]),  # right
        'i': np.array([0, env._default_teleop_speed]),  # up
        'k': np.array([0, -env._default_teleop_speed])  # down
    }

    keys_grab_actions = {
        'g': np.array([500.]),  # grab acceleration max
    }

    SCALE = 0.1  # noise is 10% of velocity norm
    SLOW_DOWN_ALPHA = 0.75  # how much to slow down

    all_keys = list(keys_actions.keys()) + list(keys_grab_actions.keys())
    for key in all_keys:
        user_input.register_callback(KI(key, KI.ON.pressed), lambda ui, ki: None)

    last_vel = np.array([0., 0.])
    last_grab_acc = np.array([0.])

    def model_forward_fn(model: Model, obs: AttrDict, goal: AttrDict, user_input_state=None):
        nonlocal last_vel
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
            policy_name=np.array([["keys_teleop"]]),
            policy_switch=np.array([[False]]),
        )

    return model_forward_fn


def get_pygame_mouse_teleop_fn(posact_to_vel_fn=None):
    def pygame_mouse_teleop_fn(env, user_input: UserInput):
        nonlocal posact_to_vel_fn

        keys_grab_actions = {
            'g': np.array([1.]),  # grab acceleration max
        }

        if posact_to_vel_fn is None:
            posact_to_vel_fn = get_posact_to_velact_fn(MAX_VEL=env._default_teleop_speed)

        grid_size = env.grid_size

        all_keys = list(keys_grab_actions.keys())
        for key in all_keys:
            user_input.register_callback(KI(key, KI.ON.pressed), lambda ui, ki: None)

        def model_forward_fn(model: Model, obs: AttrDict, goal: AttrDict, user_input_state=None):
            grab_acc = np.array([[0.]])

            if user_input_state is None:
                user_input_state = user_input.read_input()

            if pygame.mouse.get_focused():
                mouse_pos = np.asarray(pygame.mouse.get_pos())
                mouse_pos[1] = grid_size[1] - mouse_pos[1]  # y axis (up down) gets flipped for inputs

                # mouse_pos = np.ascontiguousarray(np.flip(mouse_pos))
            else:
                mouse_pos = obs["position"].reshape((2,))
            mouse_pos = mouse_pos[None]

            for key, on_states in user_input_state.items():

                if key in keys_grab_actions.keys() and KI.ON.pressed in on_states:
                    grab_acc += keys_grab_actions[key]

            targ = AttrDict(target=AttrDict(position=mouse_pos[None], grab_binary=grab_acc[None]))
            if isinstance(obs.get_one(), torch.Tensor):
                targ.leaf_modify(lambda arr: to_torch(arr, device=model.device, check=True))
            ac = posact_to_vel_fn(obs, targ)
            return AttrDict(
                action=ac[:, 0],
                policy_type=torch.tensor([[255]], dtype=torch.uint8, device=ac.device) if isinstance(obs.get_one(), torch.Tensor) else np.array([[-1]]),
                policy_name=np.array([["mouse_teleop"]]),
                policy_switch=np.array([[False]]),
                # policy_name=torch.tensor([["mouse_teleop"]], device=ac.device, dtype=torch.) if isinstance(obs.get_one(), torch.Tensor) else np.array([['mouse_teleop']])
            ) & targ.leaf_apply(lambda arr: arr[:, 0])

        return model_forward_fn
    return pygame_mouse_teleop_fn


def pygame_key_teleop_step(env, act, last_act, gamma):
    running = True
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and (
                event.key in [pygame.K_ESCAPE, pygame.K_q]
        ):
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            env.reset()

        if event.type == pygame.KEYDOWN and event.key == pygame.K_i:
            act[1] = 75.
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_k:
            act[1] = -75.
        elif event.type == pygame.KEYUP and event.key in (pygame.K_i, pygame.K_k):
            act[1] = 0.

        if event.type == pygame.KEYDOWN and event.key == pygame.K_j:
            act[0] = -75.
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_l:
            act[0] = 75.
        elif event.type == pygame.KEYUP and event.key in (pygame.K_j, pygame.K_l):
            act[0] = 0.

        if event.type == pygame.KEYDOWN and event.key == pygame.K_g:
            act[2] = 1000.
        elif event.type == pygame.KEYUP and event.key == pygame.K_g:
            act[2] = 0.

    # print(act)
    smoothed_act = gamma * act[:2] + (1 - gamma) * last_act[:2]
    smoothed_act = np.concatenate([smoothed_act, act[2:]])

    act_dict = AttrDict(action=smoothed_act[None])
    last_act = smoothed_act

    obs, goal, done = env.step(act_dict)

    return obs, goal, act, last_act, done, running
