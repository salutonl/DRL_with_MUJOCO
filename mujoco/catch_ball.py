import numpy as np
import mujoco_py
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.utils import seeding
from gym import error, spaces
import random


class CatchBallEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.random_seed = random.uniform(0.3, 0.2)
        print(self.random_seed)
        mujoco_env.MujocoEnv.__init__(self, 'catchball.xml', 5)
        self.model = mujoco_py.load_model_from_path('/Users/mac/opt/anaconda3/envs/spinningup/lib/python3.7/site-packages/gym/envs/mujoco/assets/catchball.xml')
        self.frame_skip = 2
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}
        self.times = 3
        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        self._set_action_space()
        action = self.action_space.sample()
        observation, _reward, done, _ = self.step(action)
        self._set_observation_space(observation)
        self.seed()
        utils.EzPickle.__init__(self)

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        low = np.full(observation.shape, -float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
        self.observation_space = space
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def mass_center(self, model, sim):
        mass = np.expand_dims(model.body_mass, 1)
        xpos = sim.data.xipos
        return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def _get_obs(self):

        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        center_mass_inertia = self.sim.data.cinert.flat.copy()
        center_mass_vel = self.sim.data.cvel.flat.copy()

        actuator_forces = self.sim.data.qfrc_actuator.flat.copy()

        external_force = self.sim.data.cfrc_ext.flat.copy()

        return np.concatenate((
            position[2:],
            velocity,
            center_mass_inertia,
            center_mass_vel,
            actuator_forces,
            external_force
        ))

    def ctrl_cost(self, action, control_weight):
        control_cost = control_weight * np.sum(np.square(self.sim.data.ctrl))
        return control_cost

    def get_pos(self, name):
        return self.data.get_body_xpos(name)

    def get_ball(self):
        x = self.get_pos('right_hand')[0]-self.get_pos('target')[0]
        y = self.get_pos('right_hand')[1]-self.get_pos('target')[1]
        z = self.get_pos('right_hand')[2]-self.get_pos('target')[2]
        # z = abs(z)
        # y = abs(y)
        # x = abs(x)
        dis = x*x + y*y +z*z
        if dis <= 0.010:
            return 100000
        else:

            return 0
                # -(dis**0.5)

    def done(self):
        x = self.get_pos('right_hand')[0]-self.get_pos('target')[0]
        y = self.get_pos('right_hand')[1]-self.get_pos('target')[1]
        z = self.get_pos('right_hand')[2]-self.get_pos('target')[2]
        # z = abs(z)
        # y = abs(y)
        # x = abs(x)
        dis = x*x + y*y +z*z
        if self.get_pos('target')[2] < 0.965:
            return True
        if dis <= 0.010:
            return True
        else:
            return False

    def step(self, action):

        action = [x*(1+self.random_seed) for x in action]
        self.do_simulation(action, self.frame_skip)
        actuator_forces = self.sim.data.qfrc_actuator.flat.copy()
        actuator_forces = [abs(x)*(1 + self.random_seed*0.0002) for x in actuator_forces]
        actuator_forces_cost = sum(actuator_forces)

        observation = self._get_obs()

        reward = self.get_ball()
                 # - self.ctrl_cost(action, 0.1) - actuator_forces_cost

        done = self.done()

        info = {
            'done': done,
            'reward': reward,
            'qpos': self.sim.data.qpos.flat.copy(),
            'target_position': self.get_pos('target')
        }

        return observation, reward, done, info

    def reset(self):

        i = self.times % 3

        low = -1e-2
        high = 1e-2
        qpos = self.init_qpos + self.np_random.uniform(
            low=low, high=high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(
            low=low, high=high, size=self.model.nv)
        self.set_state(qpos, qvel)
        x_v = random.uniform(-3.5, -5.8)
        y_v = random.uniform(0.2, -0.8)
        z_v = random.uniform(2.85, 2.8)
        self.sim.data.set_joint_qvel('target_x', x_v)
        self.sim.data.set_joint_qvel('target_y', y_v)
        self.sim.data.set_joint_qvel('target_z', z_v)
        self.times += 1
        observation = self._get_obs()
        return observation
