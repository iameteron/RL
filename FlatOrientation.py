import gym
import random
import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

class FlatOrientation(gym.Env):
    def __init__(
            self, 
            t_start: int = 0, 
            t_end: int = 1,
            dt: float = 0.1, 
            R1: int = 1000,
            R2: int = 1000,
            max_action: int = 25,
            observable_states: list = [0, 1, 2],
            ic_boundaries: list = [[-0.5, 0.5], [-0.5, 0.5]],
            integration: str = 'RK45'
    ):

        self.t_start = t_start
        self.t_end = t_end
        self.dt = dt
        self.R1 = R1
        self.R2 = R2
        self.max_action = max_action
        self.ic_boundaries = np.array(ic_boundaries)
        self.observable_states = observable_states
        low = np.array([0, -5, -5])
        high = np.array([1, 5, 5])
        self.observation_space = gym.spaces.Box(
            low=low[observable_states], 
            high=high[observable_states], 
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-self.max_action, high=self.max_action, shape=(1,), dtype=np.float32
        )
        self.integration = integration

    def reset(self, seed=None, options=None):
        self.state = np.array(
            [0] + [np.random.uniform(x[0], x[1]) for x in self.ic_boundaries]
        )
        return self.state, {}
    
    def step(self, u):
        t_curr = self.state[0]
        t_next = self.state[0] + self.dt
        x_curr = self.state[1:]
        df_dt = lambda t, x: rhs(t, x, u)

        x_next = self.integrate(df_dt, t_curr, t_next, x_curr, u)
        
        self.state = np.hstack((t_next, x_next)) 

        if np.isclose(t_next, self.t_end):
            reward = - self.R1 * (self.state[1] - np.pi)**2 - self.R2 * self.state[2]**2
            reward -= self.dt * u[0]**2
            done = True

        else:
            # reward = - self.R1 * (self.state[1] - np.pi)**2 - self.R2 * self.state[2]**2
            # reward -= self.dt * u[0]**2
            reward = - self.dt * u[0]**2
            done = False

        return self.state, reward, False, done, {}

    def integrate(self, df_dt, t_curr, t_next, x_curr, u):
        if self.integration == 'Euler':
            x_next = rhs(t_curr, x_curr, u) * self.dt + x_curr

        else:
            sol = solve_ivp(df_dt, (t_curr, t_next), x_curr, method=self.integration) #, t_eval=t_eval)
            x_next = sol.y.T[-1]

        return x_next
            
    
def rhs(t, x, u):
    return np.array([x[1], u[0]])


def plot_u(env, agent):
    traj = agent.get_trajectory(env, prediction=True, initial_state=np.array([0, 0, 0]))
    u = np.array(traj['actions'])
    t = np.arange(env.t_start, env.t_end + env.dt, env.dt)

    new_u = np.repeat(u, 2)
    new_t = np.repeat(t, 2)
    new_t = new_t[1:-1]

    plt.plot(new_t, new_u, label='u')
    plt.legend()


def plot_sheaf(env, agent, n=1, initial_state=None):
    for _ in range(n):
        traj = agent.get_trajectory(env, prediction=True, initial_state=initial_state)
        t = np.arange(env.t_start, env.t_end + env.dt, env.dt)
        states = np.array(traj['states'])
        x1 = states[:, 1]
        x2 = states[:, 2]

        print(x1[-1] - np.pi, x2[-1])
        
        kwargs_x1 = {'color': 'blue'}
        kwargs_x2 = {'color': 'orange'}
        plt.plot(t, x1, **kwargs_x1)
        plt.plot(t, x2, **kwargs_x2)

    plt.plot([], [], label=r'$ x_1 $', **kwargs_x1)
    plt.plot([], [], label=r'$ x_2 $', **kwargs_x2)
    plt.title('Bundle of Trajectories')
    plt.xlabel('t')
    plt.legend()
