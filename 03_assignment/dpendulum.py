from pendulum import Pendulum
import numpy as np
from numpy import pi
import time
    

class DPendulum:
    ''' Discrete Pendulum environment. Joint angle, velocity and torque are discretized
        with the specified steps. Joint velocity and torque are saturated. 
        Guassian noise can be added in the dynamics. 
        Cost is -1 if the goal state has been reached, zero otherwise.
    '''
    def __init__(self, ndu=3, vMax=8., uMax=2, dt=0.2, ndt=1, noise_stddev=0):
        self.pendulum = Pendulum(1,noise_stddev)
        self.pendulum.DT  = dt
        self.pendulum.NDT = ndt
        self.dt = dt        # time step
        self.nq = 1
        self.nv = 1
        self.nu = 1
        self.nx = self.nq + self.nv
        self.ndu = ndu        # Number of discretization steps for joint torque
        self.uMax = uMax    # Max torque (u in [-umax,umax])
        self.vMax = vMax    # Max velocity (v in [-vmax,vmax])

        self.control_map = np.linspace(-uMax, uMax, ndu)        

        self.max_episodes = 200
        self.episode_counter = 0


    def map_control(self, iu):
        return self.control_map[iu]

    def reset(self, x0=None, random=False):
        if x0 is None: 
            if random:
                q0 = np.random.uniform(-np.pi, np.pi)
                v0 = np.random.uniform(0., 1.)
            else:
                q0 = np.pi
                v0 = 0.
            x0 = np.array([q0,v0])
        self.x = x0
        self.episode_counter = 0
        return x0

    def step(self, iu):
        u = self.map_control(iu[0])

        x = self.x.copy()
     
        self.x   = self.dynamics(x, u)
        self.episode_counter += 1

        done = False
        if self.episode_counter == self.max_episodes:
            done = True
        
        cost = self.compute_cost(x, u)
        
        return self.x, cost, done
    
    def compute_cost(self, x, u, terminal=False):
        cost = (10 * x[0]**2 + 0.1 * x[1]**2 + 0.01*u **2)*self.dt        
        return cost

    def render(self):
        q = self.x[0]
        self.pendulum.display(np.array([q,]))
        time.sleep(self.pendulum.DT)

    def dynamics(self, x, u):
        x, _ = self.pendulum.dynamics(x, u)
        return x
  