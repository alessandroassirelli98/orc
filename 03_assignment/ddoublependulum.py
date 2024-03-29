from pendulum import Pendulum
import numpy as np
from numpy import pi
import time
import example_robot_data as erd
import pinocchio as pin


class DDoublePendulum:
    ''' Discrete Pendulum environment. Joint angle, velocity and torque are discretized
        with the specified steps. Joint velocity and torque are saturated. 
        Guassian noise can be added in the dynamics. 
        Cost is -1 if the goal state has been reached, zero otherwise.
    '''
    def __init__(self, ndu=3, vMax1=8., vMax2=13., uMax=2, dt=0.2, ndt=1, noise_stddev=0):
        self.robot = erd.load("double_pendulum")
        self.model = self.robot.model
        self.data = self.robot.data
        self.dt = dt        # time step
        self.nq = self.robot.nq
        self.nv = self.robot.nv
        self.nu = 1
        self.nx = self.nq + self.nv
        self.ndu = ndu        # Number of discretization steps for joint torque
        self.uMax = uMax    # Max torque (u in [-umax,umax])
        self.vMax1 = vMax1    # Max velocity (v in [-vmax,vmax])
        self.vMax2 = vMax2    # Max velocity (v in [-vmax,vmax])

        self.control_map = np.linspace(-uMax, uMax, ndu)

        self.max_episodes = 200
        self.episode_counter = 0

        try:
            import webbrowser
            self.viz = pin.visualize.MeshcatVisualizer(self.robot.model, self.robot.collision_model, self.robot.visual_model)
            self.viz.initViewer()
            webbrowser.open(self.viz.viewer.url())
            self.viz.loadViewerModel()
            self.viz.display(self.robot.q0)
            time.sleep(3)
        except:
            self.viz=None


    def map_control(self, iu):
        return self.control_map[iu]

    def reset(self, x0=None, random=False):
        if x0 is None: 
            if random:
                q0 = np.array([np.random.uniform(-np.pi, np.pi), 0.])
                v0 = np.array([np.random.uniform(-1., 1.), 0.])
            else:
                q0 = np.array([np.pi, 0.])
                v0 = np.array([0., 0.])
            x0 = np.concatenate([q0,v0])
        self.x = x0
        self.episode_counter = 0
        return x0

    def step(self, iu):
        u = np.array([self.map_control([iu[0]])[0], 0]) # Underactuation

        x = self.x.copy()

        self.x   = self.dynamics(x, u)
        self.episode_counter += 1

        done = False
        if self.episode_counter == self.max_episodes:
            done = True

        cost = self.compute_raw_cost(x, u) 
        
        return self.x, cost, done

    def compute_raw_cost(self, x, u):
        cost = (10*x[0]**2 + 0.1 * x[2]**2 + 0.01*u[0] **2 + \
                    10*x[1]**2 + 0.1 * x[3]**2) * self.dt
        return cost

    def render(self):
        q = self.x[:self.nq]
        self.viz.display(q)
        time.sleep(self.dt)

    def dynamics(self, x, u):
        modulePi = lambda th: (th+np.pi)%(2*np.pi)-np.pi

        dt = self.dt
        nq = self.nq
        nv = self.nv
        data = self.data
        model = self.model

        q = x[:nq]
        v = x[nq:]
        ddq = pin.aba(model, data, q, v, u)

        v = v + ddq * dt
        v = np.clip(v, [-self.vMax1, -self.vMax2], [self.vMax1, self.vMax2])

        q = q + v * dt
        q = modulePi(q)

        x = np.concatenate([q, v])
        return x