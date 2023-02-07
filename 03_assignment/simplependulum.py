from pendulum import Pendulum
import numpy as np
from numpy import pi
import time
import example_robot_data as erd
import pinocchio as pin


class DSimplePendulum:
    ''' Discrete Pendulum environment. Joint angle, velocity and torque are discretized
        with the specified steps. Joint velocity and torque are saturated. 
        Guassian noise can be added in the dynamics. 
        Cost is -1 if the goal state has been reached, zero otherwise.
    '''
    def __init__(self, ndu=3, vMax1=8., uMax=2, dt=0.02, ndt=1, noise_stddev=0):
        self.robot = erd.load("double_pendulum")
        model = self.robot.model
        freeze_id = [model.getJointId('joint2')]
        # Lock second joint
        self.collision_model = self.robot.collision_model
        self.visual_model = self.robot.visual_model

        geom_models = [self.visual_model, self.collision_model]
        self.model, geometric_models_reduced = pin.buildReducedModel(
                                                model,
                                                list_of_geom_models=geom_models,
                                                list_of_joints_to_lock=freeze_id,
                                                reference_configuration=self.robot.q0) 
        self.data = self.model.createData()
        self.visual_model = geometric_models_reduced[0]
        self.collision_model = geometric_models_reduced[1]

        self.dt = dt        # time step
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = 1
        self.nx = self.nq + self.nv
        self.ndu = ndu        # Number of discretization steps for joint torque
        self.uMax = uMax    # Max torque (u in [-umax,umax])
        self.vMax1 = vMax1    # Max velocity (v in [-vmax,vmax])

        self.control_map = np.linspace(-uMax, uMax, ndu)
        
        max_state = np.array([np.pi, vMax1])
        self.max_cost = self.compute_raw_cost(max_state, [uMax])

        self.max_episodes = 200
        self.episode_counter = 0

        try:
            import webbrowser
            self.viz = pin.visualize.MeshcatVisualizer(self.model, self.collision_model, self.visual_model)
            self.viz.initViewer()
            webbrowser.open(self.viz.viewer.url())
            self.viz.loadViewerModel()
            self.viz.display(np.array([np.pi]))
            time.sleep(3)
        except:
            self.viz=None


    def map_control(self, iu):
        return self.control_map[iu]

    def reset(self,x0=None):
        if x0 is None: 
            x0 = np.array([np.pi,0])
        self.x = x0
        self.episode_counter = 0
        return x0

    def step(self, iu):
        u = self.map_control([iu[0]]) # Underactuation

        x = self.x.copy()

        self.x   = self.dynamics(x, u)
        self.episode_counter += 1

        done = False
        if self.episode_counter == self.max_episodes:
            done = True

        cost = self.compute_raw_cost(x, u) #/ self.max_cost
        
        return self.x, cost, done

    def compute_raw_cost(self, x, u, terminal=False):
        cost = (x[0]**2 + 0.1 * x[1]**2 + 0.001*u[0] **2) * self.dt

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
        v = np.clip(v, -self.vMax1, self.vMax1)

        q = q + v * dt
        q = modulePi(q)

        x = np.concatenate([q, v])
        return x
    
    def plot_V_table(self, V):
        ''' Plot the given Value table V '''
        import matplotlib.pyplot as plt
        Q,DQ = np.meshgrid([self.d2cq(i) for i in range(self.nq)], 
                            [self.d2cv(i) for i in range(self.nv)])
        plt.pcolormesh(Q, DQ, V.reshape((self.nv,self.nq)), cmap=plt.cm.get_cmap('Blues'))
        plt.colorbar()
        plt.title('V table')
        plt.xlabel("q")
        plt.ylabel("dq")
        plt.show()
        
    def plot_policy(self, pi):
        ''' Plot the given policy table pi '''
        import matplotlib.pyplot as plt
        Q,DQ = np.meshgrid([self.d2cq(i) for i in range(self.nq)], 
                            [self.d2cv(i) for i in range(self.nv)])
        plt.pcolormesh(Q, DQ, pi.reshape((self.nv,self.nq)), cmap=plt.cm.get_cmap('RdBu'))
        plt.colorbar()
        plt.title('Policy')
        plt.xlabel("q")
        plt.ylabel("dq")
        plt.show()
        
    def plot_Q_table(self, Q):
        ''' Plot the given Q table '''
        import matplotlib.pyplot as plt
        X,U = np.meshgrid(range(Q.shape[0]),range(Q.shape[1]))
        plt.pcolormesh(X, U, Q.T, cmap=plt.cm.get_cmap('Blues'))
        plt.colorbar()
        plt.title('Q table')
        plt.xlabel("x")
        plt.ylabel("u")
        plt.show()
    
