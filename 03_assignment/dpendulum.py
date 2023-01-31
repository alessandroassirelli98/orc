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
        self.goal = np.array([0.,0.])
        
        max_state = np.array([np.pi, vMax])
        self.max_cost = self.compute_cost(max_state, uMax)

        self.max_episodes = 200
        self.episode_counter = 0


    def map_control(self, iu):
        return self.control_map[iu]

    def reset(self,x0=None):
        if x0 is None: 
            q0 = np.pi
            v0 = 0
            x0 = np.array([q0,v0])
        self.x = x0
        self.episode_counter = 0
        return x0

    def step(self, iu):
        u = self.map_control(iu[0])

        x = self.x.copy()

        cost = self.compute_cost(x, u) / self.max_cost
        
        self.x   = self.dynamics(x, u)
        self.episode_counter += 1

        done = True if (self.x == np.array([0.,0.])).all() else False
        if self.episode_counter == self.max_episodes:
            done = True
        
        return self.x, cost, done
    
    def compute_cost(self, x, u):
        cost = (10 * x[0]**2 + 0.1 * x[1]**2 + 0.001*u **2)
        return cost

    def render(self):
        q = self.x[0]
        self.pendulum.display(np.array([q,]))
        time.sleep(self.pendulum.DT)

    def dynamics(self, x, u):
        x, _ = self.pendulum.dynamics(x, u)
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
    
if __name__=="__main__":
    print("Start tests")
    env = DPendulum()
    nq = env.nq
    nv = env.nv
    
    # sanity checks
    for i in range(nq*nv):
        x = env.i2x(i)
        i_test = env.x2i(x)
        if(i!=i_test):
            print("ERROR! x2i(i2x(i))=", i_test, "!= i=", i)
        
        xc = env.d2c(x)
        x_test = env.c2d(xc)
        if(x_test[0]!=x[0] or x_test[1]!=x[1]):
            print("ERROR! c2d(d2c(x))=", x_test, "!= x=", x)
        xc_test = env.d2c(x_test)
        if(np.linalg.norm(xc-xc_test)>1e-10):
            print("ERROR! xc=", xc, "xc_test=", xc_test)
    print("Tests finished")
    