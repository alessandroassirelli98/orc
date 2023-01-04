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
    def __init__(self, nu=3, vMax=8, uMax=2, dt=0.2, ndt=1, noise_stddev=0):
        self.pendulum = Pendulum(1,noise_stddev)
        self.pendulum.DT  = dt
        self.pendulum.NDT = ndt
        self.vMax = vMax    # Max velocity (v in [-vmax,vmax])
        self.nq = 1
        self.nv = 1
        self.nx = self.nq + self.nv
        self.nu = nu        # Number of discretization steps for joint torque
        self.uMax = uMax    # Max torque (u in [-umax,umax])
        self.dt = dt        # time step
        self.control_map = np.linspace(-uMax, uMax, nu)
        self.goal = np.array([0.,0.])

        self.m = 1
        self.l = 1
        self.g = 9.81

    def map_control(self, iu):
        return self.control_map[iu]

    def reset(self,x0=None):
        if x0 is None: 
            q0 = np.pi*(np.random.rand(self.nq)*2-1)
            v0 = self.vMax*(np.random.rand(self.nv)*2-1)
            x0 = np.concatenate([q0,v0])
        self.x = x0
        return x0

    def step(self, iu):
        reward = - (self.x[0]**2 + 0.1 * self.x[1]**2 + 0.001*self.control_map[iu] **2)
        self.x   = self.dynamics(self.x,iu)
        done = True if (self.x == np.array([0.,0.])).all() else False
        return self.x, reward, done

    def render(self):
        q = self.x[0]
        self.pendulum.display(np.array([q,]))
        time.sleep(self.pendulum.DT)

    def dynamics(self, x, iu):
        m = self.m
        l = self.l
        g = self.g
        dt = self.dt

        thdot = x[1] + (3/(2*l) * g * np.sin(x[0]) + 3/(m*l**2)*self.control_map[iu])*dt
        th = x[0] + thdot*dt
        x = np.array([th, thdot])
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
    