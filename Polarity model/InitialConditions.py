
import numpy as np
from Models import *
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class InitialCondition:

    def set_initial_condition(self, N):
        return
    
class random_ini:
    
    per=0.01
    
    def set_initial_condition(self, N):
        Wper = (np.concatenate([self.per*np.random.rand(N), 0*np.random.rand(N), self.per*np.random.rand(N)])).transpose()
        W0 = Wper;
        return W0


# class around_steadystate:
    
#     def set_initial_condition(self, N):
#         #ein=5;eein=0;pin=0.2
        
# #        sol_timeseries = solve_ivp(EgfrPtp, [0, 500], [ein, eein, pin], args=(), dense_output=True)
# #        
# #        [ess,eess,pss] = sol_timeseries.y[:,-1]
        
#         #e = np.ones(N);
#         #ess,pss=0.2,0.2
#         #We = (np.concatenate([ess*e, eess*e, pss*e])).transpose()
#         #We = np.array([ce*e; ge*e]).transpose()

#         Wper = (np.concatenate([self.per*np.random.rand(N), 0*np.random.rand(N), self.per*np.random.rand(N)])).transpose()
#         W0 = Wper;
#         return W0
    

class around_steadystate:
    
    per=0.01
    def set_initial_condition(self, N):
        f=EgfrPtp()
        ein=0.1;eein=0;pin=0.2
        sol_timeseries = solve_ivp(f.reaction, [0, 500], [ein, eein, pin], args=([0]), dense_output=True)
#        
        [ess,eess,pss] = sol_timeseries.y[:,-1]
        
        ep=sol_timeseries.y[0]
        pa=sol_timeseries.y[2]
        
        plt.figure()
        plt.title('Finding uniforn SS')
        plt.plot(sol_timeseries.t,ep,'r-',lw=2.0,label='EGFRp')
        plt.plot(sol_timeseries.t,pa,'k-',lw=2.0,label='PTPRGa')
        plt.legend()
        plt.show()
        
        e = np.ones(N);
        # ess,pss=0.2,0.2
        We = (np.concatenate([ess*e, eess*e, pss*e])).transpose()
        #We = np.array([ce*e; ge*e]).transpose()
        random_num=np.random.rand(N)
        Wper = (np.concatenate([self.per*np.random.rand(N), np.zeros(N), self.per*np.random.rand(N)])).transpose()
        W0 = We + Wper;
        if len(np.argwhere(W0<0))!=0:
            W0=np.nan
            print('Negative numbers in initial values')
        else:
            return W0

