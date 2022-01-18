
import numpy as np

from scipy.integrate import solve_ivp
import sdeint
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from LaplaceBeltramiOperator import *
from InitialConditions import *
from Experiments import *
from colormaps import parula_map


class ReactionDiffusion1D:
    
    R = 2; A = np.pi*R**2; L = 2*np.pi*R;
    N = 20 # Number of boundary nodes
    tF = 300;
    dt=0.01
    t_eval = np.arange(0,tF,dt)
    folder = 'C:\\Users\\nandan\\Dropbox\\Caesar\\EGFR memory paper_after science\\figure1\\egfrptp_sde\\20210802\\data\\'
   
    save_data = None
    
    add_noise = True
    noise_ampl=0.02

    def __init__(self, model, initial_condition, lbo,stimulus):
        
        self.model = model
        self.lbo = lbo
        self.initial_condition = initial_condition
        self.stimulus=stimulus
    
    def initialize_system(self):
        
        self.stimulus.N=self.N
        self.cellmem = np.linspace(0,self.L,self.N)
        self.dsigma=self.cellmem[1]-self.cellmem[0]
        # self.d1=self.model.kexc1*self.dsigma**2
        self.d1=self.model.kexc1
        self.d2=0
        # self.d3=self.model.kexc2*self.dsigma**2
        self.d3=self.model.kexc2
        
        self.Z = np.zeros(3*self.N)
        self.Z = self.initial_condition.set_initial_condition(self.N)
        self.Stimulus=np.zeros((self.tF,self.N))
                
    def get_input_profile(self):
        for t in range(self.tF):
            self.Stimulus[t]=self.stimulus.add_stimulus(t)
    def get_gradient_steepness(self):
        
        self.stimulus_steepness=(self.Stimulus[:,int(0.5*self.N)-1]-self.Stimulus[:,0])/100
        self.stimulus_steepness=self.stimulus_steepness/self.stimulus.egft
#        perc_cropped=perc[t_beg1:t_end1]
#        perc_cropped=(perc_cropped/perc_cropped[0])*100
        
    def F_det(self,t,W):
        
        LB = self.lbo.set_matrix(self.N)
        LB = (1/self.dsigma**2)*LB;
        A = block_diag(self.d1*LB, self.d2*LB, self.d3*LB);
#        
        y = [W[:self.N],W[self.N:2*self.N],W[2*self.N:]]
        stimulus_input=self.stimulus.add_stimulus(t)

        ffe, ffee, gge = self.model.reaction(t, y, stimulus_input)
        
        return np.matmul(A,W) + np.concatenate([ffe, ffee, gge]).transpose();
    
    def F_stocha(self,W,t):
        
        LB = self.lbo.set_matrix(self.N)
        LB = (1/self.dsigma**2)*LB;
        A = block_diag(self.d1*LB, self.d2*LB, self.d3*LB);
#        
        y = [W[:self.N],W[self.N:2*self.N],W[2*self.N:]]
        stimulus_input=self.stimulus.add_stimulus(t)

        ffe, ffee, gge = self.model.reaction(t, y, stimulus_input)
        
        return np.matmul(A,W) + np.concatenate([ffe, ffee, gge]).transpose();
    
    def G_stocha(self,W,t):
        
        G1 = np.diag(self.noise_ampl*np.ones(self.N))
        G2 = G3 = np.zeros((self.N,self.N))
        
        G = block_diag(G1, G2, G3)
        return G
    
    def simulate(self):
        
        self.initialize_system()
        sol_det = solve_ivp(self.F_det, [0,self.tF], self.Z, t_eval=self.t_eval);
        
        if self.add_noise==True:
            sol_stocha = sdeint.itoint(self.F_stocha, self.G_stocha, self.Z, tspan=self.t_eval)
        else:
            sol_stocha=None
        
        self.get_input_profile()
        self.get_gradient_steepness()
        
        return sol_det,sol_stocha

    
    def plot_kymo(self,ep,tt,label):
        
        fig,ax = plt.subplots()
        im = ax.imshow(ep.T, extent=[1,self.N+1,len(ep.T),0],cmap=parula_map, aspect = 'auto',vmin=0,vmax=np.max(ep))
        ax.figure.colorbar(im)
        ax.set_ylabel('Time(min)',fontsize=20)
        ax.set_xlabel('Plama Membrane bins',fontsize=20)
        ax.set_xticks(np.arange(1.5,self.N+1.5,2))
        ax.set_xticklabels(np.arange(1,self.N+1,2),fontsize=20)
        ax.set_yticks(np.arange(0,len(ep.T),5000))
        ax.set_yticklabels(np.arange(0,self.tF,50),fontsize=20)
#        ax.set_title('@EGFRt'+str(egfrt)+'_spatial profile')
        for t in tt:
            ax.axhline(y=int(t/rd.dt),color='r')
        
        if not self.save_data is None:
            plt.savefig(os.path.join(self.folder,'Kymograph_'+label+'_'+type(self.stimulus).__name__+'_seed('+str(seed_int)+').png'))    
            np.save(os.path.join(self.folder,'Kymograph_'+label+'_'+type(self.stimulus).__name__+'_seed('+str(seed_int)+').npy'),ep)            
        plt.show()
    
    def plot_cellmem(self,ep, time):
        
        egfrp_q = ep[:,time]
        
        fig  = plt.figure(figsize=(5,5))        
        #norm = mpl.colors.Normalize(0,1) 
        
        t = np.linspace(0,self.L,self.N+1)
        r = np.linspace(0.9,1,2)  
        rg, tg = np.meshgrid(r,t)      
        rg,epg=  np.meshgrid(r,egfrp_q)  
        
        ax=fig.add_subplot(111, projection='polar')
        im = plt.pcolormesh(t, r, epg.T, vmin=0,vmax=epmax)  
        fig.colorbar(im)
        ax.set_yticklabels([])   
        ax.set_xticklabels([])  
        ax.spines['polar'].set_visible(False) 
#        ax.set_title('@EGFRt_'+str(egfrt)+'_initial state')
        
        if not self.save_data is None:
            np.save(os.path.join(self.folder,'Cell membrane_egfrp_'+type(self.stimulus).__name__+'_@'+str(time)+'(a.u).npy'),epg)
            plt.savefig(os.path.join(self.folder,'Cell membrane_'+type(self.stimulus).__name__+'_@'+str(time)+'(a.u)_from python.png'))
        plt.show()
    
    def plot_timeseries(self,ep, bin_front,bin_back):
        
        plt.figure()
        plt.figure(figsize=(9,3))
        plt.plot(self.t_eval,ep[bin_front],'k-',lw=3.0)
        plt.plot(self.t_eval,ep[bin_back],'k--',lw=3.0)
        #plt.xlabel('time')
        plt.ylabel('EGFRp',fontsize=20)
        plt.xlabel('Time(min)',fontsize=20)
        
        # for i in range(len(self.vlines)):
        #     plt.axvline(x=self.vlines[i][0], color='g',ls='--')  
        plt.xlim(0,self.t_eval[-1])
#        plt.ylim(0,0.6)
#        plt.xticks([])
        plt.show()
        
    def plot_spatialprofile(self,ep):
        
        bins_=np.arange(1,self.N+1,1)
        av_ep_perbin=np.mean(ep,axis=1)
        
        plt.figure()
        plt.figure(figsize=(9,3))
        plt.plot(bins_,av_ep_perbin,'k-',lw=3.0)
        plt.ylabel('EGFRp per bin',fontsize=20)
        plt.xlabel('Plama Membrane bins',fontsize=20)
        plt.xlim(1,self.N)
#        plt.xticks([])
        plt.show()
        
    def plot_gradient_profile(self,tt):
        
        colours=['darkgreen','g','lime','r']
        
        bins_=np.arange(1,self.N+1,1)
        egf_profile1=self.Stimulus[tt[0]]
        egf_profile2=self.Stimulus[tt[1]]
        egf_profile3=self.Stimulus[tt[2]]
        # egf_profile4=self.Stimulus[tt[3]]
        
        plt.figure()
        plt.figure(figsize=(9,3))
        plt.plot(bins_,egf_profile1,'-',color=colours[0],lw=3.0)
        plt.plot(bins_,egf_profile2,'-',color=colours[1],lw=3.0)
        plt.plot(bins_,egf_profile3,'-',color=colours[2],lw=3.0)
        # plt.plot(bins_,egf_profile4,'-',color=colours[3],lw=3.0)
        plt.ylabel('EGF',fontsize=20)
        plt.xlabel('Cell contour',fontsize=20)
        plt.xlim(1,self.N)
#        plt.xticks([])
        if not self.save_data is None:
            np.save(os.path.join(self.folder,'Stimulus_egf_'+type(self.stimulus).__name__+'_seed('+str(seed_int)+').npy'),self.Stimulus)     
        plt.show()
    
    def plot_gradient_steepness(self,tt):
        t_beg=tt[0]
        t_end=tt[-2]
        
        stimulus_steepness_cropped = self.stimulus_steepness[t_beg:t_end]
        stimulus_steepness_cropped=(stimulus_steepness_cropped/stimulus_steepness_cropped[0])*100
        
        plt.figure()
        plt.plot(self.t_eval[t_beg:t_end]/self.dt,stimulus_steepness_cropped,'k-',lw=3.0)
        plt.xlim(t_beg,t_end)
        plt.xlabel('Time(min)',fontsize=20)
        plt.ylabel('Gradient steepness(%)',fontsize=20)
        if not self.save_data is None:
            plt.savefig(os.path.join(self.folder,'Gradient steepness_egf_'+type(self.stimulus).__name__+'.png'))    
            np.save(os.path.join(self.folder,'Gradient steepness_egf_'+type(self.stimulus).__name__+'.npy'),stimulus_steepness_cropped)     
        plt.show()

        
if __name__ == '__main__':
    
    seed_int = np.random.randint(1000000)
    
    # seed_int=180988 # 1 gradient
    
    # seed_int=350106 # 2 gradients same
    # seed_int=481606 # 2 gradients opposite
    
    # seed_int=997281 # 1 gradient  @ihss
    # seed_int=59179 # 2 gradient from opposite @ihss
    # seed_int=598078 # 2 gradient from same @ihss

    np.random.seed(seed_int)
    print('random seed used ',seed_int)
    
    
    lbo = Periodic()
    model = EgfrPtp()
    initial_condition = random_ini()
    
    stimulus=single_gradient()
    # stimulus=seq_gradient_2_static()
    # stimulus=seq_gradient_2_dynamic()
    # stimulus=seq_gradient_3_criticality()
    
    Stimu_strength = [0.2]#np.arange(0.0, 0.51, 0.01)
    
    tt=[10,50,70]
    # tt=[10,70,150,210]
    # tt=[10,50,70]
    
    print(type(model).__name__)
    

    for stimu_strength in Stimu_strength:
        stimu_strength = np.round(stimu_strength,2)
        print(stimu_strength)
        stimulus.egft=stimu_strength
        
        rd = ReactionDiffusion1D(model, initial_condition, lbo,stimulus)
        sol_det, sol_stocha=rd.simulate()
        
        ep=sol_det.y[:rd.N]
        ep = np.roll(ep,0,axis=0)
        epmax=np.max(ep)
        
        eep=sol_det.y[rd.N:2*rd.N]
        
        
        rd.plot_gradient_profile(tt)
        # rd.plot_gradient_steepness(tt)
        
        rd.plot_kymo(ep,tt,label='egfrp')
        rd.plot_kymo(eep,tt,label='eegfrp')
        
        rd.plot_timeseries(ep,10,0)
        
        if not sol_stocha is None:
            ep_stocha=sol_stocha[:,:rd.N].T
            ep_stocha = np.roll(ep_stocha,0,axis=0)
            epmax=np.max(ep_stocha)
            eep_stocha=sol_stocha[:,rd.N:2*rd.N].T
            
            rd.plot_kymo(ep_stocha,tt,label='egfrp')
            rd.plot_kymo(eep_stocha,tt,label='eegfrp')
            
