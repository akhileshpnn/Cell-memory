import numpy as np
from scipy.signal import gaussian


class Experiment:
    def add_stimulus(self, t):
        return


class single_gradient:
    
    t_beg=10
    t_end=70
    
    # beg_range=2; end_range=5#dynamic gradient
    beg_range=2; end_range=3#less dynamic   
       
    def add_stimulus(self, t):
        
        egf_start=self.egft
        egf_end=0.75*self.egft
        
        egft = ((egf_end-egf_start)/(self.t_end-self.t_beg))*(t-self.t_end) + egf_end
        extend=((self.end_range-self.beg_range)*(t-self.t_beg)**2/(self.t_end-self.t_beg)**2)+self.beg_range
        if self.t_beg<=t<=self.t_end:
            ligand = egft*gaussian(self.N,extend,sym=False)
            ligand=np.roll(ligand,-1)
        else:
            ligand = np.zeros(self.N)
            
            
        return ligand

class seq_gradient_2_dynamic:
    # t_beg1=10;t_end1=70
    # t_beg2=100;t_end2=160;
    
    t_beg1=10;t_end1=70
    t_beg2=150;t_end2=210;
    
    # beg_range=2; end_range=5 #dynamic gradient
    # beg_range2=2; end_range2=5 #dynamic gradient
    
    beg_range=2; end_range=3 # less dynamic
    beg_range2=2; end_range2=3 # less dynamic

    def add_stimulus(self, t):
        
        egf_start=self.egft
        egf_end=0.75*self.egft
        
        if t<self.t_beg2:
            egft = ((egf_end-egf_start)/(self.t_end1-self.t_beg1))*(t-self.t_end1) + egf_end
        else:
            egft = ((egf_end-egf_start)/(self.t_end2-self.t_beg2))*(t-self.t_end2) + egf_end
        
        extend1=((self.end_range-self.beg_range)*(t-self.t_beg1)**2/(self.t_end1-self.t_beg1)**2)+self.beg_range
        extend2=((self.end_range2-self.beg_range2)*(t-self.t_beg2)**2/(self.t_end2-self.t_beg2)**2)+self.beg_range2

        if self.t_beg1<=t<=self.t_end1:
            ligand = egft*gaussian(self.N,extend1,sym=False)
        elif self.t_beg2<=t<=self.t_end2:
            ligand = egft*gaussian(self.N,extend2,sym=False)
            ligand=np.roll(ligand,-10)   
        else:
            ligand = np.zeros(self.N)

        return ligand
    

class seq_gradient_2_static:
    
    t_beg1=10;t_end1=70
    t_beg2=100;t_end2=160
    
    beg_range=3; end_range=3 #static gradient
    beg_range2=3; end_range2=3 #static gradient
    
    # beg_range=2; end_range=2 #static gradient
    # beg_range2=2; end_range2=2 #static gradient

    def add_stimulus(self, t):
        
        extend1=((self.end_range-self.beg_range)*(t-self.t_beg1)**2/(self.t_end1-self.t_beg1)**2)+self.beg_range
        extend2=((self.end_range2-self.beg_range2)*(t-self.t_beg2)**2/(self.t_end2-self.t_beg2)**2)+self.beg_range2

        if self.t_beg1<=t<=self.t_end1:
            ligand = self.egft*gaussian(self.N,extend1,sym=False)
        elif self.t_beg2<=t<=self.t_end2:
            ligand = self.egft*gaussian(self.N,extend2,sym=False)
            # ligand=np.roll(ligand,-10)   
        else:
            ligand = np.zeros(self.N)

        return ligand


class seq_gradient_3_ihss:
    t_beg1=10;t_end1=60
    t_beg2=120;t_end2=140;
    t_beg3=180;t_end3=240;
    
    beg_range1=1; end_range1=3 #dynamic gradient
    beg_range2=2; end_range2=2 #static gradient
    beg_range3=1; end_range3=3 #dynamic gradient
    
    def add_stimulus(self, t):
        
        egf_start=self.egft
        egf_end=0.75*self.egft
        
        if t<self.t_beg2:
            egft = ((egf_end-egf_start)/(self.t_end1-self.t_beg1))*(t-self.t_end1) + egf_end
        elif self.t_beg2<=t<self.t_beg3:
            egft = ((egf_end-egf_start)/(self.t_end2-self.t_beg2))*(t-self.t_end2) + egf_end
        else:
            egft = ((egf_end-egf_start)/(self.t_end3-self.t_beg3))*(t-self.t_end3) + egf_end
        
        extend1=((self.end_range1-self.beg_range1)*(t-self.t_beg1)**2/(self.t_end1-self.t_beg1)**2)+self.beg_range1
        extend2=((self.end_range2-self.beg_range2)*(t-self.t_beg2)**2/(self.t_end2-self.t_beg2)**2)+self.beg_range2
        extend3=((self.end_range3-self.beg_range3)*(t-self.t_beg3)**2/(self.t_end3-self.t_beg3)**2)+self.beg_range3

        if self.t_beg1<=t<=self.t_end1:
            ligand = egft*gaussian(self.N,extend1,sym=False)
            # ligand=np.roll(ligand,-3)
        elif self.t_beg2<=t<=self.t_end2:
            ligand = egft*gaussian(self.N,extend2,sym=False)
            # ligand=np.roll(ligand,-3)
        elif self.t_beg3<=t<=self.t_end3:
            ligand = egft*gaussian(self.N,extend3,sym=False)
            ligand=np.roll(ligand,8)
        else:
            ligand = np.zeros(self.N)

        return ligand
    
class seq_gradient_3_criticality:
    t_beg1=10;t_end1=70
    t_beg2=120;t_end2=140;
    t_beg3=180;t_end3=240;
    
    beg_range1=2; end_range1=5 #dynamic gradient
    beg_range2=3; end_range2=3 #static gradient
    beg_range3=2; end_range3=5 #dynamic gradient
    
    def add_stimulus(self, t):
        
        egf_start=self.egft
        egf_end=0.75*self.egft
        
        if t<self.t_beg2:
            egft = ((egf_end-egf_start)/(self.t_end1-self.t_beg1))*(t-self.t_end1) + egf_end
        elif self.t_beg2<=t<self.t_beg3:
            egft = ((egf_end-egf_start)/(self.t_end2-self.t_beg2))*(t-self.t_end2) + egf_end
        else:
            egft = ((egf_end-egf_start)/(self.t_end3-self.t_beg3))*(t-self.t_end3) + egf_end
        
        extend1=((self.end_range1-self.beg_range1)*(t-self.t_beg1)**2/(self.t_end1-self.t_beg1)**2)+self.beg_range1
        extend2=((self.end_range2-self.beg_range2)*(t-self.t_beg2)**2/(self.t_end2-self.t_beg2)**2)+self.beg_range2
        extend3=((self.end_range3-self.beg_range3)*(t-self.t_beg3)**2/(self.t_end3-self.t_beg3)**2)+self.beg_range3

        if self.t_beg1<=t<=self.t_end1:
            ligand = egft*gaussian(self.N,extend1,sym=False)
            # ligand=np.roll(ligand,-3)
        elif self.t_beg2<=t<=self.t_end2:
            ligand = egft*gaussian(self.N,extend2,sym=False)
            # ligand=np.roll(ligand,-3)
        elif self.t_beg3<=t<=self.t_end3:
            ligand = egft*gaussian(self.N,extend3,sym=False)
            ligand=np.roll(ligand,8)
        else:
            ligand = np.zeros(self.N)

        return ligand
    
