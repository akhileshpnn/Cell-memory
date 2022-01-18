import numpy as np


class EgfrPtp:
    
    # egfrt=1.1; # monostable
   
    # egfrt=1.27; #criticality
    # egfrt=1.265; #criticality for 2 gradient from same direction
    
    egfrt=1.35; #ihss
    
    # egfrt=1.85; #ihss
    
    
    ptprgt=1.0;ptpn2t=1.0;
    b2=1.1;g1=1.9;a1=0.001;a2=0.3;a3=0.7;b1=11;k1=0.5;k2=0.5;g2=0.1
    
    kon=0.05;koff=5.6*kon
    
    kexc1=0.008 #um^2/sec
    kexc2=0.008 #um^2/sec

    
    def reaction(self, t, y, stimulus_input):
        ep, eep, pa = y 
        e = (self.egfrt-ep-eep)
        egft=stimulus_input
        
        n2a_qss=self.ptpn2t*(self.k1+self.b2*(np.sum(ep)+np.sum(eep)))/(self.k1+self.k2+self.b2*(np.sum(ep)+np.sum(eep)))        
        dep=e*(self.a1*e+self.a2*ep+self.a3*eep)-self.g1*pa*ep-self.g2*n2a_qss*ep-self.kon*ep**2*(egft-eep)+ 0.5*self.koff*eep
        deep=self.kon*(ep**2 + e**2)*(egft-eep)-self.koff*eep
        dpa=self.k1*(self.ptprgt-pa)-self.k2*pa-self.b1*pa*(ep+eep)
        return [dep,deep,dpa]
