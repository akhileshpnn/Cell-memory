
import numpy as np
def pressure_pro(egfrp):
    
    kprot=0.08 #nN/um2 
    
    pprot=kprot*(egfrp-np.mean(egfrp))/(np.max(egfrp)-np.mean(egfrp))
    pprot[np.argwhere(pprot<0)]=0
    return pprot

def pressure_ret(egfrp):
    
    kret=0.05 #nN/um2

    pret=-kret*(np.mean(egfrp)-egfrp)/(np.mean(egfrp)-np.max(egfrp))
    pret[np.argwhere(pret<0)]=0
    pret=-pret
        
 
    return pret

def pressure_area(A,A0,mem_len,egfrp):

    karea = 0.02
    
    parea=karea*(A0-A)
    return parea*np.ones(mem_len)

def pressure_tension(cur,rc):

    kten=0.1

    cur = np.round(cur,2)
    pten=kten*(cur-1/rc)    
    
    return pten


def area(Z):
    inside=np.argwhere(Z<0)
    A=len(inside)
    return A


    
    
    

