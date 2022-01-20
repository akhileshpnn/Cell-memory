
import numpy as np
from derivFunc import *

def f_potential(y,grid,velocity,dsigma):
    
    Y=np.reshape(y,np.shape(grid[0]))
#Approximate the convective term dimension by dimension.
    delta = np.zeros(np.shape(grid[0]));
    stepBoundInv = 0;
    for i in range(np.shape(velocity)[2]):
    
    # Get upwinded derivative approximations.
        # [ derivL, derivR ] = upwindFirstFirst(Y, i, dsigma)
        [ derivL, derivR ] = upwindFirstENO2(Y, i, dsigma)
    
    #Figure out upwind direction.
        v = velocity[:,:,i]
        flowL = (v < 0)*1
        flowR = (v > 0)*1
    
    #Approximate convective term with upwinded derivatives
        deriv = derivL*flowR + derivR*flowL;
    
        delta = delta + deriv*v;
    
    # stepBoundInv = stepBoundInv + max(abs(v(:))) / grid.dx(i);
    ydot = -np.ndarray.flatten(delta)
    
    return ydot


def f_visco(l,y,Ptot,grid,velocity,dsigma):
    
    kc=0.1 #nN/um3
    tauc=0.08 #nN/um3
    
    L=np.reshape(l,np.shape(grid[0]))
    Y=np.reshape(y,np.shape(grid[0]))

    delta = np.zeros(np.shape(grid[0]));
    
    for i in range(np.shape(velocity)[2]):
    
    # Get upwinded derivative approximations.
        # [ derivL, derivR ] = upwindFirstFirst(Y, i, dsigma)
        [ derivL, derivR ] = upwindFirstENO2(Y, i, dsigma)
    
    #Figure out upwind direction.
        v = velocity[:,:,i];
        flowL = (v < 0)*1;
        flowR = (v > 0)*1;
    
    #Approximate convective term with upwinded derivatives
        deriv = derivL*flowR + derivR*flowL;
    
        delta = delta + deriv*v;
    
    
    delta=delta+(kc/tauc)*L - (1/tauc)*Ptot
    ldot = -np.ndarray.flatten(delta)    
    return ldot
