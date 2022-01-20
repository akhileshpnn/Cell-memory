import numpy as np
from plot_functions import *
from utils_ import *
import os
from derivFunc import *


def reinit(y,grid,sgnFactor,dsigma,apply_subcell_fix=None,check_efficiency=None):
    
    eps=1e-16
    robust_small_epsilon = 1e6 * eps;
    
    Dim,Nbx,Nby=np.shape(grid)
    
    Y=np.reshape(y,(Nby,Nby))
    # Y = sio.loadmat(os.path.join(folder,'matlab.mat'))['data']
    Yint=Y.copy()
#Approximate the convective term dimension by dimension.
    delta = np.zeros((Nbx,Nby));
    S = smearedSign(Y, sgnFactor=0);
    
    Deriv = np.zeros((Nbx,Nby,Dim))
    for i in range(Dim):
    
    # Get upwinded derivative approximations.
        # [ derivL, derivR ] = upwindFirstFirst(Y, i, dsigma)
        [ derivL, derivR ] = upwindFirstENO2(Y, i, dsigma)
    
    #Figure out upwind direction.
        flowL = ((S*derivR <= 0)&(S*derivL <= 0))*1
        flowR = ((S*derivR >= 0)&(S*derivL >= 0))*1;
        
        flows = ((S*derivR <  0)&(S*derivL >  0))*1
        # flows = ((S*derivR >  0)&(S*derivL <  0))*1
        # flows_flattened= np.ndarray.flatten(flows)
        conv_indxs=np.argwhere(flows==1)
        if len(conv_indxs)!=0:
            s = np.zeros(np.shape(flows));
            s[conv_indxs[:,0],conv_indxs[:,1]] = S[conv_indxs[:,0],conv_indxs[:,1]]*(abs(derivR[conv_indxs[:,0],conv_indxs[:,1]]) - abs(derivL[conv_indxs[:,0],conv_indxs[:,1]]))/ (derivR[conv_indxs[:,0],conv_indxs[:,1]] - derivL[conv_indxs[:,0],conv_indxs[:,1]]);
            
            flowL[conv_indxs[:,0],conv_indxs[:,1]] = (flowL[conv_indxs[:,0],conv_indxs[:,1]] | (s[conv_indxs[:,0],conv_indxs[:,1]] < 0)*1);
            flowR[conv_indxs[:,0],conv_indxs[:,1]] = (flowR[conv_indxs[:,0],conv_indxs[:,1]] | (s[conv_indxs[:,0],conv_indxs[:,1]] >= 0)*1);        
    
    #Approximate convective term with upwinded derivatives
        deriv = derivL*flowR + derivR*flowL;
        Deriv[:,:,i]=deriv
    
    # Deriv[:,:,0]=np.gradient(Y,axis=0)
    # Deriv[:,:,1]=np.gradient(Y,axis=1)
    
    mag = np.zeros((Nbx,Nby));
    for i in range(Dim):
        mag = mag + Deriv[:,:,i]**2;
    mag = np.maximum(np.sqrt(mag),1e-16*np.ones((Nbx,Nby)))

    delta = -S;
    
    for i in range(Dim):
        v = S*Deriv[:,:,i]/mag
        delta = delta + v*Deriv[:,:,i];
    
    # delta=np.nan_to_num(delta)
    
    apply_subcell_fix=True 
    robust_subcell=True
    if apply_subcell_fix==True:
        denom = np.zeros(np.shape(Yint));
        for d in range(Dim): 
            dx_inv = 1/dsigma
            if d==0:
                Y_p=np.roll(Yint,shift=-1,axis=d);Y_p[-1]=Y_p[-2]
                Y_m=np.roll(Yint,shift=1,axis=d);Y_m[0]=Y_m[1]
                diff2 = (0.5 * dx_inv * (Y_p- Y_m))**2;
            elif d==1:
                Y_p=np.roll(Yint,shift=-1,axis=d);Y_p[:,-1]=Y_p[:,-2]
                Y_m=np.roll(Yint,shift=1,axis=d);Y_m[:,0]=Y_m[:,1]
                diff2 = (0.5 * dx_inv * (Y_p- Y_m))**2;
            
            if robust_subcell==True:
                if d==0:
                    short_diff2 = (dx_inv * (Yint[1:,:] - Yint[:-1,:]))**2;
                    diff2[1:,:]=np.maximum(diff2[1:,:],short_diff2)
                    diff2[:-1,:]=np.maximum(diff2[:-1,:],short_diff2)
                    diff2 = np.maximum(diff2, robust_small_epsilon**2);
                elif d==1:
                    short_diff2 = (dx_inv * (Yint[:,1:] - Yint[:,:-1]))**2;
                    diff2[:,1:]=np.maximum(diff2[:,1:],short_diff2)
                    diff2[:,:-1]=np.maximum(diff2[:,:-1],short_diff2)
                    diff2 = np.maximum(diff2, robust_small_epsilon**2);            
            
            denom=denom+diff2
        
        denom = np.sqrt(denom);
        
        D = Yint / denom;
        
        # near = isNearInterface(Y0);
        near=boundary_dilation(Yint)
        # nan_indxs=np.argwhere(np.isnan(delta))
        # near[nan_indxs[:,0],nan_indxs[:,1]]=1
        
        delta = (delta*(np.logical_not(near).astype(int))+ (S*abs(Y) - D) / dsigma*near);
            
    
    # stepBoundInv = stepBoundInv + max(abs(v(:))) / grid.dx(i);
    ydot = -np.ndarray.flatten(delta)
    
    if check_efficiency==True:
        return ydot, mag
    else:
        return ydot



def smearedSign(data, sgnFactor):
    # s = data / np.sqrt(data**2 + sgnFactor)
    s = data / np.sqrt(data**2 + 1e-16)
    return s
