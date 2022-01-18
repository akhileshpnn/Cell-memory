from addGhostExtrapolate import *

def upwindFirstFirst(data, dim, dx):
    Nx,Ny=np.shape(data)
    dxInv = 1 / dx;

    #Add ghost cells.
    # gdata = getGhostExtrapolate(data,dim,width=1)
    gdata = getPeriodicExtrapolate(data,dim,width=1)

    
    if dim==0:
        deriv = dxInv * (gdata[1:,:] - gdata[:-1,:]);
        derivL = deriv[:-1,:];
        derivR = deriv[1:,:];
    elif dim==1:
        deriv = dxInv * (gdata[:,1:] - gdata[:,:-1]);
        derivL = deriv[:,:-1];
        derivR = deriv[:,1:];
        
    return [derivL,derivR]

def upwindFirstENO2(data, dim,dx):
    Nx,Ny=np.shape(data)

    dxInv = 1 / dx;

    #Add ghost cells.
    gdata = getGhostExtrapolate(data,dim,width=2)  
    
    if dim==0:
        D1 = dxInv * (gdata[1:,:] - gdata[:-1,:]);
        D2 = 0.5*dxInv * (D1[1:,:] - D1[:-1,:]);
        D1 = D1[1:-1,:]
        
        dL = np.zeros((Nx,Ny,2));
        dR = np.zeros((Nx,Ny,2));
        
        for i in range(2):
            dL[:,:,i]=D1[:-1,:]
            dR[:,:,i]=D1[1:,:]
        
        dL[:,:,0]=dL[:,:,0]+dx*D2[:-2,:]
        dL[:,:,1]=dL[:,:,1]+dx*D2[2:,:]
        
        dR[:,:,0]=dR[:,:,0]-dx*D2[1:-1,:]
        dR[:,:,1]=dR[:,:,1]-dx*D2[2:,:]
        
        
        D2abs=abs(D2)
        smallerL = (D2abs[:-1,:]) < (D2abs[1:,:]);
        smallerR = (D2abs[:-1,:]) > (D2abs[1:,:]);
    
        derivL = dL[:,:,0]*smallerL[:-1,:] + dL[:,:,1]*smallerR[:-1,:];
        derivR = dR[:,:,0]*smallerL[1:,:] + dR[:,:,1]*smallerR[1:,:];
    
    elif dim==1:
        D1 = dxInv * (gdata[:,1:] - gdata[:,:-1]);
        D2 = 0.5*dxInv * (D1[:,1:] - D1[:,:-1]);
        D1 = D1[:,1:-1]
        
        dL = np.zeros((Nx,Ny,2));
        dR = np.zeros((Nx,Ny,2));
        
        for i in range(2):
            dL[:,:,i]=D1[:,:-1]
            dR[:,:,i]=D1[:,1:]
        
        dL[:,:,0]=dL[:,:,0]+dx*D2[:,:-2]
        dL[:,:,1]=dL[:,:,1]+dx*D2[:,2:]
        
        dR[:,:,0]=dR[:,:,0]-dx*D2[:,1:-1]
        dR[:,:,1]=dR[:,:,1]-dx*D2[:,2:]
        
        
        D2abs=abs(D2)
        smallerL = (D2abs[:,:-1]) < (D2abs[:,1:]);
        smallerR = (D2abs[:,:-1]) > (D2abs[:,1:]);
    
        derivL = dL[:,:,0]*smallerL[:,:-1] + dL[:,:,1]*smallerR[:,:-1];
        derivR = dR[:,:,0]*smallerL[:,1:] + dR[:,:,1]*smallerR[:,1:];
        
    return [derivL,derivR]
        
        
        
