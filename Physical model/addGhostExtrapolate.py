import numpy as np

def getGhostExtrapolate(Y,dim,width):

    slopeMultiplier = +1;    
    Nx,Ny=np.shape(Y)
    
    if dim==0:
        Y_out = np.pad(Y, [(width, width), (0, 0)], mode='constant')
        slopTop=Y[0]-Y[1]
        slopTop = slopeMultiplier*abs(slopTop)*np.sign(Y[0])
        slopBottom=Y[-1]-Y[-2]
        slopBottom = slopeMultiplier*abs(slopBottom)*np.sign(Y[-1])
        for i in range(width):            
            Y_out[width-1+i]=Y_out[width-i]+slopTop
            Y_out[Nx+width+i]=Y_out[Nx-1+width+i]+slopBottom

    elif dim==1:
        Y_out = np.pad(Y, [(0, 0), (width, width)], mode='constant')
        slopTop=Y[:,0]-Y[:,1]
        slopTop = slopeMultiplier*abs(slopTop)*np.sign(Y[:,0])
        slopBottom=Y[:,-1]-Y[:,-2]
        slopBottom = slopeMultiplier*abs(slopBottom)*np.sign(Y[:,-1])
        for i in range(width):            
            Y_out[:,width-1+i]=Y_out[:,width-i]+slopTop
            Y_out[:,Ny+width+i]=Y_out[:,Ny-1+width+i]+slopBottom
    return Y_out

def getPeriodicExtrapolate(Y,dim,width):
    
    Nx,Ny=np.shape(Y)
    
    if dim==0:
        Y_out = np.pad(Y, [(width, width), (0, 0)], mode='constant')
        Y_out[0]=Y_out[-2]
        Y_out[-1]=Y_out[1]

    elif dim==1:
        Y_out = np.pad(Y, [(0, 0), (width, width)], mode='constant')
        Y_out[:,0]=Y_out[:,-2]
        Y_out[:,-1]=Y_out[:,1]
    return Y_out
    
    
