import numpy as np
from scipy.spatial import distance



def curvarure(N,dsigma):
    [Nx, Ny]=N

    Nxx=np.gradient(Nx,dsigma,axis=0) 
    Nyy=np.gradient(Ny,dsigma,axis=1) 
    
    return Nxx+Nyy


def normal_vectors(Y,dsigma):
    
    U,V=partial_derivatives(Y,dsigma)
    
    mag=np.maximum(np.sqrt(U**2+V**2),1e-16*np.ones(np.shape(Y)))
    Nx=U/mag
    Ny=V/mag
    
    N=[Nx,Ny]
    
    return N

def partial_derivatives(Y,dsigma):
    
    Yx=np.gradient(Y,dsigma, axis=0) # x derivative
    Yy=np.gradient(Y,dsigma, axis=1) # y derivative
    
    return [Yx,Yy]
    
def boundary_dilation(Y0):
    from scipy import ndimage
    cell = Y0.copy()
    cell[cell<0]=True
    cell[(cell>0)*(cell!=1)]=False 
    dilation_index = 5
    cell_dilated = ndimage.grey_dilation(cell,structure=np.ones((dilation_index, dilation_index))) -1
    region_between = cell_dilated - cell
    return region_between


def set_cm(mask_cell):
    asd = np.roll(mask_cell, 1, axis=0)+np.roll(mask_cell, -1, axis=0)+np.roll(mask_cell, 1, axis=1)+np.roll(mask_cell, -1, axis=1)
    asd = asd + np.roll(np.roll(mask_cell, 1, axis=0),1,axis=1)+np.roll(np.roll(mask_cell, -1, axis=0),1,axis=1)+np.roll(np.roll(mask_cell, 1, axis=0),-1,axis=1)+np.roll(np.roll(mask_cell, -1, axis=0),-1,axis=1)
    asd[asd==8]=0
    asd[asd>0]=1
    asd[mask_cell==0]=0
    return asd

def masking_cell(Y):
    
    Ymask=np.zeros(np.shape(Y))
    threshold=0
    xy=np.where(Y<threshold)
    Ymask[xy]=1
    return Ymask

def mask_boundary(Y):
    boundary= np.zeros(np.shape(Y), dtype='float32')   
    mask_cell=masking_cell(Y)
    boundary=set_cm(mask_cell)
    return boundary



#https://stackoverflow.com/questions/58377015/counterclockwise-sorting-of-x-y-data

def sort_xy(membrane_indxs):
    
    membrane_indxs_sorted = np.zeros(np.shape(membrane_indxs))
    
    x= membrane_indxs[0]
    y= membrane_indxs[1]
    
    centroid=np.mean(membrane_indxs,axis=1).astype(int)
    
    x0 = centroid[0]
    y0 = centroid[1]

    r = np.sqrt((x-x0)**2 + (y-y0)**2)

    # angles = np.where((y-y0) > 0, np.arcsin((x-x0)/r), 2*np.pi-np.arcsin((x-x0)/r))
    angles = np.where((y-y0) > 0, np.arcsin((x-x0)/r), 2*np.pi-np.arcsin((x-x0)/r))

    mask = np.argsort(angles)

    x_sorted = x[mask]
    y_sorted = y[mask]
    
    membrane_indxs_sorted[0]=x_sorted
    membrane_indxs_sorted[1]=y_sorted
    membrane_indxs_sorted=tuple(membrane_indxs_sorted.astype(int))
    

    return membrane_indxs_sorted,centroid,angles

def membrane_indx_anticlockwise(membrane_annotated):
    membrane_indxs=np.where(~np.isnan(membrane_annotated))
    aa = np.argsort(membrane_annotated[membrane_indxs])
    membrane_indxs = np.array(np.transpose(membrane_indxs))
    membrane_indxs=membrane_indxs[aa]
    membrane_indxs = np.transpose(membrane_indxs)
    membrane_indxs=tuple(membrane_indxs.astype(int))
    return membrane_indxs


def activity_rotation(grid,membrane_indxs,pip3,signal_pos):
    mem_len=len(membrane_indxs[0])
    if signal_pos==None:
        pip3=np.zeros(mem_len)
    else:    
        Xm=grid[0][membrane_indxs]
        Ym=grid[1][membrane_indxs]
        coords=[signal_pos]
        mem_pos=np.zeros((mem_len,2))
        mem_pos[:,0]=Xm
        mem_pos[:,1]=Ym
        dist=distance.cdist(coords, mem_pos, 'euclidean')
        membrane_indxs = np.array(np.transpose(membrane_indxs))
        closest_px=np.argmin(dist)
        pip3_max_at=np.argmax(pip3)
        shift_by=pip3_max_at-closest_px
        pip3=np.roll(pip3,-shift_by)
    
    return pip3

def nearest_neighbour_extrapolation(X,membrane_indxs,Nb):
    
    [vmx,vmy,ptot] = X
    Vmx=np.zeros((Nb,Nb))*np.nan
    Vmy=np.zeros((Nb,Nb))*np.nan
    Ptot=np.zeros((Nb,Nb))*np.nan
    
    Vmx[membrane_indxs]=vmx  
    Vmy[membrane_indxs]=vmy    
    Ptot[membrane_indxs]=ptot    
    
    membrane_indxs = np.array(np.transpose(membrane_indxs))
    len_mem=len(vmx)    
    for i in range(Nb):
        for j in range(Nb):
            coords=[(i,j)]
            dist=distance.cdist(coords, membrane_indxs, 'euclidean')
            membrane_px=membrane_indxs[np.argmin(dist)]
            if np.isnan(Vmx[i,j]):                
                Vmx[i,j]=Vmx[int(membrane_px[0]),int(membrane_px[1])]
            if np.isnan(Vmy[i,j]):
                Vmy[i,j]=Vmy[int(membrane_px[0]),int(membrane_px[1])]
            if np.isnan(Ptot[i,j]):
                Ptot[i,j]=Ptot[int(membrane_px[0]),int(membrane_px[1])]
    return [Vmx,Vmy,Ptot]

def activity_overlaying(egfrp,pm_bins):
    membrane_overlayed = np.zeros(np.shape(pm_bins))*np.nan
    for i in range(len(egfrp)):
        membrane_overlayed[np.where(pm_bins==i)]=egfrp[i]
    return membrane_overlayed

# def nearest_neighbour_extrapolation(xm,membrane_indxs,Nb):
    
#     X=np.zeros((Nb,Nb))*np.nan
#     X[membrane_indxs]=xm    
#     membrane_indxs = np.array(np.transpose(membrane_indxs))
#     len_mem=len(xm)    
#     for i in range(Nb):
#         for j in range(Nb):
#             if np.isnan(X[i,j]):
#                 coords=[(i,j)]
#                 dist=distance.cdist(coords, membrane_indxs, 'euclidean')
#                 membrane_px=membrane_indxs[np.argmin(dist)]
#                 X[i,j]=X[int(membrane_px[0]),int(membrane_px[1])]
#     return X 

def assigning_receptor_activity(angles,R):
    ctrl_angles=np.linspace(-180,180,len(R))
    R_assigned=np.zeros(len(angles))   
    for i in range(1,len(ctrl_angles)):
        indxs=np.argwhere(np.logical_and(angles<ctrl_angles[i],angles>ctrl_angles[i-1])==True)
        R_assigned[indxs]=R[i]
    R_assigned[-1]=R_assigned[-2]
    return R_assigned


def signal_pos(activity,centroid,L):
    x0,y0=centroid
    
    max_of_activity_at=np.argwhere(activity==np.max(activity))
    if len(max_of_activity_at)>1:
        max_of_activity_at = max_of_activity_at[1][0]
    else:
        max_of_activity_at = max_of_activity_at[0]
    # center_of_mass=ndimage.measurements.center_of_mass(activity)       
    # max_of_activity_at=int(center_of_mass[0])
    theta = np.linspace(-np.pi,np.pi,len(activity))
    angle=theta[max_of_activity_at]
    
    # h1_sign=np.sign(np.sin(np.linspace(0,2*np.pi,len(activity))))
    # h2_sign=np.sign(np.cos(np.linspace(0,2*np.pi,len(activity))))
    # h1=int(h1_sign[max_of_activity_at]*L*np.sin(angle))
    # h2=int(h2_sign[max_of_activity_at]*L*np.cos(angle))
    quadrants = np.linspace(0,len(activity),5)
    if quadrants[0]<=max_of_activity_at<quadrants[1]:
        x_sign=-1;y_sign=1;
    elif quadrants[1]<=max_of_activity_at<quadrants[2]:
        x_sign=1;y_sign=1;
    elif quadrants[2]<=max_of_activity_at<quadrants[3]:
        x_sign=1;y_sign=-1;
    elif quadrants[3]<=max_of_activity_at<=quadrants[4]:
        x_sign=-1;y_sign=-1;
    
    h1=abs(int(L*np.sin(angle)))
    h2=abs(int(L*np.cos(angle)))
    
    signal_center=(abs(x0+x_sign*h2),abs(y0+y_sign*h1))
    
    return signal_center
        
