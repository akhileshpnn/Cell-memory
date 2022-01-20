import numpy as np
import matplotlib.pyplot as plt
from plot_functions import *
from scipy import signal
from scipy.spatial import distance
from scipy import ndimage
from utils_ import *
from pressure_functions import *
from termConvection import *
from termreinit import *
from membrane_index import *

Nb=200;dim=2;L=40

#defining grid
x=np.linspace(0,L,Nb);y=np.linspace(0,L,Nb)
dsigma=x[1]-x[0]
grid=np.meshgrid(x, y,indexing='ij')

center= [20, 7]  
radius = 2

def initialize_potential(grid,center,radius):    
    Y0 = np.zeros((Nb,Nb))
    for i in range(len(center)):
      Y0 = Y0 + (grid[i] - center[i])**2;
    Y0 = np.sqrt(Y0) - radius
    return Y0

def pressure_total(egfrp,A,mem_len,cur):
    
    pprot=pressure_pro(egfrp)
    pret=pressure_ret(egfrp)
    parea=pressure_area(A,A0,mem_len,egfrp)
    pten=pressure_tension(cur,radius)
    ptot=pprot+pret+parea-pten
    
    return [pprot,pret,parea,pten,ptot] 

def velocity_update(visco_l,ptot,Nv):
        
    kc=0.1 #nN/um3
    tauc=0.08 #nN/um3
    taua=0.1 #nNs/um3  
    
    [Nxm, Nym] = Nv
    
    [lx,ly]=visco_l
    ptotx=ptot*Nxm;ptoty=ptot*Nym
       
    vmx = -(kc/tauc)*lx + (1/tauc +1/taua)*ptotx
    vmy = -(kc/tauc)*ly + (1/tauc +1/taua)*ptoty
    
    return [vmx,vmy]   


def reinitialization(y, grid, dsigma, plot_reint=None,check_efficiency=None):
    
    NN=200  # |Phi|<=1.75 can be successfully reinitialized 
    Mod=[]
    Ys_reinit=[]
    sgnFactor=0#dsigma**2
    # a0=area(np.reshape(y, (Nb,Nb)))
    
    for i in range(NN):
        ydot,mag=reinit(y,grid,sgnFactor,dsigma,check_efficiency=True)
        y=y+0.001*ydot
        Mod.append(np.mean(mag))
        if plot_reint==True:
            Ys_reinit.append(np.reshape(y, (N,N)))
    # a=area(np.reshape(y, (Nb,Nb)))
    # print('area loss is '+str(a0-a))
    if plot_reint==True:
        plot3Dseq(Ys_reinit,grid)
    
    if check_efficiency==True:
        plt.figure()
        plt.plot(np.arange(1,NN+1),Mod,'k-')
        plt.axhline(y=1,ls='--',color='grey')
        plt.ylim(0,2)
        plt.xlim(0,NN)
        plt.ylabel('|Phi|')
        plt.xlabel('Iterations')
        plt.show()
    return y    


tF = 250
dt=0.01
t_eval = np.arange(0,tF,dt)
save = None


#initializing system
Y0=initialize_potential(grid,center,radius)
y0=np.ndarray.flatten(Y0)        
velocity0=np.zeros((Nb,Nb,dim))

L0x=np.zeros((Nb,Nb));L0y=np.zeros((Nb,Nb)) #viscoelastic state
l0x=np.ndarray.flatten(L0x);l0y=np.ndarray.flatten(L0y)  


y=y0.copy();Y=Y0.copy()
velocity=velocity0.copy()
lx=l0x.copy();ly=l0y.copy()

A0=area(Y0)

print(folder_save)

# loading kymograph
seed_int=180988 #Figure 1E
shift=None
print('seed '+str(seed_int))
egfrp_kymo = np.load(os.path.abspath(os.getcwd())+'\\Kymographs\\'+'Figure1D_Kymograph_Ep_seed('+str(seed_int)+').npy')

tt=[1000,7000]
if shift==True:
    egfrp_kymo = np.roll(egfrp_kymo,shift=-4,axis=0)

plot_kymo(egfrp_kymo,tt)

j=0
for i in range(1,len(t_eval)):   
    print(i,end='\r')
    
    Y=np.reshape(y,(Nb,Nb))    
    Lx=np.reshape(lx,(Nb,Nb));Ly=np.reshape(ly,(Nb,Nb))

    cell_mask=masking_cell(Y)    
    membrane_annotated,pm_bins = membrane_sorted(cell_mask)
    # first_index = np.argwhere(membrane_annotated==0)
    membrane_indxs=membrane_indx_anticlockwise(membrane_annotated)
    mem_len=len(membrane_indxs[0])
              
    egfrp=egfrp_kymo[:,i]
    mask_overlayed = activity_overlaying(egfrp,pm_bins)
    egfrp = mask_overlayed[membrane_indxs]
    
    Nv=normal_vectors(Y,dsigma)
    [Nvx,Nvy]=Nv
 
    A=area(Y)
    curvature_field=curvarure(Nv,dsigma)
    cur=curvature_field[membrane_indxs]
               
    [pprot,pret,parea,pten,ptot]  = pressure_total(egfrp,A,mem_len,cur)
        
    Nvxm, Nvym = Nvx[membrane_indxs], Nvy[membrane_indxs]
    visco_l = [Lx[membrane_indxs],Ly[membrane_indxs]]    
    vm=velocity_update(visco_l,ptot,[Nvxm, Nvym])
    [vmx,vmy]=vm
    
    [velocity[:,:,0],velocity[:,:,1],Ptot] = nearest_neighbour_extrapolation([vmx,vmy,ptot],membrane_indxs,Nb)
    
    Ptotx=Ptot*Nvx;Ptoty=Ptot*Nvy
    
    ydot = f_potential(y, grid, velocity, dsigma)
    y = y + dt * ydot
 
    lxdot = f_visco(lx,lx,Ptotx,grid,velocity,dsigma)
    lydot = f_visco(ly,ly,Ptoty,grid,velocity,dsigma)
    lx = lx + dt * lxdot
    ly = ly + dt * lydot
    
    y=reinitialization(y, grid, dsigma, check_efficiency=None)
    

    if i%1==0:  # frequency of plotting
        if i==10:
            save_grid=True
        else:
            save_grid=None
        
        pressure = np.zeros((Nb,Nb,dim))
        visco = np.zeros((Nb,Nb,dim))
        pressure[:,:,0]=Ptotx;pressure[:,:,1]=Ptoty
        visco[:,:,0]=Lx;visco[:,:,1]=Ly
        
        plot_membrane(mask_overlayed,lim=L,f=j)        
        
        # save_membrane_mask(mask_overlayed,save_grid,f=j)
        # save_other_quants(grid,Y,membrane_indxs,pressure,visco,velocity,save_grid,f=j)
        j=j+1
