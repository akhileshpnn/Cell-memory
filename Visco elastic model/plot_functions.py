
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as animation
from skimage import feature
import os
import scipy
from colormaps import parula_map


folder_num=4
folder_save = 'C:\\Users\\nandan\\Desktop\\soma_temp\\Viscoelastic simulations\\20210806\\'+str(folder_num)+'\\'
# folder_save = 'X:\\Temp_Storage(June2020)\\Akhilesh\\ViscoElasticModel\\20210601\\'+str(folder_num)+'\\'
# folder_save= '\\\\billy.storage.mpi-dortmund.mpg.de\\abt2\\group\\agkoseska\\Temp_Storage(June2020)\\Akhilesh\\ViscoElasticModel\\20210618\\'+str(folder_num)+'\\'


def plot3D(Y,grid):
    
    fig, ax = plt.subplots()
    ax = plt.axes(projection='3d')
    ax.plot_surface(grid[0], grid[1], Y, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
    plt.show()
    
def plot3Dseq(Ys,grid,save=None):
    
    X=grid[:,:,0]
    Y=grid[:,:,1]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot = [ax.plot_surface(X, Y, Ys[0], color='0.75', rstride=1, cstride=1)]
    ax.set_zlim(-0.5,1)
    ani = animation.FuncAnimation(fig, update_plot, len(Ys), fargs=(grid, Ys, plot,ax), interval=1000/10)
    ani.save(os.path.join(folder_save,'reinitialization.mp4',writer='ffmpeg',fps=10))   

def update_plot(frame_number, grid, zarray, plot,ax):
    plot[0].remove()
    plot[0] = ax.plot_surface(grid[:,:,0], grid[:,:,1], zarray[frame_number], rstride=1, cstride=1,cmap='viridis', edgecolor='none')


def plot_potential2D(Y,lim,f,save=None):
    
    fig, ax = plt.subplots()
    ax.set_title(int(f*5)+0)
    ax.imshow(Y.T,extent=[0,lim,0,lim],origin='lower',cmap=parula_map)
#    ax.imshow(signal.T,extent=[0,lim,0,lim],origin='lower',cmap='gray')
    # ax.imshow(Y,extent=None,origin='lower',cmap='binary')
    if save==True:
         plt.savefig(os.path.join(folder_save,'bdry_image_'+str(f).zfill(5)+'.png'), dpi=300, quality='95', bbox_inches='tight', pad_inches=0)
    else:
         plt.show()
         
def plot_kymo(ep_kymo,tt):
    
    N,Tmax = np.shape(ep_kymo)    
    fig,ax = plt.subplots()
    im = ax.imshow(ep_kymo.T, extent=[1,N+1,len(ep_kymo.T),0],cmap=parula_map, aspect = 'auto',vmin=0,vmax=np.max(ep_kymo))
    ax.figure.colorbar(im)
    ax.set_ylabel('Time',fontsize=20)
    ax.set_xlabel('Plama Membrane bins',fontsize=20)
    ax.set_xticks(np.arange(1.5,N+1.5))
    ax.set_xticklabels(np.arange(1,N+1))
    
    for t in tt:
        plt.axhline(y=t,color='r')
    
    plt.show()





def plot_membrane_velocity(Vm,grid,membrane_mask,membrane_indxs,lim,f,save=None):
    
    [Vmx,Vmy]=Vm
    X=grid[0]
    Y=grid[1]
    
    fig, ax = plt.subplots()
    ax.set_title("velocity@boundary")
    # ax.imshow(membrane_mask,extent=[0,lim,0,lim],origin='upper',cmap='binary')
    ax.imshow(membrane_mask.T,extent=[0,lim,0,lim],origin='lower',cmap='binary')
    M = np.hypot(Vmx, Vmy)
    Q = ax.quiver(X[membrane_indxs], Y[membrane_indxs], Vmx[membrane_indxs], Vmy[membrane_indxs],M, units='xy',angles='xy', pivot='tail', width=1/len(membrane_mask))
    # qk = ax.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',coordinates='figure')
    fig.colorbar(Q)
    if save==True:
        plt.savefig(os.path.join(folder_save,'vel_image_'+str(f).zfill(5)+'.png'), dpi=300, quality='95', bbox_inches='tight', pad_inches=0)
    else:
        plt.show()


def plot_viscoelastic_direction(Lv,grid,membrane_mask,membrane_indxs,lim,f,save=None):
    
    [Lvx,Lvy]=Lv
    X=grid[0]
    Y=grid[1]
    
    fig, ax = plt.subplots()
    ax.set_title("visco@boundary")
    ax.imshow(membrane_mask.T,extent=[0,lim,0,lim],origin='lower',cmap='binary')
    #ax.set_xlim(-5,5);ax.set_ylim(-5,5)
    M = np.hypot(Lvx, Lvy)
    Q = ax.quiver(X[membrane_indxs], Y[membrane_indxs], Lvx[membrane_indxs], Lvy[membrane_indxs],M, units='xy',angles='xy', pivot='tail', width=1/len(membrane_mask))
    # qk = ax.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',coordinates='figure')
    fig.colorbar(Q)
    if save==True:
        plt.savefig(os.path.join(folder_save,'visco_image_'+str(f).zfill(5)+'.png'), dpi=300, quality='95', bbox_inches='tight', pad_inches=0)
    else:
        plt.show()


def plot_pressure_profile(ptot):
    [ptotx,ptoty]=ptot
    
    plt.figure()
    plt.plot(ptotx,'b-',label='x')
    plt.plot(ptoty,'r-',label='y')
    # plt.ylim(-0.1,0.1)
    plt.xlabel('membrane')
    plt.ylabel('pressure')
    plt.legend()
    plt.show()        

def plot_pip3(pip3):
    plt.figure()
    plt.plot(np.arange(0,len(pip3)),pip3,'k-',lw=2.0)
    plt.show()

def plot_normal(N,grid,membrane_mask,membrane_indxs,lim,f,save=None):
    
    [Nx,Ny]=N
    X=grid[0]
    Y=grid[1]
    
    fig, ax = plt.subplots()
    ax.set_title("Normal@boundary")
    ax.imshow(membrane_mask.T,extent=[0,lim,0,lim],origin='lower',cmap='binary')
    # ax.imshow(membrane_mask)
    M = np.hypot(Nx, Ny)
#    Q = ax.quiver(X[membrane_indxs], Y[membrane_indxs], Nx[membrane_indxs], Ny[membrane_indxs], M, units='xy',angles='xy', pivot='tail', width=1/len(membrane_mask), scale=1)
    Q = ax.quiver(X[membrane_indxs], Y[membrane_indxs], Nx[membrane_indxs], Ny[membrane_indxs], M,scale_units='xy',angles='xy',scale=1, pivot='tail', width=0.009,color='b')
    fig.colorbar(Q)
#    plt.colorbar(im)

    if save==True:
        plt.savefig(os.path.join(folder_save,'normal_image_'+str(f).zfill(5)+'.png'), dpi=300, quality='95', bbox_inches='tight', pad_inches=0)
    else:
        plt.show()

def plot_membrane_pressure(Pm,grid,membrane_mask,membrane_indxs,lim,f,save=None):
    
    [Pmx,Pmy]=Pm
    X=grid[0]
    Y=grid[1]
    
    fig, ax = plt.subplots()
    ax.set_title("pressure@boundary")
    # ax.imshow(membrane_mask,extent=[0,lim,0,lim],origin='upper',cmap='binary')
    ax.imshow(membrane_mask.T,extent=[0,lim,0,lim],origin='lower',cmap='binary')
    M = np.hypot(Pmx, Pmy)
    Q = ax.quiver(X[membrane_indxs], Y[membrane_indxs], Pmx[membrane_indxs], Pmy[membrane_indxs],M, units='xy',angles='xy', pivot='tail', width=1/len(membrane_mask))
    # qk = ax.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',coordinates='figure')
    fig.colorbar(Q)
    if save==True:
        plt.savefig(os.path.join(folder_save,'pres_image_'+str(f).zfill(5)+'.png'), dpi=300, quality='95', bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
        
#def plot_all_vectors(Pm,Lm,Vm,grid,membrane_mask,membrane_indxs,pip3,lim,f,save=None):
#    
#    X=grid[0]
#    Y=grid[1]
#    
#    [Pmx,Pmy]=Pm
#    [Lmx,Lmy]=Lm
#    [Vmx,Vmy]=Vm
#    
##    membrane_mask_plot=membrane_mask.copy()*np.nan
##    membrane_mask_plot[membrane_indxs]=pip3
#    
##    signal_mask = np.zeros(np.shape(membrane_mask))*np.nan
##    signal_mask[signal]=2
##    
#    fig, ax = plt.subplots()
#    # ax.set_title("vectors@boundary")
#    ax.set_title(int(f*10)+0)
#    ax.imshow(membrane_mask.T,extent=[0,lim,lim,0],origin='lower',cmap=parula_map)
##    ax.imshow(signal_mask.T,extent=[0,lim,0,lim],origin='lower',cmap='binary')
#    # ax.scatter(X[signal],Y[signal],c='yellow')
##    Q1=ax.quiver(X[membrane_indxs], Y[membrane_indxs], Pmx[membrane_indxs], Pmy[membrane_indxs],scale_units='xy',angles='xy',scale=1, pivot='tail', width=0.009,color='r')
##    Q2=ax.quiver(X[membrane_indxs], Y[membrane_indxs], Lmx[membrane_indxs], Lmy[membrane_indxs],scale_units='xy',angles='xy',scale=1, pivot='tail', width=0.009,color='g')
##    Q3=ax.quiver(X[membrane_indxs], Y[membrane_indxs], Vmx[membrane_indxs], Vmy[membrane_indxs],scale_units='xy',angles='xy',scale=1, pivot='tail', width=0.009,color='b')
#    
#    # Q1=ax.quiver(X[membrane_indxs], Y[membrane_indxs], Pmx[membrane_indxs], Pmy[membrane_indxs], pivot='tail',color='r')
#    # Q2=ax.quiver(X[membrane_indxs], Y[membrane_indxs], Lmx[membrane_indxs], Lmy[membrane_indxs], pivot='tail',color='g')
#    # Q3=ax.quiver(X[membrane_indxs], Y[membrane_indxs], Vmx[membrane_indxs], Vmy[membrane_indxs],pivot='tail',color='b')
#    
#    
#    # plt.quiverkey(Q1, 10, 10, .02, 'pressure', coordinates='data')
#    # plt.quiverkey(Q2, 10, 9, .02, 'visco', coordinates='data')
#    # plt.quiverkey(Q3, 10, 8, .2, 'velocity', coordinates='data')
##    ax.set_xlim(5,15)
##    ax.set_ylim(5,15)
#    
#    if save==True:
#        plt.savefig(os.path.join(folder_save,'vel_image_'+str(f).zfill(5)+'.png'), dpi=300, quality='95', bbox_inches='tight', pad_inches=0)
#    else:
#        plt.show()

def plot_all_vectors(Pm,Lm,Vm,grid,membrane_mask,membrane_indxs,signal_pos,lim,f):
    X=grid[0]
    Y=grid[1]
    
    [Pmx,Pmy]=Pm
    [Lmx,Lmy]=Lm
    [Vmx,Vmy]=Vm
    
    fig, ax = plt.subplots()
    # ax.set_title("vectors@boundary")
    ax.set_title(int(f*10)+0)
#    ax.imshow(membrane_mask,extent=[0,lim,0,lim],origin='upper',cmap=parula_map)
    ax.imshow(membrane_mask.T,extent=[0,lim,lim,0],cmap=parula_map)
    if not signal_pos is None:
        ax.scatter([signal_pos[0]],[signal_pos[1]])
    
#    Q1=ax.quiver(X[membrane_indxs], Y[membrane_indxs], Pmx[membrane_indxs], Pmy[membrane_indxs],scale_units='xy',angles='xy',scale=1, pivot='tail', width=0.009,color='r')
#    Q2=ax.quiver(X[membrane_indxs], Y[membrane_indxs], -Lmx[membrane_indxs], -Lmy[membrane_indxs],scale_units='xy',angles='xy',scale=1, pivot='tail', width=0.009,color='g')
    Q3=ax.quiver(X[membrane_indxs], Y[membrane_indxs], Vmx[membrane_indxs], Vmy[membrane_indxs],scale_units='xy',angles='xy',scale=1, pivot='tail', width=0.009,color='b')

    plt.show()
    
def plot_membrane(membrane_mask,lim,f):
    fig, ax = plt.subplots()
    # ax.set_title("vectors@boundary")
    ax.set_title(int(f*10)+0)
    im = ax.imshow(membrane_mask,extent=[0,lim,lim,0],cmap=parula_map,vmin=0,vmax=0.6)
    plt.colorbar(im)
    plt.show()

def save_cell_mask(cell_mask,f):

    np.save(os.path.join(folder_save,'cell_mask_'+str(f)+'.npy'),cell_mask)

def save_membrane_mask(membrane_mask,save_grid,f):
    np.save(os.path.join(folder_save,'membrane_mask_'+str(f)+'.npy'),membrane_mask)

def save_other_quants(grid,potential,membrane_indxs,pressure,visco,velocity,save_grid,f):
    
    if save_grid==True:
        np.save(os.path.join(folder_save,'grid_'+str(f)+'.npy'),grid)
    
    np.save(os.path.join(folder_save,'potential_'+str(f)+'.npy'),potential)
    np.save(os.path.join(folder_save,'membrane_indxs_'+str(f)+'.npy'),membrane_indxs)
    # np.save(os.path.join(folder_save,'pressure_'+str(f)+'.npy'),pressure)
    np.save(os.path.join(folder_save,'visco_'+str(f)+'.npy'),visco)
    # np.save(os.path.join(folder_save,'velocity_'+str(f)+'.npy'),velocity)

def save_signal_mask(signal_mask,f):
    np.save(os.path.join(folder_save,'signal_mask_'+str(f)+'.npy'),signal_mask)
    

        

def plot_vectorfield(V,grid,f,save=None):
    
    [Vx,Vy]=V
    X=grid[0]
    Y=grid[1]
    
    fig, ax = plt.subplots()
    ax.set_title("vector field")
    M = np.hypot(Vx, Vy)
    Q = ax.quiver(X, Y, Vx, Vy, M, units='x', pivot='tail', width=1/len(Vx), scale=1)
    fig.colorbar(Q)

    if save==True:
        plt.savefig(os.path.join(folder_save,'vector_field_'+str(f).zfill(5)+'.png'), dpi=300, quality='95', bbox_inches='tight', pad_inches=0)
    else:
        plt.show()

