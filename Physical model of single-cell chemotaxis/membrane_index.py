
from utils_ import *
import scipy

def pm_nuc(cell_mask):
    cmd = set_pixel_dist(cell_mask)
    ret = cmd
    ret[cell_mask == 0] = np.nan
    ret = ret/np.nanmax(ret)
    return ret

def set_pixel_dist(cell_mask):
    cm = set_cm(cell_mask)
    cmd = np.nan * np.ones(np.shape(cell_mask), dtype='float32')
    cell_xy = np.array(np.where(cell_mask == 1)).T
    cm_array = np.array(np.where(cm == 1)).T
    for x, y in cell_xy:
        cmd[x, y] = np.min(scipy.spatial.distance.cdist([[x, y]], cm_array, metric='euclidean'))
    return cmd

def sb_mask_equal_area(pn, n_bins=2):
    sb = np.nan * np.ones(np.shape(pn))
    ixs_cmd_px = np.where(~np.isnan(pn))
    ixs_sort = np.argsort(pn[ixs_cmd_px])
    ixs_sort = ixs_sort[len(ixs_sort) % n_bins:]
    ixs_split = np.split(ixs_sort, n_bins)
    for i in range(n_bins):
        ixs = np.array(ixs_cmd_px)[:, ixs_split[i]]
        sb[ixs[0], ixs[1]] = i
    return sb

def calc_angles_center(sb, pn):
    # indices of cell mask pixels
    pn_indices_xy = np.vstack(np.where(~np.isnan(pn)))
    cell_center = pn_indices_xy[:,np.argmax(pn[~np.isnan(pn)])]
    # median of full cell mask pixels
    # cell_center = np.int32(np.median(cell_indices_xy, 1))
    # cell_center = np.array(np.unravel_index(find(pn==np.nanmax(pn)), np.shape(pn))).T[0] # pixel furthest away from edges (cm)
    # indices of pixels within the first spatial bin (pm)
    pm_indices = np.vstack(np.where(sb == 0))  # np.unravel_index(find(sb == 0), np.shape(sb))
    # vectors from center to each pm pixel
    pm_center_coords = pm_indices - np.tile(cell_center.T, (np.shape(pm_indices)[1], 1)).T
    # angles from pm pixels towards cell center
    angle = np.arctan2(pm_center_coords[1][:], pm_center_coords[0][:])
    angles = np.nan * np.ones(np.shape(sb))
    # normalize from 0-1 starting from south, in counter-clockwise fashion
    angles[pm_indices[0][:], pm_indices[1][:]] = (angle + np.pi / 2) % (2 * np.pi)
    
    return angles, cell_center

def boarder_pixel(angles, xy):
    x = xy[0]; y = xy[1]
    max_x, max_y = np.subtract(np.shape(angles),(1,1))
    if x==0 or x==max_x or y==0 or y==max_y:
        return True
    else:
        return False

def calc_cm_rad_vals(pn, angles, cell_center):
        # coords of cm pixels (edges)
        cm_indices_xy = np.vstack(np.where(pn == 0))  # np.unravel_index(find(pn == 0), np.shape(pn))
        # cm pixel indices with angle==0
        cm_bottom_start_index = np.where(angles[cm_indices_xy[0][:], cm_indices_xy[1][:]] == 0)[0]
        if len(cm_bottom_start_index) > 1:
            # if more than 1 pixel is under zero angle, choose the one that is the closest to the cell center pixel
            cm_start_center_coords = cm_indices_xy[:,cm_bottom_start_index] - np.tile(cell_center.T, (len(cm_bottom_start_index), 1)).T
            dists = np.sqrt(np.sum(cm_start_center_coords**2, 0))
            cm_bottom_start_index = cm_bottom_start_index[np.argmin(dists)]
        else:
            cm_bottom_start_index = cm_bottom_start_index[0]
        # array of indices of cm pixels , giving unique 1D indices for each tuple in cm_indices_xy 
        indices = np.ravel_multi_index((cm_indices_xy[0][:], cm_indices_xy[1][:]), np.shape(pn))
        # 512x512 matrix where cm pixels are indexed from 1 to len(indices)
        indices_mat = -1*np.ones(np.shape(pn), dtype=int)
        indices_mat[cm_indices_xy[0][:], cm_indices_xy[1][:]] = np.arange(len(indices)) #np.ravel_multi_index((cm_indices_xy[0][:], cm_indices_xy[1][:]), np.shape(indices_mat))
        # 4xlen(indices) matrix of direct pixel neighbors
        neighs_mat_3 = [np.roll(indices_mat, 1, 0), np.roll(indices_mat, -1, 0), np.roll(indices_mat, 1, 1), np.roll(indices_mat, -1, 1)]
        neighs = np.reshape(neighs_mat_3, (4, len(pn.flat[:])))[:, indices]
        # 4xlen(indices) matrix of diagonal pixel neighbors
        neighs_mat_3_diag = [np.roll(np.roll(indices_mat, 1, 0), 1, 1), np.roll(np.roll(indices_mat, -1, 0), 1, 1) \
                               , np.roll(np.roll(indices_mat, 1, 0), -1, 1), np.roll(np.roll(indices_mat, -1, 0), -1, 1)]
        neighs_diag = np.reshape(neighs_mat_3_diag, (4, len(pn.flat[:])))[:, indices]

        order = -1*np.ones(np.shape(indices),dtype=int)
        prev_index = -1
        curr_index = cm_bottom_start_index # start from pixel at angle zero
        try:           
            for i in range(len(indices)):

                order[curr_index] = i
                ns = neighs[:, curr_index] # get neighboring pixels
                ns = ns[ns != -1]  # exclude missing neighboring pixels
                ns = ns[order[ns] == -1] # only consider unprocessed neighbors (no order id yet)
                
                if len(ns) == 0: # if no direct neighbors
                    ns = neighs_diag[:, curr_index]#consider diagonal neighbors
                    ns = ns[ns != -1]
                    ns = ns[order[ns] == -1]
                    
                    if boarder_pixel(angles,cm_indices_xy[:,curr_index]) is True and len(ns) == 0: # if pixel on boarder ignore and go back to previous and has no diagonal neighbours that isn't considered so far
                        curr_index = prev_index
                        continue
                    elif boarder_pixel(angles,cm_indices_xy[:,curr_index]) is True and len(ns) != 0: # if pixel on boarder but has a neighbour to consider
                        ns = ns
                    elif boarder_pixel(angles,cm_indices_xy[:,curr_index]) is False and len(ns)==0:  # if no option break the loop and it shows leftouts
                        curr_index = prev_index
                        continue
                if len(ns) == 1: # if only one neighbor, choose it for next
                    next_index = ns[0]
                else: # if more than one neighbor, choose the one on the right-hand side in the direction of 'movement'
                    if i == 0: # for the first pixel
                        dir_angle = [0,1] # come from the top, except if the starting pixel has three direct neighbors (there is no pixel in the direction towards the cell center)
                        # then come from the left if there is a pixel on the top-left (if not, the cm folloring will start in the opposite direction)
                        
                        if len(ns) == 3 and np.isnan(pn[cm_indices_xy[0, curr_index]-1,cm_indices_xy[1, curr_index]-1]):
                            dir_angle = [1,0]
                    else:
                        # get 'direction' from previous pixel
                        # dir_angle = cm_indices_xy[:, curr_index]-cm_indices_xy[:, prev_index]
                        dir_angle = cm_indices_xy[:, prev_index]-cm_indices_xy[:, curr_index]
                    # calculate angles between current and neighboring pixels
                    potential_angles = np.vstack((cm_indices_xy[0, ns]-cm_indices_xy[0, curr_index],
                                                  cm_indices_xy[1, ns]-cm_indices_xy[1, curr_index])).T
                    # cross product with 'direction' vector to find the most right-hand side pixel
                    next_index = ns[np.argmin(np.cross(dir_angle, potential_angles))]
                prev_index = curr_index
                curr_index = next_index
        except:
            print('problem')
        else:
            # if not all pixels are processed, there is probably some loop that was left-out, usually it is fine
            if i < len(indices)-1:
                print('left-outs')

        cm_rad_vals = np.nan * np.ones(np.shape(pn))
        cm_rad_vals.flat[indices] = order
        cm_rad_vals[cm_rad_vals == -1] = np.nan

        return cm_rad_vals
def calc_pm_rad_vals(sb, cm_rad_vals):
        pm_rad_vals = np.nan * np.ones(np.shape(sb), dtype='float32')
        pm_xy = np.array(np.where(sb >= 0)).T
        cm_rad_array = np.array(np.where(cm_rad_vals >= 0)).T
        for x, y in pm_xy:
            ind = np.argmin(scipy.spatial.distance.cdist([[x, y]], cm_rad_array, metric='euclidean'))
            i1, i2 = cm_rad_array[ind]
            pm_rad_vals[x, y] = cm_rad_vals[i1, i2]  # choose the cm_rad_val from the closest cm pixel
        pm_rad_vals = pm_rad_vals / np.nanmax(pm_rad_vals)
        pm_rad_vals[sb != 0] = np.nan

        return pm_rad_vals

def set_bin_edges(cm_rad_vals, cell_center, n_bins=20, pin_angles=False):
    if pin_angles:
        # if mod(n_bins,4)!=0:
        #     print('n_bins not divisble by 4!')
        #     return []
        cm_indices_xy = np.vstack(np.where(~np.isnan(cm_rad_vals)))
        indices = np.ravel_multi_index((cm_indices_xy[0][:], cm_indices_xy[1][:]), np.shape(cm_rad_vals))
        pi_half = np.where((cm_indices_xy[1,:] == cell_center[1]) * (cm_indices_xy[0,:] >= cell_center[0]))
        index1 = indices[pi_half][np.argmin(np.abs(cm_rad_vals.flat[indices[pi_half]] - np.nanmax(cm_rad_vals) / 4))]
        ls1 = np.linspace(0, cm_rad_vals.flat[index1], 6)
        pi = np.where((cm_indices_xy[0, :] == cell_center[0]) * (cm_indices_xy[1, :] >= cell_center[1]))
        index2 = indices[pi][np.argmin(np.abs(cm_rad_vals.flat[indices[pi]] - np.nanmax(cm_rad_vals) / 2))]
        ls2 = np.linspace(cm_rad_vals.flat[index1], cm_rad_vals.flat[index2], 6)[1:]
        pi_3halves = np.where((cm_indices_xy[1, :] == cell_center[1]) * (cm_indices_xy[0, :] < cell_center[0]))
        index3 = indices[pi_3halves][np.argmin(np.abs(cm_rad_vals.flat[indices[pi_3halves]] - np.nanmax(cm_rad_vals) * 3/4))]
        ls3 = np.linspace(cm_rad_vals.flat[index2], cm_rad_vals.flat[index3], 6)[1:]
        ls4 = np.linspace(cm_rad_vals.flat[index3], np.nanmax(cm_rad_vals), 6)[1:]
        # return np.hstack((ls1, ls2, ls3, ls4))/np.nanmax(cm_rad_vals)
        return np.hstack((ls1, ls2, ls3, ls4))
    else:
        return np.linspace(0, 1, n_bins+1)

def membrane_sorted(cell_mask):
    n_radial_bins = 20
    pn = pm_nuc(cell_mask)
    sb = sb_mask_equal_area(pn)
    angles, cell_center = calc_angles_center(sb, pn)
    cm_rad_vals = calc_cm_rad_vals(pn, angles, cell_center)
    # pm_rad_vals = calc_pm_rad_vals(sb, cm_rad_vals)
    bin_edges = set_bin_edges(cm_rad_vals, cell_center, n_radial_bins, pin_angles=True)
    bins = np.nan * np.zeros(np.shape(angles))
    for i in range(n_radial_bins):
        bins[(cm_rad_vals>=bin_edges[i]) * (cm_rad_vals<=bin_edges[i+1])] = i
    return cm_rad_vals,bins
    
