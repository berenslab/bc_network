import numpy as np


'''
"""
load weights 100 best models
"""
h5f = h5py.File('weights_100_best.h5','r')
w_100 = h5f['weights_100_best'][:]
loss_100 = h5f['loss_100_best'][:]
h5f.close()

# get w of best model
w_best = w_100[np.argmax(loss_100)]
print('best_index:',np.argmax(loss_100))
'''


def joining_ws(w_acl_bc,
                   w_bc_acl,
                   w_acg_bc,
                   w_bc_acg,
                   w_acl_acg):
    """
    put all weights in one matrix
    ---
    # BC, acl, acg [14,10,35]
    in order: w_acl_bc, w_bc_acl, w_acg_bc, w_bc_acg, w_acl_acg
    """
    w = np.zeros((59,59))
    # acl_bc
    w[:14,14:24] = w_acl_bc
    # bc_acl
    w[14:24,:14] = w_bc_acl
    # bc_acg
    w[24:,:14] = w_bc_acg
    # acg_bc
    w[:14,24:] = w_acg_bc
    # acl_acg
    w[24:,14:24] = w_acl_acg 

    return w

def un_joining_w(w_in):
    """
    inverse of joining_ws
    --- 
    returns: w_acl_bc, w_bc_acl, w_acg_bc, w_bc_acg, w_acl_acg
    """
    # acl_bc
    w_acl_bc = w_in[:14,14:24] 
    # bc_acl
    w_bc_acl = w_in[14:24,:14] 

    # bc_acg
    w_bc_acg = w_in[24:,:14] 
    # acg_bc
    w_acg_bc = w_in[:14,24:] 

    # acl_acg
    w_acl_acg = w_in[24:,14:24] 
    
    return w_acl_bc, w_bc_acl, w_acg_bc, w_bc_acg, w_acl_acg
    

def sorting(w_in, return_types=False, return_types_only=False):
    """
    sorting by OFF/ON ratio
    ---
    type 0 = off
    type 1 = on
    return: w (or: (w,) ac_l_type, ac_g_type)
    """
    # unjoin 
    w_acl_bc, w_bc_acl, w_acg_bc, w_bc_acg, w_acl_acg = un_joining_w(w_in)
    
    # sort ACs by input
    
    # LOCAL
    # sorting by OFF/ON BC (on bc->ac connection)
    sorting_acl =  np.argsort(w_bc_acl[:,:5].sum(axis=1) / w_bc_acl[:,5:].sum(axis=1))[::-1]
    # get AC types ('OFF' and 'ON' type)
    ac_l_type = np.array((w_bc_acl[:,:5].sum(axis=1) < w_bc_acl[:,5:].sum(axis=1))*1, dtype=np.float)
    #ac_l_type[ac_l_type==0]=np.nan

    #GLOBAL
    # sorting by OFF/ON BC (on bc->ac connection)
    sorting_acg =  np.argsort(w_bc_acg[:,:5].sum(axis=1) / w_bc_acg[:,5:].sum(axis=1))[::-1]
    # get AC types ('OFF' and 'ON' type)
    ac_g_type = np.array((w_bc_acg[:,:5].sum(axis=1) < w_bc_acg[:,5:].sum(axis=1))*1, dtype=np.float)
    #ac_g_type[ac_g_type==0]=np.nan

    # JOIN 
    w = joining_ws(w_acl_bc[:,sorting_acl],
                   w_bc_acl[sorting_acl],
                   w_acg_bc[:,sorting_acg],
                   w_bc_acg[sorting_acg],
                   w_acl_acg[sorting_acg][:,sorting_acl])
    
    if return_types:
        return w, ac_l_type[sorting_acl], ac_g_type[sorting_acg]
    elif return_types_only:
        return ac_l_type[sorting_acl], ac_g_type[sorting_acg]
    else:
        return w
    
def norm_full_w(w):
    """
    assumed ordering and shape:  BC, acl, acg [14,10,35]
    ---
    normalizing by 'input to one celltype' (= by rows)
    """
    
    #w_norm[14:24,:14] = (w[14:24,:14].T / w[14:24,:14].sum(axis=1)).T
    
    w_norm = np.zeros((59,59))

    # acl_bc
    w_norm[:14,14:24] = (w[:14,14:24].T/w[:14,14:24].sum(axis=1)).T
    # bc_acl
    w_norm[14:24,:14] = (w[14:24,:14].T / w[14:24,:14].sum(axis=1)).T
    
    # acg_bc
    w_norm[:14,24:] = (w[:14,24:].T / w[:14,24:].sum(axis=1)).T
    # bc_acg
    w_norm[24:,:14] = (w[24:,:14].T/ w[24:,:14].sum(axis=1)).T

    # acl_acg
    w_norm[24:,14:24] = (w[24:,14:24].T/w[24:,14:24].sum(axis=1)).T
    
    return w_norm

def get_corr(w_norm1,w_norm2):
    """
    get correlation of non-zeros elements of two matrices
    (assuming same number of zero el)
    """
    assert np.sum(w_norm1==0) == np.sum(w_norm2==0)
    
    corr = np.corrcoef(w_norm1[w_norm1>0].flatten(),w_norm2[w_norm2>0].flatten() )[0,1]
    
    return corr

def collapse_on_off(w_in):
    """
    collapsing 59 x 59 connectivity matrix to 6 x 6 ON and OFF  
    """
    
    w_temp = np.zeros((59,6))
    w_sort,ac_l_type, ac_g_type = sorting(w_in, return_types=True)
    bc_types=np.ones(14)
    bc_types[:5] = 0 
    
    mask = bc_types==0
    w_temp[:,0] = w_sort[:,0:14][:,mask].sum(axis=1)
    mask = bc_types==1
    w_temp[:,1] = w_sort[:,0:14][:,mask].sum(axis=1)
    
    mask = ac_l_type==0
    w_temp[:,2] = w_sort[:,14:24][:,mask].sum(axis=1)
    mask = ac_l_type==1
    w_temp[:,3] = w_sort[:,14:24][:,mask].sum(axis=1)
    
    mask = ac_g_type==0
    w_temp[:,4] = w_sort[:,24:][:,mask].sum(axis=1)
    mask = ac_g_type==1
    w_temp[:,5] = w_sort[:,24:][:,mask].sum(axis=1)
    
    w = np.zeros((6,6))
    mask = bc_types==0
    w[0] = w_temp[0:14][mask].sum(axis=0)
    mask = bc_types==1
    w[1] = w_temp[0:14][mask].sum(axis=0)
    
    mask = ac_l_type==0
    w[2] = w_temp[14:24][mask].sum(axis=0)
    mask = ac_l_type==1
    w[3] = w_temp[14:24][mask].sum(axis=0)
    
    mask = ac_g_type==0
    w[4] = w_temp[24:][mask].sum(axis=0)
    mask = ac_g_type==1
    w[5] = w_temp[24:][mask].sum(axis=0)
    
    return w

def get_onoff_ratios(w):
    """
    w: norm_sorted connectivity
    ---
    returns: ratios for each AC
    first local than global (10 local, 35 global)
    """
    w_acl_bc, w_bc_acl, w_acg_bc, w_bc_acg, w_acl_acg = un_joining_w(w)
    off_on_ratio_l = w_bc_acl[:,0:5].sum(axis = -1)/w_bc_acl[:,5:].sum(axis = -1)
    off_on_ratio_g = w_bc_acg[:,0:5].sum(axis = -1)/w_bc_acg[:,5:].sum(axis = -1)
    ratios_model = np.concatenate((off_on_ratio_l, off_on_ratio_g))
    
    return np.array(ratios_model)