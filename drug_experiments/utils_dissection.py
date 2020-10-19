import numpy as np
import torch
import torch.nn.functional as F


"""
helper
"""
def get_max_corr_per_trace(out, traces):
    """
    out and traces in shape (n, tpts)
    """
    corr_all = np.zeros(traces.shape[0])
    ind_all = np.zeros(traces.shape[0])
    for tracenr in range(traces.shape[0]):
        corr = []
        for outnr in range(out.shape[0]):
            corr.append(scp.stats.pearsonr(traces[tracenr], out[outnr])[0])
        corr_all[tracenr] = np.max(corr)
        ind_all[tracenr] = np.argmax(corr)
    return corr_all, ind_all

def get_corr_train(fit,data):
    """
    pairwise correlation of traces _i_i_
    """
    if type(fit)== type(torch.zeros(1)):
        fit = np.array(fit.cpu().detach())
    #data = np.array(data.cpu().detach())
    n = data.shape[0]
    corr_pred = np.zeros(n)
    for i in range(n):
        corr_pred[i] = np.corrcoef(data[i],fit[i])[0,1]
    return corr_pred

def z_score(data):
    """
    only for 1d data
    """
    data_ = np.copy(data) - np.mean(data)
    data_ = data_ / np.std(data_)
    return data_


def final_transformation(track_release, iglusnfr_kernel, steady_state_steps=0):
        """
        final convolution with iGlusNFr kernel and affine transformation
        :param track_release:
        :return:
        """
        # convolve with iGluSNFR kernel
        # treat channel as batch dimension, because all get same iGlu kernel.
        # ToDo: change this when using batched inputs!
        x = track_release.T[:, None, :]
        x = F.conv1d(x,iglusnfr_kernel)
        x = x[:, 0, steady_state_steps:]  # CD

        # normalize (mean=0, norm=1) so correlation can be computed easily
        #x = x - torch.mean(x, dim=1, keepdim=True)
        #x = x / (torch.norm(x, 2, dim=1, keepdim=True) + 1e-10)
        return x

#######################

def get_tonic_releas(response):
    """
    response: chirp respones: (n, tpts)
    """
    y = np.array(response)
    baseline = np.mean(y[:,576:640], axis=1) #576
    y = (y.T - baseline).T
    
    y_neg = np.copy(y[:,640:1792])
    y_neg[y_neg>0] = 0
    
    y_pos = np.copy(y[:,640:1792])
    y_pos[y_pos<0] = 0
    
    neg_sum = np.sum(np.abs(y_neg), axis=1)
    pos_sum = np.sum(np.abs(y_pos), axis=1)
    

    return  neg_sum/(neg_sum+pos_sum)#, y_pos,y_neg
    
    
#######################
def get_max_activation(response, adaption_tpts=576, signal_period='ON-OFF'):
    """
    made specific test stimulus
    !! take care of baseline
    ---
    response.shape: (n,tpts)
    """
    response= np.array(response)
    baseline = np.mean(response[:,adaption_tpts-10:adaption_tpts], axis=1)
    
    max_vals = np.zeros((response.shape[0], 5))
    
    if signal_period=='OFF':
        shift=192
    elif signal_period=='ON':
        shift=0
    
    if signal_period=='OFF' or signal_period=='ON':
        for i in range(5):
            max_vals[:,i] = np.max(response[:, adaption_tpts+192*(i*2)+shift:adaption_tpts+192*(i*2+1)+shift],
                              axis=1) - baseline

    elif signal_period=='ON-OFF':
        for celltype in range(14):
            if celltype<5:
                shift =192
            else:
                shift=0
            for i in range(5):
                max_vals[celltype,i] = (np.max(response[celltype, adaption_tpts+192*(i*2)+shift
                                                       :adaption_tpts+192*(i*2+1)+shift])
                                        - baseline[celltype])



    return max_vals
