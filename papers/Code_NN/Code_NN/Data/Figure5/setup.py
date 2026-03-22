import torch
import numpy as np
import h5py
import time

#simulate only the PMNs
def forwardpass_onlyp(up,dt,taup,gp,bp,wsp,Jpp):
    #outputs
    ups = []

    for t in range(T):
        up = (1 - dt/taup) * up + (dt/taup) * (gp * torch.matmul(torch.nn.functional.softplus(up), Jpp) + bp + torch.matmul(s[t,:,:], wsp))
        ups.append(up)

    #stack time dimension
    ups = torch.stack(ups)

    #compute output activations
    p = torch.nn.functional.softplus(ups)

    return p

#simulate PMNs and MNs (for training teacher)
def forwardpass(um,up,dt,taum,taup,gm,gp,bm,bp,wsp,Jpm,Jpp):
    #outputs
    ums = []
    ups = []

    #forward pass through time
    for t in range(T):
        # Neural network forward pass for one time step
        um = (1 - dt/taum) * um + (dt/taum) * (gm * torch.matmul(torch.nn.functional.softplus(up), Jpm) + bm)
        up = (1 - dt/taup) * up + (dt/taup) * (gp * torch.matmul(torch.nn.functional.softplus(up), Jpp) + bp + torch.matmul(s[t,:,:], wsp))
        ums.append(um)
        ups.append(up)

    #stack time dimension
    ums = torch.stack(ums)
    ups = torch.stack(ups)

    #compute output activations
    m = torch.nn.functional.softplus(ums)
    p = torch.nn.functional.softplus(ups)

    return m,p

#correlation between two activity traces
def calccorr(p1,p2,inds): 
    Ninds = len(inds)
    corrs = np.zeros([Ninds,B])

    p1np = p1.detach().numpy()
    p2np = p2.detach().numpy()

    for bi in range(B):
        for ii in range(Ninds):
            ind = inds[ii]
            corrs[ii,bi] = np.corrcoef(p1np[:,bi,ind],p2np[:,bi,ind])[0,1]

    return np.mean(corrs)

#mse between two activity traces
def calcmse(p1,p2,inds):
    return torch.mean(torch.pow(p1[:,:,inds]-p2[:,:,inds],2))



def loadconns():
    f = h5py.File("Data/Figure5/data.h5","r")
    #after loading, rows are postsynaptic and columns are presynaptic
    Jpm = ((f["Jpm"][:]).T).astype(np.float32)
    Jpp = ((f["Jpp"][:]).T).astype(np.float32)

    pnames = f["p"][:]
    pnames_orig = f["p_orig"][:]
    mnames = f["m"][:]
    types = f["nt"][:]
    mnorder = f["mnorder"][:]
    f.close()

    return Jpm,Jpp,pnames,pnames_orig,mnames,types,mnorder

def genpulse(Tstop,dt,pstart,pend,Trise):
    T = int(Tstop/dt)

    p = np.zeros(T)

    istart = int(pstart/dt)
    iend = int(pend/dt)+1
    irise = int(Trise/dt)

    p[istart:iend] = 1.
    p[istart:(istart+irise)] = np.sin(np.pi*np.arange(irise)/(2.*irise))
    p[(iend-irise):iend] = np.flipud(np.sin(np.pi*np.arange(irise)/(2.*irise)))
    return p

def gentargets(mnorder,B,S,dt):
    M = mnorder.shape[1]

    Tstop = 6
    Tpulserise = 1. #pulse length
    Tpulse = 2.
    dtpulse = 0.25 #time between start of one pulse and the next
    dtpulse_end = 0.125 #time between start of one pulse and the next
    tstart = 1.
    segdelay = 1.
    T = int(Tstop/dt)

    Npulse = np.max(mnorder)

    pstarts = np.zeros([2,M])
    pends = np.zeros([2,M])

    for mi in range(M):
        for bi in range(B):
            segoffset = 0
            if (bi == 1) and (mi >= int(M/2)): #A2 MNs fire later during backward
                segoffset = segdelay
            elif (bi == 0) and (mi < int(M/2)): #A1 MNs fire later during forward
                segoffset = segdelay

            pstarts[bi,mi] = tstart + (mnorder[bi,mi]-1)*dtpulse + segoffset
            pends[bi,mi] = tstart + Tpulse + (mnorder[bi,mi]-1)*dtpulse_end + segoffset

    mtarg = np.zeros([T,B,M])
    s = np.zeros([T,B,S],dtype=np.float32)

    for mi in range(M):
        for bi in range(B):
            if mnorder[bi,mi] > 0:
                mtarg[:,bi,mi] = genpulse(Tstop,dt,pstarts[bi,mi],pends[bi,mi],(pends[bi,mi]-pstarts[bi,mi])/2.)
    
    
    seg2inds = np.where(np.sum(mtarg[:,0,0:int(M/2)],1)>0)[0]
    seg1inds = np.where(np.sum(mtarg[:,0,int(M/2):M],1)>0)[0]

    square_pulse = np.zeros(T,dtype=np.float32)
    istart = int(tstart/dt)
    iend = np.max(seg2inds)
    square_pulse[istart:iend] = 1.

    for bi in range(int(np.ceil(B/2))):
        s[:,bi,0] = square_pulse

    for bi in range(int(np.ceil(B/2)),B):
        s[:,bi,1] = square_pulse

    return mtarg,s,seg1inds,seg2inds

def initJpp(Jpp0,types):
    J = np.zeros(Jpp0.shape,dtype=np.float32)
    N = J.shape[0]
    N2 = J.shape[1]
    J = np.copy(Jpp0)

    for qi in range(N):
        if 'inh' in str(types[qi]):
            J[qi,:] = -J[qi,:]
        elif 'unknown' in str(types[qi]):
            J[qi,:] = -J[qi,:]
            #J[qi,:] = np.random.choice([-1,1])*J[qi,:]
    #for qi in range(N2):
    #    wpos = J[:,qi] * (J[:,qi]>0)
    #    wneg = J[:,qi] * (J[:,qi]<0)
    #    if np.sum(wpos) > 0:
    #        wpos = wpos/np.sum(wpos)
    #    if np.sum(np.abs(wneg)) > 0:
    #        wneg = wneg/np.sum(np.abs(wneg))
    #    J[:,qi] = 1*wpos + 1*wneg
    return J

def plotprogress(track_loss,count,m,p,mtarg):
    plt.clf()
    plt.subplot2grid((4,2),(0,0),colspan=2)
    plt.semilogy(track_loss[0:count,:])
    plt.ylim(0.1,1.1*np.max(track_loss))
    plt.xlim(0,100)
    plt.ylabel("loss")

    plt.subplot(423)
    plt.imshow(mtarg.detach().numpy()[:,0,:].T)
    plt.colorbar()
    plt.ylabel("MN targets")
    plt.title("FWD")

    plt.subplot(424)
    if B > 1:
        plt.imshow(mtarg.detach().numpy()[:,1,:].T)
        plt.colorbar()
    plt.title("BWD")

    plt.subplot(425)
    plt.imshow(m.detach().numpy()[:,0,:].T)
    plt.colorbar()
    plt.ylabel("MNs")

    plt.subplot(426)
    if B > 1:
        plt.imshow(m.detach().numpy()[:,1,:].T)
        plt.colorbar()

    plt.subplot(427)
    plt.imshow(p.detach().numpy()[:,0,:].T)
    plt.ylabel("PMNs")
    plt.colorbar()
    plt.xlabel("timestep")

    plt.subplot(428)
    if B > 1:
        plt.imshow(p.detach().numpy()[:,1,:].T)
        plt.colorbar()
    plt.xlabel("timestep")

    plt.pause(.0001)
    plt.show()
    plt.tight_layout()

#load connections 
Jpm0, Jpp0, pnames, pnames_orig, mnames, types, mnorder = loadconns()

M = len(mnames)  # number of MNs
N = len(pnames)  # number of PMNs
S = 2
dt = 0.05

Ncycles = 1
dt = 0.05
tau0 = 0.2

#generate targets
mtarg, s, seg1inds, seg2inds = gentargets(mnorder, B, S, dt)
s = torch.tensor(s)
mtarg = torch.tensor(mtarg)
T = mtarg.shape[0]

taum = tau0
taup = tau0

lam = 1.
