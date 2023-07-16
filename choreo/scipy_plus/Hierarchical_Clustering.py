
import numpy as np
import scipy
import bisect
import functools
    
def PolyCost(data,imin,imax,degpolyfit=1):

    p, residuals, rank, singular_values, rcond = np.polyfit(np.array(range(imin,imax)), data[imin:imax,:], degpolyfit, rcond=None, full=True)
    cost = sum(residuals)
        
    return cost

def FourierCost(data,imin,imax,degFourierFit=1):

    nvals = imax - imin
    ncoeffs_tot = nvals // 2 + 1

    if (ncoeffs_tot <= degFourierFit +1):
        return 0.
    
    else:
        
        coeffs = scipy.fft.rfft(data[imin:imax,:],axis=0)

        residuals_F = 2*np.linalg.norm(coeffs[(degFourierFit+1):,:])**2
        if (nvals % 2 == 0):
            residuals_F -= np.linalg.norm(coeffs[ncoeffs_tot-1,:])**2
        
        residuals_F /= nvals

        return residuals_F

        # # To check :
        # recons_data =  scipy.fft.irfft(coeffs[:(degFourierFit+1),:],axis=0,n=nvals)
        # residuals_L = np.linalg.norm(recons_data-data[imin:imax,:])**2
        # print("")
        # print((nvals % 2 == 0))
        # print(residuals_F / residuals_L)
        # return residuals_L



def FindNextFuse(data,cuts,costs,allfusecosts):
    
    min_cost = 1e100

    for i in range(len(allfusecosts)):

        if (allfusecosts[i] < min_cost):
            min_cost = allfusecosts[i]
            min_fuse = i+1 # The item to remove
            
    return min_fuse,min_cost

def FindAllFuses(meas_data,ydata,OneRangeCost):
    
    yticks = ydata.copy()
    
    cuts = list(range(meas_data.shape[0]+1))
    costs = [0 for i in range(meas_data.shape[0])]
    
    allfusecosts = [OneRangeCost(meas_data,cuts[i],cuts[i+2]) for i in range(meas_data.shape[0]-1)]
    
    Hcosts = [0 for i in range(meas_data.shape[0])]
    Hcuts = []
    Hticks = []
    Hmin = []
    Hmax = []
    
    HNcur = list(range(meas_data.shape[0]))
    HNmin = []
    HNmax = []
    
    cost_sum = 0
    
    for i in range(meas_data.shape[0]-1):

        min_fuse,min_cost = FindNextFuse(meas_data,cuts,costs,allfusecosts)

        HNmin.append(HNcur[min_fuse-1])
        HNmax.append(HNcur[min_fuse])
        
        HNcur[min_fuse-1] = meas_data.shape[0] + i
        HNcur.pop(min_fuse)

        Hmin.append(yticks[min_fuse-1])
        Hmax.append(yticks[min_fuse])
        

        costs[min_fuse-1] = min_cost + costs[min_fuse-1] + costs[min_fuse]

        yticks[min_fuse-1] = (yticks[min_fuse-1] + yticks[min_fuse])/2

        cost_sum += min_cost

        chosen_cost = costs.pop(min_fuse)
        chosen_cut = cuts.pop(min_fuse)
        chosen_tick = yticks.pop(min_fuse)

        allfusecosts.pop(min_fuse-1)
        
        if (min_fuse > 1):
            allfusecosts[min_fuse-2] = OneRangeCost(meas_data,cuts[min_fuse-2],cuts[min_fuse]) - costs[min_fuse-2] - costs[min_fuse-1]
        if (min_fuse <= (len(allfusecosts))):
            allfusecosts[min_fuse-1] = OneRangeCost(meas_data,cuts[min_fuse-1],cuts[min_fuse+1]) - costs[min_fuse-1] - costs[min_fuse]

        Hcuts.append(chosen_cut)
        Hcosts.append(cost_sum)

    return Hcuts,Hcosts,Hmin,Hmax,HNmin,HNmax

def transform_cost(x): # Should map [0,1] monotonically increasingly to [0,1]
    
    # ~ return x
    return x**(1/2)

def FindLastNFuses(meas_data,ydata,n,OneRangeCost):

     Hcuts,Hcosts,Hmin,Hmax,HNmin,HNmax = FindAllFuses(meas_data,ydata,OneRangeCost)
     
     ncuts = len(Hcuts)
     nlvls = len(Hcosts)
     
     last_cuts = [Hcuts[ncuts-i-1] for i in range(n)]
     last_min = [Hmin[ncuts-i-1] for i in range(n)]
     last_max = [Hmax[ncuts-i-1] for i in range(n)]
     
     last_costs = [transform_cost(Hcosts[nlvls-i-1]/Hcosts[nlvls-1]) for i in range(n)]
     last_Cmin = [transform_cost(Hcosts[HNmin[ncuts-i-1]]/Hcosts[nlvls-1]) for i in range(n)]
     last_Cmax = [transform_cost(Hcosts[HNmax[ncuts-i-1]]/Hcosts[nlvls-1]) for i in range(n)]
     
     return last_cuts,last_costs,last_min,last_max,last_Cmin,last_Cmax

def FindFuses(meas_data,ydata=None,n=None,OneRangeCost=None):

    if (meas_data.ndim == 1):
        meas_data_mod = meas_data.reshape((-1,1))
    else:
        meas_data_mod = meas_data

    if ydata is None:
        ydata = list(range(len(meas_data)))

    if n is None:
        n = len(meas_data) - 1

    if OneRangeCost is None:
        OneRangeCost = functools.partial(PolyCost,degpolyfit=0)

    return FindLastNFuses(meas_data_mod,ydata,n,OneRangeCost)

def ColorClusters(meas_data,ydata,nc,last_cuts):
    
    color_clusters = [0,meas_data.shape[0]]

    for icolor in range(nc-1):
        
        idx = bisect.bisect(color_clusters, last_cuts[icolor])
        color_clusters.insert(idx,last_cuts[icolor])

    ydata_cluster = [ydata[color_clusters[i]] for i in range(len(color_clusters)-1)]

    return color_clusters,ydata_cluster
