"""
Used for calculating tensor invariants and tensor basis functions.
===
Distributed by: Notre Dame CICS (MIT Liscense)
- Associated publication:
url: https://www.sciencedirect.com/science/article/pii/S0021999119300464
doi: https://doi.org/10.1016/j.jcp.2019.01.021
github: https://github.com/cics-nd/rans-uncertainty
===
"""
import torch as th
import numpy as np

# Default tensor type
dtype = th.DoubleTensor

class Invariant():
    """
    Class used for invariant calculations
    """
    def getInvariants(self, s0, r0):
        """
        Calculates the invariant neural network inputs
        Args:
            s0 (DoubleTensor): [nCellx3x3] Rate-of-strain tensor -> 0.5*(k/e)(du + du')
            r0 (DoubleTensor): [nCellx3x3] Rotational tensor -> 0.5*(k/e)(du - du')
        Returns:
            invar (DoubleTensor): [nCellx5x1] Tensor containing the 5 invariant NN inputs
        """
        # Invariant Training inputs
        # For equations see Eq. 14 in paper
        # SB Pope 1975 (http://doi.org/10.1017/S0022112075003382)
        # Or Section 11.9.2 (page 453) of Turbulent Flows by SB Pope
        nCells = s0.size()[0]
        invar = th.DoubleTensor(nCells, 5).type(dtype)

        s2 = s0.bmm(s0)
        r2 = r0.bmm(r0)
        s3 = s2.bmm(s0)
        r2s = r2.bmm(s0)
        r2s2 = r2.bmm(s2)

        invar[:,0] = (s2[:,0,0]+s2[:,1,1]+s2[:,2,2]) #Tr(s2)
        invar[:,1] = (r2[:,0,0]+r2[:,1,1]+r2[:,2,2]) #Tr(r2)
        invar[:,2] = (s3[:,0,0]+s3[:,1,1]+s3[:,2,2]) #Tr(s3)
        invar[:,3] = (r2s[:,0,0]+r2s[:,1,1]+r2s[:,2,2]) #Tr(r2s)
        invar[:,4] = (r2s2[:,0,0]+r2s2[:,1,1]+r2s2[:,2,2]) #Tr(r2s2)

        # Scale invariants by sigmoid function
        # Can use other scalings here
        invar_sig = (1.0 - th.exp(-invar))/(1.0 + th.exp(-invar))
        invar_sig[invar_sig != invar_sig] = 0
        
        return invar_sig

    def getTensorFunctions(self, s0, r0):
        """
        Calculates the linear independent tensor functions for calculating the
        deviatoric  component of the Reynolds stress. Ref: S. Pope 1975 in JFM
        Args:
            s0 (DoubleTensor): [nCellsx3x3] Rate-of-strain tensor -> 0.5*(k/e)(du + du')
            r0 (DoubleTensor): [nCellsx3x3] Rotational tensor -> 0.5*(k/e)(du - du')
        Returns:
            invar (DoubleTensor): [nCellsx10x[3x3]] Tensor 10 linear independent functions
        """
        # Invariant Training inputs
        # For equations see Eq. 15 in paper
        # Or SB Pope 1975 (http://doi.org/10.1017/S0022112075003382)
        # Or Section 11.9.2 (page 453) of Turbulent Flows by SB Pope
        nCells = s0.size()[0]
        invar_func = th.DoubleTensor(nCells,10,3,3).type(dtype)

        s2 = s0.bmm(s0)
        r2 = r0.bmm(r0)
        sr = s0.bmm(r0)
        rs = r0.bmm(s0)

        invar_func[:,0] = s0
        invar_func[:,1] = sr - rs
        invar_func[:,2] = s2 - (1.0/3.0)*th.eye(3).type(dtype)*(s2[:,0,0]+s2[:,1,1]+s2[:,2,2]).unsqueeze(1).unsqueeze(1)
        invar_func[:,3] = r2 - (1.0/3.0)*th.eye(3).type(dtype)*(r2[:,0,0]+r2[:,1,1]+r2[:,2,2]).unsqueeze(1).unsqueeze(1)
        invar_func[:,4] = r0.bmm(s2) - s2.bmm(r0)
        t0 = s0.bmm(r2)
        invar_func[:,5] = r2.bmm(s0) + s0.bmm(r2) - (2.0/3.0)*th.eye(3).type(dtype)*(t0[:,0,0]+t0[:,1,1]+t0[:,2,2]).unsqueeze(1).unsqueeze(1)
        invar_func[:,6] = rs.bmm(r2) - r2.bmm(sr)
        invar_func[:,7] = sr.bmm(s2) - s2.bmm(rs)
        t0 = s2.bmm(r2)
        invar_func[:,8] = r2.bmm(s2) + s2.bmm(r2) - (2.0/3.0)*th.eye(3).type(dtype)*(t0[:,0,0]+t0[:,1,1]+t0[:,2,2]).unsqueeze(1).unsqueeze(1)
        invar_func[:,9] = r0.bmm(s2).bmm(r2) + r2.bmm(s2).bmm(r0)

        # Scale the tensor basis functions by the L2 norm
        l2_norm = th.DoubleTensor(invar_func.size(0), 10)
        l2_norm = 0
        for (i, j), x in np.ndenumerate(np.zeros((3,3))):
            l2_norm += th.pow(invar_func[:,:,i,j],2)
        invar_func = invar_func/th.sqrt(l2_norm).unsqueeze(2).unsqueeze(3)

        return invar_func
