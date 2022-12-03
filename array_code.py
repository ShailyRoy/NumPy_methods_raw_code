# ********
# This file is individualized for NetID sroy15.
# ********

import numpy as np

def arith(x):
    # https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html#arithmetic-matrix-multiplication-and-comparison-operations
    i = np.array(x)
    C = np.empty(x.shape)
    C = 2*i**2+3*i
  
    return C # replace with your implementation   

def agg(x):
    s = 0
    mn = np.array(x)*np.inf
    mx = np.array(x)*(-np.inf)
    cx = 4*x
    mx = cx.max(axis=2)
    cmx = 2*mx
    mn = cmx.min(axis=1)
    cmn = 2*mn
    s=cmn.sum()
    # https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html#calculation
    return s # replace with your implementation   
    
def bool(x):
    i = np.array(x) # matrix equiv. of the imaginary unit
    j=i**2
    k=i*2+2
    count = np.count_nonzero(j < k)
    return count # replace with your implementation   

def bcast(x):
    x1,x2 = x
    y = np.empty(x1.shape)
    y= (x1+3)*(4*x2[:x1.shape[1],:x1.shape[2]] - 4)
    # https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
    return y # replace with your implementation   

def bcast_ax(x):
    x1, x2 = x
    y = np.empty((x1.shape[0], x2.shape[0], x2.shape[1]))
    #y[:x1.shape[0],:x2.shape[0],:x2.shape[1]]= (3+x1[:x1.shape[0],:x2.shape[1]])*(4*x2[:x2.shape[0],:x2.shape[1]]-4)
    #y= (3+x1.reshape(x1.shape[0],x2.shape[1]))*(4*x2.reshape(x2.shape[0],x2.shape[1])-4)
    y[np.newaxis,np.newaxis,np.newaxis] = (3+x1[:,np.newaxis,:])*(4*x2[np.newaxis,:,:]-4)
    # https://docs.scipy.org/doc/numpy/reference/constants.html#numpy.newaxis
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.reshape.html
    return y # replace with your implementation   

def newax(x):
    y = np.empty((len(x), len(x), len(x)))
    y[np.newaxis,np.newaxis,np.newaxis] = (x[:,np.newaxis,np.newaxis]**2)*(x[np.newaxis,:,np.newaxis]*3)*(3**x[np.newaxis,np.newaxis,:])
    #print(y)
    return y # replace with your implementation   

def series_pow(x):
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html
    s = 0
    z = np.arange(0,x)
    x=np.array(z)
    s=(x**1.85).sum()
    return s # replace with your implementation   

def series_alt(x):
    s = 0
    z = np.arange(0,x)
    x=np.array(z)
    #s += (-1)**i * (i**1.85)
    s=(((-1)**x)*(x**1.85)).sum()
    return s# replace with your implementation   

def series_dbl(x):
    x1, x2 = x
    s = 0
    j=np.arange(0,x2)
    j = np.array(j)
    i=np.arange(0,x1)
    i = np.array(i)
    s = ((3*j[np.newaxis,0:x2]+3) * i[0:x1,np.newaxis]**4).sum()
    return s # replace with your implementation   

def idx(x):
    y = np.empty((int(x.shape[0]/2), x.shape[1]))
    i = np.arange(0,y.shape[0])
    j = np.arange(0,y.shape[1])
    #y[i,j] = 2*x[2*i,j] + 4*x[2*i+1,j]**2
    y[:,np.newaxis]=2*x[i*2,np.newaxis]+4*x[i*2+1,np.newaxis]**2
    # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    return y # replace with your implementation   

def hypercube(x):
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.astype.html
    y = np.empty((x, 2**x)) 
    z = np.arange(0,2**x)
    j=np.array(z)
    t =  np.arange(0,x)
    i= np.array(t)
    i = 2**i
    #k = np.empty((x, 2**x)) 
    k = ((j[np.newaxis]/i[:,np.newaxis]).astype(int)) % 2
    y[:,np.newaxis] = (-1)**k[:,np.newaxis]
    return y # replace with your implementation   

