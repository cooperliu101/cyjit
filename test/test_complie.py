from cyjit import make
inline, load, jit, build=make('my')

import numpy
from math import sqrt

inline('from libc.math cimport sqrt')

@jit('double(double[:,::1])',
     locals='''
     double s
     int i,j,m,n
     ''',
     wraparound=False,
     boundscheck=False)
def f(arr):
    s=0.0
    m=arr.shape[0]
    n=arr.shape[1]
    for i in range(m):
        for j in range(n):
            s+=sqrt(arr[i,j])
    return s
build()

arr=numpy.arange(100*100.0).reshape(100,100)
import time
st=time.clock()
print f(arr)
print time.clock()-st, 'compiled'
st=time.clock()
print f.py_func(arr)
print time.clock()-st, 'pyfunc'