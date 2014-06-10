#!/usr/bin/python
import numpy as np
import time

### Compute DFT ###
def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

x = np.random.random(1024)
np.allclose( DFT_slow(x) , np.fft.fft(x))

def time_lap(t1, t2, message):
    print message, t2 - t1
    return t2 - t1

t1 = time.time(); DFT_slow(x); t2 = time.time(); time_lap(t1, t2, "DFT_slow")
t1 = time.time(); np.fft.fft(x); t2 = time.time(); time_lap(t1, t2, "np")

#print t.timeit(2)
#t2 = timeit.Timer( 'np.fft.fft(x)', "print 'set np' )
#print t2.timeit(10)
