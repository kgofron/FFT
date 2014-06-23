import numpy as np
import time

def time_lap(t1, t2, message):
    print message, t2 - t1
    return t2 - t1

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
np.allclose(DFT_slow(x), np.fft.fft(x))

#%timeit DFT_slow(x)
#%timeit np.fft.fft(x)
t1 = time.time(); DFT_slow(x); t2 = time.time(); time_lap(t1, t2, "DFT_slow")
t1 = time.time(); np.fft.fft(x); t2 = time.time(); time_lap(t1, t2, "np")


### DFT to FFT: Exploiting Symmetry###
def FFT(x):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    
    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:  # this cutoff should be optimized
        return DFT_slow(x)
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N / 2] * X_odd,
                               X_even + factor[N / 2:] * X_odd])
    
x = np.random.random(1024)
np.allclose(FFT(x), np.fft.fft(x))

#%timeit DFT_slow(x)
#%timeit  FFT(x) 
#%timeit np.fft.fft(x)
print 
t1 = time.time(); DFT_slow(x); t2 = time.time(); time_lap(t1, t2, "DFT_slow")
t1 = time.time(); FFT(x); t2 = time.time(); time_lap(t1, t2, "FFT")    
t1 = time.time(); np.fft.fft(x); t2 = time.time(); time_lap(t1, t2, "np")
    
### Vectorized Numpy Version
def FFT_vectorized(x):
    """A vectorized, non-recursive version of the Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if np.log2(N) % 1 > 0:
        raise ValueError("size of x must be a power of 2")

    # N_min here is equivalent to the stopping condition above,
    # and should be a power of 2
    N_min = min(N, 32)
    
    # Perform an O[N^2] DFT on all length-N_min sub-problems at once
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))

    # build-up each level of the recursive calculation all at once
    while X.shape[0] < N:
        X_even = X[:, :X.shape[1] / 2]
        X_odd = X[:, X.shape[1] / 2:]
        factor = np.exp(-1j * np.pi * np.arange(X.shape[0])
                        / X.shape[0])[:, None]
        X = np.vstack([X_even + factor * X_odd,
                       X_even - factor * X_odd])

    return X.ravel()

x = np.random.random(1024)
np.allclose(FFT_vectorized(x), np.fft.fft(x))
    
    
# Large Arrays for tests 
x = np.random.random(1024 * 16)
#%timeit FFT(x)
#%timeit FFT_vectorized(x)
#%timeit np.fft.fft(x)
print
# t1 = time.time(); DFT_slow(x); t2 = time.time(); time_lap(t1, t2, "DFT_slow")
t1 = time.time(); FFT(x); t2 = time.time(); time_lap(t1, t2, "FFT")
t1 = time.time(); FFT_vectorized(x); t2 = time.time(); time_lap(t1, t2, "FFT_vectorized")    
t1 = time.time(); np.fft.fft(x); t2 = time.time(); time_lap(t1, t2, "np")

# We're now within about a factor of 10 of the FFTPACK benchmark, using only a couple dozen lines of pure Python + NumPy
# http://www.netlib.org/fftpack/fft.c

# Plotting with Matpoltlib: http://jakevdp.github.io/mpl_tutorial/tutorial_pages/tut1.html
