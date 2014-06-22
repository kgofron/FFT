# http://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html

from scipy.fftpack import fft
import numpy as np


# One dimensional discrete Fourier transforms
# Number of samplepoints
N = 600
# sample spacing
T = 1.0 / 800.0

l1 = [];
for i in range(N):
    l1.append(np.random.random() / 10)

x = np.linspace(0.0, N*T, N)
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x) + l1
yf = fft(y)
from scipy.signal import blackman
w = blackman(N)
ywf = fft(y*w)
xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
import matplotlib.pyplot as plt
plt.plot(xf, 2.0/N * np.abs(yf[0:N/2]), '-g')
plt.grid()
#plt.artis.set_label("Label")
plt.show()

plt.semilogy(xf[1:N/2], 2.0/N * np.abs(yf[1:N/2]), '-b')
plt.semilogy(xf[1:N/2], 2.0/N * np.abs(ywf[1:N/2]), '-r')
plt.grid()
plt.show()

# for Comples x sequence, additional helper functions. 
#from scipy.fftpack import fftfreq
#freq = fftfreq(np.arange(8), 0.125)
#print freq
#from scipy.fftpack import fftfreq
#x = np.arange(8)
#sf.fftshift(x)

