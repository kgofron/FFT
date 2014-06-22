#!/usr/bin/python
# python
"""
Partially based on
http://matplotlib.org/examples/pylab_examples/psd_demo.html
"""

from pylab import *
import scipy
import scipy.fftpack

dt = 0.0005
amplitude = 0.1
t = arange(0,1,dt)
nse = randn(len(t))
r = exp(-t/0.05)

cnse = convolve(nse, r)*dt
cnse = cnse[:len(t)]
s1 = amplitude * sin(2*pi*t*50)
s2 = amplitude * sin(2*pi*t*80)
s = amplitude * (s1 + s2) + cnse

# Check the scipy computation of fft
t2 = arange(0,1,dt)
acc = lambda t2: amplitude * (sin(2*pi*t*20) + sin(2*pi*t*80)) + 0.02 * scipy.random.random(len(t2))
signal = acc(t2)
FFT_acc = abs(scipy.fft(signal))
freqs = scipy.fftpack.fftfreq(signal.size, t2[1]-t2[0])
subplot(211)
plot(t2, signal)

subplot(212)
#plot(freqs, 20*log10(FFT_acc),'x')
plot(freqs, 20*log10(FFT_acc),'-g')
xlabel('Frequency [Hz]')
ylabel('20*log10(FFT)')
show()
# The basic test of 20Hz and 80Hz sin() funciton

subplot(221)
#subplot(311)
plot(t,s1, '-r', label='s1=50Hz')
plot(t,s2, '-g', label='s2=80Hz')
xlabel('Time [s]')
ylabel('Amplitude [arb. units]')
legend(loc='upper right')
xlim(0, 0.2)
# End of basic fft check

subplot(222)
#subplot(312)
plot(t,s, '-b', label='s1+s2+noise')
xlabel('Time [s]')
ylabel('Amplitude [arb. units]')
legend(loc='upper right')
xlim(0, 0.2)

subplot(223)
FFT = abs(scipy.fft(s))
freqs = scipy.fftpack.fftfreq(s.size, t[1]-t[0])
plot(freqs, 20*log10(FFT),'-g')
#plot(freqs , FFT, '-m', label='FFT')
#legend(loc='upper right')
xlabel('Frequency [Hz]')
ylabel('20*log10(FFT)')

subplot(224)
#subplot(313)
psd(s, 512, 1/dt)
xlim(0, 200)
#ylim(-1.2, 1.2)
#xlabel('x-axis')
#ylabel('y-axis')
#title('My plot')

show()
"""
Based on 
matplotlib.org/examples/pyplot_examples/psd_demo.html
"""
