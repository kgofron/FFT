#!/usr/bin/python
# python

from pylab import *

dt = 0.01
t = arange(0,20,dt)
nse = randn(len(t))
r = exp(-t/0.05)

cnse = convolve(nse, r)*dt
cnse = cnse[:len(t)]
s = 10*0.1*sin(2*pi*t) + cnse

subplot(211)
plot(t,s)
subplot(212)
psd(s, 512, 1/dt)

show()
"""
matplotlib.org/examples/pyplot_examples/psd_demo.html
"""
