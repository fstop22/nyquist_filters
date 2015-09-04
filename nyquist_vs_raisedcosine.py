#!/usr/bin/env python
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as signal
from nyquist import *
from nyquist import _FftSize
from filters import *

N = 63
Q = 4
alpha = 0.5
N_fft = _FftSize(Q)

# Design nyquist filter
hh = Nyquist(N, Q, alpha)
n = sp.linspace(-sp.ceil(N/2), sp.floor(N/2), N)

# Design raised cosine filter
h_rc = rcosfilter(N, alpha, 1, 4)[1]
h_rc = h_rc/sum(h_rc)
[w, H_rc] = sig.freqz(h_rc, 1, N_fft)

[w, HH] = sig.freqz(hh, 1, N_fft)
f = Q*w/(2*sp.pi)

# Find 6 dB point
idx = (f == 0.5).nonzero()[0]
n = sp.linspace(0, hh.size-1, hh.size)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(f, 20*sp.log10(sp.absolute(HH)), f, 20*sp.log10(abs(H_rc)))
plt.axvline(f[idx], -100, 20, color='r')
plt.axhline(20*sp.log10(sp.absolute(HH[idx])), 0, Q/2.0, color='r')
plt.grid()
plt.xlabel('Hz/Symbol Rate')
plt.ylabel('dB')
plt.title(r'Nyquist Filter vs Raised Cosine of length %i, $\alpha$ = %0.2f' % (N, alpha))
plt.subplot(2, 1, 2)
plt.plot(n, hh, 'b-o', n, h_rc, 'g-*')
plt.axis([n[0], n[-1], min(hh), max(hh)])
plt.grid()

# Design root nyquist filter
h = rootNyquist(N, Q, alpha)
n = sp.linspace(-sp.ceil(N/2), sp.floor(N/2), N)

# Design root raised cosine filter
h_rrc = rrcosfilter(N, alpha, 1, 4)[1]
h_rrc = h_rrc/sum(h_rrc)
[w, H_rrc] = sig.freqz(h_rrc, 1, N_fft)

[w, H] = sig.freqz(h, 1, N_fft)
f = Q*w/(2*sp.pi)

# Find 3 dB point
idx = (f == 0.5).nonzero()[0]
n = sp.linspace(0, hh.size-1, hh.size)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(f, 20*sp.log10(sp.absolute(H)), f, 20*sp.log10(abs(H_rrc)))
plt.axvline(f[idx], -100, 20, color='r')
plt.axhline(20*sp.log10(sp.absolute(H[idx])), 0, Q/2.0, color='r')
plt.grid()
plt.xlabel('Hz/Symbol Rate')
plt.ylabel('dB')
plt.title(r'Root Nyquist Filter vs Root Raised Cosine of length %i, $\alpha$ = %0.2f' % (N, alpha))
plt.subplot(2, 1, 2)
plt.plot(n, h, 'b-o', n, h_rrc, 'g-*')
plt.axis([n[0], n[-1], min(h), max(h)])
plt.grid()

plt.show()
