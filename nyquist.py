#! /usr/bin/env python
import scipy as sp
import scipy.signal as sig


def _FftSize(Q):

    '''
    Determines FFT size based on samples/symbol (Q) that gaurentees FFT is
    evaulated at f=0.5
    '''
    return int(sp.floor(4096.0/Q)*Q)

def _design(N, Q, alpha, desired, mu=1.0e-2, eps=1.0e12):
    
    '''
    Generalized algorithm for designing Nyquist or root Nyquist
    shape filters

    Uses remez algorithm to initially design an almost nyquist filter
    with Q samples per symbol, lenght N, and excess bandwith of alpha.
    The adjusts pass band cutoff frequency via gradient descent algorithm
    until |H(0.5)| = desired.  For nyquist filter |H(0.5)| = 0.5, 
    for root Nyquist filter |H(0.5)| = sqrt(2)/2. 

     |H(f)|   ^
              |
             1|----------
              |         | \
         ^    |         |  \
         |    |         |   \
      desired |---------|---|\
         |    |         |   | \
         v    |         |   |  \
              |         |   |   \
              +---------|---|---|--------|---> f
              0        1-a  1  1+a      Q/2
                       ---  -  ---
                        2   2   2
                      <--->
                Move passband cutoff freq until |H(0.5)| = desired
    
    Called by Nyquist and rootNyquist functions to design filters,
    respectively.
    
    Parameters
    ----------
    N : int
        Number of taps of filter
    Q : int
        Samples per symbol
    alpha : float
        Excess bandwidth factor (0 < alpha <= 1.0)
    desired : float
        Desired magnitude of filter at inflection point f = 0.5.  For nyquist filter
        set to 0.5.  For root-nyquist filter set to sqrt(2)/2
    mu : float
        Gradient descent adaptation rate
    eps : float
        Error tolerance

    Returns
    -------
    h : ndarray
        Filter coefficients
    '''

    # Initial pass and stop bands of filter
    Fs = float(Q)
    f0 = 0
    f1 = 1.0/2*(1-alpha)
    f2 = 1.0/2*(1+alpha)
    f3 = Fs/2
    F = [f0, f1, f2, f3]

    # Desired gain of pass and stop bands
    G = [1, 0]

    # Design initial filter
    N_fft = _FftSize(Q)
    h = sig.remez(N, F, G, Hz=Fs)
    [w, H] = sig.freqz(h, 1, N_fft)
    f = Q*w/(2*sp.pi)
    idx = (f == 0.5).nonzero()[0]

    # Gradient descent method to move 3 dB point of filter
    error = desired - sp.absolute(H[idx])
    while sp.absolute(error) >= eps:
        f1 = f1 * (1 + mu*error)
        F[1] = f1
        h = sig.remez(N, F, G, Hz=Fs)
        [w, H] = sig.freqz(h, 1, N_fft)
        error = desired - sp.absolute(H[idx])

    return h


def rootNyquist(N, Q, alpha, mu=1.0e-2, eps=1.0e-12):

    '''
    Design root-nyquist filter of length N taps

    Parameters
    ----------
    N : int
        Number of taps of filter
    Q : int
        Samples per symbol
    alpha : float
        Excess bandwidth factor (0 < alpha <= 1.0)
    mu : float
        Gradient descent adaptation rate
    eps : float
        Error tolerance

    Returns
    -------
    h : ndarray
        Array containing coefficients of root-nyquist filter

    References
    ----------
    "Multirate Signal Processing for Communication Systems," Fred Harris
    '''

    desired = sp.sqrt(2.0)/2.0
    return _design(N, Q, alpha, desired, mu, eps)

def Nyquist(N, Q, alpha, mu=1.0e-2, eps=1.0e-12):

    '''
    Design nyquist filter of length N taps

    Parameters
    ----------
    N : int
        Number of taps of filter
    Q : int
        Samples per symbol
    alpha : float
        Excess bandwidth factor (0 < alpha <= 1.0)
    mu : float
        Gradient descent adaptation rate
    eps : float
        Error tolerance

    Returns
    -------
    h : ndarray
        Array containing coefficients of root-nyquist filter
    '''

    desired = 0.5
    return _design(N, Q, alpha, desired, mu, eps)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N = 129
    Q = 4
    alpha = 0.25
    N_fft = _FftSize(Q)

    # Design root-nyquist filter
    h = rootNyquist(N, Q, alpha)
    n = sp.linspace(-sp.ceil(N/2), sp.floor(N/2), N)

    [w, H] = sig.freqz(h, 1, N_fft)
    f = Q*w/(2*sp.pi)

    # Find 3 dB point
    idx = (f == 0.5).nonzero()[0]

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(f, 20*sp.log10(sp.absolute(H)))
    plt.axvline(f[idx], -100, 20, color='r')
    plt.axhline(20*sp.log10(sp.absolute(H[idx])), 0, Q/2.0, color='r')
    plt.grid()
    plt.xlabel('Hz/Symbol Rate')
    plt.ylabel('dB')
    plt.title(r'Root-Nyquist Filter of length %i, $\alpha$ = %0.2f' % (N, alpha))
    plt.subplot(2, 1, 2)
    plt.stem(n, h)
    plt.axis([n[0], n[-1], min(h), max(h)])

    # Design nyquist filter
    hh = Nyquist(N, Q, alpha)
    n = sp.linspace(-sp.ceil(N/2), sp.floor(N/2), N)

    [w, HH] = sig.freqz(hh, 1, N_fft)
    f = Q*w/(2*sp.pi)

    # Find 6 dB point
    idx = (f == 0.5).nonzero()[0]
    n = sp.linspace(0, hh.size-1, hh.size)

    print hh.size
    print n.size

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(f, 20*sp.log10(sp.absolute(HH)))
    plt.axvline(f[idx], -100, 20, color='r')
    plt.axhline(20*sp.log10(sp.absolute(HH[idx])), 0, Q/2.0, color='r')
    plt.grid()
    plt.xlabel('Hz/Symbol Rate')
    plt.ylabel('dB')
    plt.title(r'Nyquist Filter of length %i, $\alpha$ = %0.2f' % (N, alpha))
    plt.subplot(2, 1, 2)
    plt.stem(n, hh)
    plt.axis([n[0], n[-1], min(hh), max(hh)])

    plt.show()
