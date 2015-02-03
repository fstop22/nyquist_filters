#!/usr/bin/env python
import unittest
import scipy as sp
import scipy.signal as sig
from nyquist import *
from nyquist import _FftSize

class nyquistFilterTestCase(unittest.TestCase):
    
    def setUp(self):
        self.N = 65
        self.Q = 4
        self.alpha = 0.25
        self.h = Nyquist(self.N, self.Q, self.alpha)

    def testLength(self):
        self.assertEqual(self.h.size, self.N)

    def testGain(self):
        self.assertAlmostEqual(1.0, self.h.sum(), places=3)

    def testInflection(self):
        N_fft = _FftSize(self.Q)
        w,H = sig.freqz(self.h, 1, N_fft)
        f = self.Q*w/(2*sp.pi)
        idx = (f == 0.5).nonzero()[0]

        self.assertAlmostEqual(sp.absolute(H[idx][0]), 0.5, places=10)

    def tearDown(self):
        pass

class rootNyquistFilterTestCase(unittest.TestCase):
    
    def setUp(self):
        self.N = 65
        self.Q = 4
        self.alpha = 0.25
        self.h = rootNyquist(self.N, self.Q, self.alpha)

    def testLength(self):
        self.assertEqual(self.h.size, self.N)

    def testGain(self):
        self.assertAlmostEqual(1.0, self.h.sum(), places=2)

    def testInflection(self):
        N_fft = _FftSize(self.Q)
        w,H = sig.freqz(self.h, 1, N_fft)
        f = self.Q*w/(2*sp.pi)
        idx = (f == 0.5).nonzero()[0]

        self.assertAlmostEqual(sp.absolute(H[idx][0]), sp.sqrt(2)/2, places=10)

    def tearDown(self):
        pass
if __name__ == "__main__":
    unittest.main()
