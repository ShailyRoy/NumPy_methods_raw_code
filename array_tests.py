# ********
# This file is individualized for NetID sroy15.
# ********

import numpy as np
import unittest as ut
import itertools as it
import inspect as ip
import array_code as ac

class ArrayTestCase(ut.TestCase):
    
    def test_arith(self):
        def _arith(x):
            y = np.empty(x.shape)
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    for k in range(x.shape[2]):
                        y[i,j,k] = 2*x[i,j,k]**2 + 3*x[i,j,k]
            return y
        
        X = [np.random.randn(3,4,5) for _ in range(3)]
        self._test_fun(ac.arith, _arith, X)

    def test_agg(self):
        def _agg(x):
            s = 0
            for i in range(x.shape[0]):
                mn = np.inf
                for j in range(x.shape[1]):
                    mx = -np.inf
                    for k in range(x.shape[2]):
                        cx = 4*x[i,j,k]
                        if cx > mx: mx = cx
                    cmx = 2*mx
                    if cmx < mn: mn = cmx
                cmn = 2*mn
                s += cmn
            return s

        X = [np.random.randn(3,4,5) for _ in range(3)]
        self._test_fun(ac.agg, _agg, X)

    def test_bool(self):
        def _bool(x):
            count = 0
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    for k in range(x.shape[2]):
                        if x[i,j,k]**2 < 2*x[i,j,k] + 2: count += 1
            return count

        X = [np.random.randn(3,4,5) for _ in range(3)]
        self._test_fun(ac.bool, _bool, X)

    def test_bcast(self):
        def _bcast(x):
            x1, x2 = x
            y = np.empty(x1.shape)
            for i in range(x1.shape[0]):
                for j in range(x1.shape[1]):
                    for k in range(x1.shape[2]):
                        y[i,j,k] = (x1[i,j,k]+3)*(4*x2[j,k] - 4)
            return y

        X = [(np.random.randn(3,4,5), np.random.randn(4,5)) for _ in range(3)]
        self._test_fun(ac.bcast, _bcast, X)

    def test_bcast_ax(self):
        def _bcast_ax(x):
            x1, x2 = x
            y = np.empty((x1.shape[0], x2.shape[0], x2.shape[1]))
            for i in range(x1.shape[0]):
                for j in range(x2.shape[0]):
                    for k in range(x2.shape[1]):
                        y[i,j,k] = (3+x1[i,k])*(4*x2[j,k]-4)
            return y

        X = [(np.random.randn(3,5), np.random.randn(4,5)) for _ in range(3)]
        self._test_fun(ac.bcast_ax, _bcast_ax, X)

    def test_newax(self):
        def _newax(x):
            y = np.empty((len(x), len(x), len(x)))
            for i in range(len(x)):
                for j in range(len(x)):
                    for k in range(len(x)):
                        y[i,j,k] = (x[i]**2)*(x[j]*3)*(3**x[k])
            return y

        X = [np.random.randn(5) for _ in range(3)]
        self._test_fun(ac.newax, _newax, X)

    def test_series_pow(self):
        def _series_pow(x):
            s = 0
            for i in range(x):
                s += (i**1.85)
            return s

        X = list(range(2,5))
        self._test_fun(ac.series_pow, _series_pow, X)

    def test_series_alt(self):
        def _series_alt(x):
            s = 0
            for i in range(x):
                s += (-1)**i * (i**1.85)
            return s

        X = list(range(2,5))
        self._test_fun(ac.series_alt, _series_alt, X)

    def test_series_dbl(self):
        def _series_dbl(x):
            x1, x2 = x
            s = 0
            for i in range(x1):
                for j in range(x2):
                    s += (3*j+3) * i**4
            return s

        X = list(it.product(range(2,5), repeat=2))
        self._test_fun(ac.series_dbl, _series_dbl, X)

    def test_idx(self):
        def _idx(x):
            y = np.empty((int(x.shape[0]/2), x.shape[1]))
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    y[i,j] = 2*x[2*i,j] + 4*x[2*i+1,j]**2
            return y

        X = [np.random.randn(4,8) for _ in range(3)]
        self._test_fun(ac.idx, _idx, X)

    def test_hypercube(self):
        def _hypercube(x):
            y = np.empty((x, 2**x))            
            for i in range(x):
                for j in range(2**x):
                    y[i,j] = (-1) ** (int(j / 2**i) % 2)
            return y

        X = list(range(1,4))
        self._test_fun(ac.hypercube, _hypercube, X)

    def _test_fun(self, fun, _fun, X):
        # Helper function used in the foregoing tests

        # Make sure no loops were used
        src = ip.getsource(fun)
        self.assertFalse("for " in src)
        self.assertFalse("while " in src)

        # Check whether output values are correct
        for x in X:
            y, _y = fun(x), _fun(x)
            if not np.allclose(y, _y):
                print("\n****************** %s mismatch ******************" % fun.__name__)
                print("x, y, _y:")
                for a in [x, y, _y]: print(a)
                self.assertTrue(np.allclose(y, _y))


if __name__ == "__main__":    
    
    test_suite = ut.TestLoader().loadTestsFromTestCase(ArrayTestCase)
    res = ut.TextTestRunner(verbosity=2).run(test_suite)
    num, errs, fails = res.testsRun, len(res.errors), len(res.failures)
    print("score: %d of %d (%d errors, %d failures)" % (num - (errs+fails), num, errs, fails))
    
