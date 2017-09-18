import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for smooth function.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func is not implemented.')

    def grad(self, x):
        """
        Computes the gradient vector at point x.
        """
        raise NotImplementedError('Grad is not implemented.')


class BaseProxOracle(object):
    """
    Base class for proximal h(x)-part in a composite function f(x) + h(x).
    """

    def func(self, x):
        """
        Computes the value of h(x).
        """
        raise NotImplementedError('Func is not implemented.')

    def prox(self, x, alpha):
        """
        Implementation of proximal mapping.
        prox_{alpha}(x) := argmin_y { 1/(2*alpha) * ||y - x||_2^2 + h(y) }.
        """
        raise NotImplementedError('Prox is not implemented.')


class BaseCompositeOracle(object):
    """
    Base class for the composite function.
    phi(x) := f(x) + h(x), where f is a smooth part, h is a simple part.
    """

    def __init__(self, f, h):
        self._f = f
        self._h = h

    def func(self, x):
        """
        Computes the f(x) + h(x).
        """
        return self._f.func(x) + self._h.func(x)

    def grad(self, x):
        """
        Computes the gradient of f(x).
        """
        return self._f.grad(x)

    def prox(self, x, alpha):
        """
        Computes the proximal mapping.
        """
        return self._h.prox(x, alpha)

    def duality_gap(self, x):
        """
        Estimates the residual phi(x) - phi* via the dual problem, if any.
        """
        return None


class BaseNonsmoothConvexOracle(object):
    """
    Base class for implementation of oracle for nonsmooth convex function.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func is not implemented.')

    def subgrad(self, x):
        """
        Computes arbitrary subgradient vector at point x.
        """
        raise NotImplementedError('Subgrad is not implemented.')

    def duality_gap(self, x):
        """
        Estimates the residual phi(x) - phi* via the dual problem, if any.
        """
        return None


class LeastSquaresOracle(BaseSmoothOracle):
    """
    Oracle for least-squares regression.
        f(x) = 0.5 ||Ax - b||_2^2
    """
    def __init__(self, matvec_Ax, matvec_ATx, b):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.b = b

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        v = self.matvec_Ax(x) - self.b
        return 0.5 * np.dot(v, v)

    def grad(self, x):
        """
        Computes the gradient vector at point x.
        """
        v = self.matvec_Ax(x) - self.b
        return self.matvec_ATx(v)


class L1RegOracle(BaseProxOracle):
    """
    Oracle for L1-regularizer.
        h(x) = regcoef * ||x||_1.
    """
    def __init__(self, regcoef):
        self.regcoef = regcoef

    def func(self, x):
        return self.regcoef * np.linalg.norm(x, 1)

    def prox(self, x, alpha):
        """
        Implementation of proximal mapping.
        prox_{alpha}(x) := argmin_y { 1/(2*alpha) * ||y - x||_2^2 + h(y) }.
        """
        n = x.shape[0]
        result = np.zeros(n)
        alpha2 = alpha * self.regcoef
        for i in range(n):
            if x[i] < -alpha2:
                result[i] = x[i] + alpha2
            elif x[i] > alpha2:
                result[i] = x[i] - alpha2

        return result


class LassoProxOracle(BaseCompositeOracle):
    """
    Oracle for 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
        f(x) = 0.5 * ||Ax - b||_2^2 is a smooth part,
        h(x) = regcoef * ||x||_1 is a simple part.
    """

    def duality_gap(self, x):
        """
        Estimates the residual phi(x) - phi* via the dual problem, if any.
        """
        Ax_b = self._f.matvec_Ax(x) - self._f.b
        return lasso_duality_gap(x, Ax_b, self._f.matvec_ATx(Ax_b), self._f.b, self._h.regcoef)


class LassoNonsmoothOracle(BaseNonsmoothConvexOracle):
    """
    Oracle for nonsmooth convex function
        0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
    """
    def __init__(self, matvec_Ax, matvec_ATx, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        v = self.matvec_Ax(x) - self.b
        return 0.5 * np.dot(v, v) + self.regcoef * np.linalg.norm(x, 1)

    def subgrad(self, x):
        """
        Computes the gradient vector at point x.
        """
        v = self.matvec_Ax(x) - self.b
        return self.matvec_ATx(v) + self.regcoef * np.sign(x)

    def duality_gap(self, x):
        """
        Estimates the residual phi(x) - phi* via the dual problem, if any.
        """
        Ax_b = self.matvec_Ax(x) - self.b
        return lasso_duality_gap(x, Ax_b, self.matvec_ATx(Ax_b), self.b, self.regcoef)

def lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef):
    """
    Estimates f(x) - f* via duality gap for
        f(x) := 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
    """

    denom = np.linalg.norm(ATAx_b, np.inf)
    if denom < 1e-16:
        mu = Ax_b
    else:
        mu = np.minimum (1, regcoef * 1. / denom) * Ax_b

    return 0.5 * np.dot(Ax_b, Ax_b) + regcoef * np.linalg.norm(x, 1) + 0.5 * np.dot(mu, mu) + np.dot(b, mu)


def create_lasso_prox_oracle(A, b, regcoef):
    matvec_Ax = lambda x: A.dot(x)
    matvec_ATx = lambda x: A.T.dot(x)
    return LassoProxOracle(LeastSquaresOracle(matvec_Ax, matvec_ATx, b),
                           L1RegOracle(regcoef))


def create_lasso_nonsmooth_oracle(A, b, regcoef):
    matvec_Ax = lambda x: A.dot(x)
    matvec_ATx = lambda x: A.T.dot(x)
    return LassoNonsmoothOracle(matvec_Ax, matvec_ATx, b, regcoef)


class lasso_barrier_oracle():
    def __init__(self, A, b, regcoef):
        self.A = A
        self.b = b
        self.regcoef = regcoef
        self.n = A.shape[1]
        self.ATA = A.T.dot(A)

    def set_constant(self,t):
        self.t = t

    def func_outer(self, x, u):
        Ax_b = self.A.dot(x) - self.b
        f_x = 0.5 * np.dot(Ax_b, Ax_b) + self.regcoef * np.sum(u)
        return f_x

    def func(self, X):
        x = X[:self.n]
        u = X[self.n:]
        Ax_b = self.A.dot (x) - self.b
        f_x = 0.5 * np.dot(Ax_b, Ax_b) + self.regcoef * np.sum(u)
        f = self.t * f_x - np.sum(np.log(u + x) + np.log(u - x))
        return f

    def grad (self, X):
        x = X[:self.n]
        u = X[self.n:]
        Ax_b = self.A.dot(x) - self.b
        ATAx_b = self.A.T.dot(Ax_b)

        dx = self.t * ATAx_b - 1./ (u + x) + 1. / (u - x)
        du = self.t * self.regcoef  - 1./ (u + x) - 1. / (u - x)
        grad = np.concatenate([dx, du])
        return grad


    def hess (self, X):
        x = X[:self.n]
        u = X[self.n:]

        vec_p2 = (u + x) * (u + x)
        vec_m2 = (u - x) * (u - x)

        dxx = 1. / (vec_p2) + 1. / (vec_m2)
        duu = 1. / (vec_p2) + 1. / (vec_m2)
        dux = 1. / (vec_p2) - 1. / (vec_m2)

        diagonal = np.concatenate([dxx, duu])
        hess = np.diag(diagonal)
        hess[:self.n, :self.n] += self.t * self.ATA
        hess += np.diag(dux, self.n) + np.diag(dux, -self.n)
        return hess
