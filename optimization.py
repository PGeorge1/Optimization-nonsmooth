from collections import defaultdict
import numpy as np
from numpy.linalg import norm, solve
from time import time
import datetime
from datetime import datetime
import scipy
from scipy import sparse
from scipy import optimize

from oracles import lasso_barrier_oracle

class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        # TODO: Implement line search procedures for Armijo, Wolfe and Constant steps.
        wolf=None

        if self._method == 'Constant':
            return self.c

        if self._method == 'Wolfe':
            wolf = scipy.optimize.linesearch.line_search_wolfe2 (f=oracle.func,
                                                                 myfprime=oracle.grad,
                                                                 xk=x_k, pk=d_k, c1=self.c1, c2=self.c2)[0]
            if wolf != None:
                return wolf

        if self._method == 'Armijo' or wolf == None:
            if previous_alpha != None:
                alpha=previous_alpha
            else:
                alpha = self.alpha_0

            phi_deriv = oracle.grad(x_k).dot(d_k)

            while oracle.func(x_k + alpha * d_k) > oracle.func(x_k) + self.c1 * alpha * phi_deriv:
                alpha = alpha / 2.0
            return alpha

        return None


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix (e.g. non-invertible matrix).
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = newton(oracle, np.zeros(5), line_search_options={'method': 'Constant', 'c': 1.0})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    n = int(x_k.shape[0] / 2)

    t0=datetime.now()
    norm_grad0=np.linalg.norm(oracle.grad(x_0))

    for iteration in range(max_iter + 1):
        #Oracle
        grad_k=oracle.grad(x_k)
        norm_grad_k=np.linalg.norm (grad_k)
        hess_k=oracle.hess(x_k)

        #Fill trace data
        if trace:
            history['time'].append((datetime.now() - t0).total_seconds())
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(norm_grad_k)
            if x_k.size < 3:
                history['x'].append(x_k)

        if display==True:
            print(u"debug info")

        #Criterium
        if norm_grad_k*norm_grad_k <= tolerance * norm_grad0*norm_grad0:
            break;
        if iteration == max_iter:
            return x_k, 'iterations_exceeded', history

        #Compute direction
        try:
            L=scipy.linalg.cho_factor(hess_k, lower=True)
            d_k=scipy.linalg.cho_solve(L,-grad_k)
        except:
            return x_k, 'computational_error', history


        alpha_1 = 1.
        alpha_2 = 1.
        for i in range(n):
            dxi = d_k[i]
            dyi = d_k[i + n]
            xi = x_k[i]
            yi = x_k[n + i]
            if dxi - dyi > 0:
                value = (yi - xi) * 1. / (dxi - dyi)
                if value < alpha_1:
                    alpha_1 = value
            if dxi + dyi < 0:
                value = (-xi - yi) * 1. / (dxi + dyi)
                if value < alpha_2:
                    alpha_2 = value
        alpha = np.min([0.95 * np.min([alpha_1, alpha_2]), 1.])

        #Line search
        alpha = line_search_tool.line_search (oracle, x_k, d_k, alpha)

        if alpha == None:
            return x_k, 'computational_error', history

        #Update x_k
        x_k = x_k + alpha * d_k

    return x_k, 'success', history



def barrier_method_lasso(A, b, reg_coef, x_0, u_0, tolerance=1e-5,
                         tolerance_inner=1e-8, max_iter=100,
                         max_iter_inner=20, t_0=1, gamma=10,
                         c1=1e-4, lasso_duality_gap=None,
                         trace=False, display=False):
    """
    Log-barrier method for solving the problem:
        minimize    f(x, u) := 1/2 * ||Ax - b||_2^2 + reg_coef * \sum_i u_i
        subject to  -u_i <= x_i <= u_i.

    The method constructs the following barrier-approximation of the problem:
        phi_t(x, u) := t * f(x, u) - sum_i( log(u_i + x_i) + log(u_i - x_i) )
    and minimize it as unconstrained problem by Newton's method.

    In the outer loop `t` is increased and we have a sequence of approximations
        { phi_t(x, u) } and solutions { (x_t, u_t)^{*} } which converges in `t`
    to the solution of the original problem.

    Parameters
    ----------
    A : np.array
        Feature matrix for the regression problem.
    b : np.array
        Given vector of responses.
    reg_coef : float
        Regularization coefficient.
    x_0 : np.array
        Starting value for x in optimization algorithm.
    u_0 : np.array
        Starting value for u in optimization algorithm.
    tolerance : float
        Epsilon value for the outer loop stopping criterion:
        Stop the outer loop (which iterates over `k`) when
            `duality_gap(x_k) <= tolerance`
    tolerance_inner : float
        Epsilon value for the inner loop stopping criterion.
        Stop the inner loop (which iterates over `l`) when
            `|| \nabla phi_t(x_k^l) ||_2^2 <= tolerance_inner * \| \nabla \phi_t(x_k) \|_2^2 `
    max_iter : int
        Maximum number of iterations for interior point method.
    max_iter_inner : int
        Maximum number of iterations for inner Newton's method.
    t_0 : float
        Starting value for `t`.
    gamma : float
        Multiplier for changing `t` during the iterations:
        t_{k + 1} = gamma * t_k.
    c1 : float
        Armijo's constant for line search in Newton's method.
    lasso_duality_gap : callable object or None.
        If calable the signature is lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef)
        Returns duality gap value for esimating the progress of method.
    trace : bool
        If True, the progress information is appended into history dictionary
        during training. Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    (x_star, u_star) : tuple of np.array
        The point found by the optimization procedure.
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every **outer** iteration of the algorithm
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0)
    u_k = np.copy(u_0)
    oracle = lasso_barrier_oracle(A, b, reg_coef)
    t_k = t_0
    n = x_0.shape[0]

    time_0 = datetime.now()

    for it in range(max_iter + 1):
        # Oracle
        f_k = oracle.func_outer(x_k, u_k)
        Ax_b = A.dot(x_k) - b
        ATAx_b = A.T.dot(Ax_b)
        duality_gap_k = lasso_duality_gap(x_k, Ax_b, ATAx_b, b, reg_coef)
        oracle.set_constant(t_k)


        # Debug info
        if display:
            print(it)

        # Fill trace data
        if trace:
            history['func'].append(f_k)
            history['time'].append((datetime.now() - time_0).total_seconds())
            history['duality_gap'].append(duality_gap_k)
            if x_k.size < 3:
                history['x'].append(x_k)


        # Criterium
        if duality_gap_k <= tolerance:
            break

        if it >= max_iter or 2.*n / t_k < 1e-16:
            return (x_k, u_k), 'iterations_exceeded', history

        # Solve
        X_star, status_inner, histery_inner = newton(oracle, np.concatenate([x_k, u_k]),
                                                     line_search_options={'method': 'Armijo', 'c1': c1},
                                                     tolerance=tolerance_inner, max_iter=max_iter_inner)

        # update
        x_k = X_star[:n]
        u_k = X_star[n:]
        t_k = t_k * gamma

    return (x_k, u_k), 'success', history


def subgradient_method(oracle, x_0, tolerance=1e-2, max_iter=1000, alpha_0=1,
                       display=False, trace=False):
    """
    Subgradient descent method for nonsmooth convex optimization.

    Parameters
    ----------
    oracle : BaseNonsmoothConvexOracle-descendant object
        Oracle with .func() and .subgrad() methods implemented for computing
        function value and its one (arbitrary) subgradient respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    alpha_0 : float
        Initial value for the sequence of step-sizes.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0)
    x_best = np.copy(x_0)

    t0 = datetime.now()

    for it in range(max_iter + 1):

        # Oracle
        f_x_k = oracle.func(x_k)
        subgrad_x_k = oracle.subgrad(x_k)
        norm_sub_grad_x_k = np.linalg.norm(subgrad_x_k)
        duality_gap_x_k = oracle.duality_gap(x_k)
        if (f_x_k < oracle.func(x_best)):
            x_best = np.copy(x_k)

        # Debug info
        if display:
            print(it)

        # Fill trace data
        if trace:
            history['func'].append(f_x_k)
            history['time'].append((datetime.now() - t0).total_seconds())
            history['duality_gap'].append(duality_gap_x_k)
            if x_k.size < 3:
                history['x'].append(x_k)

        # Criterium
        if duality_gap_x_k < tolerance:
            break

        if it >= max_iter:
            return x_best, 'iterations_exceeded', history

        # Direction
        d_k = - subgrad_x_k / norm_sub_grad_x_k

        # Line search
        alpha = alpha_0 / np.sqrt(it + 1)

        x_k = x_k + alpha * d_k

    return x_best, 'success', history



def proximal_gradient_descent(oracle, x_0, L_0=1, tolerance=1e-5,
                              max_iter=1000, trace=False, display=False, print_number_iterations=False):
    """
    Proximal gradient descent for composite optimization.

    Parameters
    ----------
    oracle : BaseCompositeOracle-descendant object
        Oracle with .func() and .grad() and .prox() methods implemented
        for computing function value, its gradient and proximal mapping
        respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    L_0 : float
        Initial value for adaptive line-search.
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0)
    L_k = 2 * np.copy(L_0)

    t0 = datetime.now()
    total_lin_iterations = 0.

    for it in range(max_iter + 1):
        # Oracle
        phi_x_k = oracle.func(x_k)
        f_x_k = oracle._f.func(x_k)
        grad_f_x_k = oracle._f.grad(x_k)
        duality_gap_x_k = oracle.duality_gap(x_k)
        L_k = np.max([L_k * 0.5, L_0])

        # Debug info
        if display:
            print(it)

        # Fill trace data
        if trace:
            history['func'].append(phi_x_k)
            history['time'].append((datetime.now() - t0).total_seconds())
            history['duality_gap'].append(duality_gap_x_k)
            if x_k.size < 3:
                history['x'].append(x_k)
            if print_number_iterations:
                history['iterations'].append(total_lin_iterations)


        # Criterium
        if duality_gap_x_k < tolerance:
            break

        if it >= max_iter:
            return x_k, 'iterations_exceeded', history

        # Line search
        while True:
            total_lin_iterations += 1
            alpha = 1. / L_k
            y = oracle.prox(x_k - alpha * grad_f_x_k, alpha)
            v = y - x_k
            if oracle._f.func(y) <= oracle._f.func(x_k) + np.dot(grad_f_x_k, v) + 0.5 * L_k * np.dot(v, v):
                x_k = y.copy()
                break
            L_k = 2 * L_k


    return x_k, 'success', history
