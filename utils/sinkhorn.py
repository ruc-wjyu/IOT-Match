# -*- coding: utf-8 -*-
"""
Rewrite ot.bregman.sinkhorn in Python Optimal Transport (https://pythonot.github.io/_modules/ot/bregman.html#sinkhorn)
using pytorch operations.
Bregman projections for regularized OT (Sinkhorn distance).
"""

import torch

M_EPS = 1e-16


def sinkhorn(a, b, C, reg=1e-1, method='sinkhorn', maxIter=100, tau=1e3,
             stopThr=1e-9, verbose=False, log=False, warm_start=None, eval_freq=10, print_freq=200, **kwargs):
    """
    Solve the entropic regularization optimal transport
    The input should be PyTorch tensors
    The function solves the following optimization problem:
    .. math::
        \gamma = arg\min_\gamma <\gamma,C>_F + reg\cdot\Omega(\gamma)
        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma\geq 0
    where :
    - C is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are target and source measures (sum to 1)
    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [1].
    Parameters
    ----------
    a : torch.tensor (na,)
        samples measure in the target domain
    b : torch.tensor (nb,)
        samples in the source domain
    C : torch.tensor (na,nb)
        loss matrix
    reg : float
        Regularization term > 0
    method : str
        method used for the solver either 'sinkhorn', 'greenkhorn', 'sinkhorn_stabilized' or
        'sinkhorn_epsilon_scaling', see those function for specific parameters
    maxIter : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error ( > 0 )
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    Returns
    -------
    gamma : (na x nb) torch.tensor
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters
    References
    ----------
    [1] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013
    See Also
    --------
    """

    if method.lower() == 'sinkhorn':
        return sinkhorn_knopp(a, b, C, reg, maxIter=maxIter,
                              stopThr=stopThr, verbose=verbose, log=log,
                              warm_start=warm_start, eval_freq=eval_freq, print_freq=print_freq,
                              **kwargs)
    elif method.lower() == 'sinkhorn_stabilized':
        return sinkhorn_stabilized(a, b, C, reg, maxIter=maxIter, tau=tau,
                                   stopThr=stopThr, verbose=verbose, log=log,
                                   warm_start=warm_start, eval_freq=eval_freq, print_freq=print_freq,
                                   **kwargs)
    elif method.lower() == 'sinkhorn_epsilon_scaling':
        return sinkhorn_epsilon_scaling(a, b, C, reg,
                                        maxIter=maxIter, maxInnerIter=100, tau=tau,
                                        scaling_base=0.75, scaling_coef=None, stopThr=stopThr,
                                        verbose=False, log=log, warm_start=warm_start, eval_freq=eval_freq,
                                        print_freq=print_freq, **kwargs)
    else:
        raise ValueError("Unknown method '%s'." % method)


def sinkhorn_knopp(a, b, C, reg=1e-1, maxIter=10, stopThr=1e-9,
                   verbose=False, log=False, warm_start=None, eval_freq=10, print_freq=200, **kwargs):
    """
    Solve the entropic regularization optimal transport
    The input should be PyTorch tensors
    The function solves the following optimization problem:
    .. math::
        \gamma = arg\min_\gamma <\gamma,C>_F + reg\cdot\Omega(\gamma)
        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma\geq 0
    where :
    - C is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are target and source measures (sum to 1)
    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [1].
    Parameters
    ----------
    a : torch.tensor (na,)
        samples measure in the target domain
    b : torch.tensor (nb,)
        samples in the source domain
    C : torch.tensor (na,nb)
        loss matrix
    reg : float
        Regularization term > 0
    maxIter : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error ( > 0 )
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    Returns
    -------
    gamma : (na x nb) torch.tensor
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters
    References
    ----------
    [1] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013
    See Also
    --------
    """
    # ywj
    a = a.double()
    b = b.double()
    C = C.double()

    device = a.device
    na, nb = C.shape

    assert na >= 1 and nb >= 1, 'C needs to be 2d'
    assert na == a.shape[0] and nb == b.shape[0], "Shape of a or b does't match that of C"
    assert reg > 0, 'reg should be greater than 0'
    assert a.min() >= 0. and b.min() >= 0., 'Elements in a or b less than 0'

    if log:
        log = {'err': []}

    if warm_start is not None:
        u = warm_start['u']
        v = warm_start['v']
    else:
        u = torch.ones(na, dtype=a.dtype).to(device) / na
        v = torch.ones(nb, dtype=b.dtype).to(device) / nb

    K = torch.empty(C.shape, dtype=C.dtype)
    # torch.div(C, -reg, out=K)
    K = torch.div(C, -reg).to(device)
    #torch.exp(K, out=K)
    K = torch.exp(K)
    b_hat = torch.empty(b.shape, dtype=C.dtype).to(device)

    it = 1
    err = 1
    '''
    # allocate memory beforehand
    KTu = torch.empty(v.shape, dtype=v.dtype).to(device)
    Kv = torch.empty(u.shape, dtype=u.dtype).to(device)
    '''
    while (it <= maxIter):  # 原版是or
        upre, vpre = u, v
        #torch.matmul(u, K, out=KTu)
        KTu = torch.matmul(u, K).to(device)
        v = torch.div(b, KTu + M_EPS).to(device)
        #torch.matmul(K, v, out=Kv)
        Kv = torch.matmul(K, v)

        u = torch.div(a, Kv + M_EPS).to(device)

        if torch.any(torch.isnan(u)) or torch.any(torch.isnan(v)) or \
                torch.any(torch.isinf(u)) or torch.any(torch.isinf(v)):
            print('Warning: numerical errors at iteration', it)
            u, v = upre, vpre
            break

        if log and it % eval_freq == 0:
            # we can speed up the process by checking for the error only all
            # the eval_freq iterations
            # below is equivalent to:
            # b_hat = torch.sum(u.reshape(-1, 1) * K * v.reshape(1, -1), 0)
            # but with more memory efficient
            b_hat = torch.matmul(u, K) * v
            err = (b - b_hat).pow(2).sum().item()
            # err = (b - b_hat).abs().sum().item()
            log['err'].append(err)

        if verbose and it % print_freq == 0:
            print('iteration {:5d}, constraint error {:5e}'.format(it, err))

        it += 1

    if log:
        log['u'] = u
        log['v'] = v
        log['alpha'] = reg * torch.log(u + M_EPS)
        log['beta'] = reg * torch.log(v + M_EPS)

    # transport plan
    P = u.reshape(-1, 1) * K * v.reshape(1, -1)
    if log:
        return P, log
    else:
        return P


def sinkhorn_stabilized(a, b, C, reg=1e-1, maxIter=1000, tau=1e3, stopThr=1e-9,
                        verbose=False, log=False, warm_start=None, eval_freq=10, print_freq=200, **kwargs):
    """
    Solve the entropic regularization OT problem with log stabilization
    The function solves the following optimization problem:
    .. math::
        \gamma = arg\min_\gamma <\gamma,C>_F + reg\cdot\Omega(\gamma)
        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma\geq 0
    where :
    - C is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are target and source measures (sum to 1)
    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [1]
    but with the log stabilization proposed in [3] an defined in [2] (Algo 3.1)
    Parameters
    ----------
    a : torch.tensor (na,)
        samples measure in the target domain
    b : torch.tensor (nb,)
        samples in the source domain
    C : torch.tensor (na,nb)
        loss matrix
    reg : float
        Regularization term > 0
    tau : float
        thershold for max value in u or v for log scaling
    maxIter : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error ( > 0 )
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    Returns
    -------
    gamma : (na x nb) torch.tensor
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters
    References
    ----------
    [1] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013
    [2] Bernhard Schmitzer. Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems. SIAM Journal on Scientific Computing, 2019
    [3] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016). Scaling algorithms for unbalanced transport problems. arXiv preprint arXiv:1607.05816.
    See Also
    --------
    """

    device = a.device
    na, nb = C.shape

    assert na >= 1 and nb >= 1, 'C needs to be 2d'
    assert na == a.shape[0] and nb == b.shape[0], "Shape of a or b does't match that of C"
    assert reg > 0, 'reg should be greater than 0'
    assert a.min() >= 0. and b.min() >= 0., 'Elements in a or b less than 0'

    if log:
        log = {'err': []}

    if warm_start is not None:
        alpha = warm_start['alpha']
        beta = warm_start['beta']
    else:
        alpha = torch.zeros(na, dtype=a.dtype).to(device)
        beta = torch.zeros(nb, dtype=b.dtype).to(device)

    u = torch.ones(na, dtype=a.dtype).to(device) / na
    v = torch.ones(nb, dtype=b.dtype).to(device) / nb

    def update_K(alpha, beta):
        """log space computation"""
        """memory efficient"""
        torch.add(alpha.reshape(-1, 1), beta.reshape(1, -1), out=K)
        torch.add(K, -C, out=K)
        torch.div(K, reg, out=K)
        torch.exp(K, out=K)

    def update_P(alpha, beta, u, v, ab_updated=False):
        """log space P (gamma) computation"""
        torch.add(alpha.reshape(-1, 1), beta.reshape(1, -1), out=P)
        torch.add(P, -C, out=P)
        torch.div(P, reg, out=P)
        if not ab_updated:
            torch.add(P, torch.log(u + M_EPS).reshape(-1, 1), out=P)
            torch.add(P, torch.log(v + M_EPS).reshape(1, -1), out=P)
        torch.exp(P, out=P)

    K = torch.empty(C.shape, dtype=C.dtype).to(device)
    update_K(alpha, beta)

    b_hat = torch.empty(b.shape, dtype=C.dtype).to(device)

    it = 1
    err = 1
    ab_updated = False

    # allocate memory beforehand
    KTu = torch.empty(v.shape, dtype=v.dtype).to(device)
    Kv = torch.empty(u.shape, dtype=u.dtype).to(device)
    P = torch.empty(C.shape, dtype=C.dtype).to(device)

    while (err > stopThr and it <= maxIter):
        upre, vpre = u, v
        torch.matmul(u, K, out=KTu)
        v = torch.div(b, KTu + M_EPS)
        torch.matmul(K, v, out=Kv)
        u = torch.div(a, Kv + M_EPS)

        ab_updated = False
        # remove numerical problems and store them in K
        if u.abs().sum() > tau or v.abs().sum() > tau:
            alpha += reg * torch.log(u + M_EPS)
            beta += reg * torch.log(v + M_EPS)
            u.fill_(1. / na)
            v.fill_(1. / nb)
            update_K(alpha, beta)
            ab_updated = True

        if log and it % eval_freq == 0:
            # we can speed up the process by checking for the error only all
            # the eval_freq iterations
            update_P(alpha, beta, u, v, ab_updated)
            b_hat = torch.sum(P, 0)
            err = (b - b_hat).pow(2).sum().item()
            log['err'].append(err)

        if verbose and it % print_freq == 0:
            print('iteration {:5d}, constraint error {:5e}'.format(it, err))

        it += 1

    if log:
        log['u'] = u
        log['v'] = v
        log['alpha'] = alpha + reg * torch.log(u + M_EPS)
        log['beta'] = beta + reg * torch.log(v + M_EPS)

    # transport plan
    update_P(alpha, beta, u, v, False)

    if log:
        return P, log
    else:
        return P


def sinkhorn_epsilon_scaling(a, b, C, reg=1e-1, maxIter=100, maxInnerIter=100, tau=1e3, scaling_base=0.75,
                             scaling_coef=None, stopThr=1e-9, verbose=False, log=False, warm_start=None, eval_freq=10,
                             print_freq=200, **kwargs):
    """
    Solve the entropic regularization OT problem with log stabilization
    The function solves the following optimization problem:
    .. math::
        \gamma = arg\min_\gamma <\gamma,C>_F + reg\cdot\Omega(\gamma)
        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma\geq 0
    where :
    - C is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are target and source measures (sum to 1)
    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix
    scaling algorithm as proposed in [1] but with the log stabilization
    proposed in [3] and the log scaling proposed in [2] algorithm 3.2
    Parameters
    ----------
    a : torch.tensor (na,)
        samples measure in the target domain
    b : torch.tensor (nb,)
        samples in the source domain
    C : torch.tensor (na,nb)
        loss matrix
    reg : float
        Regularization term > 0
    tau : float
        thershold for max value in u or v for log scaling
    maxIter : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error ( > 0 )
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    Returns
    -------
    gamma : (na x nb) torch.tensor
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters
    References
    ----------
    [1] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013
    [2] Bernhard Schmitzer. Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems. SIAM Journal on Scientific Computing, 2019
    [3] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016). Scaling algorithms for unbalanced transport problems. arXiv preprint arXiv:1607.05816.
    See Also
    --------
    """

    na, nb = C.shape

    assert na >= 1 and nb >= 1, 'C needs to be 2d'
    assert na == a.shape[0] and nb == b.shape[0], "Shape of a or b does't match that of C"
    assert reg > 0, 'reg should be greater than 0'
    assert a.min() >= 0. and b.min() >= 0., 'Elements in a or b less than 0'

    def get_reg(it, reg, pre_reg):
        if it == 1:
            return scaling_coef
        else:
            if (pre_reg - reg) * scaling_base < M_EPS:
                return reg
            else:
                return (pre_reg - reg) * scaling_base + reg

    if scaling_coef is None:
        scaling_coef = C.max() + reg

    it = 1
    err = 1
    running_reg = scaling_coef

    if log:
        log = {'err': []}

    warm_start = None

    while (err > stopThr and it <= maxIter):
        running_reg = get_reg(it, reg, running_reg)
        P, _log = sinkhorn_stabilized(a, b, C, running_reg, maxIter=maxInnerIter, tau=tau,
                                      stopThr=stopThr, verbose=False, log=True,
                                      warm_start=warm_start, eval_freq=eval_freq, print_freq=print_freq,
                                      **kwargs)

        warm_start = {}
        warm_start['alpha'] = _log['alpha']
        warm_start['beta'] = _log['beta']

        primal_val = (C * P).sum() + reg * (P * torch.log(P)).sum() - reg * P.sum()
        dual_val = (_log['alpha'] * a).sum() + (_log['beta'] * b).sum() - reg * P.sum()
        err = primal_val - dual_val
        log['err'].append(err)

        if verbose and it % print_freq == 0:
            print('iteration {:5d}, constraint error {:5e}'.format(it, err))

        it += 1

    if log:
        log['alpha'] = _log['alpha']
        log['beta'] = _log['beta']
        return P, log
    else:
        return P


def entropic_partial_wasserstein(a, b, M, reg, m=None, numItermax=100,
                                 stopThr=1e-100, verbose=False, log=False):
    r"""
    Solves the partial optimal transport problem
    and returns the OT plan

    The function considers the following problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)

        s.t. \gamma 1 \leq a \\
             \gamma^T 1 \leq b \\
             \gamma\geq 0 \\
             1^T \gamma^T 1 = m \leq \min\{\|a\|_1, \|b\|_1\} \\

    where :

    - M is the metric cost matrix
    - :math:`\Omega`  is the entropic regularization term
        :math:`\Omega=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are the sample weights
    - m is the amount of mass to be transported

    The formulation of the problem has been proposed in [3]_ (prop. 5)


    Parameters
    ----------
    a : np.ndarray (dim_a,)
        Unnormalized histogram of dimension dim_a
    b : np.ndarray (dim_b,)
        Unnormalized histograms of dimension dim_b
    M : np.ndarray (dim_a, dim_b)
        cost matrix
    reg : float
        Regularization term > 0
    m : float, optional
        Amount of mass to be transported
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    gamma : (dim_a x dim_b) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary returned only if `log` is `True`


    Examples
    --------
    >>> import ot
    >>> a = [.1, .2]
    >>> b = [.1, .1]
    >>> M = [[0., 1.], [2., 3.]]
    >>> np.round(entropic_partial_wasserstein(a, b, M, 1, 0.1), 2)
    array([[0.06, 0.02],
           [0.01, 0.  ]])

    References
    ----------
    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G.
       (2015). Iterative Bregman projections for regularized transportation
       problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.

    See Also
    --------
    ot.partial.partial_wasserstein: exact Partial Wasserstein
    """
    a = a.double()
    b = b.double()
    M = M.double()
    device = a.device
    # m = m.to(device)

    dim_a, dim_b = M.shape
    #dx = np.ones(dim_a, dtype=np.float64)
    dx = torch.ones(dim_a, dtype=a.dtype).to(device)
    #dy = np.ones(dim_b, dtype=np.float64)
    dy = torch.ones(dim_b, dtype=b.dtype).to(device)

    if len(a) == 0:
        #a = np.ones(dim_a, dtype=np.float64) / dim_a
        a = torch.ones(dim_a, dtype=a.dtype) / dim_a

    if len(b) == 0:
        #b = np.ones(dim_b, dtype=np.float64) / dim_b
        b = torch.ones(dim_b, dtype=b.dtype) / dim_b

    if m is None:
        #m = np.min((np.sum(a), np.sum(b))) * 1.0
        m = torch.min(a.sum(), b.sum()) * 1.0

    if m < 0:
        raise ValueError("Problem infeasible. Parameter m should be greater"
                         " than 0.")
    if m > torch.min(a.sum(), b.sum()):
        raise ValueError("Problem infeasible. Parameter m should lower or"
                         " equal than min(|a|_1, |b|_1).")

    log_e = {'err': []}

    # Next 3 lines equivalent to K=np.exp(-M/reg), but faster to compute
    #K = np.empty(M.shape, dtype=M.dtype)
    #K = torch.empty(M.shape, dtype=M.dtype).to(device)
    #np.divide(M, -reg, out=K)
    K = torch.div(M, -reg).to(device)
    #np.exp(K, out=K)
    K = torch.exp(K)
    K = K.to(device)
    #np.multiply(K, m / np.sum(K), out=K)
    K = K * (m / K.sum())
    # print('K shape:{}'.format(K.shape))
    # print('dx shape:{}'.format(dx.shape))
    # print('a shape:{}'.format(a.shape))
    # # temp = torch.diag(torch.min(a / torch.sum(K, dim=1), dx))
    # print('torch.sum(K, dim=1) shape:{}'.format(torch.sum(K, dim=1).shape))
    # print('a/torch.sum(K, dim=1) shape:{}'.format((a/torch.sum(K, dim=1)).shape))

    err, cpt = 1, 0
    eps = 1e-6
    while (cpt < numItermax):  # err > stopThr and
        #Kprev = K
        #K1 = np.dot(np.diag(np.minimum(a / np.sum(K, axis=1), dx)), K)
        #print('K shape:{}'.format(K.shape))
        #print('a / torch.sum(K, dim=1) shape:{}'.format((a / torch.sum(K, dim=1)).shape))
        #print('dx shape:{}'.format(dx.shape))
        K1 = torch.matmul(torch.diag(torch.min(a / (torch.sum(K, dim=1)), dx)), K)
        #K2 = np.dot(K1, np.diag(np.minimum(b / np.sum(K1, axis=0), dy)))
        K2 = torch.matmul(K1, torch.diag(torch.min(b / (torch.sum(K1, dim=0)), dy)))

        K = K2 * (m / torch.sum(K2))

        cpt = cpt + 1
        #print('K:{}'.format(K))
        if torch.any(torch.isnan(K)) or torch.any(torch.isinf(K)):
            print('Warning: numerical errors at iteration', cpt)
            break
        '''
        if cpt % 10 == 0:
            err = np.linalg.norm(Kprev - K)
            if log:
                log_e['err'].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 11)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt = cpt + 1
    log_e['partial_w_dist'] = np.sum(M * K)
    if log:
        return K, log_e
    else:
        return K
    '''
    return K



def batch_entropic_partial_wasserstein(a, b, M, reg, m=None, numItermax=10,
                                 stopThr=1e-100, verbose=False, log=False):
    r"""
    Solves the partial optimal transport problem
    and returns the OT plan

    The function considers the following problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)

        s.t. \gamma 1 \leq a \\
             \gamma^T 1 \leq b \\
             \gamma\geq 0 \\
             1^T \gamma^T 1 = m \leq \min\{\|a\|_1, \|b\|_1\} \\

    where :

    - M is the metric cost matrix
    - :math:`\Omega`  is the entropic regularization term
        :math:`\Omega=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are the sample weights
    - m is the amount of mass to be transported

    The formulation of the problem has been proposed in [3]_ (prop. 5)


    Parameters
    ----------
    a : np.ndarray (dim_a,)
        Unnormalized histogram of dimension dim_a
    b : np.ndarray (dim_b,)
        Unnormalized histograms of dimension dim_b
    M : np.ndarray (dim_a, dim_b)
        cost matrix
    reg : float
        Regularization term > 0
    m : float, optional
        Amount of mass to be transported
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    gamma : (dim_a x dim_b) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary returned only if `log` is `True`


    Examples
    --------
    >>> import ot
    >>> a = [.1, .2]
    >>> b = [.1, .1]
    >>> M = [[0., 1.], [2., 3.]]
    >>> np.round(entropic_partial_wasserstein(a, b, M, 1, 0.1), 2)
    array([[0.06, 0.02],
           [0.01, 0.  ]])

    References
    ----------
    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G.
       (2015). Iterative Bregman projections for regularized transportation
       problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.

    See Also
    --------
    ot.partial.partial_wasserstein: exact Partial Wasserstein
    """
    a = a.double()
    b = b.double()
    M = M.double()
    device = a.device
    # m = m.to(device)

    batch_size, dim_a, dim_b = M.shape
    #dx = np.ones(dim_a, dtype=np.float64)
    dx = torch.ones(batch_size, dim_a, dtype=a.dtype).to(device)
    #dy = np.ones(dim_b, dtype=np.float64)
    dy = torch.ones(batch_size, dim_b, dtype=b.dtype).to(device)

    # if len(a) == 0:
    #     #a = np.ones(dim_a, dtype=np.float64) / dim_a
    #     a = torch.ones(dim_a, dtype=a.dtype) / dim_a
    #
    # if len(b) == 0:
    #     #b = np.ones(dim_b, dtype=np.float64) / dim_b
    #     b = torch.ones(dim_b, dtype=b.dtype) / dim_b
    #
    # if m is None:
    #     #m = np.min((np.sum(a), np.sum(b))) * 1.0
    #     m = torch.min(a.sum(), b.sum()) * 1.0
    #
    # if m < 0:
    #     raise ValueError("Problem infeasible. Parameter m should be greater"
    #                      " than 0.")
    # if m > torch.min(a.sum(), b.sum()):
    #     raise ValueError("Problem infeasible. Parameter m should lower or"
    #                      " equal than min(|a|_1, |b|_1).")
    #
    # log_e = {'err': []}

    # Next 3 lines equivalent to K=np.exp(-M/reg), but faster to compute
    #K = np.empty(M.shape, dtype=M.dtype)
    #K = torch.empty(M.shape, dtype=M.dtype).to(device)
    #np.divide(M, -reg, out=K)
    K = torch.div(M, -reg).to(device)
    #np.exp(K, out=K)
    K = torch.exp(K)
    K = K.to(device)
    #np.multiply(K, m / np.sum(K), out=K)
    K = K * (m / K.sum())

    err, cpt = 1, 0

    while(cpt < numItermax):

        print('a shape:{}'.format(a.shape))
        print('torch.sum(K, dim=1) shape:{}'.format(torch.sum(K, dim=1).shape))

        K1 = torch.matmul(torch.diag(torch.min(a / torch.sum(K, dim=1), dx)), K)

        #K2 = np.dot(K1, np.diag(np.minimum(b / np.sum(K1, axis=0), dy)))
        K2 = torch.matmul(K1, torch.diag(torch.min(b / torch.sum(K1, dim=0), dy)))

        K = K2 * (m / torch.sum(K2))

        cpt = cpt + 1
        #print('K:{}'.format(K))
        if torch.any(torch.isnan(K)) or torch.any(torch.isinf(K)):
            print('Warning: numerical errors at iteration', cpt)
            break
        '''
        if cpt % 10 == 0:
            err = np.linalg.norm(Kprev - K)
            if log:
                log_e['err'].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 11)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt = cpt + 1
    log_e['partial_w_dist'] = np.sum(M * K)
    if log:
        return K, log_e
    else:
        return K
    '''
    return K

def geometricMean(alldistribT):
    """return the  geometric mean of distributions"""
    return torch.exp(torch.mean(torch.log(alldistribT), dim=1))


def geometricBar(weights, alldistribT):
    """return the weight geometric mean of distributions"""
    assert(len(weights) == alldistribT.shape[1])
    return torch.exp(torch.matmul(torch.log(alldistribT), weights.T))


def barycenter_sinkhorn(A, M, reg, weights=None, numItermax=20, verbose=False, log=False):
    r"""Compute the entropic regularized wasserstein barycenter of distributions A
     The function solves the following optimization problem:
    .. math::
       \mathbf{a} = arg\min_\mathbf{a} \sum_i W_{reg}(\mathbf{a},\mathbf{a}_i)
    where :
    - :math:`W_{reg}(\cdot,\cdot)` is the entropic regularized Wasserstein distance (see ot.bregman.sinkhorn)
    - :math:`\mathbf{a}_i` are training distributions in the columns of matrix :math:`\mathbf{A}`
    - reg and :math:`\mathbf{M}` are respectively the regularization term and the cost matrix for OT
    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [3]_
    Parameters
    ----------
    A : ndarray, shape (dim, n_hists)
        n_hists training distributions a_i of size dim
    M : ndarray, shape (dim, dim)
        loss matrix for OT
    reg : float
        Regularization term > 0
    weights : ndarray, shape (n_hists,)
        Weights of each histogram a_i on the simplex (barycentric coodinates)
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    Returns
    -------
    a : (dim,) ndarray
        Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters
    References
    ----------
    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G. (2015). Iterative Bregman projections for regularized transportation problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.
    """

    if weights is None:
        weights = torch.ones(A.shape[1]).div(A.shape[1])
    else:
        assert (len(weights) == A.shape[1])

    if log:
        log = {'err': []}

    # M = M/np.median(M) # suggested by G. Peyre
    K = torch.exp(-M / reg)

    cpt = 0
    err = 1

    UKv = torch.matmul(K, A.T.div(torch.sum(K, dim=0)).T)
    u = (geometricMean(UKv) / UKv.T).T

    while (cpt < numItermax):
        cpt = cpt + 1
        UKv = u * torch.matmul(K, A.div(K.matmul(u)))
        u = (u.T * geometricBar(weights, UKv)).T.div(UKv)

        if cpt % 10 == 1:
            err = torch.sum(torch.std(UKv, dim=1))

            # log and verbose print
            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

    return geometricBar(weights, UKv)

# a = torch.rand(10,5)
# b = torch.rand(10)
#
# D = torch.rand(10,10)
#
# res = barycenter_sinkhorn(a,D,0.1)
# print(res.shape)

# a = torch.rand(3,5)
# b = torch.rand(3,5)
# cost = torch.cdist(a,b)
# plan = entropic_partial_wasserstein(a,b,cost,reg=1.)
