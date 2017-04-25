"""LQR, iLQR and MPC."""

from deeprl_hw3.controllers import approximate_A, approximate_B
import numpy as np
import scipy.linalg
import ipdb


def simulate_dynamics_next(env, x, u):
    """Step simulator to see how state changes.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.
    u: np.array
      The command to test. When approximating B you will need to
      perturb this.

    Returns
    -------
    next_x: np.array
    """
    # set state
    env.state = np.copy(x)
    # Take a step
    new_x, _, _, _ = env._step(u, env.dt)
    return new_x


def cost_inter(env, x, u):
    """intermediate cost function

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.
    u: np.array
      The command to test. When approximating B you will need to
      perturb this.

    Returns
    -------
    l, l_x, l_xx, l_u, l_uu, l_ux. The first term is the loss, where the remaining terms are derivatives respect to the
    corresponding variables, ex: (1) l_x is the first order derivative d l/d x (2) l_xx is the second order derivative
    d^2 l/d x^2
    """
    action_dim = u.shape[0]
    state_dim = x.shape[0]

    l = np.linalg.norm(u)**2

    l_x = np.zeros(state_dim)
    l_xx = np.zeros((state_dim, state_dim))
    l_u = 2*u
    # l_uu = 2*np.ones((action_dim, action_dim))
    l_uu = 2 * np.eye(action_dim)
    l_ux = np.zeros((action_dim, state_dim))
    return l, l_x, l_xx, l_u, l_uu, l_ux


def cost_final(env, x):
    """cost function of the last step

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.

    Returns
    -------
    l, l_x, l_xx The first term is the loss, where the remaining terms are derivatives respect to the
    corresponding variables
    """
    state_dim = x.shape[0]
    mult = 1e4
    goal = env.goal
    l = mult * (np.linalg.norm(x - goal)**2)
    l_x = mult * 2 * (x - goal)
    # l_xx = mult * 2 * np.ones((state_dim, state_dim))
    l_xx = mult * 2 * np.eye(state_dim)
    return l, l_x, l_xx


def simulate(env, x0, U):
    """
    Simulates the real environment taking initial state x0
    and control inputs U
    """
    # Set initial state
    # env.state = np.copy(x0)
    tN = U.shape[0]
    d = x0.shape[0]
    X = np.zeros((tN, d))
    X[0, :] = np.copy(x0)
    cost = 0
    for i in range(tN-1):
        new_x = simulate_dynamics_next(env, X[i, :], U[i, :])  # env._step(U[i, :])
        X[i+1, :] = np.copy(new_x)
        l, _, _, _, _, _ = cost_inter(env, X[i, :], U[i, :])
        cost = cost + l * env.dt

    # Final cost
    l, _, _ = cost_final(env, X[tN-1, :])
    cost = cost + l
    return X, cost


def calc_ilqr_input(env, sim_env, tN=50, max_iter=1e6):
    """Calculate the optimal control input for the given state.


    Parameters
    ----------
    env: gym.core.Env
      This is the true environment you will execute the computed
      commands on. Use this environment to get the Q and R values as
      well as the state.
    sim_env: gym.core.Env
      A copy of the env class. Use this to simulate the dynamics when
      doing finite differences.
    tN: number of control steps you are going to execute
    max_itr: max iterations for optmization

    Returns
    -------
    U: np.array
      The SEQUENCE of commands to execute. The size should be (tN, #parameters)
    """
    x0 = np.copy(env.state)

    action_dim = env.DOF
    state_dim = x0.shape[0]
    dt = env.dt

    U = np.zeros((tN, action_dim))
    cost_list = []
    lam = 1.0  # regularization parameter
    alpha = 0.1  # line search parameter
    eps = 0.001  # convergence check parameter
    lam_update = 10  # lambda update parameter
    lam_max = 10000  # Lambda maximum parameter
    forward_pass = True

    for iteration in range(int(max_iter)):

        if forward_pass:
            # Forward pass
            X, cost = simulate(sim_env, x0, U)
            oldcost = np.copy(cost)

            # Compute fx, fu, lx, lu, lxx, lux, luu at all timesteps
            fx = np.zeros((tN, state_dim, state_dim))
            fu = np.zeros((tN, state_dim, action_dim))
            l = np.zeros((tN, 1))
            lx = np.zeros((tN, state_dim))
            lu = np.zeros((tN, action_dim))
            lxx = np.zeros((tN, state_dim, state_dim))
            lux = np.zeros((tN, action_dim, state_dim))
            luu = np.zeros((tN, action_dim, action_dim))
            for tstep in range(tN-1):
                A = approximate_A(sim_env, X[tstep], U[tstep])
                B = approximate_B(sim_env, X[tstep], U[tstep])

                fx[tstep] = np.eye(state_dim, state_dim) + A * dt
                fu[tstep] = B * dt

                l[tstep], lx[tstep], lxx[tstep], lu[tstep], luu[tstep], lux[tstep] = cost_inter(env, X[tstep], U[tstep])

                l[tstep], lx[tstep], lxx[tstep], lu[tstep], luu[tstep], lux[tstep] = l[tstep] * dt, lx[tstep] * dt, lxx[tstep] * dt, lu[tstep] * dt, luu[tstep] * dt, lux[tstep] * dt
            # Last step
            l[tN-1], lx[tN-1], lxx[tN-1] = cost_final(env, X[tN-1])
            forward_pass = False

        # Compute gain
        V = np.copy(l[tN-1])
        Vx = np.copy(lx[tN-1])
        Vxx = np.copy(lxx[tN-1])
        k = np.zeros((tN, action_dim))
        K = np.zeros((tN, action_dim, state_dim))

        for tstep in range(tN-2, -1, -1):
            Qx = lx[tstep] + np.dot(fx[tstep].T, Vx)
            Qu = lu[tstep] + np.dot(fu[tstep].T, Vx)
            Qxx = lxx[tstep] + np.dot(fx[tstep].T, np.dot(Vxx, fx[tstep]))
            Qux = lux[tstep] + np.dot(fu[tstep].T, np.dot(Vxx, fx[tstep]))
            Quu = luu[tstep] + np.dot(fu[tstep].T, np.dot(Vxx, fu[tstep]))

            # SVD stuff
            # U_svd, S, V_svd = np.linalg.svd(Quu)
            # S[S < 0] = 0
            # S += lam
            # Quu_inv = np.dot(U_svd, np.dot(np.diag(1.0/S), V_svd.T))

            Quu_eigvalues, Quu_eigvectors = np.linalg.eig(Quu)
            Quu_eigvalues[Quu_eigvalues < 0] = 0.
            Quu_eigvalues += lam
            Quu_inv = np.dot(Quu_eigvectors, np.dot(np.diag(1.0/Quu_eigvalues), Quu_eigvectors.T))

            k[tstep] = -1 * np.dot(Quu_inv, Qu)
            K[tstep] = -1 * np.dot(Quu_inv, Qux)

            Vx = Qx - np.dot(K[tstep].T, np.dot(Quu, k[tstep]))
            Vxx = Qxx - np.dot(K[tstep].T, np.dot(Quu, K[tstep]))

        # Update control signal
        U_updated = np.zeros((tN, action_dim))
        x_updated = np.copy(x0)
        for tstep in range(tN-1):
            U_updated[tstep] = U[tstep] + alpha * k[tstep] + np.dot(K[tstep], x_updated - X[tstep])
            x_updated = simulate_dynamics_next(sim_env, x_updated, U_updated[tstep])
            # ipdb.set_trace()
        # Forward simulate to get cost
        X, cost = simulate(sim_env, x0, U_updated)
        newcost = np.copy(cost)
        cost_list.append(np.copy(newcost))

        # Compare costs
        if newcost < oldcost:
            # print 'less'
            lam = lam / lam_update
            U = np.copy(U_updated)
            # oldcost = np.copy(newcost)
            forward_pass = True

            # Check convergence
            if abs(newcost - oldcost) / oldcost < eps:
                print 'Converged at iteration:', iteration
                break

            oldcost = np.copy(newcost)

        else:
            # print 'more'
            # Update lambda and do update again
            lam *= lam_update
            forward_pass = False

            if lam > lam_max:
                print 'Lambda max reached'
                break

    return U, cost_list
