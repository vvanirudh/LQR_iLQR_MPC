"""LQR, iLQR and MPC."""

import numpy as np
import scipy


def simulate_dynamics(env, x, u, dt=1e-5):
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
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    xdot: np.array
      This is the **CHANGE** in x. i.e. (x[1] - x[0]) / dt
      If you return x you will need to solve a different equation in
      your LQR controller.
    """
    # Set state
    env.state(x)
    # Take a step
    new_x = env._step(u, dt)
    # Compute xdot
    xdot = (new_x - x) / (dt)
    return xdot


def approximate_A(env, x, u, delta=1e-5, dt=1e-5):
    """Approximate A matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. You will need to perturb this.
    u: np.array
      The command to test.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    A: np.array
      The A matrix for the dynamics at state x and command u.
    """
    A = np.zeros((x.shape[0], x.shape[0]))
    # For each dimension
    for ind in range(x.shape[0]):
        # Positive step
        x[ind] += delta
        xdot_plus = simulate_dynamics(env, x, u, dt)
        # Negative step
        x[ind] -= 2*delta
        xdot_minus = simulate_dynamics(env, x, u, dt)
        # Restore
        x[ind] += delta
        A[:, ind] = (xdot_plus - xdot_minus) / (2 * delta)
    return A


def approximate_B(env, x, u, delta=1e-5, dt=1e-5):
    """Approximate B matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test.
    u: np.array
      The command to test. You will ned to perturb this.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    B: np.array
      The B matrix for the dynamics at state x and command u.
    """
    B = np.zeros((x.shape[0], u.shape[0]))
    # For each dimension
    for ind in range(u.shape[0]):
        # Positive step
        u[ind] += delta
        xdot_plus = simulate_dynamics(env, x, u, dt)
        # Negative step
        u[ind] -= 2*delta
        xdot_minus = simulate_dynamics(env, x, u, dt)
        # Restore
        u[ind] += delta
        B[:, ind] = (xdot_plus - xdot_minus) / (2 * delta)
    return B


def calc_lqr_input(env, sim_env):
    """Calculate the optimal control input for the given state.

    If you are following the API and simulate dynamics is returning
    xdot, then you should use the scipy.linalg.solve_continuous_are
    function to solve the Ricatti equations.

    Parameters
    ----------
    env: gym.core.Env
      This is the true environment you will execute the computed
      commands on. Use this environment to get the Q and R values as
      well as the state.
    sim_env: gym.core.Env
      A copy of the env class. Use this to simulate the dynamics when
      doing finite differences.

    Returns
    -------
    u: np.array
      The command to execute at this point.
    """
    # Get state of the environment
    x = env.state()
    # Get A and B
    x_sim = x.copy()
    # TODO: what control input to use when simulating dynamics?
    u_sim = np.ones((2,))
    A = approximate_A(sim_env, x_sim, u_sim)
    B = approximate_B(sim_env, x_sim, u_sim)

    # Solve ricatti
    P = scipy.linalg.solve_continuous_are(A, B, env.Q, env.R)
    K = np.dot(np.linalg.pinv(env.R), np.dot(B.T, P))

    # Compute state difference
    x_diff = x - env.goal()

    # Compute u
    u = -np.dot(K, x_diff)
    return u
