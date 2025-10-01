from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import norm

from GJ8 import gauss_jackson_8
from kepler_orbit import calculate_kepler_orbit, GM


def orbital_dynamics(t: float, y: np.ndarray, dy: np.ndarray, u: np.ndarray = None) -> np.ndarray:
    '''
    The equation of motion representing the orbit.

    r'' = -GM/|r^3| * r + u(t)

    Possible terms to put into u(t) are:

    - thrust: u(t) = m_dot / m * v_rel, in direction of travel (r', unless object can rotate)
    - atmospheric drag: u(t) = 1/(2m) * C_D(r') * rho(r) * A * |r'| ** 2, in direction of travel
    - parachute drag
    
    #### Arguments
    
    `t` (float): time since initial condition.
    `y` (np.ndarray): position vector (km)
    `dy` (np.ndarray): velocity vector (km / s)

    #### Optional Arguments

    `u` (np.ndarray): a control input (net force excluding gravity divided by mass, 
    in correct direction, kN / kg = km s^-2)
    
    #### Returns
    
    np.ndarray: acceleration vector (km / s^2)
    '''

    if u is None:
        u = np.zeros_like(y)
    
    mass = 100  # kg
    drag = (1/2 * 0.00 * 0.05 * 2.5 * norm(dy) * dy) * 0.001  # kN  # drag is set to zero for this demo
    
    return (-GM / (norm(y) ** 3)) * y - drag / mass

def runge_kutta_4(f, t_span: tuple, y0: np.ndarray, dy0: np.ndarray, h: float, u: np.ndarray = None):
    '''
    Fourth-order Runge-Kutta integrator for second-order ODEs.
    
    Solves y'' = f(t, y, y') with initial conditions y(t0) = y0, y'(t0) = dy0.
    
    #### Arguments
    
    `f` (callable): function f(t, y, dy, u) that returns y'' (acceleration)
    `t_span` (tuple): (t_start, t_end) time interval
    `y0` (np.ndarray): initial position vector
    `dy0` (np.ndarray): initial velocity vector
    `h` (float): time step size
    `u` (np.ndarray): optional control input
    
    #### Returns
    
    tuple: (t, y, dy, ddy) arrays of time, position, velocity, and acceleration
    '''
    
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / h) + 1
    
    # Initialize arrays
    t = np.linspace(t_start, t_end, n_steps)
    y = np.zeros((n_steps, len(y0)))
    dy = np.zeros((n_steps, len(dy0)))
    ddy = np.zeros((n_steps, len(dy0)))
    
    # Set initial conditions
    y[0] = y0
    dy[0] = dy0
    ddy[0] = f(t[0], y0, dy0, u)
    
    # RK4 integration loop
    for i in range(n_steps - 1):
        t_i = t[i]
        y_i = y[i]
        dy_i = dy[i]
        
        # k1 for velocity and acceleration
        k1_v = dy_i
        k1_a = f(t_i, y_i, dy_i, u)
        
        # k2 for velocity and acceleration
        k2_v = dy_i + 0.5 * h * k1_a
        k2_a = f(t_i + 0.5 * h, y_i + 0.5 * h * k1_v, dy_i + 0.5 * h * k1_a, u)
        
        # k3 for velocity and acceleration
        k3_v = dy_i + 0.5 * h * k2_a
        k3_a = f(t_i + 0.5 * h, y_i + 0.5 * h * k2_v, dy_i + 0.5 * h * k2_a, u)
        
        # k4 for velocity and acceleration
        k4_v = dy_i + h * k3_a
        k4_a = f(t_i + h, y_i + h * k3_v, dy_i + h * k3_a, u)
        
        # Update position and velocity
        y[i + 1] = y_i + (h / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        dy[i + 1] = dy_i + (h / 6.0) * (k1_a + 2 * k2_a + 2 * k3_a + k4_a)
        ddy[i + 1] = f(t[i + 1], y[i + 1], dy[i + 1], u)
    
    return t, y, dy, ddy


def main():

    # initial conditions
    R_0 = 7000                          # 7000 km
    V_0 = np.sqrt(GM / R_0)             # choose the speed for a circular orbit: 7.546053290107541 km/s
    T_0 = 2 * np.pi * R_0 / V_0

    print(f'Period of orbit: {T_0} seconds')

    t_span = (0, 100 * T_0)

    # calculate trajectory using GJ8
    print('Calculating trajectory using GJ8...')
    t, y, dy, ddy = gauss_jackson_8(orbital_dynamics,
        t_span, np.array([R_0, 0, 0]), np.array([0, V_0, 0]), 60, use_debug=True)

    # calculate circular orbit position
    print('Calculating trajectory using circular motion...')
    W = norm(V_0) / norm(R_0)
    r_circle = np.array([[R_0 * np.cos(W * t_i), R_0 * np.sin(W * t_i), 0] for t_i in t])
    errors_circle = [1e6 * np.hypot(r_circle[i, 0] - y[i, 0], r_circle[i, 1] - y[i, 1]) \
        for i in range(len(y))]

    # calculate orbit with Kepler's equation
    print('Calculating trajectory using Kepler equation...')
    r_kepler, _v_k, _a_k = calculate_kepler_orbit(t, np.array([R_0, 0, 0]), np.array([0, V_0, 0]))
    errors_kepler = [1e6 * np.hypot(r_kepler[i, 0] - y[i, 0], r_kepler[i, 1] - y[i, 1]) \
        for i in range(len(y))]

    # check kepler actually works (it does, to about 30 nm)
    kepler_vs_circle = [1e6 * np.hypot(r_kepler[i, 0] - r_circle[i, 0], r_kepler[i, 1] - r_circle[i, 1]) \
        for i in range(len(y))]

    # calculate orbit with RK4
    print('Calculating trajectory using RK4...')
    t_rk4, r_rk4, v_rk4, a_rk4 = runge_kutta_4(orbital_dynamics, t_span, np.array([R_0, 0, 0]), np.array([0, V_0, 0]), 60)
    errors_rk4 = [1e6 * np.hypot(r_rk4[i, 0] - r_kepler[i, 0], r_rk4[i, 1] - r_kepler[i, 1]) \
        for i in range(len(t_rk4))]

    # plots
    print('Plotting graphs...')
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Gauss-Jackson Orbit Propogation Results')
    fig.tight_layout(pad=3)

    # GJ8 calculated x and y
    # axs[0].plot(t / 60, y[:, 0], label='$ x(t) $')  # x(t)
    # axs[0].plot(t / 60, y[:, 1], label='$ y(t) $')  # y(t)
    # axs[0].set_xlabel('Time, $ t $ (min)')
    # axs[0].set_ylabel('Position, $ \mathbf{r} $ (km)')
    # axs[0].set_title('Position components')

    # GJ8 calculated trajectory
    axs[0].plot(y[:, 0], y[:, 1], label='GJ8')  # phase / state space
    axs[0].plot(r_rk4[:, 0], r_rk4[:, 1], label='RK4')
    axs[0].set_xlabel('x position, $ x $ (km)', fontsize=16)
    axs[0].set_ylabel('y position, $ y $ (km)', fontsize=16)
    axs[0].set_title('Trajectory', fontsize=24)
    axs[0].tick_params(labelsize=14)
    axs[0].legend(loc="upper right", fontsize=16)

    # error vs circle
    axs[1].plot(t / T_0, errors_kepler, label='GJ8')
    axs[1].plot(t_rk4 / T_0, errors_rk4, label='RK4')
    axs[1].set_xlabel('Time, $ t $ (orbits)', fontsize=16)
    axs[1].set_ylabel('Error, $ E $ (mm)', fontsize=16)
    axs[1].set_title('Error drift relative to Kepler solution', fontsize=24)
    axs[1].tick_params(labelsize=14)
    axs[1].set_yscale('log')
    axs[1].set_ylim(1e-8, 1e8)
    axs[1].axhline(errors_kepler[-1], color='k', linestyle='--')
    axs[1].axhline(errors_rk4[-1], color='r', linestyle='--')
    print(f'Final error for GJ8: {errors_kepler[-1]} mm')
    print(f'Final error for RK4: {errors_rk4[-1]} mm')
    plt.show()


if __name__ == '__main__':
    main()