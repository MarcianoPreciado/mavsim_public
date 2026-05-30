"""
compute_trim 
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        12/29/2018 - RWB
"""
import numpy as np
from scipy.optimize import minimize
from parameters.aerosonde_parameters import (
    # physical properties
    gravity, mass, rho, Jy,
    # geometry
    S_wing, b, c, AR, e,
    # motor / propeller
    V_max, KV, KQ, R_motor, i0,
    D_prop, C_Q0, C_Q1, C_Q2, C_T0, C_T1, C_T2,
    # stall model
    M, alpha0,
    # stability derivatives
    gamma1, gamma2, gamma3, gamma4, gamma5, gamma6, gamma7,
    # lift and drag
    C_L_0, C_L_alpha, C_L_q, C_L_delta_e,
    C_D_0, C_D_alpha, C_D_p, C_D_q, C_D_delta_e,
    # side force
    C_Y_0, C_Y_beta, C_Y_p, C_Y_r, C_Y_delta_a, C_Y_delta_r,
    # pitch moment
    C_m_0, C_m_alpha, C_m_q, C_m_delta_e,
    # roll moment
    C_p_0, C_p_beta, C_p_p, C_p_r, C_p_delta_a, C_p_delta_r,
    # yaw moment
    C_r_0, C_r_beta, C_r_p, C_r_r, C_r_delta_a, C_r_delta_r,
)
import parameters.aerosonde_parameters as MAV
from tools.rotations import euler_to_quaternion, quaternion_to_euler
from message_types.msg_delta import MsgDelta
import time
from math import sin, cos, tan, exp, pi, sqrt

def compute_trim(mav, Va, gamma):
    # define initial state and input

    print("Trim inputs: Va=%f, gamma=%f" % (Va, gamma))
    # set the initial conditions of the optimization
    e0 = euler_to_quaternion(0., gamma, 0.)
    state0 = np.array([[0],  # pn
                   [0],  # pe
                   [MAV.down0],  # pd
                   [Va],  # u
                   [0], # v
                   [0], # w
                   [e0.item(0)],  # e0
                   [e0.item(1)],  # e1
                   [e0.item(2)],  # e2
                   [e0.item(3)],  # e3
                   [0], # p
                   [0], # q
                   [0]  # r
                   ])
    delta0 = np.array([[0],  # elevator
                       [0],  # aileron
                       [0],  # rudder
                       [0.5]]) # throttle
    x0 = np.concatenate((state0, delta0), axis=0).flatten()
    # define equality constraints
    cons = ({'type': 'eq',
             'fun': lambda x: np.array([
                                x[3]**2 + x[4]**2 + x[5]**2 - Va**2,  # magnitude of velocity vector is Va
                                x[4],  # v=0, force side velocity to be zero
                                x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 - 1.,  # force quaternion to be unit length
                                x[7],  # e1=0  - forcing e1=e3=0 ensures zero roll and zero yaw in trim
                                x[9],  # e3=0
                                x[10],  # p=0  - angular rates should all be zero
                                x[11],  # q=0
                                x[12],  # r=0
                                ]),
             'jac': lambda x: np.array([
                                [0., 0., 0., 2*x[3], 2*x[4], 2*x[5], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 2*x[6], 2*x[7], 2*x[8], 2*x[9], 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                ])
             })
    # solve the minimization problem to find the trim states and inputs

    res = minimize(trim_objective_fun, x0, method='SLSQP', args=(mav, Va, gamma),
                   constraints=cons, 
                   options={'ftol': 1e-10, 'disp': True})
    # extract trim state and input and return
    trim_state = np.array([res.x[0:13]]).T
    trim_input = MsgDelta(elevator=res.x.item(13),
                          aileron=res.x.item(14),
                          rudder=res.x.item(15),
                          throttle=res.x.item(16))
    trim_input.print()
    print('trim_state=', trim_state.T)
    return trim_state, trim_input


def trim_objective_fun(x, mav, Va, gamma):
    x_dot_star = np.array([[0.],  # (0)
                           [0.],   # (1)
                           [-Va * sin(gamma)],   # (2)
                           [0.],    # (3)
                           [0.],    # (4)
                           [0.],    # (5)
                           [0.],    # (6)
                           [0.],    # (7)
                           [0.],    # (8)
                           [0.],    # (9)
                           [0.],    # (10)
                           [0.],    # (11)
                           [0.]])   # (12)

    state = x[0:13].reshape((-1, 1)) # must be (13,1) column vector; 1D causes Va_body to broadcast to (3,3) giving sqrt(3)*Va
    delta = MsgDelta(elevator=x.item(13), aileron=x.item(14), rudder=x.item(15), throttle=x.item(16))
    wind = np.zeros((6, 1))
    
    mav._state = state  
    mav._update_velocity_data(wind)
    forces_moments = mav._forces_moments(delta)
    xdot = mav._f(state, forces_moments)

    J = np.linalg.norm(x_dot_star[2:13]-xdot[2:13])**2.0
    return J
