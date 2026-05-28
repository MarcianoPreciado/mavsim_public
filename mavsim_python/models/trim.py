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
                   [0],  # pd
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
    # # objective function to be minimized
    # pn = x.item(0)
    # pe = x.item(1)
    # pd = x.item(2)
    # u = x.item(3)
    # v = x.item(4)
    # w = x.item(5)
    # e0 = x.item(6)
    # e1 = x.item(7)
    # e2 = x.item(8)
    # e3 = x.item(9)
    # p = x.item(10)
    # q = x.item(11)
    # r = x.item(12)
    # delta_elevator = x.item(13)
    # delta_aileron = x.item(14)
    # delta_rudder = x.item(15)
    # delta_throttle = x.item(16)

    # phi, theta, psi = quaternion_to_euler(np.array([[e0], [e1], [e2], [e3]]))
    # # Compute the helpful variables for state derivatives
    # h = -pd
    # alpha = 0
    # beta = 0
    # # True forms
    # sigmoid = (1 + exp(-M * (alpha - alpha0)) + exp(M * (beta + alpha0))) / ((1 + exp(-M * (alpha - alpha0))) * (1 + exp(M * (beta + alpha0))))
    # CL = (1 - sigmoid) * (C_L_0 + C_L_alpha * alpha) + sigmoid * 2 * np.sign(alpha) * (sin(alpha))**2 * cos(alpha)
    # CD = C_D_p + (C_L_0 + C_L_alpha * alpha)**2 / (pi * e * AR)

    # CL = C_L_0 + C_L_alpha * alpha
    # CD = C_D_0 + C_D_alpha * alpha

    # CX = -CD * cos(alpha) + CL * sin(alpha)
    # CXq = -C_D_q * cos(alpha) + C_L_q * sin(alpha)
    # CXdelta_e = -C_D_delta_e * cos(alpha) + C_L_delta_e * sin(alpha)
    # CZ = -CD * sin(alpha) - CL * cos(alpha)
    # CZq = -C_D_q * sin(alpha) - C_L_q * cos(alpha)
    # CZdelta_e = -C_D_delta_e * sin(alpha) - C_L_delta_e * cos(alpha)

    # # compute the state derivatives using the nonlinear equations of motion
    # # ignore pn, pe, pd since they are intended to be at steady rates of change
    # Tp, Qp = mav._motor_thrust_torque(mav._Va, delta_throttle)
    
    # udot = r*v - q*w - gravity * sin(theta) + rho * mav._Va**2 * S_wing / (2 * mass) * (CX + CXq * c*q/(2*mav._Va) + CXdelta_e * delta_elevator) + Tp/mass
    # vdot = p*w - r*u + gravity * cos(theta) * sin(phi) + rho * mav._Va**2 * S_wing / (2 * mass) * (C_Y_0 + C_Y_beta * beta + C_Y_p * (b / (2 * mav._Va)) * p + C_Y_r * (b / (2 * mav._Va)) * r + C_Y_delta_a * delta_aileron + C_Y_delta_r * delta_rudder)
    # wdot = q*u - p*v + gravity * cos(theta) * cos(phi) + rho * mav._Va**2 * S_wing / (2 * mass) * (CZ + CZq * c*q/(2*mav._Va) + CZdelta_e * delta_elevator)

    # phidot = p + q*sin(phi)*tan(theta) + r*cos(phi)*tan(theta)
    # thetadot = q*cos(phi) - r*sin(phi)
    # psidot = q*sin(phi)/cos(theta) + r*cos(phi)/cos(theta)

    # pdot = gamma1*p*q - gamma2*q*r + 0.5*rho*mav._Va**2*S_wing*b*(C_p_0 + C_p_beta * beta + C_p_p * (b / (2 * mav._Va)) * p + C_p_r * (b / (2 * mav._Va)) * r + C_p_delta_a * delta_aileron + C_p_delta_r * delta_rudder) + gamma3*Qp
    # qdot = gamma5*p*r - gamma6*(p**2 - r**2) + rho * mav._Va**2 *S_wing*c/2/Jy *(C_m_0 + C_m_alpha * alpha + C_m_q * (c / (2 * mav._Va)) * q + C_m_delta_e * delta_elevator)
    # rdot = gamma7*p*q - gamma1*q*r + 0.5*rho*mav._Va**2*S_wing*b*(C_r_0 + C_r_beta * beta + C_r_p * (b / (2 * mav._Va)) * p + C_r_r * (b / (2 * mav._Va)) * r + C_r_delta_a * delta_aileron + C_r_delta_r * delta_rudder) - gamma4*Qp

    # hdot = u*sin(theta) - v*cos(theta)*sin(phi) - w*cos(theta)*cos(phi)
    # hdot_set = -Va * sin(gamma)
    
    # J = (hdot - hdot_set)**2 + vdot**2 + wdot**2 + phidot**2 + thetadot**2 + psidot**2 + pdot**2 + qdot**2 + rdot**2

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

    state = x[0:13]
    delta = MsgDelta(elevator=x.item(13), aileron=x.item(14), rudder=x.item(15), throttle=x.item(16))
    wind = np.zeros((6, 1))
    
    mav._state = state
    mav._update_velocity_data(wind)
    forces_moments = mav._forces_moments(delta)
    xdot = mav._f(state, forces_moments)

    J = np.sum((x_dot_star[2:13]-xdot[2:13])**2)
    return J
