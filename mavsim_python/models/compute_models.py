"""
compute_ss_model
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        2/4/2019 - RWB
"""
import numpy as np
from scipy.optimize import minimize
from tools.rotations import euler_to_quaternion, quaternion_to_euler
import parameters.aerosonde_parameters as MAV
from parameters.simulation_parameters import ts_simulation as Ts
from message_types.msg_delta import MsgDelta
from math import cos

def compute_model(mav, trim_state, trim_input):
    # Note: this function alters the mav private variables
    A_lon, B_lon, A_lat, B_lat = compute_ss_model(mav, trim_state, trim_input)
    Va_trim, alpha_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, \
    a_V1, a_V2, a_V3 = compute_tf_model(mav, trim_state, trim_input)

    # write transfer function gains to file
    file = open('mavsim_python/models/model_coef.py', 'w')
    file.write('import numpy as np\n')
    file.write('x_trim = np.array([[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]]).T\n' %
               (trim_state.item(0), trim_state.item(1), trim_state.item(2), trim_state.item(3),
                trim_state.item(4), trim_state.item(5), trim_state.item(6), trim_state.item(7),
                trim_state.item(8), trim_state.item(9), trim_state.item(10), trim_state.item(11),
                trim_state.item(12)))
    file.write('u_trim = np.array([[%f, %f, %f, %f]]).T\n' %
               (trim_input.elevator, trim_input.aileron, trim_input.rudder, trim_input.throttle))
    file.write('Va_trim = %f\n' % Va_trim)
    file.write('alpha_trim = %f\n' % alpha_trim)
    file.write('theta_trim = %f\n' % theta_trim)
    file.write('a_phi1 = %f\n' % a_phi1)
    file.write('a_phi2 = %f\n' % a_phi2)
    file.write('a_theta1 = %f\n' % a_theta1)
    file.write('a_theta2 = %f\n' % a_theta2)
    file.write('a_theta3 = %f\n' % a_theta3)
    file.write('a_V1 = %f\n' % a_V1)
    file.write('a_V2 = %f\n' % a_V2)
    file.write('a_V3 = %f\n' % a_V3)
    file.write('A_lon = np.array([\n    [%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f]])\n' %
    (A_lon[0][0], A_lon[0][1], A_lon[0][2], A_lon[0][3], A_lon[0][4],
     A_lon[1][0], A_lon[1][1], A_lon[1][2], A_lon[1][3], A_lon[1][4],
     A_lon[2][0], A_lon[2][1], A_lon[2][2], A_lon[2][3], A_lon[2][4],
     A_lon[3][0], A_lon[3][1], A_lon[3][2], A_lon[3][3], A_lon[3][4],
     A_lon[4][0], A_lon[4][1], A_lon[4][2], A_lon[4][3], A_lon[4][4]))
    file.write('B_lon = np.array([\n    [%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f]])\n' %
    (B_lon[0][0], B_lon[0][1],
     B_lon[1][0], B_lon[1][1],
     B_lon[2][0], B_lon[2][1],
     B_lon[3][0], B_lon[3][1],
     B_lon[4][0], B_lon[4][1],))
    file.write('A_lat = np.array([\n    [%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f]])\n' %
    (A_lat[0][0], A_lat[0][1], A_lat[0][2], A_lat[0][3], A_lat[0][4],
     A_lat[1][0], A_lat[1][1], A_lat[1][2], A_lat[1][3], A_lat[1][4],
     A_lat[2][0], A_lat[2][1], A_lat[2][2], A_lat[2][3], A_lat[2][4],
     A_lat[3][0], A_lat[3][1], A_lat[3][2], A_lat[3][3], A_lat[3][4],
     A_lat[4][0], A_lat[4][1], A_lat[4][2], A_lat[4][3], A_lat[4][4]))
    file.write('B_lat = np.array([\n    [%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f]])\n' %
    (B_lat[0][0], B_lat[0][1],
     B_lat[1][0], B_lat[1][1],
     B_lat[2][0], B_lat[2][1],
     B_lat[3][0], B_lat[3][1],
     B_lat[4][0], B_lat[4][1],))
    file.write('Ts = %f\n' % Ts)
    file.close()


def compute_tf_model(mav, trim_state, trim_input):
    # trim values
    mav._state = trim_state
    mav._update_velocity_data()
    Va_trim = mav._Va
    alpha_trim = mav._alpha
    phi, theta_trim, psi = quaternion_to_euler(trim_state[6:10])

    rho = MAV.rho
    S = MAV.S_wing
    c = MAV.c
    b = MAV.b
    Va = mav._Va
    beta = mav._beta
    Cpp = MAV.C_p_p
    Cpda = MAV.C_p_delta_a
    Cmq = MAV.C_m_q
    Cma = MAV.C_m_alpha
    Cmde = MAV.C_m_delta_e
    Jy = MAV.Jy
    CD0 = MAV.C_D_0
    CDa = MAV.C_D_alpha
    CDde = MAV.C_D_delta_e
    m = MAV.mass
    g = MAV.gravity

    dTdVa = dT_dVa(mav, Va, trim_input.throttle)
    dTddt = dT_ddelta_t(mav, Va, trim_input.throttle)

    # define transfer function constants
    a_phi1 = -1/2 * rho * Va**2 * S * b * Cpp * b / (2 * Va)
    a_phi2 = rho * Va**2 * S * b * Cpda / 2
    a_theta1 = -1 * rho * Va**2 * c * S / (2 * Jy) * Cmq * c / (2 * Va)
    a_theta2 = -1 * rho * Va**2 * c * S / (2 * Jy) * Cma
    a_theta3 = rho * Va**2 * c * S / (2 * Jy) * Cmde

    # Compute transfer function coefficients using new propulsion model
    a_V1 = rho * Va * S / m * (CD0 + CDa * alpha_trim + CDde * trim_input.elevator) - dTdVa / m
    a_V2 = dTddt / m
    a_V3 = g * cos(theta_trim - alpha_trim)

    return Va_trim, alpha_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, a_V1, a_V2, a_V3


def compute_ss_model(mav, trim_state, trim_input):
    x_euler = euler_state(trim_state)
    
    A = df_dx(mav, x_euler, trim_input)
    B = df_du(mav, x_euler, trim_input)
    # extract longitudinal states (u, w, q, theta, pd)
    E1 = np.array([
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    E2 = np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ])
    E3 = np.array([
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0,	 0, 0, 0, 1, 0, 0, 0]
    ])
    E4 = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])

    A_lon = E1 @ A @ E1.T
    B_lon = E1 @ B @ E2.T
    A_lat = E3 @ A @ E3.T
    B_lat = E3 @ B @ E4.T
    
    return A_lon, B_lon, A_lat, B_lat

def euler_state(x_quat):
    # convert state x with attitude represented by quaternion
    # to x_euler with attitude represented by Euler angles
    p = x_quat[0:3]
    v = x_quat[3:6]
    e = x_quat[6:10]
    omega = x_quat[10:13]

    theta = np.array(quaternion_to_euler(e)).reshape(-1,1)
    
    x_euler = np.concatenate((p, v, theta, omega))
    return x_euler

def quaternion_state(x_euler):
    # convert state x_euler with attitude represented by Euler angles
    # to x_quat with attitude represented by quaternions

    p = x_euler[0:3]
    v = x_euler[3:6]
    theta = x_euler[6:9]
    omega = x_euler[9:12]

    e = np.array(euler_to_quaternion(theta.item(0), theta.item(1), theta.item(2))).reshape(-1,1)

    x_quat = np.concatenate((p, v, e, omega))
    return x_quat

def f_euler(mav, x_euler, delta):
    # return 12x1 dynamics (as if state were Euler state)
    # compute f at euler_state, f_euler will be f, except for the attitude states

    # need to correct attitude states by multiplying f by
    # partial of quaternion_to_euler(quat) with respect to quat
    # compute partial quaternion_to_euler(quat) with respect to quat
    # dEuler/dt = dEuler/dquat * dquat/dt
    
    # Tq(xe)
    x_quat = quaternion_state(x_euler)
    mav._state = x_quat
    mav._update_velocity_data()
    # ƒq(Tq(xe), u)
    forces_moments = mav._forces_moments(delta)
    f_quat = mav._f(x_quat, forces_moments)
    
    # dTe_dxq (Tq(xe))
    # Te is euler_to_quaternion
    # Tq is quaternion_to_euler
    # Tq(xe) is quaternion_state(x_euler)
    # So dTe_dxq is Te(quaternion_state(x_euler) + eps - Te())
    dTe_dxq_of_xq = np.zeros((x_euler.size, x_quat.size))
    Te_out = x_euler
    n = x_quat.size # number of quaternion states
    eps = 0.001
    for i in range(n):
        x_quat_i = x_quat + eps * np.eye(n)[:, [i]]
        Te_outi = euler_state(x_quat_i)
        
        dTe_dquat_i = (Te_outi - Te_out) / eps
        dTe_dxq_of_xq[:, [i]] = dTe_dquat_i

    return dTe_dxq_of_xq @ f_quat

def df_dx(mav, x_euler, delta):
    # take partial of f_euler with respect to x_euler
    eps = 0.01  # deviation

    f1 = f_euler(mav, x_euler, delta)
    N = x_euler.size
    A = np.zeros((N, N))
    for i in range(N):
        x_euler_i = x_euler + eps * np.eye(N)[:, [i]]
        df_dxi = (f_euler(mav, x_euler_i, delta) - f1) / eps
        A[:,[i]] = df_dxi
    return A


def df_du(mav, x_euler, delta):
    # take partial of f_euler with respect to input
    eps = 0.01  # deviation
    f1 = f_euler(mav, x_euler, delta)
    u = delta.to_array()
    N = x_euler.size
    M = u.size

    B = np.zeros((N, M))
    for i in range(M):
        ui = u + eps * np.eye(M)[:, [i]]
        delta_i = MsgDelta()
        delta_i.from_array(ui)
        df_dui = (f_euler(mav, x_euler, delta_i) - f1) / eps
        B[:,[i]] = df_dui

    B = B[:, 0:4]  # only take columns corresponding to control inputs, ignore gimbal inputs
    return B


def dT_dVa(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to Va
    eps = 0.01
    
    T1, _ = mav._motor_thrust_torque(Va, delta_t)
    T2, _ = mav._motor_thrust_torque(Va + eps, delta_t)
    dT_dVa = (T2 - T1) / eps

    return dT_dVa

def dT_ddelta_t(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to delta_t
    eps = 0.01

    T1, _ = mav._motor_thrust_torque(Va, delta_t)
    T2, _ = mav._motor_thrust_torque(Va, delta_t + eps)
    dT_ddelta_t = (T2 - T1) / eps

    return dT_ddelta_t
