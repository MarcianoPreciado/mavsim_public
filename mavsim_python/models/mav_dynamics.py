"""
mavDynamics 
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
mavsim_python
    - Beard & McLain, PUP, 2012
    - Update history:  
        2/24/2020 - RWB
        7/13/2023 - RWB
        1/17/2024 - RWB
"""
import numpy as np
# load message types
from message_types.msg_state import MsgState
import parameters.aerosonde_parameters as MAV
from tools.rotations import quaternion_to_rotation, quaternion_to_euler, euler_to_quaternion

class MavDynamics:
    def __init__(self, Ts):
        self._ts_simulation = Ts
        # set initial states based on parameter file
        # _state is the 13x1 internal state of the aircraft that is being propagated:
        # _state = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        # We will also need a variety of other elements that are functions of the _state and the wind.
        # self.true_state is a 19x1 vector that is estimated and used by the autopilot to control the aircraft:
        # true_state = [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        self._state = np.array([[MAV.north0],  # (0)
                               [MAV.east0],   # (1)
                               [MAV.down0],   # (2)
                               [MAV.u0],    # (3)
                               [MAV.v0],    # (4)
                               [MAV.w0],    # (5)
                               [MAV.e0],    # (6)
                               [MAV.e1],    # (7)
                               [MAV.e2],    # (8)
                               [MAV.e3],    # (9)
                               [MAV.p0],    # (10)
                               [MAV.q0],    # (11)
                               [MAV.r0],    # (12)
                               [0],   # (13)
                               [0],   # (14)
                               ])
        # initialize true_state message
        self.true_state = MsgState()

    ###################################
    # public functions
    def update(self, forces_moments):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        '''
        self._rk4_step(forces_moments)
        # update the message class for the true state
        self._update_true_state()

    def external_set_state(self, new_state):
        self._state = new_state

    ###################################
    # private functions
    def _rk4_step(self, forces_moments):
        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self._ts_simulation
        k1 = self._f(self._state[0:13], forces_moments)
        k2 = self._f(self._state[0:13] + time_step/2.*k1, forces_moments)
        k3 = self._f(self._state[0:13] + time_step/2.*k2, forces_moments)
        k4 = self._f(self._state[0:13] + time_step*k3, forces_moments)
        self._state[0:13] += time_step/6 * (k1 + 2*k2 + 2*k3 + k4)

        # normalize the quaternion
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        normE = np.sqrt(e0**2+e1**2+e2**2+e3**2)
        self._state[6][0] = self._state.item(6)/normE
        self._state[7][0] = self._state.item(7)/normE
        self._state[8][0] = self._state.item(8)/normE
        self._state[9][0] = self._state.item(9)/normE

    def _f(self, state, forces_moments):
        """
        for the dynamics xdot = f(x, u), returns f(x, u)
        """
        ##### TODO #####
        # x = np.array([[north], [east], [down], [u], [v], [w], [e0], [e1], [e2], [p], [q], [r]]).T
        inertial_pos = np.array([[state.item(0), state.item(1), state.item(2)]]).T
        uvw = np.array([[state.item(3), state.item(4), state.item(5)]]).T # body frame velocity
        eulers = np.array([[state.item(6),state.item(7), state.item(8)]]).T # euler angles
        pqr = np.array([[state.item(9), state.item(10), state.item(11)]]).T # body frame angular rates
        e = euler_to_quaternion(eulers.item(0), eulers.item(1), eulers.item(2)) # quaternion attitude



        # Extract Forces/Moments
        f = forces_moments[0:3] # forces
        m = forces_moments[3:6] # moments

        # Position Kinematics
        # pos_dot = euler_to_rotation(eulers).T @ uvw
        pos_dot = quaternion_to_rotation(e) @ uvw 

        # Position Dynamics
        uvw_dot = 1/MAV.mass * f - np.cross(pqr.T, uvw.T).T


        # rotational kinematics
        p = pqr.item(0)
        q = pqr.item(1)
        r = pqr.item(2)
        arr = np.array([[ 0, -p, -q, -r],
                       [p, 0, r, -q],
                       [q, -r, 0, p],
                       [r, q, -p, 0]])
        e_dot = 0.5 * arr @ e


        # rotatonal dynamics
        gamma = MAV.Jx*MAV.Jz - MAV.Jxz**2
        gamma1 = MAV.Jxz*(MAV.Jx - MAV.Jy + MAV.Jz)/gamma
        gamma2 = (MAV.Jz*(MAV.Jz - MAV.Jy) + MAV.Jxz**2)/gamma
        gamma3 = MAV.Jz/gamma
        gamma4 = MAV.Jxz/gamma
        gamma5 = (MAV.Jz - MAV.Jx)/MAV.Jy
        gamma6 = MAV.Jxz/MAV.Jy
        gamma7 = ((MAV.Jx - MAV.Jy)*MAV.Jx + MAV.Jxz**2)/gamma
        gamma8 = MAV.Jx/gamma

        arr1 = np.array([gamma1*p*q - gamma2*q*r, 
                         gamma5*p*4 - gamma6*(p**2-r**2),
                         gamma7*p*q - gamma1*q*r]).T

        arr2 = np.array([[gamma3, 0, gamma4],
                         [0, 1/MAV.Jy, 0],
                         [gamma4, 0, gamma8]])
        angular_dot = arr1 + arr2 @ m

        # collect the derivative of the states
        # x_dot = np.array([[north_dot, east_dot, down_dot, udot, vdot, wdot, e0, e1, e2, e3, pdot, qdot, rdot ]]).T
        x_dot = np.array([[pos_dot.item(0)],
                          [pos_dot.item(1)],
                          [pos_dot.item(2)],
                          [uvw_dot.item(0)],
                          [uvw_dot.item(1)],
                          [uvw_dot.item(2)],
                          [e_dot.item(0)],
                          [e_dot.item(1)],
                          [e_dot.item(2)],
                          [e_dot.item(3)],
                          [angular_dot.item(0)],
                          [angular_dot.item(1)],
                          [angular_dot.item(2)]])
        return x_dot

    def _update_true_state(self):
        # update the class structure for the true state:
        #   [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        phi, theta, psi = quaternion_to_euler(self._state[6:10])
        self.true_state.north = self._state.item(0)
        self.true_state.east = self._state.item(1)
        self.true_state.altitude = -self._state.item(2)
        self.true_state.Va = 0
        self.true_state.alpha = 0
        self.true_state.beta = 0
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.Vg = 0
        self.true_state.gamma = 0
        self.true_state.chi = 0
        self.true_state.p = self._state.item(10)
        self.true_state.q = self._state.item(11)
        self.true_state.r = self._state.item(12)
        self.true_state.wn = 0
        self.true_state.we = 0
        self.true_state.bx = 0
        self.true_state.by = 0
        self.true_state.bz = 0
        self.true_state.camera_az = 0
        self.true_state.camera_el = 0