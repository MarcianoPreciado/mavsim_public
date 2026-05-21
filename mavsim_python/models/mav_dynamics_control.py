"""
mavDynamics 
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
mavsim_python
    - Beard & McLain, PUP, 2012
    - Update history:  
        2/24/2020 - RWB
"""
import numpy as np
from models.mav_dynamics import MavDynamics as MavDynamicsForces
# load message types
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta
import parameters.aerosonde_parameters as MAV
from tools.rotations import quaternion_to_rotation, quaternion_to_euler

from math import sqrt, pi, exp, sin, cos
class MavDynamics(MavDynamicsForces):
    def __init__(self, Ts):
        super().__init__(Ts)
        # store wind data for fast recall since it is used at various points in simulation
        self._wind = np.array([[0.], [0.], [0.]])  # wind in NED frame in meters/sec
        # store forces to avoid recalculation in the sensors function
        self._forces = np.array([[0.], [0.], [0.]])
        self._Va = MAV.u0
        self._alpha = 0
        self._beta = 0
        # update velocity data and forces and moments
        self._update_velocity_data()
        self._forces_moments(delta=MsgDelta())
        # update the message class for the true state
        self._update_true_state()


    ###################################
    # public functions
    def update(self, delta, wind):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        '''
        # get forces and moments acting on rigid bod
        forces_moments = self._forces_moments(delta)
        super()._rk4_step(forces_moments)
        # update the airspeed, angle of attack, and side slip angles using new state
        self._update_velocity_data(wind)
        # update the message class for the true state
        self._update_true_state()

    ###################################
    # private functions
    def _update_velocity_data(self, wind=np.zeros((6,1))):
        steady_state = wind[0:3]
        gust = wind[3:6]

        # convert steady-state wind vector from world to body frame
        wind_body = quaternion_to_rotation(self._state[6:10]).T @ steady_state
        # add the gust 
        wind_body += gust
        # convert total wind to world frame
        self._wind = quaternion_to_rotation(self._state[6:10]) @ wind_body

        # velocity vector relative to the airmass ([ur , vr, wr]= ?)
        Va_body = self._state[3:6] - wind_body
        # compute airspeed (self._Va = ?)
        self._Va = np.linalg.norm(Va_body).item(0)
        # compute angle of attack (self._alpha = ?)
        ur = Va_body.item(0)
        vr = Va_body.item(1)
        wr = Va_body.item(2)
        self._alpha = np.arctan(wr / ur)
        # compute sideslip angle (self._beta = ?)
        self._beta = np.arcsin(vr / self._Va)

    def _forces_moments(self, delta):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta_a, delta_e, delta_r, delta_t)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
        # extract states (phi, theta, psi, p, q, r)
        phi, theta, psi = quaternion_to_euler(self._state[6:10])
        p, q, r = self._state[10:13, 0]
        # compute gravitational forces ([fg_x, fg_y, fg_z])
        fg_vehicle = np.array([[0, 0, MAV.mass * MAV.gravity]]).T


        # compute Lift and Drag coefficients (CL, CD)
        # Linearized forms
        """
        CL = MAV.C_L_0 + MAV.C_L_alpha * self._alpha
        CD = MAV.C_D_0 + MAV.C_D_alpha * self._alpha
        """
        # True forms
        sigmoid = (1 + exp(-MAV.M * (self._alpha - MAV.alpha0)) + exp(MAV.M * (self._alpha + MAV.alpha0))) / ((1 + exp(-MAV.M * (self._alpha - MAV.alpha0))) * (1 + exp(MAV.M * (self._alpha + MAV.alpha0))))
        CL = (1 - sigmoid) * (MAV.C_L_0 + MAV.C_L_alpha * self._alpha) + sigmoid * 2 * np.sign(self._alpha) * (sin(self._alpha))**2 * cos(self._alpha)
        CD = MAV.C_D_p + (MAV.C_L_0 + MAV.C_L_alpha * self._alpha)**2 / (pi * MAV.e * MAV.AR)
        
        # compute Lift and Drag Forces (F_lift, F_drag)
        F_lift = 0.5 * MAV.rho * self._Va**2 * MAV.S_wing * (CL + MAV.C_L_q * (MAV.c / (2 * self._Va)) * q + MAV.C_L_delta_e * delta.elevator)
        F_drag = 0.5 * MAV.rho * self._Va**2 * MAV.S_wing * (CD + MAV.C_D_q * (MAV.c / (2 * self._Va)) * q + MAV.C_D_delta_e * abs(delta.elevator))
        
        # propeller thrust and torque
        thrust_prop, torque_prop = self._motor_thrust_torque(self._Va, delta.throttle)

        # compute longitudinal forces in body frame (fx, fz)
        # Convert from stability frame to body frame
        lift_drag_body_components = np.array([[cos(self._alpha), -sin(self._alpha)], [sin(self._alpha), cos(self._alpha)]]) @ np.array([[ -F_drag, -F_lift]]).T
        fx, fz = lift_drag_body_components.item(0), lift_drag_body_components.item(1)

        # compute lateral forces in body frame (fy)
        fy = 0.5 * MAV.rho * self._Va**2 * MAV.S_wing * (MAV.C_Y_0 + MAV.C_Y_beta * self._beta + MAV.C_Y_p * (MAV.b / (2 * self._Va)) * p + MAV.C_Y_r * (MAV.b / (2 * self._Va)) * r + MAV.C_Y_delta_a * delta.aileron + MAV.C_Y_delta_r * delta.rudder)
        
        # compute logitudinal torque in body frame (My)
        My = 0.5 * MAV.rho * self._Va**2 * MAV.S_wing * MAV.c * (MAV.C_m_0 + MAV.C_m_alpha * self._alpha + MAV.C_m_q * (MAV.c / (2 * self._Va)) * q + MAV.C_m_delta_e * delta.elevator)

        # compute lateral torques in body frame (Mx, Mz)
        Mx = 0.5 * MAV.rho * self._Va**2 * MAV.S_wing * MAV.b * (MAV.C_ell_0 + MAV.C_ell_beta * self._beta + MAV.C_ell_p * (MAV.b / (2 * self._Va)) * p + MAV.C_ell_r * (MAV.b / (2 * self._Va)) * r + MAV.C_ell_delta_a * delta.aileron + MAV.C_ell_delta_r * delta.rudder)
        Mz = 0.5 * MAV.rho * self._Va**2 * MAV.S_wing * MAV.b * (MAV.C_n_0 + MAV.C_n_beta * self._beta + MAV.C_n_p * (MAV.b / (2 * self._Va)) * p + MAV.C_n_r * (MAV.b / (2 * self._Va)) * r + MAV.C_n_delta_a * delta.aileron + MAV.C_n_delta_r * delta.rudder)

        forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
        return forces_moments

    def _motor_thrust_torque(self, Va, delta_t):
        # compute thrust and torque due to propeller
        # map delta_t throttle command(0 to 1) into motor input voltage
        Vin = MAV.V_max * delta_t
        a = MAV.rho * MAV.D_prop**5 / (2 * pi)**2 * MAV.C_Q0
        b = MAV.rho * MAV.D_prop**4 / (2 * pi) * MAV.C_Q1 * Va + MAV.KV * MAV.KQ/MAV.R_motor
        c = MAV.rho * MAV.D_prop**3 * MAV.C_Q2 * Va**2 - MAV.KQ / MAV.R_motor * Vin + MAV.KQ * MAV.i0
        roots = np.roots([a, b, c])
        # Angular speed
        omega_p = max(roots[roots >= 0])

        # thrust and torque due to propeller
        torque_prop = (MAV.rho * MAV.D_prop**5 * MAV.C_Q0/4/pi**2) * omega_p**2 + (MAV.rho * MAV.D_prop**4 * MAV.C_Q1 * Va /2 /pi) * omega_p + MAV.rho * MAV.D_prop**3 * MAV.C_Q2 * Va**2
        thrust_prop = (MAV.rho * MAV.D_prop**4 * MAV.C_T0 / 4 / pi**2) * omega_p**2 + (MAV.rho * MAV.D_prop**3 * MAV.C_T1 * Va / 2 / pi) * omega_p + MAV.rho * MAV.D_prop**2 * MAV.C_T2 * Va**2

        return thrust_prop, torque_prop

    def _update_true_state(self):
        # rewrite this function because we now have more information
        phi, theta, psi = quaternion_to_euler(self._state[6:10])
        pdot = quaternion_to_rotation(self._state[6:10]) @ self._state[3:6]
        self.true_state.north = self._state.item(0)
        self.true_state.east = self._state.item(1)
        self.true_state.altitude = -self._state.item(2)
        self.true_state.Va = self._Va
        self.true_state.alpha = self._alpha
        self.true_state.beta = self._beta
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.Vg = np.linalg.norm(pdot)
        self.true_state.gamma = np.arcsin(pdot.item(2) / self.true_state.Vg)
        self.true_state.chi = np.arctan2(pdot.item(1), pdot.item(0))
        self.true_state.p = self._state.item(10)
        self.true_state.q = self._state.item(11)
        self.true_state.r = self._state.item(12)
        self.true_state.wn = self._wind.item(0)
        self.true_state.we = self._wind.item(1)
        self.true_state.bx = 0
        self.true_state.by = 0
        self.true_state.bz = 0
        self.true_state.gimbal_az = 0
        self.true_state.gimbal_el = 0
