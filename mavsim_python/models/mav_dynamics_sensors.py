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
from message_types.msg_sensors import MsgSensors
import parameters.aerosonde_parameters as MAV
import parameters.sensor_parameters as SENSOR
from models.mav_dynamics_control import MavDynamics as MavDynamicsNoSensors
from tools.rotations import quaternion_to_rotation, quaternion_to_euler, euler_to_rotation
from numpy.random import normal
from numpy import sin, cos, atan2, sqrt, exp

def gaussian_markov_next(last, k, sigma, ts):
    w = normal(0.0, sqrt(ts)*sigma)
    C = exp(-k * ts) + w
    return C*last + w
class MavDynamics(MavDynamicsNoSensors):
    def __init__(self, Ts):
        super().__init__(Ts)
        # initialize the sensors message
        self._sensors = MsgSensors()
        # random walk parameters for GPS
        self._gps_eta_n = 0.
        self._gps_eta_e = 0.
        self._gps_eta_h = 0.
        # timer so that gps only updates every ts_gps seconds
        self._t_gps = 999.  # large value ensures gps updates at initial time.

    def sensors(self):
        "Return value of sensors on MAV: gyros, accels, absolute_pressure, dynamic_pressure, GPS"
        phi, theta, psi = quaternion_to_euler(self._state[6:10])
        u = self._state.item(3)
        v = self._state.item(4)
        w = self._state.item(5)
        p = self._state.item(10)
        q = self._state.item(11)
        r = self._state.item(12)
        g = MAV.gravity
        m = MAV.mass
        F = self._forces

        # simulate rate gyros(units are rad / sec)
        eta_gyro_x = normal(SENSOR.gyro_x_bias, SENSOR.gyro_sigma)
        eta_gyro_y = normal(SENSOR.gyro_y_bias, SENSOR.gyro_sigma)
        eta_gyro_z = normal(SENSOR.gyro_z_bias, SENSOR.gyro_sigma)    
        self._sensors.gyro_x = p + eta_gyro_x
        self._sensors.gyro_y = q + eta_gyro_y
        self._sensors.gyro_z = r + eta_gyro_z

        # simulate accelerometers(units of g)
        accel_noise_vec = normal(0.0, SENSOR.accel_sigma, size=(3,1))
        g_i = np.array([[0, 0, -g]]).T
        R_b2i = euler_to_rotation(phi, theta, psi)
        R_i2b = R_b2i.T
        g_b = R_i2b @ g_i
        # Vector calculation
        accel_vec = F/m + g_b + accel_noise_vec

        self._sensors.accel_x = accel_vec.item(0)
        self._sensors.accel_y = accel_vec.item(1)
        self._sensors.accel_z = accel_vec.item(2)
        
        # simulate magnetometers
        # magnetic field in provo has magnetic declination of 12.5 degrees
        # and magnetic inclination of 66 degrees
        dec = np.radians(12.5)
        inc = -np.radians(65.7)
        
        M = np.array([[1, 0, 0]]).T
        Mi = euler_to_rotation(0, inc, dec).T @ M

        # Calculate would-be measurement onboard
        mag_eta = normal(0.0, SENSOR.mag_sigma, size=(3,1))
        mag_beta = np.ones((3,1)) * SENSOR.mag_beta
        ymag = R_i2b @ (Mi + mag_beta) + mag_eta
        
        self._sensors.mag_x = ymag.item(0)
        self._sensors.mag_y = ymag.item(1)
        self._sensors.mag_z = ymag.item(2)

        # simulate pressure sensors
        eta_abs_pres = normal(0.0, SENSOR.abs_pres_sigma)
        eta_diff_pres = normal(0.0, SENSOR.diff_pres_sigma)
        rho = MAV.rho
        h_gl = -1*self._state.item(2)
        self._sensors.abs_pressure = rho*g*h_gl + eta_abs_pres
        self._sensors.diff_pressure = rho*self._Va**2 / 2 + eta_diff_pres
        
        # simulate GPS sensor
        wn = self._wind.item(0)
        we = self._wind.item(1)
        wd = self._wind.item(2)
        Va = self._Va
        pn = self._state.item(0)
        pe = self._state.item(1)
        pd = self._state.item(2)
        ph = -pd
        if self._t_gps >= SENSOR.ts_gps:
            self._gps_eta_n = gaussian_markov_next(self._gps_eta_n, SENSOR.gps_k, SENSOR.gps_n_sigma, SENSOR.ts_gps)
            self._gps_eta_e = gaussian_markov_next(self._gps_eta_e, SENSOR.gps_k, SENSOR.gps_e_sigma, SENSOR.ts_gps)
            self._gps_eta_h = gaussian_markov_next(self._gps_eta_h, SENSOR.gps_k, SENSOR.gps_h_sigma, SENSOR.ts_gps)
            
            self._sensors.gps_n = pn + self._gps_eta_n
            self._sensors.gps_e = pe + self._gps_eta_e
            self._sensors.gps_h = ph + self._gps_eta_h

            Vgn = Va*cos(psi) + wn
            Vge = Va*sin(psi) + we
            gps_eta_V = normal(0.0, SENSOR.gps_Vg_sigma)
            gps_eta_course = normal(0.0, SENSOR.gps_course_sigma)
            self._sensors.gps_Vg = sqrt(Vgn**2 + Vge**2) + gps_eta_V
            self._sensors.gps_course = atan2(Vge, Vgn) + gps_eta_course

            self._t_gps = 0.
        else:
            self._t_gps += self._ts_simulation
        return self._sensors

    def external_set_state(self, new_state):
        self._state = new_state

    def _update_true_state(self):
        # update the class structure for the true state:
        #   [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
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
        self.true_state.bx = SENSOR.gyro_x_bias
        self.true_state.by = SENSOR.gyro_y_bias
        self.true_state.bz = SENSOR.gyro_z_bias
        self.true_state.gimbal_az = self._state.item(13)
        self.true_state.gimbal_el = self._state.item(14)