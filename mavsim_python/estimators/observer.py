"""
observer
    - Beard & McLain, PUP, 2012
    - Last Update:
        3/2/2019 - RWB
        3/4/2024 - RWB
"""
import numpy as np
import parameters.aerosonde_parameters as MAV
import parameters.control_parameters as CTRL
import parameters.sensor_parameters as SENSOR
from tools.wrap import wrap
from message_types.msg_state import MsgState
from message_types.msg_sensors import MsgSensors
from estimators.filters import AlphaFilter, ExtendedKalmanFilterContinuousDiscrete
from numpy import sqrt, arctan2, arcsin, sin, cos, tan

class Observer:
    def __init__(self, ts: float, initial_measurements: MsgSensors=MsgSensors()):
        self.Ts = ts  # sample rate of observer
        # initialized estimated state message
        self.estimated_state = MsgState()

        ##### TODO #####
        self.lpf_gyro_x = AlphaFilter(alpha=0.1, y0=initial_measurements.gyro_x)
        self.lpf_gyro_y = AlphaFilter(alpha=0.1, y0=initial_measurements.gyro_y)
        self.lpf_gyro_z = AlphaFilter(alpha=0.1, y0=initial_measurements.gyro_z)
        self.lpf_accel_x = AlphaFilter(alpha=0.1, y0=initial_measurements.accel_x)
        self.lpf_accel_y = AlphaFilter(alpha=0.1, y0=initial_measurements.accel_y)
        self.lpf_accel_z = AlphaFilter(alpha=0.1, y0=initial_measurements.accel_z)
        # use alpha filters to low pass filter absolute and differential pressure
        self.lpf_abs = AlphaFilter(alpha=0.25, y0=initial_measurements.abs_pressure)
        self.lpf_diff = AlphaFilter(alpha=0.25, y0=initial_measurements.diff_pressure)
        self.lpf_gps_n = AlphaFilter(alpha=0.75, y0=initial_measurements.gps_n)
        self.lpf_gps_e = AlphaFilter(alpha=0.75, y0=initial_measurements.gps_e)
        self.lpf_gps_course = AlphaFilter(alpha=0.0, y0=initial_measurements.gps_course)
        self.lpf_gps_Vg = AlphaFilter(alpha=0.75, y0=initial_measurements.gps_Vg)

        # ekf for phi and theta
        self.attitude_ekf = ExtendedKalmanFilterContinuousDiscrete(
            f=self.f_attitude, 
            Q=np.diag([
                (0)**2, # phi 
                (0)**2, # theta
                ]), 
            P0= np.diag([
                (0)**2, # phi
                (0)**2, # theta
                ]), 
            xhat0=np.array([
                [0.*np.pi/180.], # phi 
                [0.*np.pi/180.], # theta
                ]), 
            Qu=np.diag([
                SENSOR.gyro_sigma**2, 
                SENSOR.gyro_sigma**2, 
                SENSOR.gyro_sigma**2, 
                SENSOR.abs_pres_sigma]), 
            Ts=ts,
            N=5
            )
        # ekf for pn, pe, Vg, chi, wn, we, psi
        self.position_ekf = ExtendedKalmanFilterContinuousDiscrete(
            f=self.f_smooth, 
            Q=np.diag([
                (0.0)**2,  # pn
                (0.0)**2,  # pe
                (0.0)**2,  # Vg
                (0.0)**2, # chi
                (0.0)**2, # wn
                (0.0)**2, # we
                (0.0)**2, # psi
                ]), 
            P0=np.diag([
                (0.)**2, # pn
                (0.0)**2, # pe
                (0.0)**2, # Vg
                (0.*np.pi/180.)**2, # chi
                (0.0)**2, # wn
                (0.0)**2, # we
                (0.*np.pi/180.)**2, # psi
                ]), 
            xhat0=np.array([
                [0.0], # pn 
                [0.0], # pe 
                [0.0], # Vg 
                [0.0], # chi
                [0.0], # wn 
                [0.0], # we 
                [0.0], # psi
                ]), 
            Qu=0.*np.diag([
                SENSOR.gyro_sigma**2, 
                SENSOR.gyro_sigma**2, 
                SENSOR.abs_pres_sigma,
                np.radians(3), # guess for noise on roll
                np.radians(3), # guess for noise on pitch
                ]),
            Ts=ts,
            N=10
            )
        self.R_accel = np.diag([
                SENSOR.accel_sigma**2, 
                SENSOR.accel_sigma**2, 
                SENSOR.accel_sigma**2
                ])
        self.R_pseudo = np.diag([
                0.0,  # pseudo measurement #1 ##### TODO #####
                0.0,  # pseudo measurement #2 ##### TODO #####
                ])
        self.R_gps = np.diag([
                    SENSOR.gps_n_sigma**2,  # y_gps_n
                    SENSOR.gps_e_sigma**2,  # y_gps_e
                    SENSOR.gps_Vg_sigma**2,  # y_gps_Vg
                    SENSOR.gps_course_sigma**2,  # y_gps_course
                    ])
        self.gps_n_old = 9999
        self.gps_e_old = 9999
        self.gps_Vg_old = 9999
        self.gps_course_old = 9999

    def update(self, measurement: MsgSensors) -> MsgState:
        rho = MAV.rho
        g = MAV.gravity
        ##### TODO #####
        # estimates for p, q, r are low pass filter of gyro minus bias estimate
        self.estimated_state.p = self.lpf_gyro_x.update(measurement.gyro_x) 
        self.estimated_state.q = self.lpf_gyro_y.update(measurement.gyro_y)
        self.estimated_state.r = self.lpf_gyro_z.update(measurement.gyro_z)
        # invert sensor model to get altitude and airspeed
        abs_pressure = self.lpf_abs.update(measurement.abs_pressure) 
        diff_pressure = self.lpf_diff.update(measurement.diff_pressure)
        self.estimated_state.altitude = abs_pressure / (rho * g) 
        self.estimated_state.Va = sqrt(2 * diff_pressure / rho)
        # invert sensor model on accelerometers
        accel_x = self.lpf_accel_x.update(measurement.accel_x)
        accel_y = self.lpf_accel_y.update(measurement.accel_y)
        accel_z = self.lpf_accel_z.update(measurement.accel_z)
        phi = arctan2(accel_y, accel_z)
        r = accel_x / g
        r = min(1, max(-1,r))
        theta = arcsin(r)

        #invert sensor model on GPS
        gps_n = self.lpf_gps_n.update(measurement.gps_n)
        gps_e = self.lpf_gps_e.update(measurement.gps_e)
        gps_Vg = self.lpf_gps_Vg.update(measurement.gps_Vg)
        gps_course = self.lpf_gps_course.update(measurement.gps_course)
        # Simplified inverted model smoothing
        self.estimated_state.phi = phi
        self.estimated_state.theta = theta
        self.estimated_state.north = gps_n
        self.estimated_state.east = gps_e
        self.estimated_state.Vg = gps_Vg
        self.estimated_state.chi = gps_course
        
        # estimate phi and theta with ekf
        u_attitude=np.array([
                [self.estimated_state.p],
                [self.estimated_state.q],
                [self.estimated_state.r],
                [self.estimated_state.Va],
                ])
        xhat_attitude, P_attitude=self.attitude_ekf.propagate_model(u_attitude)
        y_accel=np.array([
                [measurement.accel_x],
                [measurement.accel_y],
                [measurement.accel_z],
                ])
        xhat_attitude, P_attitude=self.attitude_ekf.measurement_update(
            y=y_accel, 
            u=u_attitude,
            h=self.h_accel,
            R=self.R_accel)
        self.estimated_state.phi = xhat_attitude.item(0)
        self.estimated_state.theta = xhat_attitude.item(1)
        # # estimate pn, pe, Vg, chi, wn, we, psi with ekf
        # u_smooth = np.array([
        #         [self.estimated_state.q],
        #         [self.estimated_state.r],
        #         [self.estimated_state.Va],
        #         [self.estimated_state.phi],
        #         [self.estimated_state.theta],
        #         ])
        # xhat_position, P_position=self.position_ekf.propagate_model(u_smooth)
        # y_pseudo = np.array([[0.], [0.]])
        # xhat_position, P_position=self.position_ekf.measurement_update(
        #     y=y_pseudo,
        #     u=u_smooth,
        #     h=self.h_pseudo,
        #     R=self.R_pseudo)
        # # only update GPS when one of the signals changes
        # if (measurement.gps_n != self.gps_n_old) \
        #     or (measurement.gps_e != self.gps_e_old) \
        #     or (measurement.gps_Vg != self.gps_Vg_old) \
        #     or (measurement.gps_course != self.gps_course_old):
        #     y_gps = np.array([
        #             [gps_n],
        #             [gps_e],
        #             [gps_Vg],
        #             [wrap(gps_course, xhat_position.item(3))],
        #             ])
        #     xhat_position, P_position=self.position_ekf.measurement_update(
        #         y=y_gps,
        #         u=u_smooth,
        #         h=self.h_gps,
        #         R=self.R_gps)
        #     # update stored GPS signals
        #     self.gps_n_old = measurement.gps_n
        #     self.gps_e_old = measurement.gps_e
        #     self.gps_Vg_old = measurement.gps_Vg
        #     self.gps_course_old = measurement.gps_course
        # self.estimated_state.north = xhat_position.item(0)
        # self.estimated_state.east = xhat_position.item(1)
        # self.estimated_state.Vg = xhat_position.item(2)
        # self.estimated_state.chi = xhat_position.item(3)
        # self.estimated_state.wn = xhat_position.item(4)
        # self.estimated_state.we = xhat_position.item(5)
        # self.estimated_state.psi = xhat_position.item(6)
        # not estimating these
        self.estimated_state.alpha = self.estimated_state.theta
        self.estimated_state.beta = 0.0
        self.estimated_state.bx = 0.0
        self.estimated_state.by = 0.0
        self.estimated_state.bz = 0.0
        
        return self.estimated_state

    def f_attitude(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        '''
            system dynamics for propagation model: xdot = f(x, u)
                x = [phi, theta].T
                u = [p, q, r, Va].T
        '''
        phi = x.item(0)
        theta = x.item(1)
        p = u.item(0)
        q = u.item(1)
        r = u.item(2)
        Va = u.item(3)

        xdot = np.array([[p + q * sin(phi)*tan(theta) + r * cos(phi)*tan(theta)],
                         [q * cos(phi) - r * sin(phi)]])

        return xdot

    def h_accel(self, x: np.ndarray, u: np.ndarray)->np.ndarray:
        '''
            measurement model y=h(x,u) for accelerometers
                x = [phi, theta].T
                u = [p, q, r, Va].T
        '''
        phi = x.item(0)
        theta = x.item(1)
        p = u.item(0)
        q = u.item(1)
        r = u.item(2)
        Va = u.item(3)
        g = MAV.gravity
        
        y = np.array([[q*Va*sin(theta) + g*sin(theta)],
                      [r*Va*cos(theta) - p*Va*sin(theta) - g*cos(theta)*sin(theta)],
                      [-q*Va*cos(theta) - g*cos(theta)*cos(phi)]])
        return y

    def f_smooth(self, x, u):
        '''
            system dynamics for propagation model: xdot = f(x, u)
                x = [pn, pe, Vg, chi, wn, we, psi].T
                u = [p, q, r, Va, phi, theta].T
        '''
        ##### TODO #####        
        xdot = np.zeros((7,1)) 
        return xdot

    def h_pseudo(self, x: np.ndarray, u: np.ndarray)->np.ndarray:
        '''
            measurement model measurement model for wind triangale pseudo measurement: y=y(x, u)
                x = [pn, pe, Vg, chi, wn, we, psi].T
                u = [q, r, Va, phi, theta].T
            returns
                y = [pn, pe, Vg, chi]
        '''
        ##### TODO #####         
        y = np.zeros((2,1))
        return y

    def h_gps(self, x: np.ndarray, u: np.ndarray)->np.ndarray:
        '''
            measurement model for gps measurements: y=y(x, u)
                x = [pn, pe, Vg, chi, wn, we, psi].T
                u = [p, q, r, Va, phi, theta].T
            returns
                y = [pn, pe, Vg, chi]
        '''
        ##### TODO #####         
        y = np.zeros((4,1))
        return y




