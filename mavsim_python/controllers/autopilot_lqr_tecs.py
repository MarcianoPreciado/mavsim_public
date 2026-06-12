"""
autopilot block for mavsim_python
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/10/22 - RWB
"""
import numpy as np
from numpy import array, sin, cos, radians, concatenate, zeros, diag
from scipy.linalg import solve_continuous_are, inv
import parameters.control_parameters as AP
import parameters.aerosonde_parameters as MAV

from tools.wrap import wrap
import models.model_coef as M
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta

def saturate(input, low_limit, up_limit):
    if input <= low_limit:
        output = low_limit
    elif input >= up_limit:
        output = up_limit
    else:
        output = input
    return output


class Autopilot:
    def __init__(self, ts_control):
        self.Ts = ts_control
        # initialize integrators and delay variables
        self.integratorCourse = 0
        self.integratorAltitude = 0
        self.integratorAirspeed = 0
        self.integratorE = 0
        self.integratorL = 0
        self.errorCourseD1 = 0
        self.errorED1 = 0
        self.errorLD1 = 0
        self.errorAltitudeD1 = 0
        self.errorAirspeedD1 = 0
        # saturation flags (previous step) for integrator pause anti-windup
        self.delta_a_sat = False
        self.delta_e_sat = False
        self.delta_t_sat = False
        # compute LQR gains
        
        CrLat = array([[0, 0, 0, 0, 1]])
        AAlat = concatenate((
                    concatenate((M.A_lat, zeros((5,1))), axis=1),
                    concatenate((CrLat, zeros((1,1))), axis=1)),
                    axis=0)
        BBlat = concatenate((M.B_lat, zeros((1,2))), axis=0)
        # Qlat = diag ([.001 , .01 , .1 , 100, 1, 100]) # v, p, r , phi , chi , intChi
        Qlat = diag([0.01, 0.01, 0.01, 10.0, 100.0, 10.0]) # v, p, r, phi, chi, intChi
        Rlat = diag ([1 , 1]) # a, r
        Plat = solve_continuous_are(AAlat, BBlat, Qlat, Rlat)
        self.Klat = inv(Rlat) @ BBlat.T @ Plat
        
        AAlon = np.array([
            [-0.2822,  0.4946, -1.2123, -9.7979,  0.,      0.,      0.,      0.,      0.    ],
            [-0.5611, -4.4978, 24.3714, -0.4874,  0.,      0.,      0.,      0.,      0.    ],
            [ 0.1986, -3.993,  -5.2947,  0.,      0.,      0.,      0.,      0.,      0.    ],
            [ 0.,      0.,      1.0001,  0.,      0.,      0.,      0.,      0.,      0.    ],
            [ 0.0497, -0.9988,  0.,     25.,      0.,      0.,      0.,      0.,      0.    ],
            [-0.0316,  0.0276, -0.,     -0.0012,  0.,      0.,      0.,      0.,      0.    ],
            [-0.0316,  0.0276, -0.,     -1.9988,  0.,      0.,      0.,      0.,      0.    ],
            [ 0.,      0.,      0.,      0.,      1.,      0.,      0.,      0.,      0.    ],
            [ 0.9988,  0.0497,  0.,      0.,      0.,      0.,      0.,      0.,      0.    ],
        ])
        BBlon = np.array([
            [ -0.1392,  9.4813],
            [ -2.5861,  0.    ],
            [-36.1124,  0.    ],
            [  0.,       0.    ],
            [  0.,       0.    ],
            [ -0.0273,  0.9653],
            [ -0.0273,  0.9653],
            [  0.,       0.    ],
            [  0.,       0.    ],
        ])

        Qlon = diag([1, 1, 1, 1/3**3, 2, 1/10**2, 1/10**2, 1/25**2., 1/20**2.]) # u, w, q, theta, h, Edot_int, Ldot_int, h_int, Va_int
        Rlon = diag ([0.5 , 1]) # e , t
        Plon = solve_continuous_are(AAlon, BBlon, Qlon, Rlon)
        self.Klon = inv(Rlon) @ BBlon.T @ Plon
        # C_lon: maps x_lon=[u,w,q,theta,h] -> [Edot, Ldot]; embedded in AAlon rows 5-6
        self.C_lon = AAlon[5:7, 0:5]
        self.commanded_state = MsgState()

        self.Klon = np.array([[ -0.08,   2.72,  -1.36, -92.49, -22.75,  -0.26,   0.27],
 [  2.08,   0.19,  -0.03,  -3.08,  -0.45,   0.01,   0.02]])

    def update(self, cmd, state):
        Var = cmd.airspeed_command
        hr = cmd.altitude_command
        chir = wrap(cmd.course_command, state.chi)

        # lateral autopilot
        eVa = state.Va - Var
        eChi = saturate(state.chi - chir, -np.radians(15), np.radians(15))
        if not self.delta_a_sat:
            self.integratorCourse += (eChi + self.errorCourseD1) * self.Ts / 2
        self.errorCourseD1 = eChi

        eu = eVa * cos(state.alpha)
        ev = eVa * sin(state.beta)
        ew = eVa * sin(state.alpha)

        xLat = array([[ ev ] , # v
                    [ state.p] ,
                    [ state.r] ,
                    [ state.phi ] ,
                    [ eChi ] ,
                    [ self.integratorCourse ]])

        tmp = -self.Klat @ xLat
        delta_a_raw = tmp.item(0)
        delta_r_raw = tmp.item(1)
        self.delta_a_sat = abs(delta_a_raw) >= radians(30)
        delta_a = saturate(delta_a_raw, -radians(30), radians(30))
        delta_r = saturate(delta_r_raw, -radians(30), radians(30))


        # longitudinal autopilot
        hr = saturate(hr, state.altitude - 0.2*AP.altitude_zone, state.altitude + 0.2*AP.altitude_zone)
        eh = state.altitude - hr
        if not self.delta_e_sat:
            self.integratorAltitude += (eh + self.errorAltitudeD1) * self.Ts / 2
        self.errorAltitudeD1 = eh

        if not self.delta_t_sat:
            self.integratorAirspeed += (eVa + self.errorAirspeedD1) * self.Ts / 2
        self.errorAirspeedD1 = eVa

        E_err = (AP.u_star / MAV.gravity) * eu + \
                (AP.v_star / MAV.gravity) * ev +  \
                (AP.w_star / MAV.gravity) * ew +  \
                eh  # = δVa·Va*/g + δh
        D_err = (AP.u_star / MAV.gravity) * eu + \
                (AP.v_star / MAV.gravity) * ev + \
                (AP.w_star / MAV.gravity) * ew - \
                eh  # = δVa·Va*/g - δh


        xLon = array([[ eVa * cos(state.alpha) ] , # u
                      [ eVa * sin(state.alpha) ] , # w
                      [ state.q] ,
                      [ state.theta ] ,
                      [ eh ] ,
                      [ E_err ],
                      [ D_err ]])

        tmp = -self.Klon @ xLon
        delta_e_raw = tmp.item(0)
        delta_t_raw = tmp.item(1)
        self.delta_e_sat = abs(delta_e_raw) >= radians(30)
        self.delta_t_sat = not (0 < delta_t_raw < 1)
        delta_e = saturate(delta_e_raw, -radians(30), radians(30))
        delta_t = saturate(delta_t_raw, 0, 1)

        # construct control outputs and commanded states
        delta = MsgDelta(elevator=delta_e,
                         aileron=delta_a,
                         rudder=delta_r,
                         throttle=delta_t)
        self.commanded_state.altitude = cmd.altitude_command
        self.commanded_state.Va = cmd.airspeed_command
        # self.commanded_state.phi = cmd.course_command
        # self.commanded_state.theta = 0
        self.commanded_state.chi = cmd.course_command
        return delta, self.commanded_state

