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
        self.integratorE = 0
        self.integratorL = 0
        self.errorCourseD1 = 0
        self.errorED1 = 0
        self.errorLD1 = 0
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
        
        AAlon = array([
            [-0.2822,  0.4946, -1.242,   0.,      0.,      0.,      0.,    0.,    0.    ],
            [-0.5611, -4.4978, 24.9691,  0.1025,  0.,      0.,      0.,    0.,    0.    ],
            [ 0.1986, -3.993,   0.,      0.,      0.,      0.,      0.,    0.,    0.    ],
            [ 0.,      0.,      1.0001,  0.,      0.,      0.,      0.,    0.,    0.    ],
            [ 0.0497, -0.9988,  0.,     25.,      0.,      0.,      0.,    0.,    0.    ],
            [10.1811,  0.5064,  0.,      0.9988,  0.,      0.,      0.,    0.,    0.    ],
            [10.1811,  0.5064,  0.,     -0.9988,  0.,      0.,      0.,    0.,    0.    ]
        ])

        BBlon = array([
            [-0.1392,   9.4813],
            [-2.5861,   0.    ],
            [-36.1124,  0.    ],
            [ 0.,        0.    ],
            [ 0.,        0.    ],
            [ 0.,        0.    ],
            [ 0.,        0.    ]
        ])

        CClon = array([
            [10.1811,  0.5064,  0.,      0.9988,  0.,      0.,      0.,    0.,    0.    ],
            [10.1811,  0.5064,  0.,     -0.9988,  0.,      0.,      0.,    0.,    0.    ],
            [ 0.,      0.,      0.,      0.,      0.,      1.,      0.,    0.,    0.    ],
            [ 0.,      0.,      0.,      0.,      0.,      0.,      1.,    0.,    0.    ]
        ])

        DDlon = array([
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]
        ])

        Qlon = diag([10.0, 100.0, 100, 10, 50.0, 100, 100, 10, 10]) # u, w, q, theta, h, Edot, Ldot, Edot_int, Ldot_int
        Rlon = diag ([1 , 1]) # e , t
        Plon = solve_continuous_are(AAlon, BBlon, Qlon, Rlon)
        self.Klon = inv(Rlon) @ BBlon.T @ Plon
        self.commanded_state = MsgState()

    def update(self, cmd, state):
        Var = cmd.airspeed_command
        hr = cmd.altitude_command
        chir = wrap(cmd.course_command, state.chi)

        # lateral autopilot
        eVa = state.Va - Var
        eChi = saturate(state.chi - chir, -np.radians(15), np.radians(15))
        self.integratorCourse += (eChi + self.errorCourseD1) * self.Ts / 2
        self.errorCourseD1 = eChi
        if abs(eChi) < np.radians(10):
            self.integratorCourse = 0
            self.errorCourseD1 = 0

        xLat = array([[ eVa * sin(state.beta )] , # v
                    [ state.p] ,
                    [ state.r] ,
                    [ state.phi ] ,
                    [ eChi ] ,
                    [ self.integratorCourse ]])

        tmp = -self.Klat @ xLat
        delta_a = saturate(tmp.item(0), -radians(30), radians(30))
        delta_r = saturate(tmp.item(1), -radians(30), radians(30))


        # longitudinal autopilot
        hr = saturate(hr, state.altitude - 0.2*AP.altitude_zone, state.altitude + 0.2*AP.altitude_zone)
        eh = state.altitude - hr
        self.integratorE += (eh + self.errorED1) * self.Ts / 2
        self.errorED1 = eh
        # if abs(eh) < 0.1*AP.altitude_zone:
        #     self.integratorE = 0
        #     self.errorED1 = 0

        self.integratorL += (eVa + self.errorLD1) * self.Ts / 2
        self.errorLD1 = eVa
        # if abs(eVa) < 0.12:
        #     self.integratorL = 0
        #     self.errorLD1 = 0
        Edot = 0
        Ldot = 0
        xLon = array([[ eVa * cos(state.alpha) ] , # u
                      [ eVa * sin(state.alpha) ] , # w
                      [ state.q] ,
                      [ state.theta ] ,
                      [ eh ] ,
                      [ Edot ],
                      [ Ldot ],
                      [ self.integratorE ] ,
                      [ self.integratorL ]])
        tmp = -self.Klon @ xLon
        delta_e = saturate(tmp.item(0), -radians(30), radians(30))
        delta_t = saturate(tmp.item(1), 0, 1)

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

