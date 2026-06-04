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
        self.integratorAltitude = 0
        self.integratorAirspeed = 0
        self.errorCourseD1 = 0
        self.errorAltitudeD1 = 0
        self.errorAirspeedD1 = 0
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
        
        u_trim = M.x_trim.item(3) # state variable u
        w_trim = M.x_trim.item(4) # state variable w
        Va = AP.Va0
        CrLon = array([[0, 0, 0, 0, 1], [u_trim/Va, w_trim/Va, 0, 0, 0]])
        AAlon = concatenate((
                    concatenate((M.A_lon, zeros((5,2))), axis=1),
                    concatenate((CrLon, zeros((2,2))), axis=1)),
                    axis=0)
        BBlon = concatenate((M.B_lon, zeros((2, 2))), axis=0)
        # Qlon = diag ([10 , 10 , .001 , .01 , 10 , 100 , 100]) # u, w, q , theta , h, intH , intVa
        Qlon = diag([10.0, 10.0, 100, 10, 50.0, 60.0, 50.0]) # u, w, q, theta, h, intH, intVa
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
        self.integratorAltitude += (eh + self.errorAltitudeD1) * self.Ts / 2
        self.errorAltitudeD1 = eh
        if abs(eh) < 0.1*AP.altitude_zone:
            self.integratorAltitude = 0
            self.errorAltitudeD1 = 0

        self.integratorAirspeed += (eVa + self.errorAirspeedD1) * self.Ts / 2
        self.errorAirspeedD1 = eVa
        if abs(eVa) < 0.12:
            self.integratorAirspeed = 0
            self.errorAirspeedD1 = 0

        xLon = array([[ eVa * cos(state.alpha) ] , # u
                      [ eVa * sin(state.alpha) ] , # w
                      [ state.q] ,
                      [ state.theta ] ,
                      [ eh ] ,
                      [ self.integratorAltitude ] ,
                      [ self.integratorAirspeed ]])
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
        print("delta: ", delta.print())
        return delta, self.commanded_state

