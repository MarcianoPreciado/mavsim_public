"""
Class to determine wind velocity at any given moment,
calculates a steady wind speed and uses a stochastic
process to represent wind gusts. (Follows section 4.4 in uav book)
"""
import math

from tools.transfer_function import TransferFunction
import numpy as np
from math import sqrt, pi

class WindSimulation:
    def __init__(self, Ts, gust_flag = True, steady_state = np.array([[0., 0., 0.]]).T):
        # steady state wind defined in the inertial frame
        self._steady_state = steady_state
        self.gust_flag = gust_flag

        #   Dryden gust model parameters (pg 62 UAV book)
        # Nominal airspeed for gust model
        Va0 = 35. # m/s
        # Low altitude, low turbulence parameters
        sig_u = 1.06 # m/s
        sig_v = 1.06 # m/s
        sig_w = 0.7  # m/s
        L_u = 200. # m
        L_v = 200. # m
        L_w = 50.  # m

        # Dryden transfer functions (section 4.4 UAV book) - Fill in proper num and den
        self.u_w = TransferFunction(num=np.array([[sig_u * sqrt(2*Va0/pi/L_u)]]), den=np.array([[1,Va0 / L_u]]),Ts=Ts)
        self.v_w = TransferFunction(num=np.array([[1,Va0/sqrt(3)/L_v]])*sig_v*sqrt(3*Va0/pi/L_v), den=np.array([[1, 2*Va0/L_v, (Va0/L_v)**2]]),Ts=Ts)
        self.w_w = TransferFunction(num=np.array([[1, Va0/sqrt(3)/L_w]])*sig_w*sqrt(3*Va0/pi/L_w), den=np.array([[1, 2*Va0/L_w, (Va0/L_w)**2]]),Ts=Ts)
        self._Ts = Ts

    def update(self):
        # returns a six vector.
        #   The first three elements are the steady state wind in the inertial frame
        #   The second three elements are the gust in the body frame
        if self.gust_flag:
            gust = np.array([[self.u_w.update(np.random.randn())],
                            [self.v_w.update(np.random.randn())],
                            [self.w_w.update(np.random.randn())]])
        else:
            gust = np.zeros((3,1))
        return np.concatenate(( self._steady_state, gust ))

