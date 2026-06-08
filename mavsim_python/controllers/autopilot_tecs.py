"""
autopilot block for mavsim_python - Total Energy Control System
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/14/2020 - RWB
"""
import numpy as np
import parameters.control_parameters as AP
import parameters.aerosonde_parameters as MAV
import parameters.simulation_parameters as SIM
from tools.transfer_function import TransferFunction
from tools.wrap import wrap
from controllers.pi_control import PIControl
from controllers.pd_control_with_rate import PDControlWithRate
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

roll_saturation = np.radians(60)

class Autopilot:
    def __init__(self, ts_control):
        # instantiate lateral controllers
        self.roll_from_aileron = PDControlWithRate(
                        kp=AP.roll_kp,
                        kd=AP.roll_kd,
                        limit=np.radians(45))
        self.course_from_roll = PIControl(
                        kp=AP.course_kp,
                        ki=AP.course_ki,
                        Ts=ts_control,
                        limit=roll_saturation)
        self.yaw_damper = TransferFunction(
                        num=np.array([[AP.yaw_damper_kr, 0]]),
                        den=np.array([[1, AP.yaw_damper_p_wo]]),
                        Ts=ts_control)

        # instantiate TECS controllers
        self.pitch_from_elevator = PDControlWithRate(
                        kp=AP.pitch_kp,
                        kd=AP.pitch_kd,
                        limit=np.radians(45))
        # throttle gains (unitless)
        # Ku = 0.0019
        # Tu = 1.75
        # self.E_kp = 0.45 * Ku
        # self.E_ki = 0.54*Ku/Tu
        # # pitch gains
        # # Ku = 0.0004
        # # Tu = 2.58
        # Ku = 0.0001
        # Tu = 3
        # self.B_kp = 0.45 * Ku
        # self.B_ki = 0.54*Ku/Tu
        # self.B_kp = 0.0008
        # self.B_ki = 0

        scale = 0.0005
        self.B_kp = self.E_kp = 1 * scale
        self.B_ki = self.E_ki = 0.4 * scale


        # saturated altitude error
        self.h_error_max = 10  # meters
        self.E_integrator = 0
        self.B_integrator = 0
        self.E_error_d1 = 0
        self.B_error_d1 = 0
        self.delta_t_d1 = 0
        self.theta_c_d1 = 0
        self.theta_c_max = 0
        self.Ts = ts_control
        self.commanded_state = MsgState()

    def update(self, cmd, state):
	
	###### TODO ######
        # lateral autopilot
        chi_c = wrap(cmd.course_command, state.chi)
        phi_c = self.course_from_roll.update(chi_c, state.chi)
        phi_c = wrap(phi_c, state.phi)
        phi_c = saturate(phi_c, -roll_saturation, roll_saturation)
        delta_a = self.roll_from_aileron.update(phi_c, state.phi, state.p)
        
        delta_r = self.yaw_damper.update(state.r)

        # longitudinal TECS autopilot
        m = MAV.mass
        g = AP.gravity
        Vac = cmd.airspeed_command
        Va = state.Va
        hc = cmd.altitude_command
        h = state.altitude
        # Total Energy Control System
        Uerror = m*g*saturate(hc - h, -self.h_error_max, self.h_error_max)
        Kerror = 0.5*m*g*(Vac**2 - Va**2)
        
        E = Uerror + Kerror
        B = Uerror - Kerror

        self.E_integrator += (E + self.E_error_d1) * self.Ts / 2
        self.B_integrator += (B + self.B_error_d1) * self.Ts / 2
        self.E_error_d1 = E
        self.B_error_d1 = B

        delta_t = self.E_kp*E + self.E_ki*self.E_integrator
        delta_t = saturate(delta_t, 0, 2)
        theta_c = (self.B_kp*B + self.B_ki*self.B_integrator)

        # Pitch hold using the elevator H = theta(s) / theta_c(s)
        delta_e = self.pitch_from_elevator.update(theta_c, state.theta, state.q)

        # construct output and commanded states
        delta = MsgDelta(elevator=delta_e,  # placeholder for elevator command
                         aileron=delta_a,
                         rudder=delta_r,
                         throttle=delta_t)
        self.commanded_state.altitude = cmd.altitude_command                
        self.commanded_state.Va = cmd.airspeed_command
        self.commanded_state.phi = phi_c
        self.commanded_state.theta = theta_c
        self.commanded_state.chi = cmd.course_command
        return delta, self.commanded_state

    def saturate(self, input, low_limit, up_limit):
        if input <= low_limit:
            output = low_limit
        elif input >= up_limit:
            output = up_limit
        else:
            output = input
        return output
