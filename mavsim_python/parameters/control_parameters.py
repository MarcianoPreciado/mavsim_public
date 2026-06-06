import numpy as np
from numpy import sqrt, roots
import models.model_coef as TF
import parameters.aerosonde_parameters as MAV


#### TODO #####
gravity = MAV.gravity  # gravity constant
Va0 = TF.Va_trim
rho = 1.293 # density of air
sigma = 0  # low pass filter gain for derivative
Vg = TF.Va_trim  # ground speed (m/s)
m = MAV.mass
S = MAV.S_wing
b = MAV.b
c = MAV.c

x_star = TF.x_trim
inputs_star = TF.u_trim

u_star = x_star.item(3)
v_star = x_star.item(4)
w_star = x_star.item(6)

phi_star = x_star.item(7)
theta_star = x_star.item(8)
psi_star = x_star.item(9)

p_star = x_star.item(9)
q_star = x_star.item(10)
r_star = x_star.item(11)

delta_e_star = inputs_star.item(0)
delta_a_star = inputs_star.item(1)
delta_r_star = inputs_star.item(2)
delta_t_star = inputs_star.item(3)

beta_star = psi_star


CYdr = MAV.C_Y_delta_r
CYp = MAV.C_Y_p
CYr = MAV.C_Y_r
CYb = MAV.C_Y_beta
CY0 = MAV.C_Y_0
CYda = MAV.C_Y_delta_a
Cr0 = MAV.C_n_0
Crp = MAV.C_n_p
Crr = MAV.C_n_r
Crb = MAV.C_n_beta
Crda = MAV.C_n_delta_a
Crdr = MAV.C_n_delta_r
gamma1 = MAV.gamma1

# Lateral state-space model coeﬃcients from Chapter 5 table 5.1
Yv = rho*S*b*v_star/Va0 * (CYp*p_star + CYr*r_star) + \
    rho*S*c*CYb/(2*m) * sqrt(u_star**2 + v_star**2) + \
    rho*S*v_star/m*(CY0 + CYb*beta_star + CYda*delta_a_star + CYdr*delta_r_star)
Yr = - u_star + rho*S*b*Va0/(4*m)*CYp
Ydr = rho*Va0**2*S / (2*m) * CYdr
Nv = rho*S*b**2*v_star/(4*Va0) * (Crp * p_star + Crr * r_star) + \
    rho*S*b*Crb/2 * sqrt(u_star**2 + w_star**2) + \
    rho*S*b*v_star * (Cr0 + Crb*beta_star + Crda*delta_a_star + Crdr*delta_r_star)
Nr = -gamma1 * q_star + rho*S*b**2*Va0/4 * Crr
Ndr = rho*Va0**2*S*b/2 * Crdr

Wx = 10 # bandwidth scale between loops

#----------roll loop-------------
# get transfer function data for delta_a to phi
wn_roll = sqrt(TF.a_phi2) * 0.8
zeta_roll = 1.2
roll_kp = wn_roll**2 / TF.a_phi2
roll_kd = (2*zeta_roll*wn_roll - TF.a_phi1) / TF.a_phi2

#----------course loop-------------
# chi_c to phi_c (course to pitch)
wn_course = wn_roll / 4
zeta_course = 0.7
course_kp = (2 * zeta_course * wn_course * Vg) / gravity
course_ki = wn_course**2 * Vg / gravity

#----------yaw damper-------------
wn_dutch_roll = sqrt(Yv*Nr - Nv*Yr)
yaw_damper_p_wo = wn_dutch_roll/10
krs = roots([Ndr**2, 2*(Nr*Ndr + Ydr*Nv), (Yv**2 + Nr**2 + 2*Yr*Nv)])
yaw_damper_kr = krs[0]

#----------pitch loop-------------
wn_pitch = 0
zeta_pitch = 0 
pitch_kp = 0
pitch_kd = 0
K_theta_DC = 0

#----------altitude loop-------------
wn_altitude = 0
zeta_altitude = 0
altitude_kp = 0
altitude_ki = 0
altitude_zone = 10

#---------airspeed hold using throttle---------------
wn_airspeed_throttle = 0
zeta_airspeed_throttle = 0
airspeed_throttle_kp = 0
airspeed_throttle_ki = 0
