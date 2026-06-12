"""
Step-response tuning test for the TECS autopilot.
Controller: controllers/autopilot_tecs.py

Run from repo root with venv active:
    python mavsim_python/launch_files/chap06/test_tecs.py

Gains to tune:
  Lateral (parameters/control_parameters.py):
      roll_kp, roll_kd
      course_kp, course_ki
      yaw_damper_kr, yaw_damper_p_wo

  Longitudinal — pitch inner loop (control_parameters.py):
      pitch_kp, pitch_kd  (tuned via wn_pitch, zeta_pitch)

  Longitudinal — TECS energy loops (set directly in autopilot_tecs.py):
      E_kp, E_ki   → throttle (total energy error)
      B_kp, B_ki   → pitch command (energy balance error)
      h_error_max  → max altitude error fed into TECS
      theta_c_max  → pitch command saturation

The bottom-right panel shows the TECS energy errors E and B, which
directly reveal how the energy is being distributed between altitude and speed.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.fspath(Path(__file__).parents[2]))

import numpy as np
import matplotlib.pyplot as plt
import parameters.simulation_parameters as SIM
import parameters.control_parameters as AP
from models.mav_dynamics_control import MavDynamics
from models.wind_simulation import WindSimulation
from tools.signals import Signals
from message_types.msg_autopilot import MsgAutopilot
from controllers.autopilot_tecs import Autopilot
from tools.transfer_function import TransferFunction
from tools.wrap import wrap
from controllers.pi_control import PIControl
from controllers.pd_control_with_rate import PDControlWithRate

def _saturate(x, lo, hi):
    return max(lo, min(hi, x))

# --- simulation setup ---
Ts = SIM.ts_simulation
wind = WindSimulation(Ts)
mav = MavDynamics(Ts)
commands = MsgAutopilot()

# --- command signals ---
phi_command  = Signals(dc_offset=0.0,           amplitude=np.radians(30),            start_time=2.0, frequency=1/60)
chi_command   = Signals(dc_offset=0.0,          amplitude=np.radians(30),           start_time=2.0, frequency=1/60)
theta_command = Signals(dc_offset=np.radians(0), amplitude=np.radians(30),  start_time=2.0, frequency=1/60)

# --- data buffers ---
t_hist      = []
chi_hist    = []; chi_c_hist   = []
phi_hist    = []; phi_c_hist   = []
theta_hist  = []; theta_c_hist = []
alt_hist    = []; alt_c_hist   = []
Va_hist     = []; Va_c_hist    = []
da_hist     = []; de_hist      = []
dr_hist     = []; dt_hist      = []
E_hist      = []; B_hist       = []

from parameters.control_parameters import \
    wn_roll, zeta_roll, wn_pitch, zeta_pitch, wn_course, zeta_course, \
    yaw_damper_kr, yaw_damper_p_wo , pitch_kp, pitch_kd, K_theta_DC

PHI_TS = TransferFunction(
                num=np.array([[wn_roll**2]]),
                 den=np.array([[1, 2*zeta_roll*wn_roll, wn_roll**2]]),
                 Ts=Ts)
CHI_TS = TransferFunction(
                num=np.array([[2*wn_course*zeta_course, wn_course**2]]),
                den=np.array([[1, 2*zeta_course*wn_course, wn_course**2]]),
                Ts=Ts)
THETA_TS = TransferFunction(
                num=np.array([[wn_pitch**2]]),
                den=np.array([[1, 2*zeta_pitch*wn_pitch, wn_pitch**2]]),
                Ts=Ts)
R_TS = TransferFunction(
                num=np.array([[AP.yaw_damper_kr, 0]]),
                den=np.array([[1, AP.yaw_damper_p_wo]]),
                Ts=Ts)

sim_time = SIM.start_time
end_time = 10.0

phi = 0; dphi_dt = 0
chi = 0
r = 0
theta = 0; dtheta_dt = 0

print(f"Running TECS autopilot step-response test  ({end_time} s) ...")
while sim_time < end_time:
    phi_c = wrap(phi_command.square(sim_time), phi)
    chi_c = wrap(chi_command.square(sim_time), chi)
    theta_c = theta_command.square(sim_time)

    phi = PHI_TS.update(phi_c)
    chi = CHI_TS.update(chi_c)
    theta = THETA_TS.update(theta_c)
    r = R_TS.update(r)

    # append data and control signals to buffers
    t_hist.append(sim_time)
    chi_hist.append(np.degrees(chi))
    chi_c_hist.append(np.degrees(chi_c))
    phi_hist.append(np.degrees(phi))
    phi_c_hist.append(np.degrees(phi_c))
    theta_hist.append(np.degrees(theta))
    theta_c_hist.append(np.degrees(theta_c))

    sim_time += Ts

print("Plotting ...")

fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
fig.suptitle("TECS Autopilot — Step Response", fontsize=14, fontweight="bold")

def _step_plot(ax, t, ref, actual, ylabel, title, ref_label, act_label, ref_color="k"):
    ax.plot(t, ref,    color=ref_color, linestyle="--", linewidth=1.5, label=ref_label)
    ax.plot(t, actual, linewidth=1.2,                                  label=act_label)
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.set_xlabel("Time (s)");
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

t = t_hist
_step_plot(axes[0,0], t, chi_c_hist,   chi_hist,   "deg", "Course  χ",   "χ_c", "χ")
_step_plot(axes[0,1], t, phi_c_hist,   phi_hist,   "deg", "Roll  φ",     "φ_c", "φ",  ref_color="gray")
_step_plot(axes[1,0], t, theta_c_hist, theta_hist, "deg", "Pitch  θ",    "θ_c", "θ",  ref_color="purple")

for ax in axes[2, :]:
    ax.set_xlabel("Time (s)")

plt.tight_layout()
out = Path(__file__).parent / "tecs_step_response.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()
