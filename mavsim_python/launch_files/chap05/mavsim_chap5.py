"""
mavsim_python
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Last Update:
        2/2/2019 - RWB
        1/5/2023 - David L. Christiansen
        7/13/2023 - RWB
"""
import os, sys
# insert parent directory at beginning of python search path
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[2]))
# use QuitListener for Linux or PC <- doesn't work on Mac
#from tools.quit_listener import QuitListener
import numpy as np
import parameters.simulation_parameters as SIM
from models.mav_dynamics_control import MavDynamics
from models.wind_simulation import WindSimulation
from models.trim import compute_trim
from models.compute_models import compute_model, compute_ss_model
from tools.signals import Signals
from viewers.view_manager import ViewManager
import time

#quitter = QuitListener()

# initialize elements of the architecture
wind = WindSimulation(SIM.ts_simulation)
mav = MavDynamics(SIM.ts_simulation)
viewers = ViewManager(mav=True, 
                      data=True,
                      video=False, video_name='chap5.mp4')

# use compute_trim function to compute trim state and trim input
Va = 25.
gamma = 0.*np.pi/180.
trim_state, trim_input = compute_trim(mav, Va, gamma)
mav._state = trim_state  # set the initial state of the mav to the trim state
delta = trim_input  # set input to constant constant trim input

# # compute the state space model linearized about trim
compute_model(mav, trim_state, trim_input)

# Group into conjugate pairs and extract ωₙ, ζ
def evaluate_dynamics(A):
    eig = np.linalg.eigvals(A)
    for lam in eig:
        if abs(lam) < 1e-6:       # handle zero eigenvalues
            print(f"λ = {lam:.4f} (zero eigenvalue)")
            continue
        if lam.imag >= 0:          # take one from each pair
            wn  = abs(lam)         # |λ| = ωₙ
            zeta = abs(lam.real) / wn   # Re(λ)/|λ| = ζ
            period = 2*np.pi / wn if wn > 1e-6 else float('inf')  # handle zero frequency

            print(f"λ = {lam:.4f}")
            print(f"  ωₙ = {wn:.4f}  rad/s")
            print(f"  ζ  = {zeta:.4f}")
            print(f"  period = {period:.4f} s")
            stability = "stable" if lam.real < 0 else "unstable" 
            damping = "undamped" if abs(zeta) < 1e-6 else "underdamped" if zeta < 1 else "critically damped" if zeta == 1 else "overdamped"
            print(f"  mode is {stability} and {damping}")
            print()

A_lon, B_lon, A_lat, B_lat = compute_ss_model(mav, trim_state, trim_input)

print("### Longitudinal eigenvalues and modes ###")
eig_lon = np.linalg.eigvals(A_lon)
evaluate_dynamics(A_lon)

print("### Lateral eigenvalues and modes ###")
eig_lat = np.linalg.eigvals(A_lat)
evaluate_dynamics(A_lat)


# this signal will be used to excite modes
input_signal = Signals(amplitude=0.3,
                       duration=0.3,
                       start_time=5.0)
delta_e_trim = delta.elevator
delta_a_trim = delta.aileron
delta_r_trim = delta.rudder

# initialize the simulation time
sim_time = SIM.start_time
end_time = 60

# main simulation loop
print("Press 'Esc' to exit...")
while sim_time < end_time:

    # -------physical system-------------
    #current_wind = wind.update()  # get the new wind vector
    current_wind = np.zeros((6, 1))
    # this input excites the phugoid mode by adding an elevator impulse at t = 5.0 s
    # delta.elevator = delta_e_trim + input_signal.impulse(sim_time)
    # this input excites the roll and spiral divergence modes by adding an aileron doublet at t = 5.0 s
    delta.aileron = delta_a_trim + input_signal.doublet(sim_time)
    # this input excites the dutch roll mode by adding a rudder doublet at t = 5.0 s
    # delta.rudder = delta_r_trim + input_signal.doublet(sim_time)

    mav.update(delta, current_wind)  # propagate the MAV dynamics

    # -------update viewer-------------
    viewers.update(
        sim_time,
        true_state=mav.true_state,  # true states
        delta=delta,
    )
        
    # -------Check to Quit the Loop-------
    # if quitter.check_quit():
    #     break

    # -------increment time-------------
    sim_time += SIM.ts_simulation
    time.sleep(0.002) # slow down the simulation for visualization
    
viewers.close(dataplot_name="ch5_data_plot")



