# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the companion code for *Small Unmanned Aircraft: Theory and Practice* by Beard & McLain (Princeton University Press, 2012). It simulates fixed-wing MAV (Micro Air Vehicle) flight dynamics, autopilot design, state estimation, path planning, and camera geolocation. Three implementations exist side-by-side: `mavsim_python/` (active), `legacy_mavsim_python/` (older structure), and `mavsim_matlab/` + `mavsim_simulink/`.

The primary working directory for Python development is **`mavsim_python/`**.

## Setup

**Always use the local virtual environment.** Before running any Python commands, source it:

```bash
source /Users/marciano/Repos/mavsim/venv/bin/activate
```

The venv is at `/Users/marciano/Repos/mavsim/venv` and already has all required packages installed (numpy, scipy, matplotlib, pyqtgraph, pyqt6, pyopengl, pynput). Do not use the system Python or install packages globally.

## Running Simulations

Each chapter has a launch file. Run from the repo root with the venv active:

```bash
python mavsim_python/launch_files/chap02/mavsim_chap2.py   # coordinate frames & visualization
python mavsim_python/launch_files/chap03/mavsim_chap3.py   # rigid body dynamics
python mavsim_python/launch_files/chap04/mavsim_chap4.py   # forces & moments
python mavsim_python/launch_files/chap05/mavsim_chap5.py   # trim & linear models
python mavsim_python/launch_files/chap06/mavsim_chap6.py   # autopilot
python mavsim_python/launch_files/chap07/mavsim_chap7.py   # sensors
python mavsim_python/launch_files/chap08/mavsim_chap8.py   # state estimation (EKF)
python mavsim_python/launch_files/chap10/mavsim_chap10.py  # waypoint/orbit following
python mavsim_python/launch_files/chap11/mavsim_chap11.py  # path manager
python mavsim_python/launch_files/chap12/mavsim_chap12.py  # path planning (RRT)
python mavsim_python/launch_files/chap13/mavsim_chap13_point_camera.py   # gimbal pointing
python mavsim_python/launch_files/chap13/mavsim_chap13_orbit_target.py   # target orbit
```

Each launch file adds `mavsim_python/` to `sys.path`, so imports like `from models.mav_dynamics import MavDynamics` work from any launch file.

Press **Esc** (or **Command-Q** on Mac) to exit the simulation window.

## Architecture

### Module Structure (`mavsim_python/`)

```
models/         — MAV physical simulation (dynamics, sensors, camera, wind, trim)
controllers/    — Autopilot implementations (PID, LQR, TECS)
estimators/     — State observers and EKF-based geolocation
planners/       — Path follower, path manager, RRT path planner
viewers/        — PyQtGraph-based 3D/2D visualization
message_types/  — Data classes passed between modules
parameters/     — Aerosonde aircraft, simulation, sensor, camera, control parameters
tools/          — Rotation utilities, signal generators, transfer functions
launch_files/   — Chapter-specific simulation entry points (chapXX/)
```

### MavDynamics Inheritance Chain

Each chapter builds on the previous dynamics layer:

```
MavDynamics (mav_dynamics.py)          — 13-state RK4 integrator (position, velocity, quaternion, angular rates)
  └─ MavDynamics (mav_dynamics_control.py)  — adds aerodynamic forces/moments, wind, Va/alpha/beta
       └─ MavDynamics (mav_dynamics_sensors.py) — adds IMU, GPS, pressure sensor simulation
            └─ MavDynamics (mav_dynamics_camera.py) — adds gimbal azimuth/elevation states (indices 13, 14)
```

Import the appropriate layer for the chapter being implemented. Later chapters use `mav_dynamics_camera.MavDynamics`.

### State Vector

The internal `_state` is 15×1 in NumPy column format:

| Index | Variable | Description |
|-------|----------|-------------|
| 0–2   | pn, pe, pd | NED position (meters) |
| 3–5   | u, v, w | Body-frame velocity (m/s) |
| 6–9   | e0–e3 | Unit quaternion attitude |
| 10–12 | p, q, r | Angular rates (rad/s) |
| 13–14 | gimbal_az, gimbal_el | Gimbal angles (chap13 only) |

The `MsgState` object (`true_state`) exposes derived quantities: `Va`, `alpha`, `beta`, `phi`, `theta`, `psi`, `chi`, `Vg`, `wn`, `we`, `altitude`, `camera_az`, `camera_el`.

### Coordinate Frame Convention

All positions use **NED (North-East-Down)**. Altitude is positive upward (`altitude = -pd`). Rotations use the `tools/rotations.py` utilities: `euler_to_rotation`, `quaternion_to_euler`, `quaternion_to_rotation`, `Euler2Quaternion`.

### Message Types (Data Flow)

Modules communicate through message objects in `message_types/`:
- `MsgState` — full MAV state (true or estimated)
- `MsgDelta` — control surface deflections (`delta_a`, `delta_e`, `delta_r`, `delta_t`, `gimbal_az`, `gimbal_el`)
- `MsgSensors` — raw sensor readings (gyro, accel, GPS, pressure)
- `MsgAutopilot` — commanded course/altitude/airspeed
- `MsgPath` — current path segment (line or orbit)
- `MsgWaypoints` — list of waypoints
- `MsgCamera` — pixel measurements (`pixel_x`, `pixel_y`)

### Main Simulation Loop Pattern

```python
while sim_time < SIM.end_time:
    measurements = mav.sensors()
    estimated_state = observer.update(measurements)   # or mav.true_state for debugging
    waypoints = path_planner.update(...)              # when requested
    path = path_manager.update(waypoints, ...)
    autopilot_commands = path_follower.update(path, estimated_state)
    delta, commanded_state = autopilot.update(autopilot_commands, estimated_state)
    current_wind = wind.update()
    mav.update(delta, current_wind)
    viewers.update(...)
    sim_time += SIM.ts_simulation
```

### Visualization

`ViewManager` (`viewers/view_manager.py`) is the unified viewer interface. Instantiate with boolean flags for which views to enable: `mav`, `path`, `waypoint`, `map`, `camera`, `sensors`, `geo`, `data`. Underlying viewers use PyQtGraph for 3D animation and 2D data plots.

### Parameters

- `parameters/aerosonde_parameters.py` — Aerosonde UAV physical constants (mass=11 kg, Va0=25 m/s, all aero coefficients)
- `parameters/simulation_parameters.py` — `ts_simulation=0.01s`, `end_time=400s`, plot refresh rates
- `parameters/control_parameters.py` — PID/LQR autopilot gains
- `parameters/sensor_parameters.py` — noise variances for IMU, GPS, barometer
- `parameters/camera_parameters.py` — focal length, pixel noise, gimbal limits

### Student Implementation Points

Files with `##### TODO #####` markers are the student implementation targets for each chapter assignment. Instructions for each chapter are in `mavsim_python/launch_files/chapXX/instructions_chapXX.txt`. Key files to implement:

- Chap 3–4: `models/mav_dynamics.py`, `models/mav_dynamics_control.py`
- Chap 5: `models/trim.py`, `models/compute_models.py`
- Chap 6: `controllers/autopilot.py` (or `autopilot_lqr.py` / `autopilot_tecs.py`), tune `parameters/control_parameters.py`
- Chap 7: `models/mav_dynamics_sensors.py`
- Chap 8: `estimators/observer.py`, `estimators/filters.py`
- Chap 10–11: `planners/path_follower.py`, `planners/path_manager.py`
- Chap 12: `planners/rrt_straight_line.py`, `planners/rrt_dubins.py`
- Chap 13: `models/gimbal.py`, `estimators/geolocation.py`
