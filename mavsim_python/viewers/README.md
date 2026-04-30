# viewers/

PyQtGraph-based 3D animation and 2D data-plot visualization for the mavsim simulation.

## Setup

Always run from the **repo root** with the virtual environment active:

```bash
source /Users/marciano/Repos/mavsim/venv/bin/activate
```

Each launch file in `launch_files/chapXX/` already does the `sys.path` insertion needed for
`viewers.*` imports. When writing a standalone script outside `launch_files/`, add this near the top:

```python
import os, sys
from pathlib import Path
sys.path.insert(0, os.fspath(Path(__file__).parent / "mavsim_python"))  # adjust depth as needed
```

---

## Quick start — SpaceCraftViewer

`spacecraft_viewer.py` is the simplest self-contained viewer. Run this example from the repo root:

```python
# example_spacecraft.py  (place this file at repo root)
import os, sys
from pathlib import Path
sys.path.insert(0, os.fspath(Path(__file__).parent / "mavsim_python"))

import pyqtgraph as pg
from message_types.msg_state import MsgState
from viewers.spacecraft_viewer import SpaceCraftViewer
import time

state = MsgState()
state.altitude = 100.0

app = pg.QtWidgets.QApplication.instance() or pg.QtWidgets.QApplication(sys.argv)
viewer = SpaceCraftViewer(app=app)

sim_time, dt, end_time = 0.0, 0.01, 10.0
while sim_time < end_time:
    state.north += 2.0 * dt
    state.east  += 1.0 * dt
    state.psi   += 0.2 * dt
    viewer.update(state)
    viewer.process_app()
    time.sleep(dt)
    sim_time += dt
```

```bash
python example_spacecraft.py
```

`SpaceCraftViewer` reads six fields from the state object: `north`, `east`, `altitude`,
`phi`, `theta`, `psi`. All are provided by `MsgState` with sensible defaults.

---

## ViewManager — unified interface for simulation chapters

All chapter launch files (`chap02`–`chap13`) use `ViewManager` instead of instantiating
individual viewers directly. It owns the single `QApplication` instance and wires up whichever
views are needed via boolean flags.

### Construction

```python
from viewers.view_manager import ViewManager

viewers = ViewManager(
    mav=True,       # 3-D MAV animation (chapters 2–9)
    path=True,      # MAV + current path segment (chapter 10)
    waypoint=True,  # MAV + waypoints (chapter 11)
    map=True,       # MAV + waypoints + world map (chapter 12)
    camera=True,    # MAV + camera FOV + projected target (chapter 13)
    planning=True,  # separate RRT tree viewer (chapter 12)
    sensors=True,   # sensor time-series plots (chapter 7)
    data=True,      # state/control time-series plots (chapters 6–13)
    geo=True,       # target geolocation error plots (chapter 13)
    video=False,    # screen-capture to .mp4
    video_name='out.mp4',
)
```

Flags are mutually exclusive for the 3-D view (`camera` > `map` > `waypoint` > `path` > `mav`);
the 2-D plot viewers (`data`, `sensors`, `geo`) can coexist with any 3-D view.

### Update (call every sim step)

```python
viewers.update(
    sim_time,
    true_state=mav.true_state,
    estimated_state=estimated_state,   # None if no estimator
    commanded_state=commanded_state,   # None if no autopilot
    delta=delta,                       # MsgDelta, for data_view
    measurements=measurements,         # MsgSensors, for sensor_view
    path=path,                         # MsgPath, for path/waypoint/map views
    waypoints=waypoints,               # MsgWaypoints
    map=world_map,                     # MsgWorldMap
    target=target,                     # for camera view (chapter 13)
    camera=camera,                     # camera model object (chapter 13)
    estimated_target=estimated_target, # for geo view (chapter 13)
)
```

Pass `None` for any argument that is not used by the active flags — `ViewManager` checks flags
before accessing the argument.

### Chapter 12 — planning tree update

```python
viewers.update_planning_tree(
    waypoints=waypoints,
    map=world_map,
    waypoints_not_smooth=waypoints_not_smooth,
    tree=tree,
    radius=radius,
)
```

### Teardown

```python
viewers.close(dataplot_name='data.png', sensorplot_name='sensors.png')
```

`save_plots=True` must be set in the constructor for image saving to have any effect.

---

## Viewer reference

| Class | File | Used by | What it shows |
|---|---|---|---|
| `SpaceCraftViewer` | `spacecraft_viewer.py` | Appendix C | 3-D spacecraft box (standalone, no `ViewManager`) |
| `MavViewer` | `mav_viewer.py` | chap 2–5 | 3-D MAV STL model, follows aircraft position |
| `MavAndPathViewer` | `mav_path_viewer.py` | chap 10 | MAV + current path segment (line or arc) |
| `MAVAndWaypointViewer` | `mav_waypoint_viewer.py` | chap 11 | MAV + waypoint list |
| `MAVWorldViewer` | `mav_world_viewer.py` | chap 12 | MAV + waypoints + building map |
| `MAVWorldCameraViewer` | `mav_world_camera_viewer.py` | chap 13 | MAV + world + camera FOV cone |
| `PlannerViewer` | `planner_viewer.py` | chap 12 | Separate window: RRT search tree + map |
| `DataViewer` | `data_viewer.py` | chap 6–13 | 5-row time-series: position, airspeed, attitude, rates, control surfaces |
| `SensorViewer` | `sensor_viewer.py` | chap 7–8 | 4-row time-series: gyros, accelerometers, GPS, pressure |
| `GeolocationViewer` | `geolocation_viewer.py` | chap 13 | True vs. estimated target NED error plots |
| `CameraViewer` | `camera_viewer.py` | chap 13 | Matplotlib window: projected target bounding box in pixel coordinates |

### Draw helpers (internal)

These are instantiated by the viewer classes above — you do not use them directly:

| File | Purpose |
|---|---|
| `draw_mav_stl.py` | Loads `aircraft1.stl` and renders the MAV mesh |
| `draw_mav.py` | Polygon-primitive MAV (fallback, not used by default) |
| `draw_spacecraft.py` | Box mesh for the spacecraft viewer |
| `draw_path.py` | Line/arc path segment rendering |
| `draw_waypoints.py` | Waypoint sphere markers |
| `draw_map.py` | Building-block world map |
| `draw_camera_fov.py` | Camera field-of-view cone |
| `draw_target.py` | Ground target box |
| `video_writer.py` | Screen-region capture → MP4 via mss + cv2 |

---

## Coordinate frame note

All position inputs use **NED (North-East-Down)**. Altitude is positive upward
(`altitude = -pd`). The viewers internally convert NED → ENU for PyQtGraph rendering.

---

## Keyboard / window controls

- **Esc** or **Command-Q** — close the simulation window and end the loop
- Mouse drag — rotate the 3-D view
- Mouse wheel — zoom
- Right-click drag — pan
