"""
mavsim_python: drawing tools
    - Beard & McLain, PUP, 2012
    - Update history:
        1/13/2021 - TWM
        7/13/2023 - RWB
        1/16/2024 - RWB
"""
import numpy as np
import pyqtgraph.opengl as gl
from tools.rotations import euler_to_rotation
from tools.drawing import rotate_points, translate_points, points_to_mesh


class DrawMav:
    def __init__(self, state, window, scale=10):
        """
        Draw the MAV.

        The input to this function is a (message) class with properties that define the state.
        The following properties are assumed:
            state.north  # north position
            state.east  # east position
            state.altitude   # altitude
            state.phi  # roll angle
            state.theta  # pitch angle
            state.psi  # yaw angle
        """
        self.unit_length = scale
        sc_position = np.array([[state.north], [state.east], [-state.altitude]])  # NED coordinates
        # attitude of mav as a rotation matrix R from body to inertial
        R_bi = euler_to_rotation(state.phi, state.theta, state.psi)
        # convert North-East Down to East-North-Up for rendering
        self.R_ned = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        # get points that define the non-rotated, non-translated MAV and the mesh colors
        self.sc_points, self.sc_index, self.sc_meshColors = self.get_sc_points()
        self.sc_body = self.add_object(
            self.sc_points,
            self.sc_index,
            self.sc_meshColors,
            R_bi,
            sc_position)
        window.addItem(self.sc_body)  # add MAV to plot     

    def update(self, state):
        sc_position = np.array([[state.north], [state.east], [-state.altitude]])  # NED coordinates
        # attitude of mav as a rotation matrix R from body to inertial
        R_bi = euler_to_rotation(state.phi, state.theta, state.psi)
        self.sc_body = self.update_object(
            self.sc_body,
            self.sc_points,
            self.sc_index,
            self.sc_meshColors,
            R_bi,
            sc_position)

    def add_object(self, points, index, colors, R, position):
        rotated_points = rotate_points(points, R)
        translated_points = translate_points(rotated_points, position)
        translated_points = self.R_ned @ translated_points
        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = points_to_mesh(translated_points, index)
        object = gl.GLMeshItem(
            vertexes=mesh,  # defines the triangular mesh (Nx3x3)
            vertexColors=colors,  # defines mesh colors (Nx1)
            drawEdges=True,  # draw edges between mesh elements
            smooth=False,  # speeds up rendering
            computeNormals=False)  # speeds up rendering
        return object

    def update_object(self, object, points, index, colors, R, position):
        rotated_points = rotate_points(points, R)
        translated_points = translate_points(rotated_points, position)
        translated_points = self.R_ned @ translated_points
        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = points_to_mesh(translated_points, index)
        object.setMeshData(vertexes=mesh, vertexColors=colors)
        return object

    def get_sc_points(self):
        """"
            Points that define the MAV, and the colors of the triangular mesh
            Define the points on the MAV following information in Appendix C.3
        """
        fuse_h = 10
        fuse_w = 10
        fuse_l1 = 16
        fuse_l2 = 8
        fuse_l3 = 31
        wing_l = 13
        wing_w = 42
        tail_h = 10
        tailwing_l = 8
        tailwing_w = 22
        scaling_factor = 5 / (fuse_l1 + fuse_l3)  # scale the points so that the MAV is a reasonable size in the viewer
        # points are in XYZ coordinates
        #   define the points on the MAV according to Appendix C.3
        points = self.unit_length * np.array([
            [fuse_l1, 0, 0],
            [fuse_l2, fuse_w/2, -fuse_h/2],
            [fuse_l2, -fuse_w/2, -fuse_h/2],
            [fuse_l2, -fuse_w/2, fuse_h/2],
            [fuse_l2, fuse_w/2, fuse_h/2],
            [-fuse_l3, 0, 0],
            [0, wing_w/2, 0],
            [-wing_l, wing_w/2, 0],
            [-wing_l, -wing_w/2, 0],
            [0, -wing_w/2, 0],
            [tailwing_l - fuse_l3, tailwing_w/2, 0],
            [-fuse_l3, tailwing_w/2, 0],
            [-fuse_l3, -tailwing_w/2, 0],
            [tailwing_l - fuse_l3, -tailwing_w/2, 0],
            [tailwing_l - fuse_l3, 0, 0],
            [-fuse_l3, 0, -tail_h]
            ]).T * scaling_factor
        # point index that defines the mesh
        index = np.array([
            [0,1,2], #nose top
            [0,2,3], #nose port
            [0,3,4], #nose bottom
            [0,1,4], #nose starboard
            [1,2,5], #fuse top
            [2,3,5], #fuse port
            [3,4,5], #fuse bottom
            [1,4,5], #fuse starboard
            [14,5,15], #tail
            [6,7,9], # wing 1
            [7,8,9], #wing 2
            [10,11,13], #tailwing 1
            [11,12,13], #tailwing 2
            ])
        #   define the colors for each face of triangular mesh
        red = np.array([1., 0., 0., 1])
        green = np.array([0., 1., 0., 1])
        blue = np.array([0., 0., 1., 1])
        yellow = np.array([1., 1., 0., 1])
        meshColors = np.empty((13, 3, 4), dtype=np.float32)
        meshColors[0] = red  # nose top
        meshColors[1] = red  # nose port
        meshColors[2] = red  # nose bottom
        meshColors[3] = red  # nose star
        meshColors[4] = yellow  # fuse top
        meshColors[5] = yellow  # fuse port
        meshColors[6] = yellow  # fuse bottom
        meshColors[7] = yellow  # fuse starboard
        meshColors[8] = blue    # tail
        meshColors[9] = green   # wing 1
        meshColors[10] = green  # wing 2
        meshColors[11] = blue   # tailwing 1
        meshColors[12] = blue   # tailwing 2
        return points, index, meshColors

