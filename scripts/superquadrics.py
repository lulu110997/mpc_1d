#!/usr/bin/python3

import math
import numpy as np
from pyquaternion import Quaternion

class SuperquadricObject:
    """
    Class for defining the properties of a superquadric object
    """

    def __init__(self, a=None, b=None, c=None, eps1=1.0, eps2=1.0, pose=None):  # , obstacle_type=None):
        """
        The isopotential arising from an object expressed as a superquadric model depends on the relative position of the point in question.
        Generally, the isopotential is equal 1 at the contour of the object and increases as the point moves away from the object. Within the object, the isopotenial is less than 1.
        While superquadrics can be used to express any symmetric profile, the simplification of objects to surrounding objects is sufficient in most cases.
        The approximation of a recangular object with known width, height and depth as an ellipsoid is embedded to the class and further shapes can be added as needed.

                                     /  /  xB_1  \ (2/eps2)    /  xB_2  \ (2/eps2) \ (eps2/eps1)     /  xB_3  \ (2/eps1)
            F(a,b,c,eps1,eps2,xB) =  |  |--------|          +  |--------|          |             +   |--------|
                                     \  \    a   /             \    b   /          /                 \    c   /

        Args:
            a        : float  | ellipsoid radius of the object corresponding to x-axis
            b        : float  | ellipsoid radius of the object corresponding to y-axis
            c        : float  | ellipsoid radius of the object corresponding to z-axis
            eps1,eps2: fload  | shape parameters of the superquadric
            pose     : tuple  | absolute pose of the object (xyz position as np.array, qxyz quaternion as a Quaternion)
        """

        self.a = a
        self.b = b
        self.c = c
        self.eps1 = eps1
        self.eps2 = eps2

        if not (pose is None):
            self.x_abs = pose[0]
            if not type(pose[1]) == "pyquaternion.quaternion.Quaternion":
                self.q_abs = Quaternion(pose[1])
            else:
                self.q_abs = pose[1]

    def update_scene(self, x_abs, q_abs=np.array([1, 0, 0, 0]), p_abs=None):
        """
        Setter for updating the object's pose and the point in question

        Args:
            x_abs: np array   | The vector x_abs is the position vector of the object's center with respect to the world frame
            q_abs: Quaternion | The quaternion q_abs is the orientation of the object in question with respect to the world frame
            p_abs: np array   | The vector p_abs is the position vector of the point in question with respect to the world frame
        """
        self.x_abs = x_abs
        if not type(q_abs) == "pyquaternion.quaternion.Quaternion":
            self.q_abs = Quaternion(q_abs)
        else:
            self.q_abs = q_abs
        self.p_abs = p_abs
        return
        qw = q_abs.w
        qx = q_abs.x
        qy = q_abs.y
        qz = q_abs.z

        self.A = np.array([
            [
                qx ** 2 - qy ** 2 - qz ** 2 + qw ** 2,
                2 * (qx * qy + qz * qw),
                2 * (qx * qz - qy * qw)
            ],
            [
                2 * (qx * qy - qz * qw),
                -qx ** 2 + qy ** 2 - qz ** 2 + qw ** 2,
                2 * (qy * qz + qx * qw)
            ],
            [
                2 * (qx * qz + qy * qw),
                2 * (qy * qz - qx * qw),
                -qx ** 2 - qy ** 2 + qz ** 2 + qw ** 2
            ]
        ])

        x_rel = self.x_abs - self.p_abs

        self.H_1 = np.matmul(self.A[0], x_rel.reshape(-1, 1) / self.a)
        self.H_2 = np.matmul(self.A[1], x_rel.reshape(-1, 1) / self.b)
        self.H_3 = np.matmul(self.A[2], x_rel.reshape(-1, 1) / self.c)

        # derivative of H_i with respect to the object's own coordinates
        self.nabla_H_1_obj = np.array([
            [(qw ** 2 + qx ** 2 - qy ** 2 - qz ** 2) / self.a],
            [2 * (qw * qz + qx * qy) / self.a],
            [- 2 * (qw * qy - qx * qz) / self.a],
            [- (2 * qx * (p_abs[0] - x_abs[0]) + 2 * qy * (p_abs[1] - x_abs[1]) + 2 * qz * (
                        p_abs[2] - x_abs[2])) / self.a],
            [(2 * qy * (p_abs[0] - x_abs[0]) - 2 * qx * (p_abs[1] - x_abs[1]) + 2 * qw * (
                        p_abs[2] - x_abs[2])) / self.a],
            [- (- 2 * qz * (p_abs[0] - x_abs[0]) + 2 * qw * (p_abs[1] - x_abs[1]) + 2 * qx * (
                        p_abs[2] - x_abs[2])) / self.a]
        ])

        self.nabla_H_2_obj = np.array([
            [- 2 * (qw * qz - qx * qy) / self.b],
            [(qw ** 2 - qx ** 2 + qy ** 2 - qz ** 2) / self.b],
            [2 * (qw * qx + qy * qz) / self.b],
            [- (2 * qy * (p_abs[0] - x_abs[0]) - 2 * qx * (p_abs[1] - x_abs[1]) + 2 * qw * (
                        p_abs[2] - x_abs[2])) / self.b],
            [- (2 * qx * (p_abs[0] - x_abs[0]) + 2 * qy * (p_abs[1] - x_abs[1]) + 2 * qz * (
                        p_abs[2] - x_abs[2])) / self.b],
            [(2 * qw * (p_abs[0] - x_abs[0]) + 2 * qz * (p_abs[1] - x_abs[1]) - 2 * qy * (
                        p_abs[2] - x_abs[2])) / self.b]
        ])

        self.nabla_H_3_obj = np.array([
            [2 * (qw * qy + qx * qz) / self.c],
            [- 2 * (qw * qx - qy * qz) / self.c],
            [(qw ** 2 - qx ** 2 - qy ** 2 + qz ** 2) / self.c],
            [(- 2 * qz * (p_abs[0] - x_abs[0]) + 2 * qw * (p_abs[1] - x_abs[1]) + 2 * qx * (
                        p_abs[2] - x_abs[2])) / self.c],
            [- (2 * qw * (p_abs[0] - x_abs[0]) + 2 * qz * (p_abs[1] - x_abs[1]) - 2 * qy * (
                        p_abs[2] - x_abs[2])) / self.c],
            [- (2 * qx * (p_abs[0] - x_abs[0]) + 2 * qy * (p_abs[1] - x_abs[1]) + 2 * qz * (
                        p_abs[2] - x_abs[2])) / self.c]
        ])

        # derivative of H_i with respect to the point of interest's own coordinates
        self.nabla_H_1_pt = np.array([
            [- (qw ** 2 + qx ** 2 - qy ** 2 - qz ** 2) / self.a],
            [-  2 * (qw * qz + qx * qy) / self.a],
            [2 * (qw * qy - qx * qz) / self.a],
            [0],
            [0],
            [0]
        ])

        self.nabla_H_2_pt = np.array([
            [2 * (qw * qz - qx * qy) / self.b],
            [- (qw ** 2 - qx ** 2 + qy ** 2 - qz ** 2) / self.b],
            [-  2 * (qw * qx + qy * qz) / self.b],
            [0],
            [0],
            [0]
        ])

        self.nabla_H_3_pt = np.array([
            [-  2 * (qw * qy + qx * qz) / self.c],
            [2 * (qw * qx - qy * qz) / self.c],
            [- (qw ** 2 - qx ** 2 - qy ** 2 + qz ** 2) / self.c],
            [0],
            [0],
            [0]
        ])

        # compute the inside-outside function and its derivatives
        self.__compute_F()
        self.__compute_nabla_F_obj()
        self.__compute_nabla_F_pt()

    def __compute_F(self):
        """
        Calculate the isopotential value of the superquadric object in canoncial form (xB)

                                     /  /  xB_1  \ (2/eps2)    /  xB_2  \ (2/eps2) \ (eps2/eps1)     /  xB_3  \ (2/eps1)
            F(a,b,c,eps1,eps2,xB) =  |  |--------|          +  |--------|          |             +   |--------|
                                     \  \    a   /             \    b   /          /                 \    c   /

        Returns:
            FaxB: float | isopotential value at the given relative position vector (xB)
        """
        self.F = ((self.H_1) ** (2 / self.eps2) + (self.H_2) ** (2 / self.eps2)) ** (self.eps2 / self.eps1) + (
            self.H_3) ** (2 / self.eps1)

    def __compute_nabla_F_obj(self):
        """
        Partial derivative of the inside-outside function at the current object's pose and point in question with respect to the object's own world coordinates
        Returns: np array | gradient of the inside-outside function with respect to the object's own world coordinates
        """

        # copied from matlab > vic_nabla > nabla_F_gen_obj_sub

        subterm = (self.H_1 ** (2 / self.eps2) + self.H_2 ** (2 / self.eps2)) ** (self.eps2 / self.eps1 - 1)

        self.nabla_F_obj = np.array([
            (2 * self.H_3 ** (2 / self.eps1 - 1) * self.nabla_H_3_obj[0]) / self.eps1 + (self.eps2 * subterm * (
                        (2 * self.H_2 ** (2 / self.eps2 - 1) * self.nabla_H_2_obj[0]) / self.eps2 + (
                            2 * self.H_1 ** (2 / self.eps2 - 1) * self.nabla_H_1_obj[0]) / self.eps2)) / self.eps1,
            (2 * self.H_3 ** (2 / self.eps1 - 1) * self.nabla_H_3_obj[1]) / self.eps1 + (self.eps2 * subterm * (
                        (2 * self.H_2 ** (2 / self.eps2 - 1) * self.nabla_H_2_obj[1]) / self.eps2 + (
                            2 * self.H_1 ** (2 / self.eps2 - 1) * self.nabla_H_1_obj[1]) / self.eps2)) / self.eps1,
            (2 * self.H_3 ** (2 / self.eps1 - 1) * self.nabla_H_3_obj[2]) / self.eps1 + (self.eps2 * subterm * (
                        (2 * self.H_2 ** (2 / self.eps2 - 1) * self.nabla_H_2_obj[2]) / self.eps2 + (
                            2 * self.H_1 ** (2 / self.eps2 - 1) * self.nabla_H_1_obj[2]) / self.eps2)) / self.eps1,
            (2 * self.H_3 ** (2 / self.eps1 - 1) * self.nabla_H_3_obj[3]) / self.eps1 + (self.eps2 * subterm * (
                        (2 * self.H_2 ** (2 / self.eps2 - 1) * self.nabla_H_2_obj[3]) / self.eps2 + (
                            2 * self.H_1 ** (2 / self.eps2 - 1) * self.nabla_H_1_obj[3]) / self.eps2)) / self.eps1,
            (2 * self.H_3 ** (2 / self.eps1 - 1) * self.nabla_H_3_obj[4]) / self.eps1 + (self.eps2 * subterm * (
                        (2 * self.H_2 ** (2 / self.eps2 - 1) * self.nabla_H_2_obj[4]) / self.eps2 + (
                            2 * self.H_1 ** (2 / self.eps2 - 1) * self.nabla_H_1_obj[4]) / self.eps2)) / self.eps1,
            (2 * self.H_3 ** (2 / self.eps1 - 1) * self.nabla_H_3_obj[5]) / self.eps1 + (self.eps2 * subterm * (
                        (2 * self.H_2 ** (2 / self.eps2 - 1) * self.nabla_H_2_obj[5]) / self.eps2 + (
                            2 * self.H_1 ** (2 / self.eps2 - 1) * self.nabla_H_1_obj[5]) / self.eps2)) / self.eps1,
        ])

    def __compute_nabla_F_pt(self):
        """
        Partial derivative of the isopotential with respect to the relative position vector xB at the given point xB
        Returns: np array | gradient of the isopotential at the given point
        """

        # copied from matlab > vic_nabla > nabla_F_gen_pt_sub

        subterm = (self.H_1 ** (2 / self.eps2) + self.H_2 ** (2 / self.eps2)) ** (self.eps2 / self.eps1 - 1)

        self.nabla_F_pt = np.array([
            (2 * self.H_3 ** (2 / self.eps1 - 1) * self.nabla_H_3_pt[0]) / self.eps1 + (self.eps2 * subterm * (
                        (2 * self.H_2 ** (2 / self.eps2 - 1) * self.nabla_H_2_pt[0]) / self.eps2 + (
                            2 * self.H_1 ** (2 / self.eps2 - 1) * self.nabla_H_1_pt[0]) / self.eps2)) / self.eps1,
            (2 * self.H_3 ** (2 / self.eps1 - 1) * self.nabla_H_3_pt[1]) / self.eps1 + (self.eps2 * subterm * (
                        (2 * self.H_2 ** (2 / self.eps2 - 1) * self.nabla_H_2_pt[1]) / self.eps2 + (
                            2 * self.H_1 ** (2 / self.eps2 - 1) * self.nabla_H_1_pt[1]) / self.eps2)) / self.eps1,
            (2 * self.H_3 ** (2 / self.eps1 - 1) * self.nabla_H_3_pt[2]) / self.eps1 + (self.eps2 * subterm * (
                        (2 * self.H_2 ** (2 / self.eps2 - 1) * self.nabla_H_2_pt[2]) / self.eps2 + (
                            2 * self.H_1 ** (2 / self.eps2 - 1) * self.nabla_H_1_pt[2]) / self.eps2)) / self.eps1,
            [0],
            [0],
            [0],
        ])

    def get_eps1(self):
        return self.eps1

    def get_eps2(self):
        return self.eps2

    def get_abc(self):
        return self.a, self.b, self.c

    def get_F(self):
        """
        Getter for the value of the inside-outside function at the given point computed with function "__compute_F"
        Returns: float | value of the inside-outside function at the given point
        """
        return self.F

    def get_nabla_F_obj(self):
        """
        Getter for the isopotential's partial derivatives at the given point xB computed with function "__compute_nabla_F"
        Returns: np array | gradient of the isopotential at the given position vector (xB)
        """
        return self.nabla_F_obj

    def get_nabla_F_pt(self):
        """
        Getter for the isopotential's partial derivatives at the given point xB computed with function "__compute_nabla_F"
        Returns: np array | gradient of the isopotential at the given position vector (xB)
        """
        return self.nabla_F_pt

    def get_pose(self):
        """
        Getter for the object's pose
        Returns: tuple(np.array, Quaternion) | (xyz position, wxyz quaternion)
        """
        return (self.x_abs, self.q_abs)

    def plot_sq(self, ax, colour, plot_type='3D', alpha=0.2):
        """
        Plot
        """

        X, Y, Z = self.get_mesh()

        if plot_type == '2D':
            # https://stackoverflow.com/questions/43482191/matplotlib-axes-plot-vs-pyplot-plot
            return ax.plot(X, Z, alpha=alpha, linewidth=.3, color=colour)
        elif plot_type == '3D':
            return ax.plot_surface(X, Y, Z, alpha=alpha, color=colour)
        elif plot_type == 'animate':
            return X, Y, Z
        else:
            raise 'Undefined plot type. Should be either animeate, 2D or 3D'

    def get_mesh(self, pos=None, quat=None):
        """
        Get expansion dimensions of object based on given pose
        """
        if pos is None:
            pos_x = self.x_abs[0]
            pos_y = self.x_abs[1]
            pos_z = self.x_abs[2]

            qw = self.q_abs.w
            qx = self.q_abs.x
            qy = self.q_abs.y
            qz = self.q_abs.z

        else:
            pos_x = pos[0]
            pos_y = pos[1]
            pos_z = pos[2]

            qw = quat[0]
            qx = quat[1]
            qy = quat[2]
            qz = quat[3]

        scos = lambda theta, eps: math.copysign(abs(math.cos(theta)) ** eps, math.cos(theta))
        ssin = lambda theta, eps: math.copysign(abs(math.sin(theta)) ** eps, math.sin(theta))

        WN = 20  # 40
        NN = 40  # 80

        w_array = np.linspace(-math.pi, math.pi, WN)
        n_array = np.linspace(-math.pi / 2, math.pi / 2, NN)
        w_mesh, n_mesh = np.meshgrid(w_array, n_array)

        X = np.zeros(w_mesh.shape)
        Y = np.zeros(w_mesh.shape)
        Z = np.zeros(w_mesh.shape)

        for (i, j), __ in np.ndenumerate(w_mesh):
            w = w_mesh[i, j]
            n = n_mesh[i, j]

            X[i, j] = self.a * scos(n, self.eps1) * scos(w, self.eps2)
            Y[i, j] = self.b * scos(n, self.eps1) * ssin(w, self.eps2)
            Z[i, j] = self.c * ssin(n, self.eps1)
        R = np.zeros((3, 3))

        # TODO: Check if R is self.A
        R[0, 0] = 1 - 2 * qy * qy - 2 * qz * qz
        R[0, 1] = 2 * qx * qy - 2 * qz * qw
        R[0, 2] = 2 * qx * qz + 2 * qy * qw

        R[1, 0] = 2 * qx * qy + 2 * qz * qw
        R[1, 1] = 1 - 2 * qx * qx - 2 * qz * qz
        R[1, 2] = 2 * qy * qz - 2 * qx * qw

        R[2, 0] = 2 * qx * qz - 2 * qy * qw
        R[2, 1] = 2 * qy * qz + 2 * qx * qw
        R[2, 2] = 1 - 2 * qx * qx - 2 * qy * qy

        T = np.zeros((4, 4))
        T[0:3, 0:3] = R
        T[0:3, 3] = np.array([pos_x, pos_y, pos_z]).transpose()
        T[3, :] = np.array([[0, 0, 0, 1]])

        # TODO: Surely these operations can be vectorised for optimisation
        # xyz_matrix = np.stack(np.copy(X), np.copy(Y), np.copy(Z))
        for (i, j), __ in np.ndenumerate(X):
            xp = np.array([[X[i, j], Y[i, j], Z[i, j], 1]]).transpose()
            xp = T @ xp
            X[i, j] = xp[0]
            Y[i, j] = xp[1]
            Z[i, j] = xp[2]

        return X, Y, Z
