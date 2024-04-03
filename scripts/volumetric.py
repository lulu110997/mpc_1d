import numpy as np
import superquadrics as sq
import sympy as sym

class VolumetricDistance():
    """
    Estimates the distance between two superquadric objects' contours.
    Based on two initially given instances of superquadric objects, the class includes functions
    to compute all attributes concerning the distance between the two objects.
    The class assumes one stationary object and one moving object (e.g., obstacle and end-effector) and
    can be extended to two moving objects in the future.
    """

    def __init__(self, SuperquadricObject_Dynamic, SuperquadricObject_Static, distance_type=None):
        """
        For two volumetric objects, the distance is approximated using the approach for rigid body radial euclidean distance defined by [Badawy2007]

        The rbr Euclidean distance takes into consideration the possible difference between the obstacle and
        manoeuvring object shapes, sizes and orientations as the inside-outside function, F, is calculated for each object.

        Args:
            SuperquadricObject_Instances: SuperquadricObject | Objects expressed as superquadric models, Instance1 is the moving object and Instance2 is the stationary object
        """

        self.sqObj_dyn = SuperquadricObject_Dynamic  # dynamic object, referred to as end-effector
        self.sqObj_stat = SuperquadricObject_Static  # static object, referred to as obstacle
        self.distance_type = distance_type
        self.h, self.h_dot = self.distance_derivative()

    def update_scene(self, x_dyn_abs, q_dyn_abs, x_stat_abs, q_stat_abs):
        self.r_dyn_stat = np.linalg.norm(x_dyn_abs - x_stat_abs)

        self.nabla_r_dyn_stat = np.array([
            [(x_dyn_abs[0] - x_stat_abs[0]) / self.r_dyn_stat],
            [(x_dyn_abs[1] - x_stat_abs[1]) / self.r_dyn_stat],
            [(x_dyn_abs[2] - x_stat_abs[2]) / self.r_dyn_stat],
            [0],
            [0],
            [0],
        ])

        self.sqObj_dyn.update_scene(x_dyn_abs, q_dyn_abs, x_stat_abs)

        self.sqObj_stat.update_scene(x_stat_abs, q_stat_abs, x_dyn_abs)

        if self.distance_type == "outside":
            self.__compute_outside_distance()
            self.__compute_nabla_outside_distance()
        elif self.distance_type == "inside":
            self.__compute_inside_distance()
            self.__compute_nabla_inside_distance()
        elif self.distance_type == "attractive":
            self.__compute_attractive_distance()
            self.__compute_nabla_attractive_distance()

    def distance_derivative(self):
        """
        Symbolically compute distance and its derivative
        Returns: tuple | distance_function and distance_derivative_function
        """
        x_ee = sym.symbols('x_ee:3')
        x_obs = sym.symbols('x_obs:3')
        a_ee = sym.symbols('a_ee:3')
        a_obs = sym.symbols('a_obs:3')
        eps_ee = sym.symbols('eps_ee:2')
        eps_obs = sym.symbols('eps_obs:2')
        x_ee = sym.Matrix([[x_ee[0], x_ee[1], x_ee[2]]])
        x_obs = sym.Matrix([[x_obs[0], x_obs[1], x_obs[2]]])
        a_ee = sym.Matrix([[a_ee[0], a_ee[1], a_ee[2]]])
        a_obs = sym.Matrix([[a_obs[0], a_obs[1], a_obs[2]]])
        eps_ee = sym.Matrix([[eps_ee[0], eps_ee[1]]])
        eps_obs = sym.Matrix([[eps_obs[0], eps_obs[1]]])

        # Implicit equation of a superquadric
        Fee = (((x_obs[0]-x_ee[0])/a_ee[0])**(2/eps_ee[1]) + ((x_obs[1]-x_ee[1])/a_ee[1])**(2/eps_ee[1]))**(eps_ee[1]/eps_ee[0]) + ((x_obs[2]-x_ee[2])/a_ee[2])**(2/eps_ee[0])
        Fobs = (((x_ee[0]-x_obs[0])/a_obs[0])**(2/eps_obs[1]) + ((x_ee[1]-x_obs[1])/a_obs[1])**(2/eps_obs[1]))**(eps_obs[1]/eps_obs[0]) + ((x_ee[2]-x_obs[2])/a_obs[2])**(2/eps_obs[0])


        # Rigid body radial distance
        hx = sym.sqrt((x_ee[0] - x_obs[0]) ** 2 +
                      (x_ee[1] - x_obs[1]) ** 2 +
                      (x_ee[2] - x_obs[2]) ** 2) * (1 - Fee**(-0.5) - Fobs**(-0.5))

        hx_dot = sym.diff(hx, x_ee)

        hx = sym.lambdify([x_ee, x_obs, a_ee, a_obs, eps_ee, eps_obs], expr=hx)
        hx_dot = sym.lambdify([x_ee, x_obs, a_ee, a_obs, eps_ee, eps_obs], expr=hx_dot)
        return hx, hx_dot

    def get_h(self):
        abc_ee = self.sqObj_dyn.get_abc()
        abc_obs = self.sqObj_stat.get_abc()
        eps_ee = self.sqObj_dyn.get_eps1(), self.sqObj_dyn.get_eps2()
        eps_obs = self.sqObj_stat.get_eps1(), self.sqObj_stat.get_eps2()
        return self.h(self.sqObj_dyn.get_pose()[0], self.sqObj_stat.get_pose()[0], abc_ee, abc_obs, eps_ee, eps_obs)

    def get_h_dot(self):
        abc_ee = self.sqObj_dyn.get_abc()
        abc_obs = self.sqObj_stat.get_abc()
        eps_ee = self.sqObj_dyn.get_eps1(), self.sqObj_dyn.get_eps2()
        eps_obs = self.sqObj_stat.get_eps1(), self.sqObj_stat.get_eps2()
        return self.h_dot(self.sqObj_dyn.get_pose()[0], self.sqObj_stat.get_pose()[0], abc_ee, abc_obs, eps_ee, eps_obs)

    # def __compute_outside_distance(self):
    #     """
    #     The rigid body radial euclidean distance for two objects outside of each other is calculated as follows:
    #
    #     d_12 = r_12 - r_1 - r_2 (2 represents the obstacle object)
    #
    #     ==>
    #
    #     d_12 = | r_12 | * (1 - F_1 ^ (-eps1_1/2) - F_2 ^ (-eps1_2/2))
    #
    #     with r_12 being the translational distance between the i-th and j-th objects' centres
    #     and F_i being the inside-outside function of the i-th object with the j-th object's centre used as point of interest.
    #     """
    #     self.outside_distance = self.r_dyn_stat * (
    #                 1 - self.sqObj_dyn.get_F() ** (-self.sqObj_dyn.get_eps1() / 2) - self.sqObj_stat.get_F() ** (
    #                     -self.sqObj_stat.get_eps1() / 2))
    #
    # def __compute_nabla_outside_distance(self):
    #     """
    #     Compute partial derivative of rigid body radial euclidean outside distance
    #     """
    #     self.nabla_outside_distance = (
    #             (self.nabla_r_dyn_stat * (1 - self.sqObj_dyn.get_F() ** (
    #                         -self.sqObj_dyn.get_eps1() / 2) - self.sqObj_stat.get_F() ** (
    #                                                   -self.sqObj_stat.get_eps1() / 2)))
    #             + self.r_dyn_stat * ((self.sqObj_dyn.get_eps1() / 2) * self.sqObj_dyn.get_F() ** (
    #                 -self.sqObj_dyn.get_eps1() / 2 - 1) * self.sqObj_dyn.get_nabla_F_obj()
    #                                  + (self.sqObj_stat.get_eps1() / 2) * self.sqObj_stat.get_F() ** (
    #                                              -self.sqObj_stat.get_eps1() / 2 - 1) * self.sqObj_stat.get_nabla_F_pt())
    #     )
    #
    # def __compute_inside_distance(self):
    #     """
    #     The rigid body radial euclidean distance between two objects, while one object (1) is inside the other's object (2) is calculated as follows:
    #
    #     d_12 = r_2 - r_1 - r_12 (2 represents the workspace object)
    #
    #     ==>
    #
    #     d_12 = | r_12 | * (F_2 ^ (-eps1_2/2) - F_1 ^ (-eps1_1/2) - 1)
    #
    #     with r_12 being the translational distance between the i-th and j-th objects' centres
    #     and F_i being the inside-outside function of the i-th object with the j-th object's centre used as point of interest.
    #     """
    #     self.inside_distance = self.r_dyn_stat * (
    #                 self.sqObj_stat.get_F() ** (-self.sqObj_stat.get_eps1() / 2) - self.sqObj_dyn.get_F() ** (
    #                     -self.sqObj_dyn.get_eps1() / 2) - 1)
    #
    # def __compute_nabla_inside_distance(self):
    #     """
    #     Compute partial derivative of rigid body radial euclidean inside distance
    #     """
    #     self.nabla_inside_distance = (
    #             (self.nabla_r_dyn_stat * (
    #                         self.sqObj_stat.get_F() ** (-self.sqObj_stat.get_eps1() / 2) - self.sqObj_dyn.get_F() ** (
    #                             -self.sqObj_dyn.get_eps1() / 2) - 1))
    #             + self.r_dyn_stat * ((- self.sqObj_stat.get_eps1() / 2) * self.sqObj_stat.get_F() ** (
    #                 -self.sqObj_stat.get_eps1() / 2 - 1) * self.sqObj_stat.get_nabla_F_pt()
    #                                  + (self.sqObj_dyn.get_eps1() / 2) * self.sqObj_dyn.get_F() ** (
    #                                              -self.sqObj_dyn.get_eps1() / 2 - 1) * self.sqObj_dyn.get_nabla_F_obj())
    #     )
    #
    # def __compute_attractive_distance(self):
    #     """
    #     The rigid body radial euclidean distance for attraction of object 1 to object 2 is expressed as the distance to the object 2 opposite side and is calculated as follows :
    #
    #     d_12 = r_12 - r_1 + r_2 (2 represents the attractive goal object)
    #
    #     ==>
    #
    #     d_12 = | r_12 | * (1 - F_1 ^ (-eps1_1/2) + F_2 ^ (-eps1_2/2))
    #
    #     with r_12 being the translational distance between the i-th and j-th objects' centres
    #     and F_i being the inside-outside function of the i-th object with the j-th object's centre used as point of interest.
    #     """
    #     self.attractive_distance = self.r_dyn_stat * (
    #                 1 - self.sqObj_dyn.get_F() ** (-self.sqObj_dyn.get_eps1() / 2) + self.sqObj_stat.get_F() ** (
    #                     -self.sqObj_stat.get_eps1() / 2))
    #
    # def __compute_nabla_attractive_distance(self):
    #     """
    #     Compute partial derivative of rigid body radial euclidean attractive distance
    #     """
    #     self.nabla_attractive_distance = (
    #             (self.nabla_r_dyn_stat * (1 - self.sqObj_dyn.get_F() ** (
    #                         -self.sqObj_dyn.get_eps1() / 2) + self.sqObj_stat.get_F() ** (
    #                                                   -self.sqObj_stat.get_eps1() / 2)))
    #             + self.r_dyn_stat * ((self.sqObj_dyn.get_eps1() / 2) * self.sqObj_dyn.get_F() ** (
    #                 -self.sqObj_dyn.get_eps1() / 2 - 1) * self.sqObj_dyn.get_nabla_F_obj()
    #                                  - (self.sqObj_stat.get_eps1() / 2) * self.sqObj_stat.get_F() ** (
    #                                              -self.sqObj_stat.get_eps1() / 2 - 1) * self.sqObj_stat.get_nabla_F_pt())
    #     )
    #
    # def _get_outside_distance(self):
    #     """
    #     Getter for the rbr-Euclidean Distance value between the contour of the given objects
    #     at the given absolute position vectors computed with function "__compute_rbrEdistance"
    #     Returns: float | rbr-Euclidean Distance value between the contour of the given objects
    #                      at the given absolute position vectors (pos_dyn, pos_stat)
    #     """
    #     return self.outside_distance
    #
    # def _get_nabla_outside_distance(self):
    #     """
    #     Getter for the rbr-Euclidean Distance's partial derivatives at the given absolute position
    #     vectors computed with function "__compute_nabla_rbrEdistance"
    #     Returns: np array | gradient of the rbr-Euclidean Distance at the given absolute position vectors (pos_dyn, pos_stat)
    #     """
    #     return self.nabla_outside_distance
    #
    # def _get_inside_distance(self):
    #     """
    #     Getter for the rbr-Euclidean Distance value between the contour of the given objects
    #     at the given absolute position vectors computed with function "__compute_rbrEdistance"
    #     Returns: float | rbr-Euclidean Distance value between the contour of the given objects
    #                      at the given absolute position vectors (pos_dyn, pos_stat)
    #     """
    #     return self.inside_distance
    #
    # def _get_nabla_inside_distance(self):
    #     """
    #     Getter for the rbr-Euclidean Distance's partial derivatives at the given absolute position
    #     vectors computed with function "__compute_nabla_rbrEdistance"
    #     Returns: np array | gradient of the rbr-Euclidean Distance at the given absolute position vectors (pos_dyn, pos_stat)
    #     """
    #     return self.nabla_inside_distance
    #
    # def _get_attractive_distance(self):
    #     """
    #     Getter for the rbr-Euclidean Distance value between the contour of the given objects
    #     at the given absolute position vectors computed with function "__compute_rbrEdistance"
    #     Returns: float | rbr-Euclidean Distance value between the contour of the given objects
    #                      at the given absolute position vectors (pos_dyn, pos_stat)
    #     """
    #     return self.attractive_distance
    #
    # def _get_nabla_attractive_distance(self):
    #     """
    #     Getter for the rbr-Euclidean Distance's partial derivatives at the given absolute position
    #     vectors computed with function "__compute_nabla_rbrEdistance"
    #     Returns: np array | gradient of the rbr-Euclidean Distance at the given absolute position vectors (pos_dyn, pos_stat)
    #     """
    #     return self.nabla_attractive_distance
    #
    # def get_distance(self):
    #     """
    #     Getter for distance.
    #
    #     Returns: float | inside distance if workspace boundary, else obtains outside distance for normal obstacles
    #     """
    #     if self.distance_type == "outside":
    #         return self._get_outside_distance()
    #     elif self.distance_type == "inside":
    #         return self._get_inside_distance()
    #     elif self.distance_type == "attractive":
    #         return self._get_attractive_distance()
    #     else:
    #         print(self.distance_type)
    #         raise "sq obstacle type is incorrect"
    #
    # def get_nabla_distance(self):
    #     """
    #     Getter for distance.
    #
    #     Returns: float | inside distance if workspace boundary, else obtains outside distance for normal obstacles
    #     """
    #     if self.distance_type == "outside":
    #         return self._get_nabla_outside_distance()
    #     elif self.distance_type == "inside":
    #         return self._get_nabla_inside_distance()
    #     elif self.distance_type == "attractive":
    #         return self._get_nabla_attractive_distance()
    #     else:
    #         print(self.distance_type)
    #         raise "sq distance type is incorrect"
    #
    # def get_dist_centres(self):
    #     return self.r_dyn_stat
    #
    # def get_nabla_dist_centres(self):
    #     return self.nabla_r_dyn_stat