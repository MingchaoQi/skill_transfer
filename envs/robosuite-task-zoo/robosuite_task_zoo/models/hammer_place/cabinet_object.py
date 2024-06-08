import numpy as np
from robosuite.models.objects import MujocoXMLObject, CompositeObject
from robosuite.utils.mjcf_utils import xml_path_completion, array_to_string, find_elements, CustomMaterial, add_to_dict, RED, GREEN, BLUE
from robosuite.models.objects import BoxObject
import robosuite.utils.transform_utils as T


import pathlib
absolute_path = pathlib.Path(__file__).parent.absolute()


class CabinetObject(MujocoXMLObject):
    def __init__(
            self,
            name,
            joints=None):

        super().__init__(str(absolute_path) + "/" + "cabinet.xml",
                         name=name, joints=None, obj_type="all", duplicate_collision_geoms=True)

        self.cabinet_base = self.naming_prefix + "base"
        self.cabinet_drawer_link = self.naming_prefix + "drawer_link"
        self.cabint_joint = self.naming_prefix + "goal_slidey"

    # @property
    # def bottom_offset(self):
    #     return np.array([0, 0, -2 * self.height])
    #
    # @property
    # def top_offset(self):
    #     return np.array([0, 0, 2 * self.height])
    #
    # @property
    # def horizontal_radius(self):
    #     return self.length * np.sqrt(2)

    def _set_cabinet_damping(self, damping):
        """
        Helper function to override the cabinet friction directly in the XML

        Args:
            damping (float): damping parameter to override the ones specified in the XML
        """
        hinge = find_elements(root=self.worldbody, tags="joint", attribs={"name": self.cabint_joint}, return_first=True)
        hinge.set("damping", array_to_string(np.array([damping])))

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of door handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"slide_handle": self.naming_prefix + "slide_handle"})
        dic.update({"base_center": self.naming_prefix + "base_center"})
        return dic
