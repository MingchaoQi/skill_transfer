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

    @property
    def bottom_offset(self):
        return np.array([0, 0, -2 * self.height])

    @property
    def top_offset(self):
        return np.array([0, 0, 2 * self.height])
        
    @property
    def horizontal_radius(self):
        return self.length * np.sqrt(2)

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
        return dic
