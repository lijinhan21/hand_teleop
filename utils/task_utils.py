import io
import os
import xml.dom.minidom
import xml.etree.ElementTree as ET
from utils.mjcf_utils import find_elements, _element_filter, sort_elements, add_prefix, recolor_collision_geoms, add_material, array_to_string
from utils.transform_utils import mat2quat, euler2mat
from utils.xml_utils import MujocoXML
from utils.objects_utils import MujocoObject, MujocoXMLObject
import numpy as np

class Task(MujocoXML):
    """
    Task class to represent a task in Mujoco.
    """

    def __init__(self, robot_xml, arena_xml, object_xmls=[], object_names=[]):
        super(Task, self).__init__('./assets/base.xml')
        
        self.merge_robot(MujocoXML(robot_xml))
        self.merge_arena(MujocoXML(arena_xml))
        
        assert len(object_xmls) == len(object_names), "Number of object xmls and object names must match!"
        objects = []
        for i in range(len(object_xmls)):
            object_xml = object_xmls[i] 
            name = object_names[i]
            objects.append(MujocoXMLObject(
                object_xml,
                name=name,
                joints=[dict(type="free", damping="0.0005")],
                obj_type="all",
                duplicate_collision_geoms=True,
            ))
        self.merge_objects(objects)
        
        # Define filter method to automatically add a default name to visual / collision geoms if encountered
        group_mapping = {
            None: "col",
            "0": "col",
            "1": "vis",
        }
        ctr_mapping = {
            "col": 0,
            "vis": 0,
        }
        def _add_default_name_filter(element, parent):
            # Run default filter
            filter_key = _element_filter(element=element, parent=parent)
            # Also additionally modify element if it is (a) a geom and (b) has no name
            if element.tag == "geom" and element.get("name") is None:
                group = group_mapping[element.get("group")]
                element.set("name", f"g{ctr_mapping[group]}_{group}")
                ctr_mapping[group] += 1
            # Return default filter key
            return filter_key
        
        self._elements = sort_elements(root=self.root, element_filter=_add_default_name_filter)
        self._elements["root_body"] = self._elements["root_body"][0]
        self._elements["bodies"] = (
            [self._elements["root_body"]] + self._elements["bodies"]
            if "bodies" in self._elements
            else [self._elements["root_body"]]
        )

    def merge_robot(self, robot_xml):
        """
        Merges a robot xml into the task xml.

        Args:
            robot_xml (MujocoXML): robot xml to merge
        """
        self.merge(robot_xml, merge_body="default")

    def merge_arena(self, arena_xml):
        """
        Merges a table xml into the task xml.

        Args:
            table_xml (MujocoXML): table xml to merge
        """
        self.merge(arena_xml, merge_body="default")

    def merge_objects(self, mujoco_objects):
        """
        Adds object models to the MJCF model.

        Args:
            mujoco_objects (list of MujocoObject): objects to merge into this MJCF model
        """
        for mujoco_obj in mujoco_objects:
            # Make sure we actually got a MujocoObject
            assert isinstance(mujoco_obj, MujocoObject), "Tried to merge non-MujocoObject! Got type: {}".format(
                type(mujoco_obj)
            )
            # Merge this object
            self.merge_assets(mujoco_obj)
            self.worldbody.append(mujoco_obj.get_obj())
            
    def set_base_xpos(self, pos):
        """
        Places the robot on position @pos.

        Args:
            pos (3-array): (x,y,z) position to place robot base
        """
        self._elements["root_body"].set("pos", array_to_string(pos))

    def set_base_ori(self, rot):
        """
        Rotates robot by rotation @rot from its original orientation.

        Args:
            rot (3-array): (r,p,y) euler angles specifying the orientation for the robot base
        """
        # xml quat assumes w,x,y,z so we need to convert to this format from outputted x,y,z,w format from fcn
        rot = mat2quat(euler2mat(rot))[[3, 0, 1, 2]]
        self._elements["root_body"].set("quat", array_to_string(rot))
        
    def set_object_xpos(self, pos, name):
        """
        Places the object on position @pos.

        Args:
            pos (3-array): (x,y,z) position to place object
        """
        # print("elements:", self._elements.keys())
        # print("bodies:", len(self._elements["bodies"]))
        # for body in self._elements["bodies"]:
        #     print(body.get("name"))
        name += '_main'
        print("worldbody:")
        for body in self.worldbody:
            print(body.get("name"))
            if body.get("name") == name:
                body.set("pos", array_to_string(pos))
                print("pos:", np.array(body.get("pos")))
        
    def get_base_xpos(self):
        """
        Returns the position of the robot base.

        Returns:
            np.array: (x,y,z) position of robot base
        """
        return np.array(self._elements["root_body"].get("pos")) 