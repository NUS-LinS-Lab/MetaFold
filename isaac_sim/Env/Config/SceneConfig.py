import numpy as np
import torch
from omni.kit.viewport.utility import get_active_viewport

class SceneConfig:
    def __init__(self,room_usd_path="./data/Cloth-Simulation/Assets/Scene/FlatGrid.usd",pos:np.ndarray=None,ori:np.ndarray=None,scale:np.ndarray=None):
        if pos is None:
            self.pos=np.array([0,0,0])
        else:
            self.pos=pos
        if ori is None:
            self.ori=np.array([0,0,0])
        else:
            self.ori=ori
        if scale is None:
            self.scale=np.array([1,1,1])
        else:
            self.scale=scale
        self.room_usd_path=room_usd_path

        # vp_api = get_active_viewport()
        # vp_api.__getattribute__('horizontalAperture').Set(36.0)