from .locm import LOCM
from collections import defaultdict
from typing import Set, Dict, List
from traces import *
import pandas as pd

class LOCM3(LOCM):

    def __init__(self,timeout:int=600, debug: Dict[str, bool]=None):
        super().__init__(timeout, debug)
        pass

    def extract_model(self, tracelist: List[Trace], sorts: Dict = None):
        sorts = self._get_sorts(tracelist)
        obj_trace_list, TM_list = self.trace_to_transition_matrix(self, tracelist, sorts)
        OSM_list =  self.TM_to_OSM(TM_list)
        fluents, actions = self.OSM_to_action(OSM_list)


        pass

    def TM_to_OSM(self, TM_list: List[pd.DataFrame]):
        OSM_list = []
        for sort, TM in enumerate(TM_list):
            OSM_by_event = []
            for event, row in TM.iterrows():
                ceA = self.get_ceA(TM, event)
                osm = OSM(sort, event, ceA)
                OSM_by_event.append(osm)
            OSM_list.append(OSM_by_event)
        return OSM_list
    
    def get_ceA(self, TM, event):
        ceA = set()
        for i, row in TM.iterrows():
            for j, val in row.items():
                if val == 1 and j!= event:
                    ceA.add(j)
        return ceA
    
    def OSM_to_action(self, OSM_list: List[List[OSM]]):
        actions = set()
        fluents = set()

        for sort, OSM_by_event in enumerate(OSM_list):
            for osm in OSM_by_event:
                event = osm.event
                action = LearnedAction(event.action, sort)
                actions.add(action)

                for obj in event.obj_params:
                    fluent = Fluent(obj.name, sort)
                    fluents.add(fluent)


    
    def pre_works():
        pass

    
     


    
