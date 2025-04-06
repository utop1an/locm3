from .locm import LOCM
from collections import defaultdict
from typing import Set, Dict, List
from pddl.actions import LearnedAction, LearnedLiftedFluent
from pddl.model import LearnedModel
from traces import OSM, Trace, Event
import pandas as pd

class LOCM3(LOCM):

    def __init__(self,timeout:int=600, debug: Dict[str, bool]=None):
        super().__init__(timeout, debug)
        pass

    def extract_model(self, trace_list: List[Trace], types: Dict = None):
        sorts = self._get_sorts(trace_list, types)
        obj_trace_list, TM_list = self.trace_to_transition_matrix(self, trace_list, sorts)
        OSM_list =  self.TM_to_OSM(TM_list)
        fluents, actions = self.OSM_to_action(self,OSM_list, sorts)
        model = LearnedModel(fluents,actions, sorts)

        return model

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
    
    def get_ceA(self, TM: pd.DataFrame, event: Event):
        """
        Get the set of events that are consecutively after the given evenit.
        Except the event itself.
        
        Parameters
        ----------
        TM : pd.DataFrame
            Transition matrix.
        event : Event
            Event to get the set of events after it.

        Returns
        -------
        ceA : Set
            Set of events that are consecutively after the given event.
        """
        ceA = set()
        for i, row in TM.iterrows():
            for j, val in row.items():
                if val == 1 and j!= event:
                    ceA.add(j)
        return ceA
    
    def OSM_to_action(self, OSM_list: List[List[OSM]], sorts: Dict):
        actions = {}
        fluents = set()

        for sort, OSM_by_event in enumerate(OSM_list):
            for osm in OSM_by_event:
                event = osm.event
                if event.action.name not in actions:
                    action = LearnedAction(event.action, [sorts[obj] for obj in event.action.obj_params])
                else:
                    action = actions[event.action.name]

                f1 = LearnedLiftedFluent(
                    f"m{event.action.name}.{event.position}.1",
                    [sorts[obj] for i,obj in enumerate(event.action.obj_params) if i in event.position ],
                    [pos for pos in event.position],
                )
                f0 = LearnedLiftedFluent(
                    f"m{event.action.name}.{event.position}.0",
                    [sorts[obj] for i,obj in enumerate(event.action.obj_params) if i in event.position ],
                    [pos for pos in event.position],
                )
                fluents.add(f1)
                fluents.add(f0)

                
                action.update_precond(f0)
                action.update_add(f1)
                action.update_delete(f0)

                for event_after in osm.del_events:
                    if event_after.action.name not in actions:
                        action_after = LearnedAction(event_after.action, [sorts[obj] for obj in event_after.action.obj_params])
                    else:
                        action_after = actions[event_after.action.name]
                    
                    action_after.update_precond(f1)
                    action_after.update_add(f0)
                    action_after.update_delete(f1)
        
        return fluents, set(actions.values())

                


    

    
     


    
