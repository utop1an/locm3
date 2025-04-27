# Learning action models

## LOCM (macq version) revision

- LOCM cannot handle duplicate objs in an action, e.g. rovers (communicate_soil_data rover1?rover general?lander **waypoint4?waypoint waypoint4?waypoint** waypoint2?waypoint)
- Implementation error
    - checking parameter flaws: macq only checks APs that are included in the hyp dict, not all the APs in/out a state
    - storing learned fluents by action name: learned fluents should be stored by APs not action names, e.g. unstack.2: s1->s2, unstack.1: s3->s1, then for fluents block_s1, it is the corresponds to different argument pos of the same action unstack
- (optional) locm paper said learn hyp on transition set and test on obj traces, but macq learn and test both on obj traces

The duplicate obj problem is not going to be fixed in LOCM/LOCM2, since they never considered this problem. We plan to work on this in the LOCM3.

After fixing the implementation errors in macq, we can get 100% executability when testing on the trainning traces (except Rovers domain which have duplicate objs in the plan)



## LOCM2 revision?

- split transitions error?

