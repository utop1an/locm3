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

- errors when checking the validity of the splitted transitions
    - the iLOCM version checks pair of transitions, which is incorrect
    - we should check with the example event seqs with respect to splitted transition, to avoid dead ends

Problems in handling duplicate objs
- more FSMs and states derived
- more state params derived
- lower exe

## POLOCM2

### experiment design

#### Current

- runtime
- accuracy/error of Learned Transition Matrix
- exe: comparing polocm2 to locm2

#### Later

- runtime
- acc/err
- cross exe

## POLOCM2-BASELINE

- need a proper way to find example event seqs
    - in the previous implementation, we only find pairs of events from the PO transition matrix
    - which is not enough for the fixed LOCM/LOCM2
    - event seqs with respect to the final transition matrix is required

- the current way of finding event seqs has problems?
    - tie-breaking issue
    - sometimes correct sometimes wrong for the same domain, rand issue

## Cross Executability 

- issues:
    - because we ignore the unseen effs (if a precond is not satisfied in the true_effs, but it has never been seen, we assume it's true)
        - many mistakes in the plans generated from the gt_domain as wel
    - when generating a plan, we break when there is no applicable action
        - shorter/0-len plans

- from the result of locm2
    - exe_on_l <<< exe_on_gt
        - is it a good metric?

fix:
    - option1: 
        - using prefix-seqs only for generating new-seqs from learned domain to test exe on ground truth domain, 
        - use ground truth (exisiting plans) to test exe on learned domain directly
        - settings:
            for each existing plan -> gen new seqs -> test exe on gt_domain -> take average to get exe_gt
            for each plan -> test exe on l_domain -> take average to get exe_l
            
        - still have the risks of generating new invalid seqs (we don't have a problem instance which defines init, goal, objs, no invariants can be generated? but still can use inequalities to constrain it?)
        - but result looks good

    - option2:
        - use both valid and invalid plans from the gt_domain to test on the l_domain
        - settings:
            for each domain and instance -> generate several valid seqs -> test on l_domain
            for each domain and instance -> generate several invalid seqs (the last action is invalid) -> test on l_domain final_exe = [ exe / (1-(1/len))]
            (random walks, avoid similar nodes to be explored)
        
        - random walk may gen seqs with transitions that never appeared in the training data, but still valid -> exe_valid < exe_l
            (adding invariants? currently sas+ is used to generate seqs, which already included some mutex info?)
        - maybe not generate another random walk for invalid, just replace the last action with an invalid one?
        - increase the number of invalid actions? but how to define invalid actions after invalid actions? [pickA stackBC (pickB?putA?pickD?)]
        - use first fail exe? no...

    
    - option3:
        - adding constraints when generating plans
        - type/equality/inequality constraints
        
        - woking...
        - still depends on (transitions/actions in the) evidenced plans

    - adding invariants constraints when generating plans
        - handcraft? domain specific, time consuming
        - invariants finder from fast_downward? strips based in variants found (is it better than sas+? no mapping from sas+ to strips atoms/predicates... working on using strips+invariants to gen new seqs)
            - before applying an action -> for each atom in the add effs -> check if it violates the true_effs (atoms in the state) against invariants, or the violations are deleted by current action -> reject action if violates

    

    






