
from extract import POLOCM2, POLOCM, POLOCM2BASELINE, POLOCMBASELINE
from evaluator import ExecutabilityEvaluator
import numpy as np
from utils import set_timer_throw_exc, GeneralTimeOut, read_plan, read_json_file
from multiprocessing import Pool, Lock
import os
import argparse
import logging
import datetime
import random
from collections import defaultdict

DEBUG = False
SEED = 42
OUTPUT_DIR = "./output"


lock= Lock()

# Setup logger
def setup_logger(log_file):
    logger = logging.getLogger('experiment_logger')
    logger.setLevel(logging.DEBUG)

    # Create a file handler for logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)

    return logger

def run_single_experiment(cplex_dir,cplex_threads, extraction_type, learning_obj, dod , test_data, invalid_test_suffixes, logger):
    """Runs a single experiment given the necessary parameters."""
    domain = learning_obj['domain']
    traces = learning_obj['traces']
    all_po_traces = learning_obj['po_traces']

    logger.info(f"Running {domain}-lo.{learning_obj['id']}-{dod} ...")

    extractions = {
        'p2': POLOCM2,
        'p': POLOCM,
        'p2b': POLOCM2BASELINE,
        'pb': POLOCMBASELINE,
    }
    extraction = extractions[extraction_type]

    
    try:
        index_by_dod = int(dod* 10 -1)
        po_traces = all_po_traces[index_by_dod]
        actual_dod = sum([poat.flex for poat in po_traces]) / len(po_traces)
        runtime,tps, fps, tns, fns,acceptance_rate , invalid_acceptance_rate, remark = solve(
            cplex_dir,
            cplex_threads, 
            extraction,
            po_traces,
            traces,
            domain,
            test_data, 
            invalid_test_suffixes,
        )


    except GeneralTimeOut as t:
        runtime, tps, fps, tns, fns,acceptance_rate , invalid_acceptance_rate, remark = (600,0,0), 0,0,0,0,0, 0, f"Timeout"
    except Exception as e:
        runtime, tps, fps, tns, fns,acceptance_rate , invalid_acceptance_rate, remark = (0, 0, 0), 0,0,0, 0, 0, 0, e
        logger.error(f"Error during experiment for domain {domain}: {e}")

    polocm_time, locm2_time, locm_time = runtime
    logger.info(f"{domain}-lo.{learning_obj['id']}-{dod}  DONE")

    result_data = {
        'id': learning_obj['id'],
        'dod': dod,
        'actual_dod': actual_dod,
        'domain': domain,
        'index': learning_obj['index'],
        'total_length': learning_obj['total_length'],
        'len%': learning_obj['len%'],
        'runtime': sum(runtime),
        'polocm_time': polocm_time,
        'locm2_time': locm2_time,
        'locm_time': locm_time,
        'tp': tps,
        'fp': fps,
        'tn': tns,
        'fn': fns,
        'acceptance_rate': acceptance_rate,
        'invalid_acceptance_rate': invalid_acceptance_rate,  # Placeholder for invalid acceptance rate
        'remark': remark
    }
    write_result_to_csv(dod, extraction_type, result_data)
    return




def write_result_to_csv(dod, extraction_type, result_data):
    """Writes the result data to a CSV file in a thread-safe manner."""
    csv_file_path = os.path.join(OUTPUT_DIR, f"results_{dod}_{extraction_type}.csv")
    with lock:
        file_exists = os.path.exists(csv_file_path)
        with open(csv_file_path, 'a') as csv_file:
            if not file_exists:
                headers = result_data.keys()
                csv_file.write(','.join(headers) + '\n')

            values = [str(result_data[key]) for key in result_data.keys()]
            csv_file.write(','.join(values) + '\n')


@set_timer_throw_exc(num_seconds=600, exception=GeneralTimeOut, max_time=600)
def solve(cplex_dir,cplex_threads, extraction,po_traces ,traces, domain_name, test_data, invalid_test_suffixes):
    try: 
        remark = []
        extraction_method = extraction(cplex_dir, cplex_threads)
        
    
        model, TM , runtime = extraction_method.extract_model(po_traces)
    
        pddl_model = model.to_pddl_domain(domain_name)
        golden_TM = extraction_method.get_TM_list(traces)
    
        tps, fps, tns, fns, r = get_AP_accuracy(TM, golden_TM)
        if r:
            remark.append(r)
        acceptance_rate, invalida_acceptance_rate, r = get_acceptance_rate(pddl_model, test_data, invalid_test_suffixes)
        if r:
            remark.append(r)

        if len(remark)==0:
            remark = ['Success']
    except MemoryError as me:
        print(f"MemoryError: {me}. Terminating extraction method.")
        extraction_method.terminate()
        return (600,0,0), 0,0,0,0,0, 0, f"Memout"
    except Exception as e:
        print(f"Error: {e}")
        extraction_method.terminate()
        return (0,0,0), 0,0, 0,0,0,0, e
    return runtime, tps, fps, tns, fns, acceptance_rate, invalida_acceptance_rate, " ".join(remark)


def get_AP_accuracy(TM, golden_TM):
    if (len(TM)==0):
        return 0,1, "AP Empty"
    if (len(TM) != len(golden_TM)):
        return 0,1, "AP Invalid Length"
    tps = 0
    fps = 0
    fns = 0
    tns = 0

    def get_golden_TM(TM_cols):
        for golden_tm in golden_TM:
            if set(golden_tm.columns) == TM_cols:
                return golden_tm
        return None
    for sort, m1 in enumerate(TM):
        m1_cols = set(m1.columns)
        golden_tm = get_golden_TM(m1_cols)
        assert golden_tm is not None, f"Golden TM for sort {sort}: {m1_cols} not found in golden_TM"
        
        m1 = m1.reindex(index=golden_tm.index, columns=golden_tm.columns) 
        m1 = np.where(m1>0, 1, 0)
        
        l1 = m1.flatten()

        m2 = np.where(golden_tm>0, 1,0)
        l2 = m2.flatten()
      
        # print(f"sort{sort}-AP array [learned]: {l1}")
        # print(f"sort{sort}-AP array [ground ]: {l2}")
        # acc.append(sum(l1==l2)/len(l1))
        # fpr.append(np.sum((l2==0)& (l1==1))/len(l1)) # one side error rate
        tps+= np.sum((l1 == 1) & (l2 == 1))
        fps += np.sum((l1 == 1) & (l2 == 0))
        
        tns += np.sum((l1 == 0) & (l2 == 0))
        fns+= np.sum((l1 == 0) & (l2 == 1))
        
    return tps, fps, tns, fns, None


def get_acceptance_rate(learned_domain, test_data, invalid_test_suffixes):
   
    try:
        evaluator = ExecutabilityEvaluator(learned_domain)
        valid_res = []
        invalid_res = []
        
        for problem, trace in test_data.items():
            valid_acceptance, invalid_acceptance = evaluator.get_acceptance_rate(trace, invalid_test_suffixes[problem])
            valid_res.append(valid_acceptance)
            # Only considert invalid acceptance if the valid seq is accepted
            if valid_acceptance == 1:
                invalid_res.append(invalid_acceptance)
        if len(valid_res) == 0:
            valid = 0
        else:
            valid = sum(valid_res) / len(valid_res)
        if len(invalid_res) == 0:
            invalid = 0
        else:
            invalid = sum(invalid_res) / len(invalid_res)
    except Exception as e:
        return 0,0, "Error in acceptance rate calculation" + str(e)
    return valid, invalid, None


def read_files(input_filepath, test_filepath, invalid_suffixes_filepath):
    
    
    TRAIN_DATA  = read_json_file(input_filepath)

    plain_traces = defaultdict(lambda: defaultdict())
    with open(test_filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            details = line.split("&&")

            domain_name = details[0]
            problem_name = details[2]
            plan = details[-1]

            plain_traces[domain_name][problem_name]= read_plan(plan)
    TEST_DATA = plain_traces

    invalid_suffixes = defaultdict(lambda: defaultdict(list))

    with open(invalid_suffixes_filepath, "r") as f:
        lines = f.readlines()
        for line in lines:
            details = line.split("&&")
            domain_name = details[0]
            problem_name = details[1]
            plan = details[2]
            
            invalid_suffixes[domain_name][problem_name].append(read_plan(plan)[0])
    INVALID_TEST_SUFFIXES = invalid_suffixes
    return TRAIN_DATA, TEST_DATA, INVALID_TEST_SUFFIXES

def experiment(cplex_dir, experiment_threads, cplex_threads, extraction_type, dod, train_data, test_data, invalid_test_suffixes):
    log_filename = f"dod{dod}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join("./logs", log_filename)
    logger = setup_logger(log_filepath)
    logger.info("Experiment Start...")
    logger.info(f"Using {experiment_threads} threads for parallel processing.")

    missing = {
        0.9: [581, 582, 584, 585, 586, 587, 881, 882, 884, 885, 887, 888],
    }


    tasks = []
    for item in train_data:
        if missing:
            if item['id'] not in missing[dod]:
                continue
        domain = item['domain']
        print(f"adding task domain: {domain}, id: {item['id']}, dod: {dod}")
        tasks.append((cplex_dir,cplex_threads, extraction_type, item, dod, test_data[domain], invalid_test_suffixes[domain], logger))
        
        
    
    if DEBUG:
        tasks = random.sample(tasks, 10)

    if experiment_threads > 1:
        logger.info("Running experiment in multiprocessing...")
        with Pool(processes=experiment_threads, maxtasksperchild=1) as pool:
            pool.starmap(run_single_experiment, tasks)
     
    else:
        logger.info("Running experiment in sequential...")
        for task in tasks:
            r= run_single_experiment(*task)
  
    logger.info("Experiment completed.")

def main(args):
    global DEBUG, OUTPUT_DIR
    input_filepath = args.i
    output_dir = args.o
    test_filepath = args.t
    invalid_suffixes_filepath = args.v
    experiment_threads = args.et
    cplex_threads = args.ct
    dod = args.d
    cplex_dir = args.cplex
    extraction_type = args.e

    if extraction_type not in ['p2', 'p', 'p2b', 'pb']:
        print(f"Invalid extraction type {extraction_type}. Choose from ['p2', 'p', 'p2b','pb']")
        return


    DEBUG = args.debug


    dods = [0.1,0.2,0.3,0.4,0.5, 0.6,0.7,0.8,0.9,1]
    if dod not in dods:
        print(f"Invalid dod {dod}. Choose from {dods}")
        return

    if experiment_threads < 1 or cplex_threads <1:
        print("Invalid number of threads. Choose a number greater than 0")
        return

    if not os.path.exists(cplex_dir):
        print("No cplex solver provided, defualt pulp solver will be used for MLP")
        cplex_dir = "default"


    if not os.path.exists(input_filepath):
        print(f"Input file {input_filepath} does not exist")
        return
    
    if not os.path.exists(test_filepath):
        print(f"Test file {test_filepath} does not exist")
        return
    
    if not os.path.exists(invalid_suffixes_filepath):
        print(f"Invalid suffixes file {invalid_suffixes_filepath} does not exist")
        return
    
    train_data, test_data, invalid_test_suffixes = read_files(input_filepath, test_filepath, invalid_suffixes_filepath)


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    OUTPUT_DIR = output_dir

    log_dir = ("./logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    experiment(cplex_dir, experiment_threads, cplex_threads, extraction_type, dod, train_data, test_data, invalid_test_suffixes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument('--i', default="./data/training_data/traces_plan_po_r10.json", type=str, help='Input trainning file name')
    parser.add_argument('--o', default="./output", type=str, help='Output directory')
    parser.add_argument('--t', type=str, default="./data/plain_traces/plain_traces.txt", help='Path to the test data')
    parser.add_argument('--v', type=str, default="./data/plain_traces/invalid_suffixes.txt", help='Path to the (invalid) test data')
    parser.add_argument('--d', type=float, default=0.1, help='dod')
    parser.add_argument('--e', type=str, default='p2', help='Type of extraction')
    parser.add_argument('--et', type=int, default=4, help='Number of threads for experiment')
    parser.add_argument("--cplex", type=str, default="/opt/cplex/cplex/bin/x86-64_linux/cplex", help="Path to cplex solver")
    parser.add_argument('--ct', type=int, default=2, help='Number of threads for cplex')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    main(args)