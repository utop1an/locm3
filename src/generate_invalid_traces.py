import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import logging
from planner import InvalidPlanner

REPEAT = 10


def generate_invalid_traces(domain_filepath,taskfile_path, valid_trace):
    output_data = []
    planner = InvalidPlanner(domain_filepath, taskfile_path, num_traces=REPEAT)

    invalid_suffixes = planner.generate_invalid_suffixes(valid_trace, duplicate_actions=True, new_actions=True)
    

    return invalid_suffixes

def main(args):
    global REPEAT
    inputfile_path = "./data/plain_traces/plain_traces.txt"
    outputfile_path = "./data/plain_traces/invalid_suffixes.txt"
    task_dir = "./data/goose-benchmarks/tasks"
    
    REPEAT = args.r

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Generating traces from raw plans in {inputfile_path}")



    with open(inputfile_path, 'r') as file:
        res =[]
        for line in file:
            raw = line.strip().split('&&')
            logging.info(f"Processing {raw[0]}-{raw[2]}")
            domain_name = raw[0]
            valid_trace = raw[-1].split(',')
            domain_filepath = os.path.join(task_dir, domain_name, "domain.pddl")
            taskfile_path = os.path.join(task_dir, domain_name, "training/easy", raw[2] )

            invalid_suffixes = generate_invalid_traces(domain_filepath, taskfile_path, valid_trace)
            for invalid_suffix in invalid_suffixes:
                res.append(f"{domain_name}&&{raw[2]}&&{invalid_suffix}\n")


    logging.info("start writing...")
    # Write the results to the output file
    with open(outputfile_path, 'w') as file:  # Line buffered
        for line in res:
            file.write(line)
            file.flush()

    
    logging.info(f"Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse plain traces from raw plans and generate random traces")
 
    parser.add_argument("--r", type=int, default=10, help="repeat times")
    args = parser.parse_args()
    main(args)

"""
Directly run to get traces from goose-benchmark plans.


"""