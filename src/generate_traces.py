import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import logging
import multiprocessing
from planner import RandomPlanner



def process_domain(domain, task_dir, num_traces, rdPlannerTimeout, max_objects):

    logging.basicConfig(level=logging.INFO, format='[%(processName)s] %(message)s')
    logging.info(f"Generating traces for domain: {domain}")
    domain_filepath = os.path.join(task_dir, domain, "domain.pddl")

    training_dir = os.path.join(task_dir, domain, "training/easy")
    output_data = []


    for task_file in os.listdir(training_dir):
        if not task_file.endswith(".pddl"):
            continue
        task_filepath = os.path.join(task_dir, domain, "training/easy", task_file)
        if not os.path.exists(task_filepath):
            continue
        plan_filepath = os.path.join(training_dir.replace("tasks", "solutions"), task_file.replace(".pddl", ".plan"))

        with open(plan_filepath, 'r') as f:
            plan_length = sum(1 for line in f if not line.startswith(";"))

        planner = RandomPlanner(domain_filepath, task_filepath, trace_len=plan_length, num_traces=num_traces, max_time=rdPlannerTimeout)
        number_of_objects = len(planner.task.objects)
        if max_objects is not None and number_of_objects > max_objects:
            continue

        random_walks = planner.generate_traces()

        for trace in random_walks:
            trace_data = f"{domain}&&{'rand'}&&{task_file}&&{'easy'}&&{number_of_objects}&&{len(trace)}&&{','.join(trace)}\n"
            output_data.append(trace_data)

        invalid_random_walks = planner.generate_traces(is_valid_trace=False)
        for invalid_trace in invalid_random_walks:
            invalid_trace_data = f"{domain}&&{'invalid'}&&{task_file}&&{'easy'}&&{number_of_objects}&&{len(invalid_trace)}&&{','.join(invalid_trace)}\n"
            output_data.append(invalid_trace_data)

    logging.info(f"{domain} done...")
    return output_data

def main(args):
    input_path = args.i
    output_path = args.o
    trace_length = args.l
    rdPlannerTimeout = args.t
    num_traces = args.n
    max_objects = args.m

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Generating traces from raw plans in {input_path}")

    if not os.path.exists(input_path):
        logging.error(f"Input path {input_path} does not exist")
        return

    task_dir = os.path.join(input_path, "tasks")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_file = os.path.join(output_path, "random_walks.txt")

    # Use multiprocessing Pool to parallelize domain processing
    with multiprocessing.Pool(processes=4) as pool:
        # Collect all domains first
        domains = [domain for domain in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, domain))]
        
        # Map each domain to the process_domain function
        results = pool.starmap(
            process_domain,
            [(domain, task_dir, num_traces, rdPlannerTimeout, max_objects) for domain in domains],
            chunksize=1
        )


    # Write the results to the output file
    with open(output_file, 'w', buffering=1) as file:  # Line buffered
        for result in results:
            for line in result:
                file.write(line)

    
    logging.info(f"Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse plain traces from raw plans and generate random traces")
    parser.add_argument("--i", type=str, default="./data/goose-benchmarks", help="Directory containing raw plans and task pddls")
    parser.add_argument("--o", type=str, default="./data/plain_traces", help="Output file path")
    parser.add_argument("--l", type=int, default=50, help="Length of the random traces")
    parser.add_argument("--t", type=int, default=30, help="Timeout for random planner generating traces per task in seconds")
    parser.add_argument("--n", type=int, default=10, help="Number of random traces per task")
    parser.add_argument("--m", type=int, default=None, help="Maximum number of objects in the tasks")
    args = parser.parse_args()
    main(args)

"""
Directly run to get traces from goose-benchmark plans.


"""