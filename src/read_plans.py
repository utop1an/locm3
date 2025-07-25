import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import logging
from planner import RandomPlanner


def generate_trace(solution_dir, planner):
    try:
        plan = parse_plan_file(solution_dir, planner)
    except Exception as e:
        logging.error(e)
        return None
    return plan

def parse_plan_file(file, planner):
    with open(file, 'r') as f:
        lines = f.readlines()
    plan = []
    for line in lines:
        if line.startswith(";"):
            break
        action = line.strip().strip("()").split(" ")
        action_name = action[0]
        op = planner.task.get_action(action_name)
        assert op is not None, f"Action {action_name} not found in domain"

        args = action[1:]
        arg_types = [p.type_name for p in op.parameters]
        arg_with_types = [arg + "?"+ t for arg, t in zip(args, arg_types)]
        new_action = f"({action_name} {' '.join(arg_with_types)})"
        plan.append(new_action)
    return plan


def process_domain(domain, solution_dir, task_dir):
    logging.info(f"Generating traces for domain: {domain}")
    domain_filepath = os.path.join(task_dir, domain, "domain.pddl")

    traning_dir = os.path.join(solution_dir, domain, "training/easy")
    output_data = []

    for plan_file in os.listdir(traning_dir):
        if not plan_file.endswith(".plan"):
            continue
        task_file = plan_file.replace(".plan", ".pddl")
        plan_filepath = os.path.join(traning_dir, plan_file)
        task_filepath = os.path.join(task_dir, domain, "training/easy", task_file)

        if not os.path.exists(task_filepath):
            continue

        planner = RandomPlanner(domain_filepath, task_filepath)
        number_of_objects = len(planner.task.objects)

        plan = generate_trace(plan_filepath, planner)
        plan_data = f"{domain}&&{task_file}&&{number_of_objects}&&{len(plan)}&&{','.join(plan)}\n"
        output_data.append(plan_data)
        

    logging.info(f"{domain} done...")
    return output_data

def write_output(output_file, results):
    with open(output_file, 'a', buffering=1) as file:  # Line buffered
        for result in results:
            for line in result:
                file.write(line)

def main(args):
    input_path = args.i
    output_path = args.o

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Generating traces from raw plans in {input_path}")

    if not os.path.exists(input_path):
        logging.error(f"Input path {input_path} does not exist")
        return

    solution_dir = os.path.join(input_path, "solutions")
    task_dir = os.path.join(input_path, "tasks")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_file = os.path.join(output_path, "plain_traces.txt")

    domains = [domain for domain in os.listdir(solution_dir) if os.path.isdir(os.path.join(solution_dir, domain))]
    for domain in domains:
        logging.info(f"Processing domain: {domain}")
        res = process_domain(domain, solution_dir, task_dir)
        write_output(output_file,res)

    logging.info(f"Writting traces to {output_file}")
    # Write the results to the output file
    

    
    logging.info(f"Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse plain traces from raw plans and generate random traces")
    parser.add_argument("--i", type=str, default="./data/goose-benchmarks", help="Directory containing raw plans and task pddls")
    parser.add_argument("--o", type=str, default="./data/plain_traces", help="Output file path")
    args = parser.parse_args()
    main(args)

"""
Directly run to get traces from goose-benchmark plans.
"""