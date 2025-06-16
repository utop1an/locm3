import os
import pandas as pd
import random
import json
import argparse
from traces import *
from pddl import TypedObject, ActionSignature

SEED=42
random.seed(SEED)
REPEAT = 5
TRACELENGTH = [10,15,30,50,75,100]
NUMBEROFTRACES =[1,3,5,10,25,50,100]
COUNTER = 0
FLEX = [0.1,0.2,0.3,0.4,0.5, 0.6,0.7,0.8,0.9,1]
PO = False

def write_to_file(output_data, file_path):
    print(f"Writing to file {file_path} with {len(output_data)} traning data")
    try:
        with open(file_path, 'w', buffering=1) as file:  # Line buffered
            json.dump(output_data, file)
    except Exception as e:
        print(f"Error writing to file {file_path}: {e}")

def sample_combined(df, number_of_traces):
    
    types = df['type'].unique()
    p1 = df[df['type'] == types[0]]
    p2 = df[df['type'] == types[1]]

    p1_traces = p1.sample(n=number_of_traces//2, random_state=SEED)
    p2_traces = p2.sample(n=number_of_traces-len(p1_traces), random_state=SEED)

    return pd.concat([p1_traces, p2_traces])


def generate_trace(domain, df, number_of_traces,combined, overall_length, trace_length=None):
    global COUNTER
    output = []
    for i in range(REPEAT):
        number_of_traces = min(number_of_traces, len(df))
        if combined == "combined":
            rows = sample_combined(df, number_of_traces)
        elif combined == "plan":
            rows = df[df['type'] == 'plan'].sample(n=number_of_traces)
        elif combined == "random":
            rows = df[df['type'] == 'rand'].sample(n=number_of_traces, random_state=SEED)
        
      
        traces = []
        total_length = 0
        number_of_objects = 0
        for r, row in rows.iterrows():
            plain_trace = row['trace'].split(',')
            if len(plain_trace)<5:
                continue
            if trace_length is not None:
                if (len(plain_trace) <= trace_length):
                    rand_trace = plain_trace
                else:
                    # rand_start = random.randint(0, len(plain_trace) - trace_length)
                    # rand_trace = plain_trace[rand_start:rand_start + trace_length]
                    rand_trace = plain_trace[:trace_length]
            else:
                rand_trace = plain_trace
            trace = []
            
            total_length += len(rand_trace)
            number_of_objects += int(row['number_of_objects'])
            for plain_op in rand_trace:
                op = plain_op.strip('()').split(' ')
                trace.append({'action': op[0], 'objs': op[1:]})
            traces.append(trace)
            if total_length + len(rand_trace) >= 1000:
                break
        if len(traces) == 0:
            continue
        
        if PO:
            poats = get_PO_data(traces)
        else:
            poats = []

        output_obj = {
            'id': COUNTER,
            'domain': domain,
            'index': i,
            'total_length': total_length,
            'traces': traces,
            'poats': poats,
            'number_of_objects': int(number_of_objects/len(traces)),
            'len%': total_length/overall_length
        }
        output.append(output_obj)
        COUNTER += 1
    return output

def get_PO_data(raw_traces):
    poats = []
    traces = []
    for raw_trace in raw_traces:
        steps = []
        for i, raw_step in enumerate(raw_trace):
            action_name = raw_step['action']
            obj_names = raw_step['objs']
            objs = [TypedObject('na', obj) for obj in obj_names]
            action = ActionSignature(action_name, objs)
            step = Step(State(), action, i)
            steps.append(step)
        trace = Trace(steps)
        traces.append(trace)


    for flex in FLEX:
        pos = []
        inds = []
        for trace in traces:
            po_trace = trace.to_partial_ordered_trace(flex)
            po = []
            ind = []
            
            for po_step in po_trace:

                if (type(po_step) != PartialOrderedStep):
                    continue
                ind.append(po_step.index)
                po.append(po_step.successors)
            pos.append(po)
            inds.append(ind)
        poat = {
            'actual_flex': po_trace.flex,
            'inds': inds,
            'pos': pos
        }
        poats.append(poat)
        
    return poats

def get_overall_length(df):
    overall_length = 0
    for i, row in df.iterrows():
        plain_trace = row['trace'].split(',')
        overall_length += len(plain_trace)
    return overall_length

def main(args):
    global REPEAT, CONVERTOR, PO
    input_filepath = args.i
    output_dir = args.o
    repeat = args.r
    combined = args.c
    po = args.po
    if po:
        PO = True
    REPEAT = repeat

    
    if combined not in ["combined", "plan", "random"]:
        print("Invalid combination type. Choose from combined, plan, random")
        return

    if not os.path.exists(input_filepath):
        print(f"Input file {input_filepath} does not exist")
        return
    

    headers = ['domain', 'type', 'problem_name', "difficulty", "number_of_objects", 'plan_len', 'trace']
    input_data = pd.DataFrame(columns=headers)
    with open(input_filepath, 'r') as file:
        for line in file:
            raw = line.strip().split('&&')

            if (raw[-1]!='Error' and raw[-1]!='TimeoutError' and raw[-1]!='TraceSearchTimeOut'):
                input_data.loc[len(input_data)] = raw
    
    output_data = []
    domains = input_data['domain'].unique() 
    for domain in domains:
        df = input_data[input_data['domain'] == domain]
        overall_length = get_overall_length(df)
        for length in TRACELENGTH:
            for num in NUMBEROFTRACES:
                if num > len(df):
                    break
                if length * num > 1000:
                    break
                output = generate_trace(domain, df, num, combined,overall_length, trace_length=length )
                output_data= output_data + output
    
    output_filename_pre = 'traces'
    output_filename_pre += f'_{combined}_r{REPEAT}'
    output_filename = f'{output_filename_pre}.json'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    write_to_file(output_data, os.path.join(output_dir, output_filename))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create json traces from plain traces')
    parser.add_argument('--i', type=str, default="./data/plain_traces/plain_traces.txt", help='Input plain trace file path')
    parser.add_argument('--o', type=str,default="./data/training_data", help='Output directory')
    parser.add_argument('--r', type=int, default=1, help='Number of times to repeat the generation')
    parser.add_argument('--c', type=str, default="plan", help='Generate traces combining plans and random walks or only plans or only random walks')
    parser.add_argument('--po', action='store_true', help='Generate PO data')
    args = parser.parse_args()

    main(args)

"""
Directly run to generate training data with no PO data.

run with:
    --po to generate PO data
"""
