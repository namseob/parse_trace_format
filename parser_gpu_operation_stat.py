import argparse
import json
from collections import OrderedDict
import operator

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

import datetime
import re

### event lists ###
'''
Duration, Complete, Instant, Counter, Flow, Object, Metadata ...
'''

# event descriptions 
EV_NAME = 'name'
EV_CAT  = 'cat'
EV_PH   = 'ph'
EV_TS   = 'ts'
EV_DUR  = 'dur'
EV_TTS  = 'tts'
EV_PID  = 'pid'
EV_TID  = 'tid'
EV_ARGS = 'args'

# ev_ph category
EV_PH_DURATION_BEGIN   = 'B'
EV_PH_DURATION_END     = 'E'

EV_PH_COMPLETE         = 'X'

EV_PH_INSTANT          = 'I'

EV_PH_COUNTER          = 'C'

EV_PH_FLOW_EVENT_START = 's'
EV_PH_FLOW_EVENT_STEP  = 't'
EV_PH_FLOW_EVENT_END   = 'f'

EV_PH_OBJECT_CREATED   = 'N'
EV_PH_OBJECT_SNAPSHOT  = 'O'
EV_PH_OBJECT_DESTROYED = 'D'

EV_PH_METADATA         = 'M'

# ev cat category
EV_CAT_TENSOR       = 'Tensor'
EV_CAT_OP           = 'Op'

# ev_name category
EV_NAME_PROC_NAME = 'process_name'
EV_NAME_MEMCPY_HTOD = 'MEMCPYHtoD'
EV_NAME_MEMCPY_DTOH = 'MEMCPYDtoH'

# ev_args category
EV_ARGS_NAME                 = 'name'
EV_ARGS_OP                   = 'op'
EV_ARGS_SNAPSHOT             = 'snapshot'

# ev_args_snapshot category
EV_ARGS_SNAPSHOT_TENSOR_DESC 	= 'tensor_description'

# ev_args_snapshot_tensor-description category
EV_ARGS_SNAPSHOT_TENSOR_DESC_DTYPE				= 'dtype'
EV_ARGS_SNAPSHOT_TENSOR_DESC_SHAPE 				= 'shape'
EV_ARGS_SNAPSHOT_TENSOR_DESC_ALLOC_DESC   = 'allocation_description'

# ev_args_snapshot_tensor-desciption_allocation-description category
EV_ARGS_SNAPSHOT_TENSOR_DESC_ALLOC_DESC_REQ_BYTES           = 'requested_bytes:'
EV_ARGS_SNAPSHOT_TENSOR_DESC_ALLOC_DESC_ALLOC_BYTES         = 'allocated_bytes:'

# ev_args_name category (for parsing)
EV_ARGS_NAME_GPU_TENSORS          = "/job:localhost/replica:0/task:0/device:GPU:0 Tensors"
EV_ARGS_NAME_GPU_ALL_COMPUTE      = "/device:GPU:0/stream:all Compute"

# mega bytes
ONE_MEGA = 1024*1024

#def extract_process_name(trace_line, proc_name, type):
def extract_pid(trace_line, ev_name, ev_ph, ev_args_name):
    pid = -1

    #print(ev_name, ev_ph)
    for trace_ev in trace_line:
        if (trace_ev[EV_NAME] == ev_name and trace_ev[EV_PH] == ev_ph):
            args = trace_ev[EV_ARGS]
            #print(args)
            pid = trace_ev[EV_PID]

            if args[EV_ARGS_NAME] == ev_args_name:
                return pid
                # print(pid)

    return pid

def extract_memcpy_htod_args_name(trace_line, memcpy_htod):
	args_name_list = []

	for trace_ev in trace_line:
		if (trace_ev[EV_NAME] == memcpy_htod and trace_ev[EV_PH] == EV_PH_COMPLETE):
			args = trace_ev[EV_ARGS]

			args_name_list.append((args[EV_ARGS_NAME], int(trace_ev[EV_TS]), int(trace_ev[EV_DUR])))

	return args_name_list

def show_memcpy_memory_stats(trace_line, requested_mem_str, allocated_mem_str):
    requested_mem_bytes = -1
    allocated_mem_bytes = -1

    mem_stats_list           = []
    mem_stats_list_exception = []

    for trace_ev in trace_line:
      if (trace_ev[EV_PH] == EV_PH_OBJECT_SNAPSHOT and trace_ev[EV_CAT] == EV_CAT_TENSOR):
            tensor_name = trace_ev[EV_NAME]
            args = trace_ev[EV_ARGS]

            snapshot_desc = args[EV_ARGS_SNAPSHOT]
            tensor_desc = snapshot_desc[EV_ARGS_SNAPSHOT_TENSOR_DESC]
            #print('>>>', tensor_name)
            #print(tensor_desc)

            #print(type(tensor_desc))
            if tensor_desc.find('cuda_host_bfc') > 0:
                  tensor_desc_list = tensor_desc.split(' ')
                  #print(tensor_desc_list)

                #i = tensor_desc_list.index('allocator_name:')
                #allocator_name = tensor_desc_list[i+1]

                #print(allocator_name)
                #print("===============")

                #if 'cuda_host_bfc\n' in allocator_name:
                  i = tensor_desc_list.index(requested_mem_str)
                  requested_mem_bytes = tensor_desc_list[i+1]

                  i = tensor_desc_list.index(allocated_mem_str)
                  allocated_mem_bytes = tensor_desc_list[i+1]

                  #print(requested_mem_bytes, allocated_mem_bytes)
                  mem_stats_list.append((tensor_name, int(requested_mem_bytes), int(allocated_mem_bytes)))
            else:
                mem_stats_list.append((tensor_name, 0, 0))
                mem_stats_list_exception.append(tensor_name)
                #print(tensor_name)

    sorted_mem_stats_list = sorted(mem_stats_list, key=operator.itemgetter(2))
    sorted_mem_stats_list_exception = sorted(mem_stats_list_exception, key=operator.itemgetter(2))

    return sorted_mem_stats_list, sorted_mem_stats_list_exception

def show_memory_stats(trace_line, pid, requested_mem_str, allocated_mem_str):
    requested_mem_bytes = -1
    allocated_mem_bytes = -1

    mem_stats_list           = []
    mem_stats_list_exception = []

    for trace_ev in trace_line:
      if (trace_ev[EV_PID] == pid and trace_ev[EV_PH] == EV_PH_OBJECT_SNAPSHOT and 
            trace_ev[EV_CAT] == EV_CAT_TENSOR):
            tensor_name = trace_ev[EV_NAME]
            args = trace_ev[EV_ARGS]

            snapshot_desc = args[EV_ARGS_SNAPSHOT]
            tensor_desc = snapshot_desc[EV_ARGS_SNAPSHOT_TENSOR_DESC]
            #print('>>>', tensor_name)
            #print(tensor_desc)

            # print(type(tensor_desc))
            if tensor_desc.find(EV_ARGS_SNAPSHOT_TENSOR_DESC_ALLOC_DESC) > 0:
                tensor_desc_list = tensor_desc.split(' ')
                #print(tensor_desc_list)

                i = tensor_desc_list.index(requested_mem_str)
                requested_mem_bytes = tensor_desc_list[i+1]

                i = tensor_desc_list.index(allocated_mem_str)
                allocated_mem_bytes = tensor_desc_list[i+1]

                #print(requested_mem_bytes, allocated_mem_bytes)
                mem_stats_list.append((tensor_name, int(requested_mem_bytes), int(allocated_mem_bytes)))
            else:
                mem_stats_list.append((tensor_name, 0, 0))
                mem_stats_list_exception.append(tensor_name)
                #print(tensor_name)

    sorted_mem_stats_list = sorted(mem_stats_list, key=operator.itemgetter(2))
    sorted_mem_stats_list_exception = sorted(mem_stats_list_exception, key=operator.itemgetter(2))

    return sorted_mem_stats_list, sorted_mem_stats_list_exception

def show_list(list_item):
    for name, req_bytes, alloc_bytes in list_item:
        s = "{0:100s} {1:12d} {2:12d}".format(name, req_bytes, alloc_bytes)
        print(s)

def show_operation(trace_line, pid, mem_stats_list):
    ops_list = []
    mem_stats_dict = OrderedDict()
    for name, req_bytes, alloc_bytes in mem_stats_list:
        mem_stats_dict[name] = int(alloc_bytes)

    for trace_ev in trace_line:
        if (trace_ev[EV_PID] == pid and 
            trace_ev[EV_PH] == EV_PH_COMPLETE and 
            trace_ev[EV_CAT] == EV_CAT_OP ):
            ts   = int(trace_ev[EV_TS])
            dur  = int(trace_ev[EV_DUR])

            args = trace_ev[EV_ARGS]
            name = args[EV_ARGS_NAME]
            #if name.find("edge") != -1:
            if "edge" in name:
              name = re.sub("edge_\d*_", '', name)
            op   = args[EV_ARGS_OP]

            alloc_bytes = mem_stats_dict[name]
            if (alloc_bytes < 0):
                alloc_bytes = 0 
            
            ops_list.append((ts, op, name, dur, alloc_bytes))
    
    return ops_list
    

def plot_mem_stats(mem_stats_list):
    tensor_names    = []
    req_mem_bytes   = []
    alloc_mem_bytes = []

    for name, req_bytes, alloc_bytes in mem_stats_list:
        tensor_names.append(name)
        req_mem_bytes.append(int(req_bytes/ONE_MEGA))
        alloc_mem_bytes.append(int(alloc_bytes/ONE_MEGA))
        #req_mem_bytes.append(req_bytes)
        #alloc_mem_bytes.append(alloc_bytes)

    print('req bytes', sum(req_mem_bytes))
    print('alloc bytes', sum(alloc_mem_bytes))
    print('diff', sum(alloc_mem_bytes) - sum(req_mem_bytes))

    plt.grid(True)
    plt.plot(req_mem_bytes, 'g-', alloc_mem_bytes, 'r-')

    #y_pos = np.arange(len(tensor_names))
    #plt.barh(y_pos, tensor_names, align='center', alpha = 0.5)
    #plt.yticks(y_pos, tensor_names)

    plt.show()

def memcpy_stats(filename):
	with open(filename, 'rt') as trace_file:
		trace_data = json.load(trace_file, object_pairs_hook=OrderedDict)
		trace_line = trace_data['traceEvents']

		memcpy_args_name_list = extract_memcpy_htod_args_name(trace_line, EV_NAME_MEMCPY_HTOD)
		mem_stats_list, mem_stats_list_exception = show_memcpy_memory_stats(trace_line, 
                          requested_mem_str = EV_ARGS_SNAPSHOT_TENSOR_DESC_ALLOC_DESC_REQ_BYTES,
                          allocated_mem_str = EV_ARGS_SNAPSHOT_TENSOR_DESC_ALLOC_DESC_ALLOC_BYTES)

		#show_list(mem_stats_list)

		pid_memcpy = extract_pid(trace_line, EV_NAME_MEMCPY_HTOD, EV_PH_COMPLETE, "")
		#print(pid_memcpy)

		ops_list = show_operation(trace_line, pid_memcpy, mem_stats_list)
		len_ops_list = len(ops_list)
		integrated_dur = ops_list[0][3]
		for i in range(0, len_ops_list):
			cur_ops = ops_list[i]
			cur_ts, cur_op, cur_name, cur_dur, cur_alloc_bytes = cur_ops
			if i < len_ops_list-1:
				next_ops = ops_list[i+1]
				next_ts, next_op, next_name, next_dur, next_alloc_bytes = next_ops
				    
				if cur_name == next_name:
					integrated_dur += next_dur
				else:
					cur_datetime = datetime.datetime.fromtimestamp(cur_ts/1000000)
					s = "{0:30s} {1:30s} {2:100s} {3:6d} {4:16d}".format(str(cur_datetime), cur_op, cur_name, integrated_dur, cur_alloc_bytes)
					print(s)
					integrated_dur = next_dur
			elif i == len_ops_list-1:
				cur_datetime = datetime.datetime.fromtimestamp(cur_ts/1000000)
				s = "{0:30s} {1:30s} {2:100s} {3:6d} {4:16d}".format(str(cur_datetime), cur_op, cur_name, integrated_dur, cur_alloc_bytes)
				print(s)

def compute_stats(filename, ev_name, ev_ph, ev_args_name_compute, ev_args_name_tensor):
    with open(filename, 'rt') as trace_file:
        trace_data = json.load(trace_file, object_pairs_hook=OrderedDict)
        trace_line = trace_data['traceEvents']
        # print(trace_line)

        pid_compute = extract_pid(trace_line, ev_name, ev_ph, ev_args_name_compute)
        #print('pid_gpu_all_compute=', pid_gpu_all_compute)
        pid_tensor = extract_pid(trace_line, ev_name, ev_ph, ev_args_name_tensor)
        #print('pid_gpu_tensor=', pid_gpu_tensor)
        mem_stats_list, mem_stats_list_exception = show_memory_stats(
                          trace_line, 
                          pid_tensor, 
                          requested_mem_str = EV_ARGS_SNAPSHOT_TENSOR_DESC_ALLOC_DESC_REQ_BYTES,
                          allocated_mem_str = EV_ARGS_SNAPSHOT_TENSOR_DESC_ALLOC_DESC_ALLOC_BYTES)
        
        #show_list(mem_stats_list)
        #print('<<<<----------------------->>>>')
        #for name in mem_stats_list_exception:
        #    print(name)
        #plot_mem_stats(mem_stats_list)

        ops_list = show_operation(trace_line, pid_compute, mem_stats_list)
        len_ops_list = len(ops_list)
        integrated_dur = ops_list[0][3]
        for i in range(0, len_ops_list):
          cur_ops = ops_list[i]
          cur_ts, cur_op, cur_name, cur_dur, cur_alloc_bytes = cur_ops
          if i < len_ops_list-1:
            next_ops = ops_list[i+1]
            next_ts, next_op, next_name, next_dur, next_alloc_bytes = next_ops
				    
            if cur_name == next_name:
              integrated_dur += next_dur
            else:
              cur_datetime = datetime.datetime.fromtimestamp(cur_ts/1000000)
              s = "{0:30s} {1:30s} {2:100s} {3:6d} {4:16d}".format(str(cur_datetime), cur_op, cur_name, integrated_dur, cur_alloc_bytes)
              print(s)
              integrated_dur = next_dur
          elif i == len_ops_list-1:
            cur_datetime = datetime.datetime.fromtimestamp(cur_ts/1000000)
            s = "{0:30s} {1:30s} {2:100s} {3:6d} {4:16d}".format(str(cur_datetime), cur_op, cur_name, integrated_dur, cur_alloc_bytes)
            print(s)
        '''
        for ts, op, name, dur, alloc_bytes in ops_list:
            #s = "{0:30s} {1:100s} {2:16d} {3:6d} {4:16d}".format(op,name,ts,dur, alloc_bytes)
            s = "{0:16d} {1:30s} {2:100s} {3:6d} {4:16d}".format(ts, op, name, dur, alloc_bytes)
            print(s)
        '''


if __name__  == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', 
        type=str, 
        help="trace file to be parsed")

    args = parser.parse_args()
    # print(args.json_file)
    memcpy_stats(args.json_file)
    print("===================================================")
    compute_stats(args.json_file, EV_NAME_PROC_NAME, EV_PH_METADATA, EV_ARGS_NAME_GPU_ALL_COMPUTE, EV_ARGS_NAME_GPU_TENSORS)
