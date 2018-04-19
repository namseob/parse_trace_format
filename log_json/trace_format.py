import argparse
import json
import operator
import re

from collections import OrderedDict

# event descriptions
EV_NAME = 'name'
EV_CAT	= 'cat'
EV_PH		= 'ph'
EV_TS		= 'ts'
EV_DUR	= 'dur'
EV_TTS	=	'tts'
EV_PID	=	'pid'
EV_TID	=	'tid'
EV_ARGS	=	'args'

'''
[ event types ]
Duration, Complete, Instant, Counter, Flow, Object, Metadata ...
'''
## event phase(EV_PH) lists for event types
DURATION_BEGIN = 'B'
DURATION_END		=	'E'

COMPLETE				=	'X'

INSTANT				=	'I'

COUNTER				=	'C'

FLOW_START	=	's'
FLOW_STEP	=	't'
FLOW_END	=	'f'

OBJECT_CREATED		=	'N'
OBJECT_SNAPSHOT	=	'O'
OBJECT_DESTROYED	=	'D'

METADATA					=	'M'

## event category(EV_CAT)
OP					=	'Op'
MEMORY			=	'Memory'  # have pid for allocator
DATAFLOW		=	'DataFlow'
TENSOR			=	'Tensor'

## event name(EV_NAME) lists
#### for 'OP'
PROCESS_NAME	=	'process_name'
MEMCPY_HTOD = 'MEMCPYHtoD'
MEMCPY_DTOH = 'MEMCPYDtoH'
#### for 'MEMORY'
GPU = 'GPU_0_bfc'
CUDA_HOST = 'cuda_host_bfc'
CPU = 'cpu'
#### for 'DATAFLOW'
#### for 'TENSOR'


## event args(EV_ARGS) lists
#### for 'METADATA'
'''
[METADATA args description]
name(value)
'''
GPU_SCHEDULE	=	'/job:localhost/replica:0/task:0/device:GPU:0 Compute'
GPU_COMPUTE 	= '/device:GPU:0/stream:all Compute'
GPU_MEMCPY 		= '/device:GPU:0/memcpy Compute'
GPU_TENSOR		=	'/job:localhost/replica:0/task:0/device:GPU:0 Tensors'

CPU_COMPUTE	=	'/job:localhost/replica:0/task:0/device:CPU:0 Compute'
CPU_TENSOR	=	'/job:localhost/replica:0/task:0/device:CPU:0 Tensors'

ALLOCATOR = 'Allocators'

#### for 'OBJECT_SNAPSHOT'
'''
[OBJECt_SNAPSHOT args description]
snapshot(dict)
'''
SNAPSHOT = 'snapshot'

####### args in snapshot
TENSOR_DESC = 'tensor_description'

####### args in tensor_desc
DTYPE = 'dtype'
SHAPE = 'shape'
ALLOC_DESC	=	'allocation_description'

######## allocation description
REQ_BYTES	=	'requested_bytes:'
ALLOC_BYTES	=	'allocated_bytes:'

def extract_pid_for_process_name(events, desc):
	pid = -1

	for event in events:
		if event[EV_PH] == METADATA:
			args = event[EV_ARGS]

			if args[EV_NAME] == desc:
				pid = event[EV_PID]
				return pid
	return pid

def extract_name_list(events, ev_cat):
	ev_cat_list = []
	for event in events:
		try:
			#print(event[EV_CAT], ev_cat)
			if event[EV_CAT] == ev_cat:
				#print(event)
				ev_cat_list.append(event[EV_NAME])
		except:
			pass
	
	return sorted(list(set(ev_cat_list)))

def extract_events_for_pid(events, pid):
	event_list = []
	for event in events:
		if event[EV_PID] == pid and event[EV_PH] != METADATA:
			event_list.append(event)
	
	return sorted(event_list, key=operator.itemgetter('ts'))

def extract_inputs_for_ops(schedule_list, ops_list):
	ops_with_inputs_list = []
	for op_event in ops_list:
		op_name = op_event[2]
		#print(op_name)
		for sch_event in schedule_list:
			try:
				if op_name == sch_event[EV_ARGS][EV_NAME]:
					keys = sch_event[EV_ARGS].keys()
					input_list = []
					for key in keys:
						if key.find("input") != -1:
							#print(key)
							input_list.append(sch_event[EV_ARGS][key])
				
					#print(input_list)
					op_with_input = (op_event, input_list)
					ops_with_inputs_list.append(op_with_input)
			except:
				pass
				
	print_list(ops_with_inputs_list)
	return ops_with_inputs_list
	
def extract_mem_stat_for_ops(ops_list, tensor_list):
	snapshot_tensor_list = []
	for tensor_event in tensor_list:
		if tensor_event[EV_PH] == OBJECT_SNAPSHOT:
			snapshot_tensor_list.append(tensor_event)
	
	#print_list(snapshot_tensor_list)
	ops_with_memstat_list = []
	
	for op_event in ops_list:
		if op_event[EV_PH] == COMPLETE:
			#print(op_event)
			op_name = op_event[EV_ARGS][EV_NAME]
			if "edge" in op_name:
				op_name = re.sub("edge_\d*_", '', op_name)

			#print(op_name)
			#print(tensor_event[EV_NAME])
			for tensor_event in snapshot_tensor_list:
				if op_name == tensor_event[EV_NAME]:
					#print(op_event)
					#print(tensor_event)
					
					tensor_desc = tensor_event[EV_ARGS][SNAPSHOT][TENSOR_DESC]
					tensor_desc = tensor_desc.split()
					#print(tensor_event[EV_ARGS][SNAPSHOT])
					#print(tensor_desc)

					try:
						i = tensor_desc.index(REQ_BYTES)
						req_mem_bytes = tensor_desc[i+1]
						i = tensor_desc.index(ALLOC_BYTES)
						alloc_mem_bytes = tensor_desc[i+1]
						ops_with_memstat_list.append((op_event[EV_TS], op_event[EV_NAME], op_event[EV_ARGS][EV_NAME], op_event[EV_DUR], req_mem_bytes, alloc_mem_bytes))
					except:
						pass

	#print_list(ops_with_memstat_list)
	
	return ops_with_memstat_list
	
def print_list(show_list):
	for val in show_list:
		print(val)
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--json_file', type=str)

	args = parser.parse_args()

	####
	with open(args.json_file, 'rt') as trace_file:
		data_dict = json.load(trace_file, object_pairs_hook=OrderedDict)
		#print(data_dict)
		events_dict = data_dict['traceEvents']
		#print(event_dict)

		#name_list = extract_name_list(events_dict, TENSOR)
		#print_list(name_list)

		gpu_schedule_pid = extract_pid_for_process_name(events_dict, GPU_SCHEDULE)
		gpu_memcpy_pid = extract_pid_for_process_name(events_dict, GPU_MEMCPY)
		gpu_compute_pid = extract_pid_for_process_name(events_dict, GPU_COMPUTE)
		gpu_tensor_pid = extract_pid_for_process_name(events_dict, GPU_TENSOR)
		cpu_compute_pid = extract_pid_for_process_name(events_dict, CPU_COMPUTE)
		cpu_tensor_pid = extract_pid_for_process_name(events_dict, CPU_TENSOR)
		allocator_pid = extract_pid_for_process_name(events_dict, ALLOCATOR)

		gpu_schedule_list = extract_events_for_pid(events_dict, gpu_schedule_pid)
		gpu_memcpy_list = extract_events_for_pid(events_dict, gpu_memcpy_pid)
		gpu_compute_list = extract_events_for_pid(events_dict, gpu_compute_pid)
		gpu_tensor_list = extract_events_for_pid(events_dict, gpu_tensor_pid)
		cpu_compute_list = extract_events_for_pid(events_dict, cpu_compute_pid)
		cpu_tensor_list = extract_events_for_pid(events_dict, cpu_tensor_pid)
		allocator_list = extract_events_for_pid(events_dict, allocator_pid)

		#print_list(gpu_schedule_list)
		#extract_mem_stat_for_ops(gpu_schedule_list, gpu_tensor_list)
		gpu_compute_ops_with_memstat_list = extract_mem_stat_for_ops(gpu_compute_list, gpu_tensor_list)
		gpu_memcpy_ops_with_memstat_list = extract_mem_stat_for_ops(gpu_memcpy_list, cpu_tensor_list)
		gpu_memcpy_ops_with_memstat_list.extend(extract_mem_stat_for_ops(gpu_memcpy_list, gpu_tensor_list))
		#print_list(gpu_memcpy_ops_with_memstat_list)
		extract_inputs_for_ops(gpu_schedule_list, gpu_compute_ops_with_memstat_list)
