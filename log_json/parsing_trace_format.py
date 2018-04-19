import argparse
import json
import operator
import re
import pandas as pd

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

MB = 1024*1024
'''
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
'''

def print_list(show_list):
	for val in show_list:
		print(val)
		
def write_to_excel(ops_list, filename, columns_format=[]):
	if len(columns_format) != 0:
		values = pd.DataFrame(ops_list, columns=columns_format)
		values.to_excel(filename, sheet_name='Sheet1', columns=columns_format, startrow=0, startcol=0)
	else:
		values = pd.DataFrame(ops_list)
		values.to_excel(filename, sheet_name='Sheet1', startrow=0, startcol=0)

def extract_refined_op_name(input_op_name):
	new_input_op_name = ""
	#input_op_name = re.sub("edge_\d*_", '', input_op_name)
	for i in range(len(input_op_name)-1, 0, -1):
		if input_op_name[i] == '/':
			new_input_op_name = input_op_name[0:i]
			break

	return new_input_op_name

def extract_snapshot_tensor(tensor_list):
	snapshot_tensor_list = []
	for tensor_event in tensor_list:
		if tensor_event[EV_PH] == OBJECT_SNAPSHOT:
			snapshot_tensor_list.append(tensor_event)
	
	return snapshot_tensor_list

def extract_req_and_alloc_bytes_for_tensor(tensor_desc):
	alloc_mem_bytes = -1
	req_mem_bytes = -1
	tensor_desc = tensor_desc.split()

	try:
		r_i = tensor_desc.index(REQ_BYTES)
		req_mem_bytes = tensor_desc[r_i+1]
		a_i = tensor_desc.index(ALLOC_BYTES)
		alloc_mem_bytes = tensor_desc[a_i+1]
	except:
		#print(tensor_desc)
		alloc_mem_bytes = req_mem_bytes
		pass

	return int(req_mem_bytes), int(alloc_mem_bytes)

def extract_req_bytes_for_ops(target_op_name, tensor_list):
	req_bytes = -1

	snapshot_tensor_list = extract_snapshot_tensor(tensor_list)
	for tensor_event in snapshot_tensor_list:
		#print(tensor_event)
		op_name = tensor_event[EV_NAME]
		tensor_desc = tensor_event[EV_ARGS][SNAPSHOT][TENSOR_DESC]
		#print(op_name, tensor_desc)
		if op_name == target_op_name:
			req_bytes, alloc_bytes = extract_req_and_alloc_bytes_for_tensor(tensor_desc)
			#print(req_bytes)
	
	return req_bytes
	
#######################################################################
# output : list[op dict]
#######################################################################
def extract_pid_for_process_name(events, desc):
	pid = -1

	for event in events:
		if event[EV_PH] == METADATA:
			args = event[EV_ARGS]

			if args[EV_NAME] == desc:
				pid = event[EV_PID]
				return pid
	return pid

def extract_events_for_pid(events, pid):
	event_list = []
	for event in events:
		if event[EV_PID] == pid and event[EV_PH] != METADATA:
			event_list.append(event)
	
	return sorted(event_list, key=operator.itemgetter('ts'))

	
#######################################################################
# output : list[op_with_mem_stat tuple]
#######################################################################
def extract_mem_stat_for_ops(ops_list, tensor_list):
	snapshot_tensor_list = extract_snapshot_tensor(tensor_list)
	ops_with_memstat_list = []
	
	start_ts = ops_list[0][EV_TS]
	total_dur = ops_list[0][EV_DUR]
	len_ops_list = len(ops_list)
	for i in range(0, len_ops_list):
		op_event = ops_list[i]
		if op_event[EV_PH] == COMPLETE:
			#print(op_event)
			op_name = op_event[EV_ARGS][EV_NAME]
			if "edge" in op_name:
				op_name = re.sub("edge_\d*_", '', op_name)

			#print(op_name)
			#print(tensor_event[EV_NAME])
			for tensor_event in snapshot_tensor_list:
				if op_name == tensor_event[EV_NAME]:
					tensor_desc = tensor_event[EV_ARGS][SNAPSHOT][TENSOR_DESC]
					#print(tensor_desc)

					try:
						if i < len_ops_list-1:
							cur_op_event = op_event
							cur_op_name = op_name

							next_op_event = ops_list[i+1]
							next_op_name = next_op_event[EV_ARGS][EV_NAME]
							if "edge" in next_op_name:
								next_op_name = re.sub("edge_\d*_", '', next_op_name)
							next_op_dur = next_op_event[EV_DUR]

							req_mem_bytes, alloc_mem_bytes = extract_req_and_alloc_bytes_for_tensor(tensor_desc)
							#print(cur_op_name, total_dur)

							if cur_op_name == next_op_name:
								total_dur += next_op_dur
							else:
								ops_with_memstat_list.append((start_ts, op_event[EV_NAME], op_event[EV_ARGS][EV_NAME], total_dur, alloc_mem_bytes, req_mem_bytes))

								start_ts = next_op_event[EV_TS]
								total_dur = next_op_dur
						elif i == len_ops_list-1:
							ops_with_memstat_list.append((start_ts, op_event[EV_NAME], op_event[EV_ARGS][EV_NAME], total_dur, alloc_mem_bytes, req_mem_bytes))
							
					except:
						#print(op_event, tensor_event)
						pass

	#print_list(ops_with_memstat_list)
	return ops_with_memstat_list

#######################################################################
# output : list[op_with_mem_stat tuple, input_list]
#######################################################################
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
				#print(op_event, sch_event)
				pass
				
	#print_list(ops_with_inputs_list)
	return ops_with_inputs_list

#######################################################################
# output : list[op_with_mem_stat tuple, input_list, read/write tuple]
#######################################################################
def extract_read_write_amount_for_memcpy_ops(memcpy_ops_list):
	ops_with_read_write_amount_list = []
	for op_with_memstat in memcpy_ops_list:
		write_amout = 0
		read_amount = 0

		req_bytes = int(op_with_memstat[5])
		write_amount = req_bytes
		read_amount = write_amount

		op_with_read_write_amount = []
		op_with_read_write_amount.append(op_with_memstat)
		op_with_read_write_amount.append([])
		op_with_read_write_amount.append((read_amount, write_amount))

		ops_with_read_write_amount_list.append(op_with_read_write_amount)

	return ops_with_read_write_amount_list

#######################################################################
# output : list[op_with_mem_stat tuple, input_list, read/write tuple]
#######################################################################
def extract_read_write_amount_for_compute_ops(compute_ops_with_inputs_list, gpu_tensor_list, cpu_tensor_list):
	ops_with_read_write_amount_list = []
	for op_with_inputs in compute_ops_with_inputs_list:
		read_amount = 0
		write_amount = 0

		op_with_memstat = op_with_inputs[0]
		req_bytes = int(op_with_memstat[5])
		#print(req_bytes)
		write_amount = req_bytes

		input_list = op_with_inputs[1]
		#print(input_list)
		for input_op_name in input_list:
			#print(input_op_name)
			input_req_bytes = extract_req_bytes_for_ops(input_op_name, gpu_tensor_list)
			if input_req_bytes == -1:
				#print(input_op_name)
				refined_input_op_name = extract_refined_op_name(input_op_name)
				#print(refined_input_op_name)
				input_req_bytes = extract_req_bytes_for_ops(refined_input_op_name, cpu_tensor_list)
			#print(input_req_bytes)
				
			read_amount += input_req_bytes
		#print(read_amount, write_amount)

		op_with_read_write_amount = []
		op_with_read_write_amount.extend(op_with_inputs)
		op_with_read_write_amount.append((read_amount, write_amount))
		#print(op_with_read_write_amount)
		ops_with_read_write_amount_list.append(op_with_read_write_amount)

	return ops_with_read_write_amount_list


##########################################################################################
# output : list[ts, op_type, op_name, total_dur, allocated_memory, input_list, read memory, write memory, memory_usage, memory_bandwidth]
##########################################################################################
def extract_metrics_for_ops(ops_with_read_write_amount_list):
	op_with_metrics_list = []

	for op_with_read_write_amount in ops_with_read_write_amount_list:
		op_with_memstat_tuple = op_with_read_write_amount[0]
		input_list = op_with_read_write_amount[1]
		read_write_tuple = op_with_read_write_amount[2]

		op_with_metric = []

		op_with_metric.extend(list(op_with_memstat_tuple[0:5])) #from 'ts' to 'alloc_mem'

		op_with_metric.append(input_list)
		op_with_metric.append(read_write_tuple[0]/MB)
		op_with_metric.append(read_write_tuple[1]/MB)

		memory_usage = (read_write_tuple[0] + read_write_tuple[1])/MB #MByte
		op_with_metric.append(memory_usage)

		total_dur = op_with_memstat_tuple[3]/1000000 #sec
		memory_bandwidth = memory_usage/total_dur
		op_with_metric.append(memory_bandwidth/1024) #GB/s
		#print(op_with_metric)

		op_with_metrics_list.append(op_with_metric)

	return op_with_metrics_list

	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--json_file', type=str)

	args = parser.parse_args()

	####
	with open(args.json_file, 'rt') as trace_file:
		data_dict = json.load(trace_file, object_pairs_hook=OrderedDict)
		events_dict = data_dict['traceEvents']

		'''
		#name_list = extract_name_list(events_dict, TENSOR)
		#print_list(name_list)
		'''
		#### extract events ####
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


		#### extract memstat, metrics ####
		gpu_memcpy_ops_with_memstat_list = extract_mem_stat_for_ops(gpu_memcpy_list, cpu_tensor_list)
		gpu_memcpy_ops_with_memstat_list.extend(extract_mem_stat_for_ops(gpu_memcpy_list, gpu_tensor_list))
		#print_list(gpu_memcpy_ops_with_memstat_list)
		gpu_memcpy_ops_with_read_write_amount_list = extract_read_write_amount_for_memcpy_ops(gpu_memcpy_ops_with_memstat_list)

		gpu_compute_ops_with_memstat_list = extract_mem_stat_for_ops(gpu_compute_list, gpu_tensor_list)
		gpu_compute_ops_with_inputs_list = extract_inputs_for_ops(gpu_schedule_list, gpu_compute_ops_with_memstat_list)
		gpu_compute_ops_with_read_write_amount_list = extract_read_write_amount_for_compute_ops(gpu_compute_ops_with_inputs_list, gpu_tensor_list, cpu_tensor_list)
		
		merge_ops_with_read_write_amount_list = gpu_memcpy_ops_with_read_write_amount_list + gpu_compute_ops_with_read_write_amount_list
		merge_ops_with_metrics_list = extract_metrics_for_ops(merge_ops_with_read_write_amount_list)
		print_list(merge_ops_with_metrics_list)

		columns_format = ['ts', 'op_type', 'op', 'total_dur(μs)', 'allocated_memory', 'input_list', 'read_memory(MB)', 'write_memory(MB)', 'memory_usage(MB)', 'memory_bandwidth(GB/s)']
		write_to_excel(merge_ops_with_metrics_list, args.json_file.split('.')[0] + '.xlsx', columns_format)
