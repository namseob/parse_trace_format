import operator
import re
from . import constant
from . import extract_layer_stat

MB = 1024*1024

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
		if tensor_event[constant.EV_PH] == constant.OBJECT_SNAPSHOT:
			snapshot_tensor_list.append(tensor_event)
	
	return snapshot_tensor_list

def extract_req_and_alloc_bytes_for_tensor(tensor_desc):
	alloc_mem_bytes = -1
	req_mem_bytes = -1
	tensor_desc = tensor_desc.split()

	try:
		r_i = tensor_desc.index(constant.REQ_BYTES)
		req_mem_bytes = tensor_desc[r_i+1]
		a_i = tensor_desc.index(constant.ALLOC_BYTES)
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
		op_name = tensor_event[constant.EV_NAME]
		tensor_desc = tensor_event[constant.EV_ARGS][constant.SNAPSHOT][constant.TENSOR_DESC]
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
		if event[constant.EV_PH] == constant.METADATA:
			args = event[constant.EV_ARGS]

			if args[constant.EV_NAME] == desc:
				pid = event[constant.EV_PID]
				return pid
	return pid

def extract_events_for_pid(events, pid):
	event_list = []
	for event in events:
		if event[constant.EV_PID] == pid and event[constant.EV_PH] != constant.METADATA:
			event_list.append(event)
	
	return sorted(event_list, key=operator.itemgetter('ts'))

	
#######################################################################
# output : list[op_with_mem_stat tuple]
#######################################################################
def extract_mem_stat_for_ops(ops_list, tensor_list):
	snapshot_tensor_list = extract_snapshot_tensor(tensor_list)
	ops_with_memstat_list = []
	
	start_ts = ops_list[0][constant.EV_TS]
	total_dur = ops_list[0][constant.EV_DUR]
	len_ops_list = len(ops_list)
	for i in range(0, len_ops_list):
		op_event = ops_list[i]
		if op_event[constant.EV_PH] == constant.COMPLETE:
			#print(op_event)
			op_name = op_event[constant.EV_ARGS][constant.EV_NAME]
			if "edge" in op_name:
				op_name = re.sub("edge_\d*_", '', op_name)

			#print(op_name)
			#print(tensor_event[EV_NAME])
			for tensor_event in snapshot_tensor_list:
				if op_name == tensor_event[constant.EV_NAME]:
					tensor_desc = tensor_event[constant.EV_ARGS][constant.SNAPSHOT][constant.TENSOR_DESC]
					#print(tensor_desc)

					try:
						if i < len_ops_list-1:
							cur_op_event = op_event
							cur_op_name = op_name

							next_op_event = ops_list[i+1]
							next_op_name = next_op_event[constant.EV_ARGS][constant.EV_NAME]
							if "edge" in next_op_name:
								next_op_name = re.sub("edge_\d*_", '', next_op_name)
							next_op_dur = next_op_event[constant.EV_DUR]

							req_mem_bytes, alloc_mem_bytes = extract_req_and_alloc_bytes_for_tensor(tensor_desc)
							#print(cur_op_name, total_dur)

							if cur_op_name == next_op_name:
								total_dur += next_op_dur
							else:
								ops_with_memstat_list.append((start_ts, op_event[constant.EV_NAME], op_event[constant.EV_ARGS][constant.EV_NAME], total_dur, alloc_mem_bytes, req_mem_bytes))

								start_ts = next_op_event[constant.EV_TS]
								total_dur = next_op_dur
						elif i == len_ops_list-1:
							ops_with_memstat_list.append((start_ts, op_event[constant.EV_NAME], op_event[constant.EV_ARGS][constant.EV_NAME], total_dur, alloc_mem_bytes, req_mem_bytes))
							
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
				if op_name == sch_event[constant.EV_ARGS][constant.EV_NAME]:
					keys = sch_event[constant.EV_ARGS].keys()
					input_list = []
					for key in keys:
						if key.find("input") != -1:
							#print(key)
							input_list.append(sch_event[constant.EV_ARGS][key])
				
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

#def plot_gpu_memory(events, merge_ops_list):
def extract_memory_with_ops(events, merge_ops_list):
	# x : ts
	# y : memory
	gpu_memory_list = []
	max_gpu_memory = 0
	for event in events:
		try:
			if event[constant.EV_CAT] == constant.MEMORY and event[constant.EV_NAME] == constant.GPU:
				ts = int(event[constant.EV_TS])
				gpu_memory = int(event[constant.EV_ARGS][constant.GPU])
				gpu_memory_list.append((ts, gpu_memory))
				if max_gpu_memory < gpu_memory:
					max_gpu_memory = gpu_mempry
		except:
			#print(event)
			pass
	
	sorted_by_ts_gpu_memory_list = sorted(gpu_memory_list, key=operator.itemgetter(0))
	sorted_by_ts_merge_ops_list = sorted(merge_ops_list, key=operator.itemgetter(0))

	memory_with_ops_list = []
	i = 0
	for ts, gpu_memory in sorted_by_ts_gpu_memory_list:
		ops_ts = sorted_by_ts_merge_ops_list[i][0]
		op_name = ""
		if ops_ts <= ts:
			op_name = sorted_by_ts_merge_ops_list[i][2]
			i += 1
		memory_with_ops_list.append(list((ts, gpu_memory, op_name)))

	#print_list(memory_with_ops_list)
	return memory_with_ops_list
