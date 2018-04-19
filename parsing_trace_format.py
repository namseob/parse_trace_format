import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

from collections import OrderedDict
from trace_util import parsing, constant, util



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
		gpu_schedule_pid = parsing.extract_pid_for_process_name(events_dict, constant.GPU_SCHEDULE)
		gpu_memcpy_pid = parsing.extract_pid_for_process_name(events_dict, constant.GPU_MEMCPY)
		gpu_compute_pid = parsing.extract_pid_for_process_name(events_dict, constant.GPU_COMPUTE)
		gpu_tensor_pid = parsing.extract_pid_for_process_name(events_dict, constant.GPU_TENSOR)
		cpu_compute_pid = parsing.extract_pid_for_process_name(events_dict, constant.CPU_COMPUTE)
		cpu_tensor_pid = parsing.extract_pid_for_process_name(events_dict, constant.CPU_TENSOR)
		allocator_pid = parsing.extract_pid_for_process_name(events_dict, constant.ALLOCATOR)

		gpu_schedule_list = parsing.extract_events_for_pid(events_dict, gpu_schedule_pid)
		gpu_memcpy_list = parsing.extract_events_for_pid(events_dict, gpu_memcpy_pid)
		gpu_compute_list = parsing.extract_events_for_pid(events_dict, gpu_compute_pid)
		gpu_tensor_list = parsing.extract_events_for_pid(events_dict, gpu_tensor_pid)
		cpu_compute_list = parsing.extract_events_for_pid(events_dict, cpu_compute_pid)
		cpu_tensor_list = parsing.extract_events_for_pid(events_dict, cpu_tensor_pid)
		allocator_list = parsing.extract_events_for_pid(events_dict, allocator_pid)


		#### extract memstat, metrics ####
		gpu_memcpy_ops_with_memstat_list = parsing.extract_mem_stat_for_ops(gpu_memcpy_list, cpu_tensor_list)
		gpu_memcpy_ops_with_memstat_list.extend(parsing.extract_mem_stat_for_ops(gpu_memcpy_list, gpu_tensor_list))
		#print_list(gpu_memcpy_ops_with_memstat_list)
		gpu_memcpy_ops_with_read_write_amount_list = parsing.extract_read_write_amount_for_memcpy_ops(gpu_memcpy_ops_with_memstat_list)

		gpu_compute_ops_with_memstat_list = parsing.extract_mem_stat_for_ops(gpu_compute_list, gpu_tensor_list)
		gpu_compute_ops_with_inputs_list = parsing.extract_inputs_for_ops(gpu_schedule_list, gpu_compute_ops_with_memstat_list)
		gpu_compute_ops_with_read_write_amount_list = parsing.extract_read_write_amount_for_compute_ops(gpu_compute_ops_with_inputs_list, gpu_tensor_list, cpu_tensor_list)
		
		merge_ops_with_read_write_amount_list = gpu_memcpy_ops_with_read_write_amount_list + gpu_compute_ops_with_read_write_amount_list
		merge_ops_with_metrics_list = parsing.extract_metrics_for_ops(merge_ops_with_read_write_amount_list)
		#print_list(merge_ops_with_metrics_list)

		memory_with_ops_list = parsing.extract_memory_with_ops(events_dict, merge_ops_with_metrics_list)

		util.write_to_excel(merge_ops_with_metrics_list, memory_with_ops_list, args.json_file.split('.')[0] + '.xlsx')
