import pandas as pd
from . import extract_layer_stat

def print_list(show_list):
	for val in show_list:
		print(val)

def write_to_excel(ops_list, memory_with_ops_list, filename):
	writer = pd.ExcelWriter(filename, engine = 'xlsxwriter')

	columns_format = ['ts', 'op_type', 'op', 'total_dur(μs)', 'allocated_memory', 'input_list', 'read_memory(MB)', 'write_memory(MB)', 'memory_usage(MB)', 'memory_bandwidth(GB/s)']
	ops_values = pd.DataFrame(ops_list, columns=columns_format)
	ops_values.to_excel(writer, sheet_name='metrics')

	columns_format = ['ts', 'memory_usage', 'op']
	memory_values = pd.DataFrame(memory_with_ops_list, columns=columns_format)
	memory_values.to_excel(writer, sheet_name = 'memory_usage')

	layer_stats_list = extract_layer_stat.vgg(ops_list)
	print_list(layer_stats_list)
	columns_format = ['layer', 'duration(ms)', 'memory_usage(MB)']
	layer_values = pd.DataFrame(layer_stats_list, columns=columns_format)
	layer_values.to_excel(writer, sheet_name = 'layer granularity')
	'''
	block_stats_list, unit_stats_list = extract_group_stats_for_resnet(ops_list)			
	columns_format = ['layer', 'duration(ms)', 'allocated_memory(MB)']
	block_values = pd.DataFrame(block_stats_list, columns=columns_format)
	block_values.to_excel(writer, sheet_name = 'block granularity')
	unit_values = pd.DataFrame(unit_stats_list, columns=columns_format)
	unit_values.to_excel(writer, sheet_name = 'unit granularity')
	'''
	writer.save()

#def plot_memory_usage(memory_with_ops_list):
