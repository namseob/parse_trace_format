from . import constant

def vgg(ops_list):
	layer_stats_list = []

	prev_op = ops_list[0]
	dur = prev_op[3]
	m_u = prev_op[8]
	for i in range(1, len(ops_list)):
		prev_op = ops_list[i-1]
		prev_op_name = prev_op[2]
		cur_op = ops_list[i]
		cur_op_name = cur_op[2]
		if cur_op_name.find('edge') == -1:
			splited_prev_op_name = prev_op_name.split('/')
			splited_cur_op_name = cur_op_name.split('/')
			print(splited_prev_op_name, splited_cur_op_name)
			if splited_cur_op_name[0].find('vgg_19') != -1:
				#print(prev_op_name, cur_op_name)
				#print(prev_op[3], cur_op[3])
				try:
					if splited_cur_op_name[2] == splited_prev_op_name[2]:
						dur += cur_op[3]
						m_u += cur_op[8]
					else:
						layer_stats_list.append((splited_prev_op_name[2], dur/1000, m_u))
						dur = cur_op[3]
						m_u = cur_op[8]
				except:
					pass
			elif splited_cur_op_name[0].find('softmax') != -1:
				dur += cur_op[3]
				m_u += cur_op[8]

			elif splited_cur_op_name[0].find('gradients') != -1:
				if splited_prev_op_name[0].find('softmax') != -1:
					layer_stats_list.append(('fc8', dur/1000, m_u))
					continue
				if splited_cur_op_name[3] == splited_prev_op_name[3]:
					dur += cur_op[3]
					m_u += cur_op[8]
				else:
					layer_stats_list.append(('b_'+splited_prev_op_name[2], dur/1000, m_u))
					dur = cur_op[3]
					m_u = cur_op[8]
	layer_stats_list.append(('b_'+splited_cur_op_name[2], dur/1000, m_u))

	return layer_stats_list

def extract_group_stats(ops_list, mode):
	block_stats_list = []
	unit_stats_list = []

	if mode == 'forward':
		prefix = ''
	elif mode == 'backward':
		prefix = 'b_'

	### front portion
	c1_ac=0; c1_dur=0
	p1_ac=0; p1_dur=0
	for op in ops_list:
		op_name=op[2]; op_dur=op[3]; op_ac=op[4]; op_bw=op[9]
		if op_name.find(EDGE) == -1:	# edge_ops filtering
			if (mode == 'forward' and op_name.find(GRADIENTS) == -1) or (mode == 'backward' and op_name.find(GRADIENTS) >= 0):
				if op_name.find(MODEL+'/conv1') != -1:
					if op_name.find(BATCHNORM) != -1 or op_name.find(CONV2D) != -1:
						c1_ac += op_ac
					c1_dur += op_dur
				elif op_name.find(MODEL+'/pool1') != -1:
					p1_ac += op_ac
					p1_dur += op_dur
		
	block_stats_list.append((prefix+'conv1', c1_dur/1000, c1_ac/MB))
	unit_stats_list.append((prefix+'conv1', c1_dur/1000, c1_ac/MB))
	block_stats_list.append((prefix+'pool1', p1_dur/1000, p1_ac/MB))
	unit_stats_list.append((prefix+'pool1', p1_dur/1000, p1_ac/MB))

	### middle portion
	dur=0; ac=0
	b_dur=0; b_ac=0
	for b_i in range(1, 5):
		for u_i in range(1, 37):
			for op in ops_list:
				op_name=op[2]; op_dur=op[3]; op_ac=op[4]; op_bw=op[9]
				if op_name.find(EDGE) == -1:	# edge_ops filtering
					if (mode == 'forward' and op_name.find(GRADIENTS) == -1) or (mode == 'backward' and op_name.find(GRADIENTS) >= 0):
						if op_name.find(BLOCK+str(b_i)+'/'+UNIT+'_'+str(u_i)+'/') != -1: #block#/unit_#
							if op_name.find(BATCHNORM) != -1 or op_name.find(CONV2D) != -1:
								ac += op_ac
							dur += op_dur	
			if dur !=0 or ac !=0:
				unit_stats_list.append((prefix+BLOCK+str(b_i)+'/'+UNIT+'_'+str(u_i), dur/1000, ac/MB))
				b_dur += dur; b_ac += ac
				dur=0; ac=0
		if b_dur != 0 or b_ac !=0:
			block_stats_list.append((prefix+BLOCK+str(b_i), b_dur/1000, b_ac/MB))
			b_dur=0; b_ac=0
	
	### end portion
	p5_ac=0; p5_dur=0
	fc_ac=0; fc_dur=0
	for op in ops_list:
		op_name=op[2]; op_dur=op[3]; op_ac=op[4]; op_bw=op[9]
		if op_name.find(EDGE) == -1:	# edge_ops filtering
			if (mode == 'forward' and op_name.find(GRADIENTS) == -1) or (mode == 'backward' and op_name.find(GRADIENTS) >= 0):
				if op_name.find(MODEL+'/logits') != -1:
					if op_name.find(BATCHNORM) != -1 or op_name.find(CONV2D) != -1:
						fc_ac += op_ac
					fc_dur += op_dur
				elif op_name.find(MODEL+'/pool5') != -1:
					p5_ac += op_ac
					p5_dur += op_dur
		
	block_stats_list.append((prefix+'pool5', p5_dur/1000, p5_ac/MB))
	unit_stats_list.append((prefix+'pool5', p5_dur/1000, p5_ac/MB))
	block_stats_list.append((prefix+'logits', fc_dur/1000, fc_ac/MB))
	unit_stats_list.append((prefix+'logits', fc_dur/1000, fc_ac/MB))

	return block_stats_list, unit_stats_list

def resnet(ops_list):
	block_stats_list, unit_stats_list = extract_group_stats(ops_list, 'forward')
	b_block_stats_list, b_unit_stats_list = extract_group_stats(ops_list, 'backward')
	block_stats_list.extend(b_block_stats_list)
	unit_stats_list.extend(b_unit_stats_list)
	#layers_stats_list.extend(extract_layer_stats(ops_list, 'backward'))

	return block_stats_list, unit_stats_list
