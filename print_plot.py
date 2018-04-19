import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

def main(csv_file):
	with open(csv_file) as f:
		df = pd.read_csv(f)	#convert DataFrame
	
		x = range(0, df.shape[0])
		ts_x = []
		for ts in df[df.columns[0]]:
			ts_x.append(ts)

		y = []
		for mem_used in df[df.columns[2]]:
			y.append(int(mem_used.replace('MiB','')))
		
		plt.xlabel("Time(ms)")
		plt.ylabel("Memory usage(MB)")
		plt.plot(x, y)
	
		prev = 0
		i = 0
		for a,b in zip(x, y):
			if b >= 1.5 * prev:
				#plt.text(a,b,str(a) + "," + str(b))
				plt.text(a,b,str(ts_x[i]) + "," + str(b))
				prev = b
			i+=1
		
	
		plt.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--csv_file', type=str, help="csv file to be parsed")

	args = parser.parse_args()
	main(args.csv_file)
