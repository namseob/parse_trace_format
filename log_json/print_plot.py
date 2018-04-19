import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

with open('./gpu_log.csv') as f:
	df = pd.read_csv(f)	#convert DataFrame

	x = range(0, df.shape[0])
	y = []
	for row in df[df.columns[2]]:
		y.append(int(row.replace('MiB','')))
	
	plt.plot(x, y)
	plt.xlabel("Time(ms)")
	plt.ylabel("Memory usage(MB)")

	plt.show()
