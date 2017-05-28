import pandas
import numpy as np
import math

year_index = 0
state_index = 1
county_index = 2
value_index = 3

all_counties = np.genfromtxt('yield_final_highquality.csv', delimiter=',')

data = np.genfromtxt('soybean_yield_filter.csv', delimiter=',')
final_data = []
for line in data:
	if(math.isnan(line[county_index])):
		continue
	if line[:-1] in all_counties[:, :-1]:
		final_data.append(line)
	else:
		print line
final_data = np.unique(final_data)
np.savetxt('soybean_final.csv',final_data, fmt=['%d','%d','%d','%.1f'], delimiter=',')
