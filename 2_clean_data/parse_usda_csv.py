import pandas
import numpy as np
import math

year_index = 0
state_index = 1
county_index = 2
value_index = 3



data = np.genfromtxt('soybean_yield_filter.csv', delimiter=',')
for line in data:
	if(math.isnan(line[county_index])):
		continue
	
