import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dist_metrics import *

PLOT_COLORS = ['#27AE60', '#D91E18', '#3498DB', '#F89406']
START_DAY = 49
DAYS_PER_PIC = 8
YEAR = 2013

def convert_day_of_year_to_datetime(year, day_of_year):
    return datetime.datetime(year, 1, 1) + datetime.timedelta(day_of_year - 1)

sal_map_data = np.load('./poster/sal_map_stats/original_model_comparison_imp_count.npz')
index = sal_map_data['index']
soy_maps, corn_maps = sal_map_data['soy_maps'], sal_map_data['corn_maps']

print 'Avg L2 Diff:', l2_dist(soy_maps, corn_maps)
print 'Avg L1 Diff:', l1_dist(soy_maps, corn_maps)
print 'Avg Perc. Diff:', perc_dist(soy_maps, corn_maps)

# Relative Importance of Photos Over Time
days_of_year = START_DAY+DAYS_PER_PIC*np.arange(soy_maps.shape[2])
dates = [convert_day_of_year_to_datetime(YEAR, day) for day in days_of_year]
fig, ax = plt.subplots()
ax.plot_date(dates, l2_dist(soy_maps, corn_maps, axis=(0,1,3)), '-', color=PLOT_COLORS[0], linewidth=2, label='L2 Dist')
ax.plot_date(dates, l1_dist(soy_maps, corn_maps, axis=(0,1,3)), '-', color=PLOT_COLORS[1], linewidth=2, label='L1 Dist')

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b, \'%y'))
ax.set_xlim(datetime.date(YEAR, 1, 1), datetime.date(YEAR, 12, 31))
ax.set_xlabel('Time of Year', fontsize=24)
ax.set_ylabel('Average L2/L1 Difference', fontsize=24)
ax.legend()
plt.show()

# Relative Importance of Bands
bar_width = 0.45
bands = np.arange(1, soy_maps.shape[3] + 1)
plt.bar(bands, l2_dist(soy_maps, corn_maps, axis=(0,1,2)), bar_width, color=PLOT_COLORS[2], alpha=0.75, label='L2 Dist')
plt.bar(bands + bar_width, l1_dist(soy_maps, corn_maps, axis=(0,1,2)), bar_width, color=PLOT_COLORS[3], alpha=0.75, label='L1 Dist')
plt.xlim(1,9)
plt.xlabel('Band', fontsize=24)
plt.ylabel('Average L2/L1 Difference', fontsize=24)
plt.legend()
plt.show()

plt.bar(np.arange(soy_maps.shape[1]), l2_dist(soy_maps, corn_maps, axis=(0,2,3)), color=PLOT_COLORS[0], alpha=0.75)
plt.show()

plt.hist(l2_dist(soy_maps, corn_maps, axis=(1,2,3)), bins=30, color=PLOT_COLORS[2], alpha=0.75)
plt.hist(l1_dist(soy_maps, corn_maps, axis=(1,2,3)), bins=30, color=PLOT_COLORS[3], alpha=0.75)
plt.show()