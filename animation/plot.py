import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import skimage

# background = cv2.imread('./save_image/gray_median.jpg')
# light = cv2.imread('./save_image/gray_light.jpg')
#
# background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
# light = cv2.cvtColor(light, cv2.COLOR_BGR2GRAY)
# match = skimage.exposure.match_histograms(light, background)
#
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# axes[1].xaxis.set_tick_params(labelsize=15)
# axes[1].yaxis.set_tick_params(labelsize=15)
# axes[0].imshow(match, cmap='gray')
# # axes[0].set_title('Grayscale Image')
# axes[1].hist(match.ravel(), 256, [0, 256])
# axes[1].set_title('Histogram distribution', size=20)
# axes[0].set_axis_off()
#
# plt.tight_layout()
# plt.show()

csv_file = pd.read_excel('./test.xlsx', index_col=0)

data = np.asarray(csv_file)[:, :-1]

mean = np.mean(data, axis=0)
std = np.std(data, axis=0)

fig, ax = plt.subplots()

# Set the bar labels
bar_labels = ['Ours', 'KF']

# Create the x position of the bars
x_pos = np.arange(len(bar_labels))

# Create the bars
# bars = ax.bar(x_pos, mean[10:12], yerr=std[10:12], align='center', alpha=0.7, ecolor='black', capsize=20, color=['blue', 'orange'], error_kw={'elinewidth':2, 'capsize':16})
bp = ax.boxplot(data[:, 8:10], patch_artist=True, notch=True, vert=1)

colors = ['#05A8AA', '#DC602E']  # LightSalmon for light orange, LightBlue for light blue

for i, patch in enumerate(bp['boxes']):
    color = colors[i]
    patch.set_facecolor('none')  # set facecolor to none for transparency
    patch.set_edgecolor(color)
    patch.set_linewidth(2)  # set edge line width

    # set whiskers color and line width
    bp['whiskers'][i * 2].set(color=color, linewidth=2)
    bp['whiskers'][i * 2 + 1].set(color=color, linewidth=2)

    # set caps color and line width
    bp['caps'][i * 2].set(color=color, linewidth=2)
    bp['caps'][i * 2 + 1].set(color=color, linewidth=2)

    # set medians color and line width
    bp['medians'][i].set(color=color, linewidth=2)

    # set fliers color
    bp['fliers'][i].set(marker='D', color=color)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('MAE (pixel)', fontsize=16)
ax.set_xticklabels(bar_labels, fontsize=16)
ax.set_title('ADE in right image', fontsize=20)
ax.tick_params(axis='y', labelsize=12)

# plt.show()
# Save the figure and show
plt.tight_layout()

# Save it in high resolution
plt.savefig('./save_image/test/ADE_right.png', dpi=300)

plt.close(fig)