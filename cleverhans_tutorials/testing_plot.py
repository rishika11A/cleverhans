
import matplotlib.pyplot as plt
import numpy as np

nb_classes = 10
img_rows = 32
img_cols = 32
nchannels = 3

grid_shape = (nb_classes, nb_classes, img_rows, img_cols,
                    nchannels)
grid_viz_data = np.zeros(grid_shape, dtype='f')
grid_viz_data_1 = np.zeros(grid_shape, dtype='f')

plt.ioff()
figure = plt.figure()
figure.canvas.set_window_title('Cleverhans: Grid Visualization')

# Add the images to the plot
num_cols = grid_viz_data.shape[0]
num_rows = grid_viz_data.shape[1]
num_channels = grid_viz_data.shape[4]
for y in range(num_rows):
  for x in range(num_cols):
  	plt.axis('off')
  	figure.add_subplot(num_rows, num_cols, (x + 1) + (y * num_cols))
  	plt.imshow(grid_viz_data[x, y, :, :, :] )
plt.axis('off')
#plt.show()
plt.savefig('cifar10_fig1')
# Draw the plot and return
#plt.savefig('cifar10_fig1')