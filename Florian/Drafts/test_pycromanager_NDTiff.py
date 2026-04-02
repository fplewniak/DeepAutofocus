from ndstorage import Dataset
import numpy as np
# Optional: visualize using matplotlib
import matplotlib.pyplot as plt

# Path to the ND-Tiff dataset directory
dataset_path = "/data3/NDTiff/NDTiff_test_data/v3/Magellan_expolore_negative_indices_and_overwritten"

# Load the dataset
ds = Dataset(dataset_path)

# Parameters to access: channel, Z index, time frame
channel_name = 0   # Replace with the actual channel name in your dataset
z_index = 0            # Zero-based index for Z-plane
time_index = 0         # Zero-based index for time point

metadata = ds.summary_metadata
for k in metadata:
    print(f'{k}:', metadata[k])
if 'Slices' in ds.summary_metadata:
    print('Slices:', metadata['Slices'])
if 'Frames' in metadata:
    print('Frames:', metadata['Frames'])
if 'IntendedDimensions' in metadata:
    print('Dim:', metadata['IntendedDimensions'])
print(ds.get_channel_names())
print(ds.get_index_keys())
# # Retrieve the image as a numpy array
# img = ds.read_image(time=time_index, z=z_index, channel=channel_name)
# img = ds.read_image(time=time_index, channel=channel_name, z=0, position=0)
#
# # Display info
# print(f"Image shape: {img.shape}")
# print(f"Image dtype: {img.dtype}")
#
#
# plt.imshow(img, cmap='gray')
# # plt.title(f"Channel: {channel_name}, Z: {z_index}, Time: {time_index}")
# plt.title(f"Channel: {channel_name}, Time: {time_index}")
# plt.show()
