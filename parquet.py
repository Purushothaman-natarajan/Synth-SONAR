import os
import pandas as pd
from PIL import Image
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa

# Define the paths
image_folder = 'path_to_your_image_folder'  # Replace with your image folder path
output_file = 'output_data.parquet'  # Output Parquet file name

# Initialize a list to store the data
data = []

# Iterate through the images and corresponding prompts
for file_name in os.listdir(image_folder):
    if file_name.endswith('.png'):
        # Image file path
        image_path = os.path.join(image_folder, file_name)
        
        # Prompt file path (assuming same name with .txt extension)
        prompt_file = os.path.join(image_folder, file_name.replace('.png', '.txt'))
        
        # Read the image
        image = Image.open(image_path)
        image_array = np.array(image)
        
        # Read the prompts
        with open(prompt_file, 'r') as file:
            prompts = file.read().splitlines()
        
        # Append each image-prompt pair to the data list
        for prompt in prompts:
            data.append([image_array, prompt])

# Convert the data to a DataFrame
columns = ['Image', 'Prompt']
df = pd.DataFrame(data, columns=columns)

# Convert the image array to bytes for storage
df['Image'] = df['Image'].apply(lambda x: x.tobytes())

# Convert the DataFrame to a PyArrow table
table = pa.Table.from_pandas(df)

# Save the table as a Parquet file
pq.write_table(table, output_file)

print(f"Data successfully saved to {output_file}")
