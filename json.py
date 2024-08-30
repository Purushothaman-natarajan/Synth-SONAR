import os
import json

# Define the paths
image_folder = 'folder/train'  # Replace with your image folder path
metadata_file = os.path.join(image_folder, 'metadata.jsonl')

# Initialize a list to store the metadata
metadata = []

# Iterate through the images and generate metadata
for file_name in os.listdir(image_folder):
    if file_name.endswith('.png'):
        # Example metadata (customize as needed)
        additional_feature = f"This is a value of a text feature you added to your image {file_name}"
        
        # Create a dictionary for each image's metadata
        entry = {
            "file_name": file_name,
            "additional_feature": additional_feature
        }
        
        # Append the entry to the metadata list
        metadata.append(entry)

# Write the metadata to a .jsonl file
with open(metadata_file, 'w') as f:
    for entry in metadata:
        f.write(json.dumps(entry) + '\n')

print(f"Metadata successfully saved to {metadata_file}")
