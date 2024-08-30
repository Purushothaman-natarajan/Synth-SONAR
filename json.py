import os
import json

# Define the paths
image_folder = r"E:\Train T1\sara"  # Replace with your image folder path
metadata_file = os.path.join(image_folder, 'metadata.jsonl')

# Initialize a list to store the metadata
metadata = []

# Iterate through the images and generate metadata
for file_name in os.listdir(image_folder):
    if file_name.endswith(('.png', '.jpg', '.jpeg')):
        # Define the corresponding text file path
        text_file_name = os.path.splitext(file_name)[0] + '.txt'
        text_file_path = os.path.join(image_folder, text_file_name)
        
        # Read the content of the text file as captions if it exists
        if os.path.exists(text_file_path):
            with open(text_file_path, 'r') as tf:
                captions = tf.read().strip().split('\n')
        else:
            captions = [f"No caption available for {file_name}"]

        # Create a dictionary for each caption associated with the image
        for caption in captions:
            entry = {
                "file_name": file_name,
                "caption": caption,
                "additional_feature": f"This is a value of a text feature you added to your image {file_name}"
            }
            
            # Append the entry to the metadata list
            metadata.append(entry)

# Write the metadata to a .jsonl file
with open(metadata_file, 'w') as f:
    for entry in metadata:
        f.write(json.dumps(entry) + '\n')

print(f"Metadata successfully saved to {metadata_file}")


import os
import json

# Define the path to the metadata file
image_folder = r"E:\Train T1\sara"  # Replace with your image folder path
metadata_file = os.path.join(image_folder, 'metadata.jsonl')

# Initialize a list to store the read metadata
metadata = []

# Read the metadata from the .jsonl file
with open(metadata_file, 'r') as f:
    for line in f:
        entry = json.loads(line.strip())  # Convert each line back to a dictionary
        metadata.append(entry)

# Example: Print out the metadata entries
for entry in metadata:
    print(entry)
