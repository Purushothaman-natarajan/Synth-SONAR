import os
import json
import argparse

def create_metadata(image_folder, metadata_file):
    """Creates a metadata.jsonl file from images and corresponding text files."""

    metadata = []

    for file_name in os.listdir(image_folder):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            text_file_name = os.path.splitext(file_name)[0] + '.txt'
            text_file_path = os.path.join(image_folder, text_file_name)

            if os.path.exists(text_file_path):
                with open(text_file_path, 'r') as tf:
                    captions = tf.read().strip().split('\n')
            else:
                captions = [f"No caption available for {file_name}"]

            for caption in captions:
                entry = {
                    "file_name": file_name,
                    "caption": caption,
                    "additional_feature": f"This is a value of a text feature you added to your image {file_name}"
                }
                metadata.append(entry)

    with open(metadata_file, 'w') as f:
        for entry in metadata:
            f.write(json.dumps(entry) + '\n')

    print(f"Metadata successfully saved to {metadata_file}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create metadata.jsonl file from images and text files.")
    parser.add_argument("image_folder", help="Path to the folder containing images.")
    args = parser.parse_args()
    create_metadata(args.image_folder, os.path.join(args.image_folder, 'metadata.jsonl'))
