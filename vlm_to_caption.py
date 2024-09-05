import json
import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForPreTraining

# Load model and processor
model_name = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForPreTraining.from_pretrained(model_name).to("cpu")

def generate_captions_vlm(image_path, caption):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    prompt = (f"Given the image and the caption: \"{caption}\", provide the following descriptions:\n"
              "1. A low-level description focusing on simple details and objects visible in the image.\n"
              "2. A high-level description interpreting the scene or conveying a broader understanding based on the image and the given caption.")

    inputs = processor(images=image, text=prompt, return_tensors="pt", padding=True, truncation=True)

    # Move inputs to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {key: value.to(device) for key, value in inputs.items()}
    model.to(device)

    # Generate responses
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=150, num_beams=5, early_stopping=True)
    
    # Decode the output
    generated_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Print the generated text for debugging
    print(f"Generated Text: {generated_text}")

    # Process the generated text
    captions = generated_text.strip().split("\n")
    if len(captions) >= 2:
        low_level_caption = captions[0].replace("ASSISTANT:", "").strip()
        high_level_caption = captions[1].replace("ASSISTANT:", "").strip()
    else:
        low_level_caption = "No low-level caption generated."
        high_level_caption = generated_text.replace("USER:", "").strip()

    return low_level_caption, high_level_caption


# Function to update JSONL file in batches and print predicted and actual captions
def update_jsonl(json_file_path, batch_size=8):
    updated_data = []
    items = []

    # Read the JSONL file line by line
    with open(json_file_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())  # Load each JSON object
            file_name = item.get('file_name', '')
            caption = item.get('caption', '')  # Use the caption as the prompt

            # Generate the prompt with additional context
            full_prompt = (f"USER: <image>\n{caption} ASSISTANT: "
                           f"Describe only the object in the image acquired through the use of synthetic aperture sonar. "
                           f"PL__*, SH__*, CGM*, ASF*, tcm* represent the object in the image, "
                           f"AS___*, AP__*, SEF* represents the background, __ may be a number with one to 5 digits.")

            # Generate captions
            image_path = os.path.join(os.path.dirname(json_file_path), file_name)
            low_level_caption, high_level_caption = generate_captions_vlm(image_path, full_prompt)

            # Print the original and generated captions
            print(f"File Name: {file_name}")
            print(f"Original Caption: {caption}")
            print(f"Generated Low-Level Caption: {low_level_caption}")
            print(f"Generated High-Level Caption: {high_level_caption}\n")

            # Update the JSON object with new captions
            item['low_level_caption'] = low_level_caption
            item['high_level_caption'] = high_level_caption
            updated_data.append(item)

    # Write the updated data back to the JSONL file
    with open(json_file_path, 'w') as f:
        for item in updated_data:
            f.write(json.dumps(item) + '\n')  # Write each object on a new line

# Path to your JSONL file
json_file_path = r"E:\Train T3\consolidate2_final\metadata.jsonl"

# Update JSONL with captions and print them in batches
update_jsonl(json_file_path, batch_size=8)
