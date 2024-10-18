import json
import os
import torch
import argparse
from PIL import Image
from transformers import AutoProcessor, AutoModelForPreTraining

def main():
    parser = argparse.ArgumentParser(description="Generate captions from side scan SONAR images using a VLM.")
    parser.add_argument("jsonl_file", help="Path to the JSONL file containing image metadata.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing images.")
    parser.add_argument("--output_file", help="Optional path to save the updated JSONL file. If not provided, it overwrites the original.")
    args = parser.parse_args()

    model_name = "llava-hf/llava-1.5-7b-hf"
    
    # Load the processor and model
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForPreTraining.from_pretrained(model_name).to("cpu")

    # Process the JSONL file and generate captions
    update_jsonl(args.jsonl_file, processor, model, args.batch_size, args.output_file)

def generate_captions_vlm(image_path, captions, processor, model):
    # Load and preprocess the image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return ["Error"] * len(captions)
    
    # Prepare text prompts for each caption
    prompts = [
        f"Given the side scan SONAR image and the caption: \"{caption}\", where PL__*, SH__*, CGM*, ASF*, TCM* represent the objects in the image, "
        f"and AS__*, AP__*, SEF* represent the background. The numbers following these abbreviations can range from one to five digits. "
        "Provide the following descriptions:\n"
        "1. A low-level description focusing on simple details and objects visible in the image.\n"
        "2. A high-level description interpreting the scene or conveying a broader understanding based on the image and the given caption."
        for caption in captions
    ]
    
    try:
        # Tokenize and process each prompt with the image
        inputs = processor(images=image, text=prompts, return_tensors="pt", padding=True, truncation=True)
    except Exception as e:
        print(f"Error processing inputs: {e}")
        return ["Error"] * len(captions)

    # Move inputs to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {key: value.to(device) for key, value in inputs.items()}
    model.to(device)

    try:
        # Generate responses
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=150, num_beams=5, early_stopping=True)
    except Exception as e:
        print(f"Error generating text: {e}")
        return ["Error"] * len(captions)
    
    # Decode the output
    try:
        generated_texts = [processor.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    except Exception as e:
        print(f"Error decoding text: {e}")
        return ["Error"] * len(captions)

    # Process the generated texts
    results = []
    for generated_text in generated_texts:
        captions = generated_text.strip().split("\n")
        if len(captions) >= 2:
            low_level_caption = captions[0].replace("ASSISTANT:", "").strip()
            high_level_caption = captions[1].replace("ASSISTANT:", "").strip()
        else:
            low_level_caption = "No low-level caption generated."
            high_level_caption = generated_text.replace("USER:", "").strip()
        
        results.append((low_level_caption, high_level_caption))
    
    return results

# Function to update JSONL file in batches and print predicted and actual captions
def update_jsonl(json_file_path, processor, model, batch_size=8, output_file=None):
    updated_data = []
    items = []

    # Use the provided output file or overwrite the input file
    output_file = output_file or json_file_path

    # Read the JSONL file line by line
    with open(json_file_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())  # Load each JSON object
            file_name = item.get('file_name', '')
            caption = item.get('caption', '')  # Use the caption as the prompt

            # Collect items in batch size
            items.append(item)

            # When we have a batch, process it
            if len(items) >= batch_size:
                process_batch(items, json_file_path, processor, model)
                items = []  # Clear after processing

        # Process remaining items
        if items:
            process_batch(items, json_file_path, processor, model)

    # Write the updated data back to the JSONL file
    with open(output_file, 'w') as f:
        for item in updated_data:
            f.write(json.dumps(item) + '\n')  # Write each object on a new line

def process_batch(items, json_file_path, processor, model):
    for item in items:
        file_name = item.get('file_name', '')
        caption = [item.get('caption', '')]  # Process a single caption

        # Generate captions
        image_path = os.path.join(os.path.dirname(json_file_path), file_name)
        low_level_caption, high_level_caption = generate_captions_vlm(image_path, caption, processor, model)[0]

        # Print the original and generated captions
        print(f"File Name: {file_name}")
        print(f"Original Caption: {caption[0]}")
        print(f"Generated Low-Level Caption: {low_level_caption}")
        print(f"Generated High-Level Caption: {high_level_caption}\n")

        # Update the JSON object with new captions
        item['low_level_caption'] = low_level_caption
        item['high_level_caption'] = high_level_caption
        updated_data.append(item)

