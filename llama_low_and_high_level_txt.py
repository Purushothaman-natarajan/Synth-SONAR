import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the LLaMA model and tokenizer from Hugging Face or your local system
model_name = "huggingface/llama-3b"  # Replace with actual LLaMA model path or Hugging Face model ID
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate captions using LLaMA
def generate_captions_llama(caption_text):
    # Construct the prompt
    prompt = f"""
    Given the image caption: "{caption_text}", please provide:
    1. A low-level description focusing on simple details and objects.
    2. A high-level description interpreting the scene or conveying a broader understanding.
    """

    # Tokenize and generate response
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=150)
    
    # Decode the output and split into low and high-level captions
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    captions = generated_text.strip().split("\n")
    low_level_caption = captions[0].replace("1. ", "").strip()
    high_level_caption = captions[1].replace("2. ", "").strip()
    
    return low_level_caption, high_level_caption

# Function to update JSON file
def update_json(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Iterate through the JSON data and update captions
    for item in data:
        original_caption = item.get('caption', '')
        
        # Generate low and high-level captions
        low_level_caption, high_level_caption = generate_captions_llama(original_caption)
        
        # Update the JSON object with new captions
        item['low_level_caption'] = low_level_caption
        item['high_level_caption'] = high_level_caption
    
    # Save the updated JSON file
    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=4)

# Path to your JSON file
json_file_path = "path_to_your_json_file.json"

# Update JSON with captions
update_json(json_file_path)
