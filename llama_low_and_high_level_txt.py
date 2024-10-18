import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model directly, keeping the model on CPU to save GPU memory
device = torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B").to(device)

# Set padding token to eos_token if the tokenizer does not have one
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Function to generate captions using LLaMA with model on CPU
def generate_captions_llama(caption_texts):
    captions_batch = []
    
    for caption_text in caption_texts:
        # Construct the prompt for each caption
        prompt = f"""
        Given the image caption: "{caption_text}", please provide:
        1. A low-level description focusing on simple details and objects.
        2. A high-level description interpreting the scene or conveying a broader understanding.
        """
        captions_batch.append(prompt)

    # Tokenize the batch of prompts and keep them on CPU
    inputs = tokenizer(captions_batch, return_tensors="pt", padding=True, truncation=True).to(device)

    # Generate responses in batch on CPU and decode them
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=150)
    
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Split the generated text into low and high-level captions
    low_level_captions, high_level_captions = [], []
    for generated_text in generated_texts:
        captions = generated_text.strip().split("\n")
        low_level_captions.append(captions[0].replace("1. ", "").strip())
        high_level_captions.append(captions[1].replace("2. ", "").strip())
    
    return low_level_captions, high_level_captions

# Function to update JSONL file in batches and print predicted and actual captions
def update_jsonl(json_file_path, batch_size=8):
    updated_data = []
    original_captions = []
    items = []

    # Read the JSONL file line by line
    with open(json_file_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())  # Load each JSON object
            original_captions.append(item.get('caption', ''))
            items.append(item)

            # Process in batches
            if len(original_captions) == batch_size:
                # Generate low and high-level captions in batch
                low_level_captions, high_level_captions = generate_captions_llama(original_captions)

                # Update each item with the generated captions
                for i in range(batch_size):
                    print(f"Original Caption: {original_captions[i]}")
                    print(f"Generated Low-Level Caption: {low_level_captions[i]}")
                    print(f"Generated High-Level Caption: {high_level_captions[i]}\n")
                    
                    items[i]['low_level_caption'] = low_level_captions[i]
                    items[i]['high_level_caption'] = high_level_captions[i]
                    updated_data.append(items[i])

                # Reset for next batch
                original_captions = []
                items = []

        # Handle remaining items if they don't fill the last batch
        if original_captions:
            low_level_captions, high_level_captions = generate_captions_llama(original_captions)

            for i in range(len(original_captions)):
                print(f"Original Caption: {original_captions[i]}")
                print(f"Generated Low-Level Caption: {low_level_captions[i]}")
                print(f"Generated High-Level Caption: {high_level_captions[i]}\n")

                items[i]['low_level_caption'] = low_level_captions[i]
                items[i]['high_level_caption'] = high_level_captions[i]
                updated_data.append(items[i])

    # Write the updated data back to the JSONL file
    with open(json_file_path, 'w') as f:
        for item in updated_data:
            f.write(json.dumps(item) + '\n')  # Write each object on a new line

# Path to your JSONL file
json_file_path = r"E:\Train T3\consolidate2_final\metadata.jsonl"

# Update JSONL with captions and print them in batches
update_jsonl(json_file_path, batch_size=8)
