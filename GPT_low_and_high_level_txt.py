import json
import openai

# Set up your OpenAI API key
openai.api_key = 'your-api-key'

# Function to generate captions using GPT model
def generate_captions(caption_text):
    # Construct prompt for low-level and high-level captions
    prompt = f"""
    Given the image caption: "{caption_text}", please provide:
    1. A low-level description focusing on simple details and objects.
    2. A high-level description interpreting the scene or conveying a broader understanding.
    """

    # Call OpenAI API to get the completion
    response = openai.Completion.create(
        engine="text-davinci-003",  # Choose the appropriate model
        prompt=prompt,
        max_tokens=150,
        temperature=0.7
    )
    
    # Extract generated captions from response
    captions = response.choices[0].text.strip().split("\n")
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
        low_level_caption, high_level_caption = generate_captions(original_caption)
        
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
