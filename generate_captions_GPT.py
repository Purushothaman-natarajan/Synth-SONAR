import json
import openai
import argparse

# Set up your OpenAI API key
# openai.api_key = 'your-api-key'
# Please refer to the OpenAI documentation for more information on using the GPT-3.5 Turbo model: https://platform.openai.com/docs/models/gpt-3-5

# Function to generate captions using GPT model
def generate_captions(caption_text):
    # Construct prompt for low-level and high-level captions
    prompt = f"""
    Given the image caption: "{caption_text}", please provide:
    1. A low-level description focusing on simple details and objects. 
    2. A high-level description interpreting the scene or conveying a broader understanding."""

    # Call OpenAI API to get the completion
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",  # Choose the appropriate model
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate low and high-level captions from a JSON file.")
    parser.add_argument("json_file", help="Path to the JSON file containing image captions.")
    args = parser.parse_args()

    openai.api_key = "your-api-key"
    update_json(args.json_file)
