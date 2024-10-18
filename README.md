# Synth-SONAR: Sonar Image Synthesis with Enhanced Diversity and Realism via Dual Diffusion Models and GPT Prompting

## Overview

**Synth-SONAR**, a sonar image generation framework that combines **Style Injection** for diversity enhancement with fine-tuning techniques. Synth-SONAR leverages **dual diffusion models** with **text-to-image synthesis** and **GPT-based prompting** to generate realistic, high-quality sonar images. The framework enhances the diversity of images through style-based modifications and ensures realism via dual diffusion processes.

## Architecture

The overall architecture of Synth-SONAR is built on the following core components:

1. **Data Ingestion and Metadata Generation**: Collect sonar images, annotate them (add captions - text descriptions), and generate captions using GPT or LLaMA models for context-aware metadata creation.
2. **Style Injection**: Perform image-to-image generation using Style Injection, where style images (e.g., sonar textures) are fused with content generated via diffusion models.
3. **Dual Diffusion Models**: Combine standard diffusion and fine-tuning approaches to enhance image fidelity, diversity, and style coherence in text to image generation.
4. **Fine-Tuning**: Utilize both text-to-image and LoRA-based fine-tuning to train models, improving the quality and realism of generated images.
5. **Retraining and Iterative Enhancement**: Continuously retrain using generated images and captions for further diversity enhancement.
6. **Clustering**: Use style-based clustering to reduce the similarity between the styles and to improve image quality.

<img src="https://github.com/Purushothaman-natarajan/Synth-SONAR/blob/main/assets/Overall%20Architecture.jpg" width="1000" />

## Style Injection in Synth-SONAR

**Style Injection** enhances the diversity of sonar images by blending stylistic elements into the generated content using a pre-trained diffusion model. The process involves three key steps:

### Step 1: Image-to-Image Generation using Style Injection

- **Style Images**: Actual sonar images are used as style references.
- **Content Generation**: The content images are generated using a Stable Diffusion model, guided by prompts from GPT models.

#### Command:

```bash
python run_styleid.py --cnt <content_img_dir> --sty <style_img_dir>
```

To run with default configuration using sample images:

```bash
python run_styleid.py --cnt data/cnt --sty data/sty --gamma 0.75 --T 1.5  # Default configuration
python run_styleid.py --cnt data/cnt --sty data/sty --gamma 0.3 --T 1.5   # High style fidelity
```

### Step 2: Fine-Tuning with Text-to-Image Training

Once the initial images are generated, fine-tune the model using a limited dataset of images and corresponding prompts. You can use standard text-to-image training or LoRA fine-tuning based on your computational resources.

#### Standard Fine-Tuning

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DIR="path_to_your_dataset" # create a folder with images and corresponding captions, then use create_metadata(data_to_json).py script to get the json file with the dataset to feed as train dataset.


accelerate launch --mixed_precision="fp16" train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-sonar-model"
```

#### Fine-Tuning with LoRA

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="path_to_your_dataset" # create a folder with images and corresponding captions, then use create_metadata(data_to_json).py script to get the json file with the dataset to feed as train dataset.

huggingface-cli login

accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=512 --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=100 --checkpointing_steps=5000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="sd-sonar-model-lora" \
  --validation_prompt="A sonar image" --report_to="wandb"
```

### Step 3: Generating More Images and Retraining

Once fine-tuning is complete, generate additional images and retrain the model for improved diversity and generalization.

## Utilities for Metadata, Caption Generation, and Style-Based Clustering

### Metadata Creation

## Preparing Image-Caption Dataset for Training

To prepare your image-caption dataset for training, follow these steps:

### Step 1: Organize Images and Captions
1. **Create a Folder:**
   - Create a folder named `dataset` or any desired name.
   - Inside this folder, place all the images you want to use for training.
   - Ensure that each image has a corresponding caption in a text file (or use a `.csv` file if preferred).

2. **Captions Format:**
   - The captions should be stored in a way that links each image to its description. You can do this by:
     - Creating a separate text file for each image (e.g., `image1.jpg` → `image1.txt` containing the caption).
     - Or creating a single `.csv` file that lists image names in one column and captions in another (e.g., `image1.jpg`, "A caption describing the image").

### Step 2: Generate JSON Metadata for Training
To convert the image and caption pairs into a dataset ready for training, use the provided `create_metadata(data_to_json).py` script, which will generate a JSON file containing metadata in the correct format.

1. **Run the Metadata Script:**
   - Execute the `create_metadata(data_to_json).py` script on your dataset folder.
   - This script will automatically read the images and their associated captions, and output a `dataset.json` file in the correct format for training.
   
2. **Expected Output:**
   - The resulting `dataset.json` will contain entries for each image with fields for:
     - Image file path.
     - Caption text.

   Example JSON entry:
   ```json
   {
     "file_name": "path_to_image/image1.jpg",
     "caption": "A caption describing the image"
   }
   ```

### Step 3: Feed the Dataset to the Model
- Once the `dataset.json` is generated, you can feed this JSON file, along with your images, into your training pipeline (e.g., a diffusion model) for tasks like text-to-image generation.

---

This process ensures a clean structure for your dataset, which can be easily ingested by machine learning models for training purposes.


You can create a `metadata.jsonl` file containing metadata for sonar images and their corresponding captions by running the `create_metadata.py` script.

#### Command:

```bash
python create_metadata.py <image_folder>
```

### Generating Captions

#### Using GPT-3.5 Turbo:

```bash
python generate_captions.py <json_file>
```

#### Using LLaMA:

```bash
python generate_captions_llama.py <jsonl_file> --batch_size 8
```

### Clustering Sonar Images by Style

The `cluster_images.py` script clusters sonar images based on stylistic features using PCA and K-Means.

#### Command:

```bash
python cluster_images.py <image_dir> --n_components 50 --n_clusters 50
```

## Setup

### Install Dependencies

```bash
# Clone the Diffusers repository
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .

# Install example-specific requirements
cd examples/text_to_image
pip install -r requirements.txt
```

### Initialize Accelerate

```bash
accelerate config
```

### Login to Hugging Face

```bash
huggingface-cli login
```

## Evaluation

We employ [Art-FID](https://github.com/matthias-wright/art-fid) and [HistoGAN](https://github.com/mahmoudnafifi/HistoGAN) for quantitative evaluation.

### Art-FID Evaluation

```bash
cd evaluation
python eval_artfid.py --sty ../data/sty_eval --cnt ../data/cnt_eval --tar ../output
```

### Histogram Loss Evaluation

```bash
cd evaluation
python eval_histogan.py --sty ../data/sty_eval --tar ../output
```

## Inference

Once the model is fine-tuned, you can perform inference as follows:

### Standard Fine-Tuned Model

```python
from diffusers import StableDiffusionPipeline
import torch

model_path = "path_to_saved_model"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

image = pipe(prompt="A sonar image of an underwater scene").images[0]
image.save("sonar_image.png")
```

### Fine-Tuned Model with LoRA

```python
from diffusers import StableDiffusionPipeline
import torch

model_path = "your_LoRA_model_path"
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

image = pipe(prompt="A sonar image").images[0]
image.save("sonar_image.png")
```

## Citation

```bibtex
@misc{natarajan2024synthsonarsonarimagesynthesis,
      title={Synth-SONAR: Sonar Image Synthesis with Enhanced Diversity and Realism via Dual Diffusion Models and GPT Prompting}, 
      author={Purushothaman Natarajan and Kamal Basha and Athira Nambiar},
      year={2024},
      eprint={2410.08612},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.08612}, 
}
```

For fine-tuning approaches and style-injection, please credit the original Hugging Face implementations:
- Hugging Face Diffusers Fine-Tuning: [train_text_to_image.py](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py)
- LoRA Fine-Tuning with PEFT: [train_text_to_image_lo)ra.py](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py)
- Style-Injection: [Style-ID](https://github.com/jiwoogit/StyleID)

----