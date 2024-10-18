# Synth-SONAR: Sonar Image Synthesis with Enhanced Diversity and Realism via Dual Diffusion Models and GPT Prompting

## Overview

This repository provides an implementation of **Synth-SONAR**, a sonar image generation framework that combines **Style Injection** for diversity enhancement with fine-tuning techniques. It leverages **text-to-image diffusion models** alongside **GPT prompting** to generate realistic, high-quality sonar images.

## Architecture

<p align="center">
<img src= "https://github.com/Purushothaman-natarajan/Synth-SONAR/blob/main/assets/Overall%20Architecture.jpg" width="1000" />
</p>

## Style Injection in Synth-SONAR

**Style Injection** enhances the diversity of the sonar images by blending stylistic elements into the generated content using a pre-trained diffusion model. The process involves three key steps:

### Step 1: Image-to-Image Generation using Style Injection
- **Style Images:** Actual sonar images are used as style references.
- **Content Generation:** The content images are generated using Stable Diffusion, guided by prompts from GPT.

Run the following command to execute the style injection process:

```bash
python run_styleid.py --cnt <content_img_dir> --sty <style_img_dir>
```

For running with default configuration using sample images, you can use:

```bash
python run_styleid.py --cnt data/cnt --sty data/sty --gamma 0.75 --T 1.5  # Default configuration
python run_styleid.py --cnt data/cnt --sty data/sty --gamma 0.3 --T 1.5   # High style fidelity
```

### Step 2: Fine-Tuning with Text-to-Image Training
After generating initial images, fine-tune the model using a limited dataset of images and corresponding prompts. This can be done using either the standard text-to-image training or LoRA, based on your computational resources.

#### Standard Fine-Tuning
To fine-tune the Stable Diffusion model on your dataset using the `train_text_to_image.py` script, run:

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DIR="path_to_your_dataset"  # Set your dataset path

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
For efficient fine-tuning with LoRA, run the `train_text_to_image_lora.py` script:

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="SONAR-SID"  # Use your dataset name

huggingface-cli login  # Log in to Hugging Face

accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=512 --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=100 --checkpointing_steps=5000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="sd-sonar-model-lora" \
  --validation_prompt="cute dragon creature" --report_to="wandb"
```

### Step 3: Generating More Images and Text with Retraining
Once the model has been fine-tuned, generate additional images and text prompts. The newly generated data can then be used to retrain the model for improved diversity and generalization. 

## Setup

### 1. Install Dependencies

Before running the scripts, make sure to install the necessary dependencies:

```bash
# Clone the Diffusers repository
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .

# Install example-specific requirements
cd examples/text_to_image
pip install -r requirements.txt
```

### 2. Initialize Accelerate

Initialize your 🤗 Accelerate environment:

```bash
accelerate config
```

### 3. Environment Variables

Set the required environment variables for the models and datasets:

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="SONAR-SID"  # Replace with your dataset
```

### 4. Login to Hugging Face

Make sure to authenticate your Hugging Face token:

```bash
huggingface-cli login
```

## Save Precomputed Inversion Features

By default, the style injection process generates a "precomputed_feats" directory that saves the DDIM inversion features of each input image. This reduces the time required for two DDIM inversions but requires significant storage (over 3 GB for each image). To avoid storage issues, disable this feature if necessary:

```bash
python run_styleid.py --precomputed ""  # Not save DDIM inversion features
```

## Evaluation

For quantitative evaluation, a set of randomly selected inputs from MS-COCO and WikiArt in the `./data` directory can be used. Before executing the evaluation code, duplicate the content and style images to match the number of stylized images first.

Run:

```bash
python util/copy_inputs.py --cnt data/cnt --sty data/sty
```

We largely employ [matthias-wright/art-fid](https://github.com/matthias-wright/art-fid) and [mahmoudnafifi/HistoGAN](https://github.com/mahmoudnafifi/HistoGAN) for evaluation.

### Art-FID Evaluation

```bash
cd evaluation;
python eval_artfid.py --sty ../data/sty_eval --cnt ../data/cnt_eval --tar ../output
```

### Histogram Loss Evaluation

```bash
cd evaluation;
python eval_histogan.py --sty ../data/sty_eval --tar ../output
```

## Inference

Once training is complete, load the fine-tuned model for inference:

### For Standard Fine-Tuned Model

```python
import torch
from diffusers import StableDiffusionPipeline

model_path = "path_to_saved_model"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

image = pipe(prompt="A sonar image of an underwater scene").images[0]
image.save("sonar_image.png")
```

### For Fine-Tuned Model with LoRA

Load the fine-tuned LoRA weights as follows:

```python
from diffusers import StableDiffusionPipeline
import torch

model_path = "sayakpaul/sd-model-finetuned-lora-t4"  # Update with your LoRA model path
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

prompt = "A naruto with green eyes and red legs."
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("naruto.png")
```

If you are loading the LoRA parameters from the Hub and the Hub repository has a base model tag, you can retrieve it like this:

```python
from huggingface_hub.repocard import RepoCard

lora_model_id = "./sd-model-finetuned-lora"
card = RepoCard.load(lora_model_id)
base_model_id = card.data.to_dict()["base_model"]

pipe = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
```

For fine-tuning approaches and style-injection, please credit the original Hugging Face implementations:
- Hugging Face Diffusers Fine-Tuning: [train_text_to_image.py](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py)
- LoRA Fine-Tuning with PEFT: [train_text_to_image_lo)ra.py](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py)
- Style-Injection: [Style-ID](https://github.com/jiwoogit/StyleID)


## Citation

If you use this code, please cite us:

```BibTeX
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
-------
