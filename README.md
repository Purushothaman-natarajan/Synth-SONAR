<!--[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
-->

# Synth-SONAR: Sonar Image Synthesis with Enhanced Diversity and Realism via Dual Diffusion Models and GPT Prompting

[Purushothaman Natarajan](https://purushothaman-natarajan.github.io/), [Kamal Basha](https://scholar.google.com/citations?user=hKUW3CwAAAAJ&hl=en&oi=sra), [Athira Nambiar](https://www.srmist.edu.in/faculty/dr-athira-m-nambiar/)

[[`Paper`](https://www.arxiv.org/pdf/2410.08612)] [[`BibTeX`](#Citation)]


## Overview

**Synth-SONAR** is a sonar image generation framework that integrates **Style Injection** for diversity enhancement with **dual diffusion models** and **GPT-based prompting** to produce realistic, high-quality sonar images. By leveraging style-based modifications and dual diffusion processes, Synth-SONAR can generate text conditioned images while maintaining a high level of realism.

## Architecture

The overall architecture of Synth-SONAR consists of the following core components:

1. **Data Ingestion and Metadata Generation**: Collect sonar images, annotate them with captions, and generate context-aware metadata using GPT or LLaMA models.
2. **Style Injection**: Perform image-to-image generation by fusing style images (e.g., sonar textures) with content generated via diffusion models.
3. **Dual Diffusion Models**: Combine standard diffusion and fine-tuning techniques to enhance image fidelity, diversity, and style coherence in text-to-image generation.
4. **Fine-Tuning**: Utilize both text-to-image and LoRA-based fine-tuning approaches to improve the quality and realism of generated images.
5. **Retraining and Iterative Enhancement**: Continuously retrain using generated images and captions to enhance diversity and realism further.
6. **Clustering**: Apply style-based clustering to reduce stylistic similarity between images and improve overall image quality.

![Overall Architecture](https://github.com/Purushothaman-natarajan/Synth-SONAR/blob/main/assets/Overall%20Architecture.jpg)

## Style Injection in Synth-SONAR

**Style Injection** enhances the diversity of sonar images by blending stylistic elements into generated content using a pre-trained diffusion model. The process involves three key steps:

### Step 1: Image-to-Image Generation using Style Injection

- **Style Images**: Utilize actual sonar images as style references.
- **Content Generation**: Generate content images using a Stable Diffusion model, guided by prompts from GPT models.

#### Create a Conda Environment

```
conda env create -f environment.yaml
conda activate Synth-SONAR
```

#### Download StableDiffusion Weights

Download the StableDiffusion weights from the [CompVis organization at Hugging Face](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)
(download the `sd-v1-4.ckpt` file), and link them:
```
ln -s <path/to/model.ckpt> models/ldm/stable-diffusion-v1/model.ckpt 
```

#### Command:

```bash
python run_styleid.py --cnt <content_img_dir> --sty <style_img_dir>
```

To run with default configurations using sample images:

```bash
python run_styleid.py --cnt data/cnt --sty data/sty --gamma 0.75 --T 1.5  # Default configuration
python run_styleid.py --cnt data/cnt --sty data/sty --gamma 0.3 --T 1.5   # High style fidelity
```

### Step 2: Fine-Tuning with Text-to-Image Training

Once the initial images are generated, fine-tune the model using a limited dataset of images and corresponding prompts. You can choose between standard text-to-image training or LoRA fine-tuning based on your computational resources.

Login to Hugging Face (for fine-tuning and using the models from Huggingface)

```bash
huggingface-cli login
```

#### Standard Fine-Tuning

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DIR="path_to_your_dataset"  # Create a folder with images and corresponding captions, then use create_metadata(data_to_json).py script to generate the JSON file.

accelerate launch --mixed_precision="fp16" ./text_to_image/train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=5000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-sonar-model"
```

#### Fine-Tuning with LoRA

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="path_to_your_dataset"  # Create a folder with images and corresponding captions, then use create_metadata(data_to_json).py script to generate the JSON file.

accelerate launch --mixed_precision="fp16" ./text_to_image/train_text_to_image_lora.py \
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

After fine-tuning is complete, generate additional images and retrain the model to enhance diversity and generalization.

-----

## Utilities for Metadata, Caption Generation, and Style-Based Clustering

### Metadata Creation

#### Preparing Image-Caption Dataset for Training

To prepare your image-caption dataset for training, follow these steps:

**Step 1: Organize Images and Captions**
1. **Create a Folder:**
   - Name the folder `dataset` (or any desired name).
   - Place all images intended for training in this folder.
   - Ensure each image has a corresponding caption in a text file (or use a `.csv` file if preferred).

2. **Captions Format:**
   - Captions should link to each image. You can do this by:
     - Creating a separate text file for each image (e.g., `image1.jpg` → `image1.txt` containing the caption).
     - Creating a single `.csv` file listing image names in one column and captions in another (e.g., `image1.jpg`, "A caption describing the image").

**Step 2: Generate JSON Metadata for Training**
To convert the image and caption pairs into a training-ready dataset, use the provided `create_metadata(data_to_json).py` script:

1. **Run the Metadata Script:**
   - Execute the `create_metadata(data_to_json).py` script with images and captions in a folder. This will automatically read images and captions and output a `metadata.jsonl` file in the correct format for training.

   #### Command:

    ```bash
    python create_metadata(data_to_json).py <image_folder>
    ```

2. **Expected Output:**
   - The `metadata.jsonl` will contain entries for each image with fields for:
     - Image file path.
     - Caption text.

   Example JSON entry:
   ```json
   {
     "file_name": "path_to_image/image1.jpg",
     "caption": "A caption describing the image"
   }
   ```

**Step 3: Feed the Dataset to the Model**
Once the `dataset.json` is generated, feed this JSON into your training pipeline for training the text-to-image generation model.

---

### Generating Captions

#### Using GPT-3.5 Turbo:

```bash
python generate_captions_GPT.py <json_file>
```

#### Using LLaMA:

```bash
python generate_captions_llama.py <jsonl_file> --batch_size 8
```

### Clustering Sonar Images by Style

Use the `cluster_images_for_style_generalization.py` script to cluster sonar images based on stylistic features using PCA and K-Means.

#### Command:

```bash
python luster_images_for_style_generalization.py <image_dir> --n_components 50 --n_clusters 50
```

## Setup

### Install Dependencies

```bash
# Install text-to-image-specific requirements
cd 'Synth-SONAR/text_to_image'
pip install -r requirements.txt
```

### Initialize Accelerate

```bash
accelerate config
```

### Login to Hugging Face (for fine-tuning and using the models from Huggingface)

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
@misc

{natarajan2024synthsonar,
  title={Synth-SONAR: Sonar Image Synthesis with Enhanced Diversity and Realism via Dual Diffusion Models and GPT Prompting},
  author={Purushothaman Natarajan, Kamal Basha, Athira Nambiar},
  year={2024},
  eprint={2408.12808},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

For fine-tuning approaches and style-injection, please credit the original Hugging Face implementations:
- Hugging Face Diffusers Fine-Tuning: [train_text_to_image.py](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py)
- LoRA Fine-Tuning with PEFT: [train_text_to_image_lo)ra.py](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py)
- Special thanks to the authors of Style-Injection: [Style-ID](https://github.com/jiwoogit/StyleID)

----
