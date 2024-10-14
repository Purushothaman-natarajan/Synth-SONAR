# Synth-SONAR: Sonar Image Synthesis with Enhanced Diversity and Realism via Dual Diffusion Models and GPT Prompting

## Overview

This repository provides an implementation of **Synth-SONAR**, a sonar image generation framework combining **Style Injection** for diversity enhancement with fine-tuning using two approaches:
1. **Hugging Face Diffusers** fine-tuning.
2. **PEFT LoRA** fine-tuning for lightweight adaptation.

Synth-SONAR leverages **text-to-image diffusion models** alongside **GPT prompting** to generate realistic, high-quality sonar images.

## Setup

1. Install dependencies via Conda:
   ```bash
   conda env create -f environment.yaml
   conda activate SynthSONAR
   ```

2. Download Stable Diffusion model weights from [Hugging Face](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original):
   ```bash
   ln -s <path/to/model.ckpt> models/ldm/stable-diffusion-v1/model.ckpt
   ```
## Style Injection in Synth-SONAR

**Style Injection** is used to add diversity to the sonar images by blending style elements into the generated content using a pre-trained diffusion model.

```bash
python run_styleid.py --cnt <content_img_dir> --sty <style_img_dir>
```

## Evaluation

### Art-FID Evaluation:
```bash
cd evaluation;
python eval_artfid.py --sty ../data/sty_eval --cnt ../data/cnt_eval --tar ../output
```

### Histogram Loss Evaluation:
```bash
cd evaluation;
python eval_histogan.py --sty ../data/sty_eval --tar ../output
```

### Fine-Tuning Approaches

Synth-SONAR supports **two fine-tuning approaches**, both adapted from the Hugging Face Diffusers library.

### 1. Fine-Tuning Using Diffusers (Standard)
This method fine-tunes Stable Diffusion for sonar images, using text-to-image conditioning. It adapts the script from Hugging Face's [`train_text_to_image.py`](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py).

```bash
python run_finetune.py --train_data <train_data_dir> --output_dir <output_dir> --use_diffusers
```

You can fine-tune various parameters such as learning rate, image resolution, and batch size to adapt the model for sonar data.

### 2. Fine-Tuning Using LoRA (Lightweight)
This method applies **PEFT LoRA** for more efficient, low-rank fine-tuning. It is adapted from the Hugging Face script [`train_text_to_image_lora.py`](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py).

```bash
python run_finetune.py --train_data <train_data_dir> --output_dir <output_dir> --use_lora --lora_rank 4
```

LoRA reduces the number of trainable parameters by using low-rank adaptations to the model's weights, which makes fine-tuning faster and less resource-intensive.

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

For fine-tuning approaches and style-injection, please credit the original Hugging Face implementations:
- Hugging Face Diffusers Fine-Tuning: [train_text_to_image.py](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py)
- LoRA Fine-Tuning with PEFT: [train_text_to_image_lo)ra.py](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py)
- Style-Injection: [Style-ID](https://github.com/jiwoogit/StyleID)

-------
