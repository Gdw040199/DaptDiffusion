<p align="center">
  <h1 align="center">DaptDiffusion: Enhancing Pixel-Level Interactive Editing with Dense-UNet and Adam Point Update in Diffusion Models</h1>
  <p align="center">
    <strong>Dawei Guan</strong>
    &nbsp;&nbsp;
  </p>
  <br>
DaptDiffusion is a diffusion model framework for image editing, which optimizes the latent space editing process based on the diffusion model by introducing the Dense-UNet feature network and the Adam point tracking method. 

## Overview of the model
![image](https://github.com/Gdw040199/DaptDiffusion/blob/main/overview.png)

## Disclaimer
This is a research project, NOT a commercial product. Users are granted the freedom to create images using this tool, but they are expected to comply with local laws and utilize it in a responsible manner. The developers do not assume any responsibility for potential misuse by users.

## News and Update
* [August 8th] v0.0.0 Release.
  * Implement Basic function of DaptDiffusion

## Installation

It is recommended to run our code on a Nvidia GPU with a linux system. Currently, it requires around 14 GB GPU memory to run our method. We will continue to optimize memory efficiency
You can also run our code on the Windows system.

To install the required libraries, simply run the following command:
```
conda env create -f environment.yaml
conda activate dragdiff
```

## Run DaptDiffusion
To start with, in command line, run the following to start the gradio user interface:
```
python3 drag_ui.py
```

Basically, it consists of the following steps:

### Case 1: Dragging Input Real Images
#### 1) train a LoRA
* Drop our input image into the left-most box.
* Input a prompt describing the image in the "prompt" field
* Click the "Train LoRA" button to train a LoRA given the input image

#### 2) do "drag" editing
* Draw a mask in the left-most box to specify the editable areas.
* Click handle and target points in the middle box. Also, you may reset all points by clicking "Undo point".
* Click the "Run" button to run our algorithm. Edited results will be displayed in the right-most box.

### Case 2: Dragging Diffusion-Generated Images
#### 1) generate an image
* Fill in the generation parameters (e.g., positive/negative prompt, parameters under Generation Config & FreeU Parameters).
* Click "Generate Image".

#### 2) do "drag" on the generated image
* Draw a mask in the left-most box to specify the editable areas
* Click handle points and target points in the middle box.
* Click the "Run" button to run our algorithm. Edited results will be displayed in the right-most box.


<!---
## Explanation for parameters in the user interface:
#### General Parameters
|Parameter|Explanation|
|-----|------|
|prompt|The prompt describing the user input image (This will be used to train the LoRA and conduct "drag" editing).|
|lora_path|The directory where the trained LoRA will be saved.|


#### Algorithm Parameters
These parameters are collapsed by default as we normally do not have to tune them. Here are the explanations:
* Base Model Config

|Parameter|Explanation|
|-----|------|
|Diffusion Model Path|The path to the diffusion models. By default we are using "runwayml/stable-diffusion-v1-5". We will add support for more models in the future.|
|VAE Choice|The Choice of VAE. Now there are two choices, one is "default", which will use the original VAE. Another choice is "stabilityai/sd-vae-ft-mse", which can improve results on images with human eyes and faces (see [explanation](https://stable-diffusion-art.com/how-to-use-vae/))|

* Drag Parameters

|Parameter|Explanation|
|-----|------|
|n_pix_step|Maximum number of steps of motion supervision. **Increase this if handle points have not been "dragged" to desired position.**|
|lam|The regularization coefficient controlling unmasked region stays unchanged. Increase this value if the unmasked region has changed more than what was desired (do not have to tune in most cases).|
|n_actual_inference_step|Number of DDIM inversion steps performed (do not have to tune in most cases).|

* LoRA Parameters

|Parameter|Explanation|
|-----|------|
|LoRA training steps|Number of LoRA training steps (do not have to tune in most cases).|
|LoRA learning rate|Learning rate of LoRA (do not have to tune in most cases)|
|LoRA rank|Rank of the LoRA (do not have to tune in most cases).|

--->
