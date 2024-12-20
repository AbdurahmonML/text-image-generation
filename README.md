# Text-Image-generation
Text-to-Image Generation is a project that focuses on generating realistic images based on textual descriptions. This project uses deep learning techniques to train a generative model that can create images corresponding to CIFAR-10 class labels such as `Airplane`, `Automobile`, `Bird`, `Cat`, `Deer`, `Dog`, `Frog`, `Horse`, `Ship`, and `Truck`. It bridges the gap between computer vision and natural language processing, enabling the model to understand textual inputs and produce coherent visual outputs.  

The project is useful in various applications, including content creation, data augmentation, and enhancing human-computer interaction.

## Architecture of the model

The Conditional UNet architecture integrates convolutional layers, self-attention mechanisms, and conditional embeddings to generate images aligned with class labels. It employs downsampling and upsampling blocks featuring Double Convolution layers with Group Normalization and optional residual connections to extract and refine features. Self-attention layers enhance spatial coherence and capture long-range dependencies, while class labels are embedded and added during both downsampling and upsampling to condition the output. During training, noise is added to the inputs, and the model learns to progressively denoise it, enabling robust and visually coherent image generation that aligns closely with the provided class labels.

## Some results:
If we go from left tp right, the images are for prompts: `Airplane`, `Automobile`, `Bird`, `Cat`, `Deer`, `Dog`, `Frog`, `Horse`, `Ship`, and `Truck`
![image](https://github.com/user-attachments/assets/054d41b7-9569-44a5-9ba4-414ec45492f5)

As we can see it's close enough to real images of that prompts, but due to lack of computational resources I didn't run it more than 250 epochs.

## How to train?
For convenience, all pieces of code were combined into a single Jupyter notebook and executed. Below are the available versions of the training code:

- **[train_full.ipynb](https://drive.google.com/file/d/1QOeQDf3s3ViqGM2LWPSpU0A0ppqKUKwu/view?usp=sharing)**: Contains the complete training workflow in one notebook.
- **[train.ipynb](https://drive.google.com/file/d/12rAATi_rHaiZwYOVcUCVmLkzQW3bUOy-/view?usp=sharing)**: A more organized version that imports all necessary functions and classes from external modules for better readability and modularity.

## Saved Weights
The trained model weights are available for download:  
[ckpt.pt](https://drive.google.com/file/d/1X-wtR3esGnamuVnUvuquj61YFUnWdSwq/view?usp=sharing)

### Loading and Training the Model
You can use the following code to load the saved weights and continue training or testing:

```python
def load(self, model_cpkt_path, model_ckpt="ckpt.pt", ema_model_ckpt="ema_ckpt.pt"):
    self.model.load_state_dict(torch.load(os.path.join(model_cpkt_path, model_ckpt)))
    self.ema_model.load_state_dict(torch.load(os.path.join(model_cpkt_path, ema_model_ckpt)))

# Initialize and train the model
diffuser = Diffusion(config.noise_steps, img_size=config.img_size, num_classes=config.num_classes)
with wandb.init(project="train_sd", group="train", config=config):
    diffuser.prepare(config)
    diffuser.load('')  # Provide the path to your checkpoint directory
    diffuser.fit(config)




