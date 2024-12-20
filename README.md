# Text-Image-generation
Text-to-Image Generation is a project that focuses on generating realistic images based on textual descriptions. This project uses deep learning techniques to train a generative model that can create images corresponding to CIFAR-10 class labels such as `Airplane`, `Automobile`, `Bird`, `Cat`, `Deer`, `Dog`, `Frog`, `Horse`, `Ship`, and `Truck`. It bridges the gap between computer vision and natural language processing, enabling the model to understand textual inputs and produce coherent visual outputs.  

The project is useful in various applications, including content creation, data augmentation, and enhancing human-computer interaction.

## Architecture of the model

The Conditional UNet architecture integrates convolutional layers, self-attention mechanisms, and conditional embeddings to generate images aligned with class labels. It employs downsampling and upsampling blocks featuring Double Convolution layers with Group Normalization and optional residual connections to extract and refine features. Self-attention layers enhance spatial coherence and capture long-range dependencies, while class labels are embedded and added during both downsampling and upsampling to condition the output. During training, noise is added to the inputs, and the model learns to progressively denoise it, enabling robust and visually coherent image generation that aligns closely with the provided class labels.






