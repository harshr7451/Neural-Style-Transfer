# Introduction
Neural Style Transfer (NST) is a deep learning technique that combines the content of one image with the
artistic style of another to generate a new, visually appealing image. Using Convolutional Neural Networks
(CNNs) like the VGG-19 model, NST extracts content features from one image and style features from another.
The process involves preprocessing the images, extracting features, calculating content and style losses, and
iteratively optimizing the generated image to minimize these losses. The resultant image retains the semantic
content of the original while adopting the stylistic elements of the reference image, oƯering creative
applications in art, social media, marketing, and design.

# VGG-19 Model Description
In the context of Neural Style Transfer (NST), the CNN model typically used is VGG-19 (Visual Geometry Group
19-layer model). VGG-19 is a deep convolutional neural network that has proven effective in image recognition
tasks. Here’s a description of the CNN model and its relevance to NST:

## Architecture:
### Layers:
VGG-19 consists of 19 layers (hence the name) organized into a sequence of
convolutional layers (with filters of size 3x3), followed by max-pooling layers for downsampling.

### Depth: 
The network architecture is deep, featuring multiple convolutional layers stacked on top
of each other, which allows it to learn increasingly complex features as information passes
through the network.

## Feature Extraction:
### Pre-trained Weights: 
VGG-19 is typically pre-trained on large-scale image datasets like
ImageNet, which contains millions of labeled images across thousands of categories. This pre-
training enables the network to capture general features useful for a wide range of image
recognition tasks.

### Feature Maps:
As an image passes through VGG-19, each layer extracts and enhances
different features, starting from simple edges and textures in early layers to more complex
shapes and patterns in deeper layers.

## Layer Utilization in NST:
### Feature Representation: 
In NST, specific layers of VGG-19 are chosen based on their ability to
capture both content and style information effectively.

### Content Features: 
Typically extracted from deeper layers (e.g., conv4_2 or conv5_2),
these layers focus on high-level semantic content such as objects and structures.

### Style Features: 
Extracted from shallower layers (e.g., conv1_1, conv2_1, conv3_1,
conv4_1, conv5_1), these layers capture stylistic elements such as textures, colors, and
patterns through their correlation matrices (Gram matrices).

## Computational Efficiency:
### Performance: 
Despite being deeper than earlier models like AlexNet, VGG-19 strikes a balance
between model complexity and computational efficiency, making it feasible for tasks like NST
where multiple image transformations are performed iteratively.

## Limitations and Adaptations:
### Memory and Computation: 
VGG-19 requires substantial memory and computation resources,
especially when applied in iterative optimization tasks like NST. This often necessitates using
GPUs for efficient training and inference.

# Installation Instructions
1) Open the NST.ipynb file in VSCode or Google Colab.
2) Upload the style and content images.
3) Provide the path of the images in the code where required.
4) Make sure the runtime type is GPU otherwise the image sizes will be small and the processing of the code will take much time.

# Required Libraries
1) PyTorch: Includes torch, torch.nn, torch.nn.functional, and torch.optim, providing core tensor
computation, neural network layers, functional APIs for operations like activations and losses, and
optimization algorithms.
2) Image and Plotting Libraries: Imports PIL.Image from the Python Imaging Library (PIL) for image
handling and matplotlib.pyplot for plotting and visualization.
3) Torchvision: Utilizes torchvision.transforms for image transformations and torchvision.models for
accessing pre-trained CNN models such as VGG-19.
4) Miscellaneous: Imports copy from the Python standard library, which is used for creating deep copies
of objects.
