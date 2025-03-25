# NYCU Computer Vision 2025 Spring HW1
"""
StudentID: 110550046
Name: 吳孟謙
"""
## Introduction
"""
The goal of this project is to solve a multi-class image classification problem with 100 object categories, using a dataset of 21,024 RGB images for training and validation, and 2,344 RGB images for testing, each belonging to one of 100 object categories.

The task comes with particular specifications:
●	No external datasets are allowed.
●	The model must be based on a ResNet architecture
●	The total number of model parameters must be less than 100M.
○	Use of pretrained weights is allowed, and modifying the backbone architecture is encouraged to improve performance.
●	The code should be PEP8-formatted
●	Weak baseline accuracy of ~0.8 and strong baseline accuracy of ~0.923

The core idea of the method is to fine-tune a pretrained ResNet152 model(~60.2M params) on the target dataset and apply targeted architectural modifications and strong regularization to adapt it to the 100-category classification task. To enhance generalization, I also include a variety of data augmentation techniques during training.
"""
## how to install
To set up the environment and install all necessary dependencies, you can use pip. It's recommended to use a virtual environment (like venv or conda) before installing:
`pip install -r requirements.txt`
## Performance snapshot
![image](https://github.com/user-attachments/assets/cf2588a7-4255-4892-87ca-a260a78da767)
