# NYCU Computer Vision 2025 Spring HW1
StudentID: 110550046  
Name: 吳孟謙
## Introduction
The goal of this project is to solve a multi-class image classification problem with 100 object categories, using a dataset of 21,024 RGB images for training and validation, and 2,344 RGB images for testing, each belonging to one of 100 object categories.  
  
The core idea of the method is to fine-tune a pretrained ResNet152 model(~60.2M params) on the target dataset and apply targeted architectural modifications and strong regularization to adapt it to the 100-category classification task. To enhance generalization, I also include a variety of data augmentation techniques during training.
## how to install
To set up the environment and install all necessary dependencies, it's recommended to use a virtual environment (such as `venv` or `conda`).

Install dependencies using `pip`:

```bash
pip install -r requirements.txt
```
## Performance snapshot
![image](https://github.com/user-attachments/assets/cf2588a7-4255-4892-87ca-a260a78da767)
