# Patch-Based Enhanced Unified Net for Accurate Segmentation of Nasopharyngeal Carcinoma in 3D MR Images

This repository includes the Nasophyrangeal Carcinoma Segmentation Python code utilized in our proposed study's analyses.  Only a limited amount of samples are available in this repository; the dataset is private and cannot be shared.

### Our code can run smoothly using these dependencies.
1. Pytorch Implementation pytorch 11.9
2. Cuda 12.3
3. Jupyter Notebook 7.3

> conda create -n your_env_name python=3.7
> 
> cconda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

   
Proposed model can be can be trained using main.ipynb jupyter notebook file. The predictive GTV mask folder contains the extrated slices segmentation results.

**Note:** If you don't use the provided dependencies. You may face some errors in the code.

**Experimental Setup**

***GPU***   NVIDIA TITAN RTX GPU 

***RAM***   32G

***OS***    Linux

***Cuda***  12.3

