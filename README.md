# A webcam artificial intelligence-based gaze-tracking algorithm

This repository provide an implementation code to replicate Figueroa S. and Morocho M. study results in their work ["A webcam artificial intelligence-based gaze-tracking algorithm"](https://drive.google.com/file/d/1Qt2q5KTAwVzkINAolY5xdCDt21Sz8Zfq/view?usp=sharing) [1] which presents a comparison analysis between our proposed model based on ResNet-50 pre-trained on ImageNet and 'benchmark' model presented in ["Efficiency in Real-Time Webcam Gaze Tracking"](https://arxiv.org/abs/2009.01270) [2] work. In addition, the source code implementation is a modification of [pperle
Pascal repository](https://github.com/pperle/gaze-tracking) where an evaluation of a monocular eye tracking set-up work is performed.

## Requirements

- ```pip install -r requirements.txt```

### Python Environment

- Python version: 3.7.1
- Conda environment used

## Updates
In this section, every update and important information about changes or improvements of the implementation will be posted here:

16/03/2023 - Traning process and results for benchmark and proposed model are provided

Nothing new...

## Dataset

MPIIFaceGaze dataset was used, to download the dataset click [here](https://perceptualui.org/research/datasets/MPIIFaceGaze/). Next, you migth need to pre-process the images by running ..... or download the preprocessed dataset by clicking [here](https://drive.google.com/file/d/1feDiiel0rxhrPLI1Xcw4Fv6N8_Ibk8Vg/view?usp=sharing).


## Pretrained Models
To replicate results, you might need to download the respective pxx.ckpt files for VGG-16 or ResNet-50 model. Please download the [pretrained_models_VGG-16](https://drive.google.com/file/d/1qv7pbBDILplEIsoVA6cKtwFzX7Ga5vYe/view?usp=sharing) or [pretrained_models_ResNet-50](https://drive.google.com/file/d/10JgeeAjLMsgg4emoOJCMkH5B2oikJb0_/view?usp=sharing).


## Training process


## Testing process


## Results

### How to plot same results presented in the paper [1].

All the plots presented in our work can be visualizeed by using the events files from 'lightning_logs_ResNet50', 'tb_logs-ResNet50' and 'tb_logs-VGG16'. For this start a new tensorboard session and:
- Run: tennsorboard --logdir=./

## Bibliography

[1] Figueroa S. and Morocho M..... "A webcam artificial intelligence-based gaze-tracking algorithm".
[2] Gudi, A., Li, X., & van Gemert, J. (2020). Efficiency in real-time webcam gaze tracking. In Computer Vision–ECCV 2020 Workshops: Glasgow, UK, August 23–28, 2020, Proceedings, Part I 16 (pp. 529-543). Springer International Publishing.
