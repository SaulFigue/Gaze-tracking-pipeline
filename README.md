# A webcam artificial intelligence-based gaze-tracking algorithm

This repository provide an implementation code to replicate Figueroa, S.; Pineda, I.; Vizcaíno, P.; Reyes-Chacón, I. and Morocho-Cayamcela, M. study results in their work ["A webcam artificial intelligence-based gaze-tracking algorithm"][1] which presents a comparison analysis between our proposed model based on ResNet-50 pre-trained on ImageNet and 'benchmark' model presented in ["Efficiency in Real-Time Webcam Gaze Tracking"](https://arxiv.org/abs/2009.01270) [2] work. In addition, the source code implementation is a modification of [pperle
Pascal repository](https://github.com/pperle/gaze-tracking) where an evaluation of a monocular eye tracking set-up work is performed.

![ProposedModelPipeline](https://raw.githubusercontent.com/SaulFigue/Gaze-tracking-pipeline/main/images/Pipeline.png)


## Requirements
- ```
     pip install -r requirements.txt
  ```

### Python Environment

- Python version: 3.7.1
- Conda environment used

## Updates
In this section, every update and important information about changes or improvements of the implementation will be posted here:

16/03/2023 - Traning process and results for benchmark and proposed model are provided

Nothing new...

## Dataset

MPIIFaceGaze dataset was used, to download the dataset click [here](https://perceptualui.org/research/datasets/MPIIFaceGaze/). Then, you migth need to pre-process the images by running 
```
python dataset/mpii_face_gaze_preprocessing.py --input_path=./MPIIFaceGaze --output_path=./data
```
or download the preprocessed dataset by clicking [here](https://drive.google.com/file/d/1feDiiel0rxhrPLI1Xcw4Fv6N8_Ibk8Vg/view?usp=sharing). Next, create a ```./data``` folder to unzip the preprocessed MPIIFaceGaze dataset.

## Pretrained Models
To replicate results, you might need to download the respective pxx.ckpt files for VGG-16 or ResNet-50 model. Please download the [pretrained_models_VGG-16](https://drive.google.com/file/d/1qv7pbBDILplEIsoVA6cKtwFzX7Ga5vYe/view?usp=sharing) or [pretrained_models_ResNet-50](https://drive.google.com/file/d/10JgeeAjLMsgg4emoOJCMkH5B2oikJb0_/view?usp=sharing). Next, create a ```./pretrained_models``` folder to unzip the pretrained models downloaded inside that rootpath.

## Training process

The current implementation works in CPU. However, it can work with GPU too. To set a gpu device, uncomment line ```191  # gpus=1```, and comment line ```192   move_metrics_to_cpu = True``` from the [train.py](train.py) file.

Once you have all the pre-processed data from ```./data``` folder and device trainer defined, you can run:

```
python train.py --vgg16=True --path_to_data=./data --validate_on_person=1 --test_on_person=0
```

to train the model with vgg16 architecture, or run:

```
python train.py --resnet50=True --path_to_data=./data --validate_on_person=1 --test_on_person=0
```

to train the model with the porposed resnet50 architecture.

## Testing process

The current implementation works in CPU. However, it can work with GPU too. To set a gpu device, uncomment line ```27  # gpus=1```, and comment line ```26   move_metrics_to_cpu = True``` from the [eval.py](eval.py) file.

In case you have trained the model from scratch, you will now have a ```./pretrained_models``` folder with ```pxx.ckpt``` files inside. Otherwise, if you dowload the pretrained modles provided in section 'Pretrained Modles', you now have a ```./pretrained_models``` folder with some ```pxx.ckpt``` inside.

After all this previous process, you can now run:
```
python eval.py --path_to_checkpoints=./pretrained_models --path_to_data=./data
```
which will take the pretrained model from Vgg16 or ResNet50 ```pxx.ckpt``` files depending on the model you have choosen.

Then, you will see all the data values presented in our paper.

## Results

Our implementation work presents a Table with all the evaluation result using mean angular error as a metric to evaluate performance and to compare both models, benchmark and proposed. Thoses values from the table demonstrate that our proposed model outperform the benchamark paper by up to ~33%.

![Table-MAE](https://raw.githubusercontent.com/SaulFigue/Gaze-tracking-pipeline/main/images/Tabla-MAE.PNG)

In addition, we have plot a pitch and yaw comparision for participan 13 using both models results.

<table>
  <tr>
    <td>
      <img src="https://raw.githubusercontent.com/SaulFigue/Gaze-tracking-pipeline/main/images/Vgg-Pitch.png" alt="Vgg-Pitch">
    </td>
    <td>
      <img src="https://raw.githubusercontent.com/SaulFigue/Gaze-tracking-pipeline/main/images/ResNet-Pitch.png" alt="ResNet-Pitch">
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://raw.githubusercontent.com/SaulFigue/Gaze-tracking-pipeline/main/images/Vgg-Yaw.png" style="display: block; max-width: 100%; height: auto;" alt="Vgg-Yaw">
    </td>
    <td>
      <img src="https://raw.githubusercontent.com/SaulFigue/Gaze-tracking-pipeline/main/images/ResNet-Yaw.png" style="display: block; max-width: 100%; height: auto;" alt="ResNet-Yaw">
    </td>
  </tr>
</table>


Finally, to evaluate the performance of the training process we build a Train/Angular-error and Train/Loss graphical representaion for both models.

<table>
  <tr>
    <td>
      <img src="https://raw.githubusercontent.com/SaulFigue/Gaze-tracking-pipeline/main/images/Train-AE-Both.png" alt="Train-AE-Both">
    </td>
    <td>
      <img src="https://raw.githubusercontent.com/SaulFigue/Gaze-tracking-pipeline/main/images/Train-loss-Both.png" alt="Train-loss-Both">
    </td>
  </tr>
</table>


### How to plot same results presented in the paper [1].

All the plots presented in our work can be visualizeed by using the events files from [lightning_logs-ResNet50](lightning_logs-ResNet50), [tb_logs-ResNet50](tb_logs-ResNet50) and [tb_logs-VGG16](tb_logs-VGG16). For this start a new tensorboard session and:
- Run: 
```
tennsorboard --logdir=./
```

## Bibliography

[1] Figueroa, S.; Pineda, I.; Vizcaíno, P.; Reyes-Chacón, I. and Morocho-Cayamcela, M. (2024). A Webcam Artificial Intelligence-Based Gaze-Tracking Algorithm.  In Proceedings of the 19th International Conference on Software Technologies, ISBN 978-989-758-706-1, ISSN 2184-2833, pages 228-235.

[2] Gudi, A., Li, X., & van Gemert, J. (2020). Efficiency in real-time webcam gaze tracking. In Computer Vision–ECCV 2020 Workshops: Glasgow, UK, August 23–28, 2020, Proceedings, Part I 16 (pp. 529-543). Springer International Publishing.
