import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchinfo import summary
from torchvision import models
from renset import resnet50


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation layer

    https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    """

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze
        self.fc = nn.Sequential(  # Excitation (similar to attention)
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FinalModel(LightningModule):
    def __init__(self, modelo, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.subject_biases = nn.Parameter(torch.zeros(15 * 2, 2))  # pitch and yaw offset for the original and mirrored participant

        ## ---------------------------------------------------------------------------
        ## ----------------------------- VGG-16 --------------------------------------
        ## ---------------------------------------------------------------------------

        """
        pre-trained VGG-16 architecture

        https://github.com/pperle/gaze-tracking/blob/main/model.py
        """

        if modelo == 'vgg16':

            self.cnn_face = nn.Sequential(
                models.vgg16(pretrained=True).features[:9],  # first four convolutional layers of VGG16 pretrained on ImageNet
                nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding='same'),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(2, 2)),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(3, 3)),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(5, 5)),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(11, 11)),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(128),
            )

            self.cnn_eye = nn.Sequential(
                models.vgg16(pretrained=True).features[:9],  # first four convolutional layers of VGG16 pretrained on ImageNet
                nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding='same'),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(2, 2)),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(3, 3)),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(4, 5)),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(5, 11)),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(128),
            )

        ## ---------------------------------------------------------------------------
        ## ---------------------------- RESNET-50 ------------------------------------
        ## ---------------------------------------------------------------------------

        elif modelo == 'resnet50':

            resnet = resnet50(pretrained=True)

            self.cnn_face = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,   # This ends at: ReLU 3-65, with Output Shape = [16,512,12,12]

            #Let's add some convolutions, ReLu and BatchNorm similar to Vgg16 to ends with 128,6,6
            
            # ----- CONVS + DILATION OPERATIONS -----------------------------------------------------
 
            nn.Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),  #Output Shape = [16,128,12,12]
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(3,3), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),  #Output Shape = [16,128,6,6]
            nn.ReLU(inplace=True),
            )

            self.cnn_eye = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2, # This ends at: ReLU 3-65, with Output Shape = [16,512,8,12]

            #Let's add some convolutions, ReLu and BatchNorm similar to Vgg16 to ends with 128,4,6
            
            # ----- CONVS + DILATION OPERATIONS -----------------------------------------------------
            
            nn.Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),  #Output Shape = [16,128,8,12]
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(2,3), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),  #Output Shape = [16,128,4,6]
            nn.ReLU(inplace=True), 
            )

        else:

            print('No model selected, default = vgg16')

            self.cnn_face = nn.Sequential(
                models.vgg16(pretrained=True).features[:9],  # first four convolutional layers of VGG16 pretrained on ImageNet
                nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding='same'),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(2, 2)),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(3, 3)),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(5, 5)),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(11, 11)),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(128),
            )

            self.cnn_eye = nn.Sequential(
                models.vgg16(pretrained=True).features[:9],  # first four convolutional layers of VGG16 pretrained on ImageNet
                nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding='same'),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(2, 2)),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(3, 3)),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(4, 5)),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(5, 11)),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(128),
            )


        self.fc_face = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6 * 6 * 128, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
        )

        self.cnn_eye2fc = nn.Sequential(
            SELayer(256),

            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            SELayer(256),

            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            SELayer(128),
        )

        self.fc_eye = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 6 * 128, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
        )

        self.fc_eyes_face = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(576, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),
            nn.Linear(256, 2),
        )

    def forward(self, person_idx: torch.Tensor, full_face: torch.Tensor, right_eye: torch.Tensor, left_eye: torch.Tensor):
        out_cnn_face = self.cnn_face(full_face)
        out_fc_face = self.fc_face(out_cnn_face)

        out_cnn_right_eye = self.cnn_eye(right_eye)
        out_cnn_left_eye = self.cnn_eye(left_eye)
        out_cnn_eye = torch.cat((out_cnn_right_eye, out_cnn_left_eye), dim=1)

        cnn_eye2fc_out = self.cnn_eye2fc(out_cnn_eye)  # feature fusion
        out_fc_eye = self.fc_eye(cnn_eye2fc_out)

        fc_concatenated = torch.cat((out_fc_face, out_fc_eye), dim=1)
        t_hat = self.fc_eyes_face(fc_concatenated)  # subject-independent term

        return t_hat + self.subject_biases[person_idx].squeeze(1)  # t_hat + subject-dependent bias term

if __name__ == '__main__':
    # from dataset.mpii_face_gaze_dataset import get_dataloaders
    # from torch.utils.tensorboard import SummaryWriter

    model = FinalModel()
    
    model.summarize(max_depth=1)

    batch_size = 16
    summary(model, [
        (batch_size, 1),
        (batch_size, 3, 96, 96),  # full face
        (batch_size, 3, 64, 96),  # right eye
        (batch_size, 3, 64, 96)  # left eye
    ], dtypes=[torch.long, torch.float, torch.float, torch.float])

