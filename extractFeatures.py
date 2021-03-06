from pycocotools.coco import COCO
import torchvision.models as model
from torch.autograd import Variable
import torch
import torch.nn as nn
import skimage.io as io
import numpy as np
import json
import pdb

# pretrained model
vgg16 = model.vgg16_bn(pretrained=True)
for param in vgg16.parameters():
    param.requires_grad = False
print("Model loaded.")

# parameter for image json file
coco = COCO('./annotations/instances_val2014.json')

# parameter for image_id json file
data = json.load(open('./visdial_0.9_val.json'))
dialogs = data['data']['dialogs']

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(*list(vgg16.features.children()))
    def forward(self, x):
        x = self.features(x)
        return x

mean = np.array([0.485, 0.456, 0.406])[:, np.newaxis, np.newaxis]
std = np.array([0.229, 0.224, 0.225])[:, np.newaxis, np.newaxis]

net = VGG16().cuda()
numOfDialogs = len(dialogs)

output = np.zeros((1000, 512, 7, 7), dtype=float)
cnt = 1
for i in range(numOfDialogs):
    imgIds = dialogs[i]['image_id']
    img = coco.loadImgs(imgIds)[0]
    I = io.imread(img['coco_url'])
    I.resize(224, 224, 3)
    I = np.transpose(I, (2, 0, 1))
    I = (I / 255 - mean) / std
    I = np.expand_dims(I, axis=0)
    output[i % 1000,:,:,:] = net(Variable(torch.from_numpy(I).float().cuda())).cpu().data.numpy()

    if i % 1000 == 999:
        np.save('vgg16_val_feature_' + str(cnt) + '.npy', output)
        output = np.zeros((1000, 512, 7, 7), dtype=float)
        cnt += 1
        print(i+1)
