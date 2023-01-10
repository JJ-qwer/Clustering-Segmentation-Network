import os
import cv2
import numpy as np
from skimage import segmentation
from evaluate import evaluate

import torch
import torch.nn as nn
from pre import *
import random
import torch.backends.cudnn as cudnn
from attention import *
import matplotlib
import matplotlib.pyplot as plt
from sparse import sparse_fuse


class Args(object):
    input_image_path_1 = 'data/Farmland_1.bmp'
    input_image_path_2 = 'data/Farmland_2.bmp'
    imgt_path = 'data/Farmland_gt.bmp'
    train_epoch = 2 ** 7
    mod_dim1 = 256
    mod_dim2 = 128
    gpu_id = 0
    min_label_num = 2
    max_label_num = 256


class MyNet(nn.Module):
    def __init__(self, inp_dim, mod_dim1, mod_dim2):
        super(MyNet, self).__init__()

        self.seq = nn.Sequential(
            CBAM(in_channels=inp_dim),

            nn.Conv2d(inp_dim, mod_dim1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim1, mod_dim2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim2),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim2, mod_dim1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim1, mod_dim2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim2),
        )

    def forward(self, x):
        return self.seq(x)


def run():
    args = Args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)  # choose GPU:0
    im1 = cv2.imread(args.input_image_path_1, 0)
    im2 = cv2.imread(args.input_image_path_2, 0)
    im_gt = cv2.imread(args.imgt_path, 0)
    if len(im1.shape) == 3:
        im1 = im1[:, :, 0].astype(np.uint8)
    else:
        im1 = im1.astype(np.uint8)

    if len(im2.shape) == 3:
        im2 = im2[:, :, 0].astype(np.uint8)
    else:
        im2 = im2.astype(np.uint8)

    if len(im_gt.shape) == 3:
        im_gt = im_gt[:, :, 0].astype(np.uint8)
    else:
        im_gt = im_gt.astype(np.uint8)
    im1 = srad(im1, 0.15).astype(np.uint8)
    im2 = srad(im2, 0.15).astype(np.uint8)
    image = dicomp(im1, im2)
    image_div = image[:]

    image_t = np.zeros([image.shape[0], image.shape[1], 3])
    image_t[:, :, 0] = image * 255
    image_t[:, :, 1] = image * 255
    image_t[:, :, 2] = image * 255
    image = image_t

    '''segmentation ML'''

    seg_map = segmentation.felzenszwalb(image, scale=4096, sigma=0.5, min_size=8)

    ylen, xlen = image_div.shape
    pix_vec = image_div.reshape([ylen * xlen, 1])
    preclassify_lab = hcluster(pix_vec, image_div)

    preclassify_lab = preprocess(image, preclassify_lab)

    seg_map = seg_map.flatten()
    seg_lab = [np.where(seg_map == u_label)[0]
               for u_label in np.unique(seg_map)]

    '''train init'''
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    tensor = image.transpose((2, 0, 1))
    tensor = tensor.astype(np.float32) / 255.0
    tensor = tensor[np.newaxis, :, :, :]
    tensor = torch.from_numpy(tensor).to(device)

    model = MyNet(inp_dim=3, mod_dim1=args.mod_dim1, mod_dim2=args.mod_dim2).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9)

    image_flatten = image.reshape((-1, 3))
    color_avg = np.random.randint(255, size=(args.max_label_num, 3))

    show = image

    '''train loop'''
    model.train()
    for batch_idx in range(args.train_epoch):
        '''forward'''
        optimizer.zero_grad()
        output = model(tensor)[0]
        output = output.permute(1, 2, 0).view(-1, args.mod_dim2)
        target = torch.argmax(output, 1)
        im_target = target.data.cpu().numpy()

        '''refine'''
        for inds in seg_lab:
            u_labels, hist = np.unique(im_target[inds], return_counts=True)
            im_target[inds] = u_labels[np.argmax(hist)]

        '''backward'''
        target = torch.from_numpy(im_target)
        target = target.to(device)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        '''show image'''
        un_label, lab_inverse = np.unique(im_target, return_inverse=True, )
        if un_label.shape[0] < args.max_label_num:  # update show
            img_flatten = image_flatten.copy()
            if len(color_avg) != un_label.shape[0]:
                color_avg = [np.mean(img_flatten[im_target == label], axis=0, dtype=np.int) for label in un_label]
            for lab_id, color in enumerate(color_avg):
                img_flatten[lab_inverse == lab_id] = color
            show = img_flatten.reshape(image.shape)
        cv2.imshow("seg_pt", normalization(show))
        cv2.waitKey(1)

        print('Loss:', batch_idx, loss.item())
        if len(un_label) < args.min_label_num:
            break

    '''save'''
    show = show[:, :, 0]
    show = np.clip(show, 0, 255)

    img_cd = sparse_fuse(show, preclassify_lab, 255 * normalization(image_div), 5, 20, 3)

    show = postprocess(img_cd)

    evaluate(im_gt, show)


if __name__ == '__main__':
    run()
