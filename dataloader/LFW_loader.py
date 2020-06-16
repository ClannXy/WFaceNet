import numpy as np
import scipy.misc
import os
import torch
import torch.utils.data
import cv2

class LFW(object):
    def __init__(self, imgl, imgr):

        self.imgl_list = imgl
        self.imgr_list = imgr

    def __getitem__(self, index):
        # imgl = scipy.misc.imread(self.imgl_list[index])
        imgl = cv2.imread(self.imgl_list[index])
        imgl = cv2.resize(imgl,IMG_SIZE)
        if len(imgl.shape) == 2:
            imgl = np.stack([imgl] * 3, 2)
        # imgr = scipy.misc.imread(self.imgr_list[index])
        imgr = cv2.imread(self.imgr_list[index])
        imgr = cv2.resize(imgr,IMG_SIZE)
        if len(imgr.shape) == 2:
            imgr = np.stack([imgr] * 3, 2)

        # imgl = imgl[:, :, ::-1]
        # imgr = imgr[:, :, ::-1]
        imglist = [imgl, imgl[:, ::-1, :], imgr, imgr[:, ::-1, :]]
        for i in range(len(imglist)):
            imglist[i] = (imglist[i] - 127.5) / 128.0
            imglist[i] = imglist[i].transpose(2, 0, 1)
        imgs = [torch.from_numpy(i).float() for i in imglist]
        return imgs

    def __len__(self):
        return len(self.imgl_list)

def parseList(root):
    with open(os.path.join(root, 'pairs.txt')) as f:
        pairs = f.read().splitlines()[1:]
    folder_name = 'lfw'
    nameLs = []
    nameRs = []
    folds = []
    flags = []
    for i, p in enumerate(pairs):
        p = p.split('\t')
        if len(p) == 3:
            nameL = os.path.join(root, folder_name, p[0], p[0] + '_' + '{:04}.jpg'.format(int(p[1])))
            nameR = os.path.join(root, folder_name, p[0], p[0] + '_' + '{:04}.jpg'.format(int(p[2])))
            fold = i // 600
            flag = 1
        elif len(p) == 4:
            nameL = os.path.join(root, folder_name, p[0], p[0] + '_' + '{:04}.jpg'.format(int(p[1])))
            nameR = os.path.join(root, folder_name, p[2], p[2] + '_' + '{:04}.jpg'.format(int(p[3])))
            fold = i // 600
            flag = -1
        nameLs.append(nameL)
        nameRs.append(nameR)
        folds.append(fold)
        flags.append(flag)
    # print(nameLs)
    return [nameLs, nameRs, folds, flags]

IMG_SIZE = (112,112)
if __name__ == '__main__':
    data_dir = 'Q:\paper\datasets\lfw'

    nl, nr, folds, flags = parseList(root=data_dir)
    testdataset = LFW(nl, nr)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=32,
                                            shuffle=False, num_workers=8, drop_last=False)
    # print(len(testdataset))
    for data in testloader:
        # print('1 iter')
        print(data[0].shape)
    pass