import torch.utils.data as data

from PIL import Image
import h5py
import os
import os.path
#import math
import scipy.io
import numpy as np
import random
import csv

def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    # print f_list
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename

def getDistortionTypeFileName(path, num):
    filename = []
    index = 1
    for i in range(0,num):
        name = '%s%s%s' % ('img',str(index),'.bmp')
        filename.append(os.path.join(path,name))
        index = index + 1
    return filename
        

class TIDFolder(data.Dataset):

    def __init__(self, root, loader, index, transform=None, target_transform=None):

        refpath = os.path.join(root, 'reference_images')
        refname = getTIDFileName(refpath,'.bmp.BMP')
        # refname = getFileName(refpath,'.BMP')

        
        # txtpath = os.path.join(root, 'mos_with_names.txt')
        # fh = open(txtpath, 'r')
        # imgnames = []
        # target = []
        # refnames_all = []
        # for line in fh:
        #     line = line.split('\n')
        #     words = line[0].split()
        #     imgnames.append((words[1]))
        #     target.append(words[0])
        #     ref_temp = words[1].split("_")
        #     refnames_all.append(ref_temp[0][1:])


        self.histlabels = []
        self.imgname=[]
        refnames_all = []
        self.csv_file = os.path.join(root, 'TIDhist.txt')
        with open(self.csv_file) as f:
            reader = f.readlines()
            for i, line in enumerate(reader):
                token = line.split("\t")
                token[0]=eval(token[0]) #LIVE去除字符串两端的引号
                self.imgname.append(token[0])
                values = np.array(token[1:11], dtype='float32')
                values /= values.sum()
                self.histlabels.append(values)

                ref_temp = token[0].split("_")
                refnames_all.append(ref_temp[0][1:])



        # labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)


        sample = []
        # for i, item in enumerate(index):
        #     print(refname[index[i]])
        for i, item in enumerate(index):
            # print(refname[index[i]])
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                # for aug in range(patch_num):
                sample.append((os.path.join(root, 'distorted_images', self.imgname[item]), self.histlabels[item]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename


def getTIDFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if suffix.find(os.path.splitext(i)[1]) != -1:
            filename.append(i[1:3])
    return filename

if __name__ == '__main__':
    liveroot = '/DATA/gaoyixuan_data/tid2013'
    index = list(range(0,25))
    random.shuffle(index)
    train_index = index[0:round(0.8*25)]
    test_index = index[round(0.8*25):25]
    trainset = TIDFolder(root = liveroot, loader = default_loader, index = train_index)
    testset = TIDFolder(root = liveroot, loader = default_loader, index = test_index)
    # img_num = {
    #     'live': list(range(0, 29)),
    #     'csiq': list(range(0, 30)),
    #     'tid2013': list(range(0, 25)),
    #     'livec': list(range(0, 1162)),
    #     'koniq-10k': list(range(0, 10073)),
    #     'bid': list(range(0, 586)),
    # }