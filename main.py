import os
o_path = os.getcwd() 
from SCNN import SCNN
import torch
import torchvision
import torch.nn as nn
from PIL import Image
import scipy.stats
import random
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from EMDLoss import EMDLoss
from SoftHistogram import SoftHistogram
from MQuantileLoss_fixedbins import MQuantileLoss
from utils import score_utils


from perceptnet.gdn import GDN
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
    
def EMD(y_true, y_pred):
    cdf_ytrue = np.cumsum(y_true, axis=-1)
    cdf_ypred = np.cumsum(y_pred, axis=-1)
    samplewise_emd = np.sqrt(np.mean(np.square(np.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return np.mean(samplewise_emd)

def JSD(y_true, y_pred):
    M=(y_true+y_pred)/2
    js=0.5*scipy.stats.entropy(y_true, M)+0.5*scipy.stats.entropy(y_pred, M)
    return js

def histogram_intersection(h1, h2):
    intersection = 0
    for i in range(len(h1)):
        intersection += min(h1[i], h2[i])
    return intersection



class myChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(myChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
 
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        # self.gdn1 = GDN(in_planes, apply_independently=True)
 
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = (self.fc2(self.relu1(self.fc1(self.avg_pool(x)))))
        max_out = (self.fc2(self.relu1(self.fc1(self.max_pool(x)))))
        out = avg_out + max_out
        return self.sigmoid(out)
 
class DBCNN(torch.nn.Module):

    def __init__(self, scnn_root, options):
        """Declare all needed layers."""
        nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.base = torchvision.models.vgg16(pretrained=True).features#保存的是fc前面的网络结构，
        
        self.features1_2=nn.Sequential(*list(self.base.children())[0:16])
        self.max=nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.features1_3=nn.Sequential(*list(self.base.children())[0:31])        
        scnn = SCNN()
        scnn = torch.nn.DataParallel(scnn).cuda()
        scnn.load_state_dict(torch.load(scnn_root))
        self.features2 = scnn.module.features
        self.projection = nn.Sequential(nn.Conv2d(128,512,1,1,0), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                                        )
        self.in_channels = 512
        self.num_bins =5
        self.histogram=SoftHistogram(n_features=self.in_channels*14*14,n_examples=1,num_bins=self.num_bins,quantiles=False)
        self.fc = torch.nn.Linear((self.in_channels*self.num_bins)*14*14, options['numbin'])
        self.CAfeatures=myChannelAttention(self.in_channels)

    def forward(self, X):
        """Forward pass of the network.
        """
        N = X.size()[0]
        
        X2=self.features2(X)
        X2=self.projection(X2)
        X2=self.max(X2)#
        X2_w=self.CAfeatures(X2)
        X1=self.features1_3(X)#
        X1=X1*X2_w
        X1 = torch.nn.functional.normalize(X1)
        X = torch.nn.functional.normalize(X1.view(N,-1))
        hist = torch.Tensor(N,X.size(1)*self.num_bins).cuda()
        for i,x in enumerate(X):
            hist[i]=self.histogram(x)
        X = self.fc(hist)
        X=F.softmax(X,dim=1)
        return X


class DBCNNManager(object):
    def __init__(self, options, path):
        """Prepare the network, criterion, solver, and data.
        Args:
            options, dict: Hyperparameters.
        """
        print('Prepare the network and data.')
        self._options = options
        self._path = path

        # Network.
        self._net = torch.nn.DataParallel(DBCNN(self._path['scnn_root'], self._options), device_ids=[0]).cuda()

        print(self._net)
        # Criterion.
        self._criterion1 = EMDLoss()
        self._criterion2 = MQuantileLoss()
        # Solver.
        self._solver = torch.optim.Adam(
                self._net.module.parameters(), lr=self._options['base_lr'],
                weight_decay=self._options['weight_decay'])

        

        crop_size = 448
        train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((448,448)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(size=crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                            std=(0.229, 0.224, 0.225))
        ])

        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((448,448)),
            # torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=crop_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
            
           
        import LIVEFolder
        train_data = LIVEFolder.LIVEFolder(
                root=self._path['live'], loader = default_loader, index = self._options['train_index'],
                transform=train_transforms)
        test_data = LIVEFolder.LIVEFolder(
                root=self._path['live'], loader = default_loader, index = self._options['test_index'],
                transform=test_transforms)
        self._train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self._options['batch_size'],
            shuffle=True, num_workers=0, pin_memory=True)
        self._test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=1,
            shuffle=False, num_workers=0, pin_memory=True)

    def train(self):
        """Train the network."""
        print('Training.')
        best_Cosine = 0.0
        best_EMD = 0.0
        best_RMSE = 0.0
        best_JSD = 0.0
        best_intersection = 0.0
        best_CC = 0.0
        
        best_MOSsrcc = 0.0
        best_MOSplcc = 0.0
        best_MOSrmse = 0.0
        best_SOSsrcc = 0.0
        best_SOSplcc = 0.0
        best_SOSrmse = 0.0
        best_epoch = None
        print('Epoch\tTest_JSD\tTest_EMD\tTest_RMSE\tTest_inter\tTest_CC\tTest_Cosine\tMOSsrcc\tMOSplcc\tMOSrmse\tSOSsrcc\tSOSplcc\tSOSrmse')
        for t in range(self._options['epochs']):
            epoch_loss = []
            for X, y in self._train_loader:
                # Data.
                X = X.cuda()
                y = y.cuda()
                # Clear the existing gradients.
                self._solver.zero_grad()
                # Forward pass.
                score = self._net(X)
                loss1 = self._criterion1(score, y.view(len(score),self._options['numbin']).detach())
                loss2 = self._criterion2(score, y.view(len(score),self._options['numbin']).detach())
                loss=self._options['numbin']*loss1+loss2
                epoch_loss.append(loss.item())
                # Backward pass.
                loss.backward()
                self._solver.step()
            JSDtest,EMDtest,RMSEtest,intersectiontest,CCtest,Cosinetest,MOSsrcc,MOSplcc,MOSrmse,SOSsrcc,SOSplcc,SOSrmse = self._consitency(self._test_loader)
            if Cosinetest >= best_Cosine:
                best_Cosine = Cosinetest
                best_EMD = EMDtest
                best_RMSE = RMSEtest
                best_JSD = JSDtest
                best_intersection = intersectiontest
                best_CC = CCtest
                #计算MOS的结果，但是MOS是直接由predicted histogram计算出来的
                best_MOSsrcc = MOSsrcc
                best_MOSplcc = MOSplcc
                best_MOSrmse = MOSrmse
                best_SOSsrcc = SOSsrcc
                best_SOSplcc = SOSplcc
                best_SOSrmse = SOSrmse
                best_epoch = t + 1
                print('*', end='')
                pwd = os.getcwd()
                modelpath = os.path.join(pwd,'models',('net_params' + '_best' + '.pkl'))
                torch.save(self._net.state_dict(), modelpath)

            print('%d\t%4.4f\t%4.4f\t%4.4f\t%4.4f\t%4.4f\t%4.4f\t\t%4.4f\t%4.4f\t%4.4f\t\t%4.4f\t%4.4f\t%4.4f' %
                  (t+1,  JSDtest,EMDtest,RMSEtest,intersectiontest,CCtest,Cosinetest,MOSsrcc,MOSplcc,MOSrmse,SOSsrcc,SOSplcc,SOSrmse))           

        print('Best at epoch %d, test srcc %f' % (best_epoch, best_Cosine))
        return best_JSD,best_EMD, best_RMSE,best_intersection,best_CC,best_Cosine, best_MOSsrcc,best_MOSplcc,best_MOSrmse, best_SOSsrcc,best_SOSplcc,best_SOSrmse


    def _consitency(self, data_loader):
        self._net.train(False)
        num_total = 0

        JSD_test = []
        JSD_all=0
        JSDtest=0
        
        EMD_test = []
        EMD_all=0
        EMDtest=0

        RMSE_all=0
        RMSE0=0
        RMSE_test=[]
        RMSEtest=0

        Cosine_all=0
        Cosine0=0
        Cosine_test=[]
        Cosinetest=0
        
        CC_all=0
        CC0=0
        CC_test=[]
        CCtest=0
        
        intersection_test = []
        intersection_all=0
        intersectiontest=0
                
        pscores_MOS = []
        tscores_MOS = []
        pscores_SOS = []
        tscores_SOS = []
        
        for X, y in data_loader:
            # Data.
            X = X.cuda()
            y = y.cuda()
            # Prediction.
            score = self._net(X)
            score=score[0].cpu()
            y=y[0].cpu()
            pscores_MOS.append(score_utils.mean_score(score.detach().numpy()))
            tscores_MOS.append(score_utils.mean_score(y.detach().numpy()))
            pscores_SOS.append(score_utils.std_score(score.detach().numpy()))
            tscores_SOS.append(score_utils.std_score(y.detach().numpy()))
            
            ##histogram
            RMSE0=np.sqrt(((score.detach().numpy() - y.detach().numpy()) ** 2).mean())#对于每张直方图，求结果
            EMD0=EMD(score.detach().numpy(),y.detach().numpy())
            JSD0=JSD(score.detach().numpy(),y.detach().numpy())
            intersection0=histogram_intersection(score.detach().numpy(),y.detach().numpy())

            X=[score.detach().numpy(),y.detach().numpy()]
            Cosine0 = (1-pairwise_distances( X, metric='cosine'))[0][1]
            CC0 = np.corrcoef(X)[0][1]

            JSD_test.append(JSD0)
            EMD_test.append(EMD0)
            RMSE_test.append(RMSE0)
            intersection_test.append(intersection0)
            Cosine_test.append(Cosine0)
            CC_test.append(CC0)

        num_total =len(EMD_test)
        for ele in range(0, len(EMD_test)):
            JSD_all = JSD_all + JSD_test[ele]  
            EMD_all = EMD_all + EMD_test[ele]  
            RMSE_all = RMSE_all + RMSE_test[ele]  
            intersection_all = intersection_all + intersection_test[ele] 
            Cosine_all = Cosine_all + Cosine_test[ele] 
            CC_all = CC_all + CC_test[ele]
        JSDtest=JSD_all/num_total
        EMDtest=EMD_all/num_total
        RMSEtest=RMSE_all/num_total
        intersectiontest=intersection_all/num_total
        Cosinetest=Cosine_all/num_total
        CCtest=CC_all/num_total
        
        ##MOS
        MOSsrcc, _ = scipy.stats.spearmanr(pscores_MOS,tscores_MOS)
        MOSplcc, _ = scipy.stats.pearsonr(pscores_MOS,tscores_MOS)
        MOSrmse=np.sqrt((((pscores_MOS)-np.array(tscores_MOS))**2).mean())
        SOSsrcc, _ = scipy.stats.spearmanr(pscores_SOS,tscores_SOS)
        SOSplcc, _ = scipy.stats.pearsonr(pscores_SOS,tscores_SOS)
        SOSrmse=np.sqrt((((pscores_SOS)-np.array(tscores_SOS))**2).mean())
        self._net.train(True)  # Set the model to training phase
        return JSDtest,EMDtest,RMSEtest,intersectiontest,CCtest,Cosinetest,MOSsrcc,MOSplcc,MOSrmse,SOSsrcc,SOSplcc,SOSrmse

def main():
    """The main function."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Train DB-CNN for BIQA.')
    parser.add_argument('--base_lr', dest='base_lr', type=float, default=1e-5,
                        help='Base learning rate for training.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=8, help='Batch size:8.')
    parser.add_argument('--epochs', dest='epochs', type=int,
                        default=50, help='Epochs for training:50.')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                        default=5e-4, help='Weight decay.')
    parser.add_argument('--dataset',dest='dataset',type=str,default='live',
                        help='dataset: live|KONIQ10K')
    parser.add_argument('--seed',  type=int, default=1998)
    
    args = parser.parse_args()
    
    seed = random.randint(10000000, 99999999)
    torch.manual_seed(seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    print("seed:", seed)
    if args.base_lr <= 0:
        raise AttributeError('--base_lr parameter must >0.')
    if args.batch_size <= 0:
        raise AttributeError('--batch_size parameter must >0.')
    if args.epochs < 0:
        raise AttributeError('--epochs parameter must >=0.')
    if args.weight_decay <= 0:
        raise AttributeError('--weight_decay parameter must >0.')

    options = {
        'base_lr': args.base_lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
        'dataset':args.dataset,
        'fc': [],
        'train_index': [],
        'test_index': [],
        'numbin':10
    }
    
    path = {
        'live': os.path.join('/home/gyx/DATA/imagehist','LIVE'),
        'fc_model': os.path.join('fc_models'),
        'scnn_root': os.path.join('pretrained_scnn','scnn.pkl'),
        'fc_root': os.path.join('fc_models','net_params_best.pkl'),
        'db_model': os.path.join('db_models')
    }
    
    
    if options['dataset'] == 'live':          
        index = list(range(0,29))
        options['numbin'] == 10

    
    
    lr_backup = options['base_lr']
    EMD_all = np.zeros((1,10),dtype=np.float)
    RMSE_all = np.zeros((1,10),dtype=np.float)
    Cosine_all = np.zeros((1,10),dtype=np.float)
    JSD_all = np.zeros((1,10),dtype=np.float)
    inter_all = np.zeros((1,10),dtype=np.float)
    CC_all = np.zeros((1,10),dtype=np.float)
    
    MOSsrcc_all = np.zeros((1,10),dtype=np.float)
    MOSplcc_all = np.zeros((1,10),dtype=np.float)
    MOSrmse_all = np.zeros((1,10),dtype=np.float)
    SOSsrcc_all = np.zeros((1,10),dtype=np.float)
    SOSplcc_all = np.zeros((1,10),dtype=np.float)
    SOSrmse_all = np.zeros((1,10),dtype=np.float)
    
    for i in range(0,10):
        #randomly split train-test set
        random.shuffle(index)
        train_index = index[0:round(0.8*len(index))]
        test_index = index[round(0.8*len(index)):len(index)]
    
        options['train_index'] = train_index
        options['test_index'] = test_index
    
        #fine-tune all model
        options['fc'] = False
        options['base_lr'] = lr_backup
        manager = DBCNNManager(options, path)
        JSD,EMD, RMSE,inter,CC,Cosine, MOSsrcc,MOSplcc,MOSrmse, SOSsrcc,SOSplcc,SOSrmse = manager.train()
        
        EMD_all[0][i] = EMD
        RMSE_all[0][i] = RMSE
        Cosine_all[0][i] = Cosine
        JSD_all[0][i] = JSD
        inter_all[0][i] = inter
        CC_all[0][i] = CC
        
        #
        MOSsrcc_all[0][i] = MOSsrcc
        MOSplcc_all[0][i] = MOSplcc
        MOSrmse_all[0][i] = MOSrmse
        SOSsrcc_all[0][i] = SOSsrcc
        SOSplcc_all[0][i] = SOSplcc
        SOSrmse_all[0][i] = SOSrmse
        print(i)
        
        
    EMD_mean = np.mean(EMD_all)
    RMSE_mean = np.mean(RMSE_all)
    Cosine_mean = np.mean(Cosine_all)
    JSD_mean = np.mean(JSD_all)
    inter_mean = np.mean(inter_all)
    CC_mean = np.mean(CC_all)
    
    MOSsrcc_mean = np.mean(MOSsrcc_all)
    MOSplcc_mean = np.mean(MOSplcc_all)
    MOSrmse_mean = np.mean(MOSrmse_all)
    SOSsrcc_mean = np.mean(SOSsrcc_all)
    SOSplcc_mean = np.mean(SOSplcc_all)
    SOSrmse_mean = np.mean(SOSrmse_all)
    print("seed:", seed)
    print(JSD_all)
    print('average JSD:%4.4f' % (JSD_mean))  
    print(EMD_all)
    print('average EMD:%4.4f' % (EMD_mean))  
    print(RMSE_all)
    print('average RMSE:%4.4f' % (RMSE_mean))  
    print(inter_all)
    print('average inter:%4.4f' % (inter_mean))  
    print(CC_all)
    print('average CC:%4.4f' % (CC_mean))  
    print(Cosine_all)
    print('average Cosine:%4.4f' % (Cosine_mean))  
    
    print(MOSsrcc_all)
    print('average MOSsrcc:%4.4f' % (MOSsrcc_mean))  
    print(MOSplcc_all)
    print('average MOSplcc:%4.4f' % (MOSplcc_mean))  
    print(MOSrmse_all)
    print('average MOSrmse:%4.4f' % (MOSrmse_mean))  
    print(SOSsrcc_all)
    print('average SOSsrcc:%4.4f' % (SOSsrcc_mean))  
    print(SOSplcc_all)
    print('average SOSplcc:%4.4f' % (SOSplcc_mean))  
    print(SOSrmse_all)
    print('average SOSrmse:%4.4f' % (SOSrmse_mean))  
    return EMD_all,RMSE_all,Cosine_all,MOSsrcc_all,MOSplcc_all,MOSrmse_all
if __name__ == '__main__':
    main()


