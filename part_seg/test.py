import argparse
import os
import torch
import datetime
import logging
import sys
import random
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from tqdm import tqdm 
from get_model_shapenet import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class PartNormalDataset(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel


        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000


    def __getitem__(self, index):
        # if index in self.cache:
        #     point_set, cls, seg = self.cache[index]
        # else:
        fn = self.datapath[index]
        cat = self.datapath[index][0]
        cls = self.classes[cat]
        cls = np.array([cls]).astype(np.int32)
        # print(fn[1])
        # exit()
        # data = np.load(fn[1]).astype(np.float32)
        data = np.loadtxt(fn[1]).astype(np.float32)
        if not self.normal_channel:
            point_set = data[:, 0:3]
        else:
            point_set = data[:, 0:6]
        seg = data[:, -1].astype(np.int32)
        if len(self.cache) < self.cache_size:
            self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]
        get_path = self.datapath[index][1].split('/')[-1].split('.')[0]
        get_category = self.datapath[index][0]
  
        return point_set, cls, seg, f'{get_category}_{get_path}'

    def __len__(self):
        return len(self.datapath)



def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args(model_name):
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default=model_name, help='model name')
    parser.add_argument('--batch_size', type=int, default=1, help='batch Size during training')
    parser.add_argument('--re_feat', type=bool, default=False)
    parser.add_argument('--epoch', default=1, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=3e-3, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Adam or SGD')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    parser.add_argument('--num_classes', type=int, default=16, help='classes number')
    parser.add_argument('--num_part', type=int, default=50, help='number of part')
    if model_name in model_list2:
        parser.add_argument('--normal', action='store_true', default=True, help='use normals')
    else:
        parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.8, help='decay rate for lr decay')

    return parser.parse_args()

def weights_init(m):
    classname = m.__class__.__name__
    try:
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
    except:
        pass


def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    exp_dir = os.path.join('./result/', f'{args.model}')
    os.makedirs(exp_dir, exist_ok=True)


    root = os.path.join('../data','shapenetcore_partanno_segmentation_benchmark_v0_normal')

    save_txt = os.path.join(exp_dir, f'{args.model}')
    os.makedirs(save_txt, exist_ok = True)

    TEST_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False )

    # getfns = TEST_DATASET.getfns()

    classifier = getmodel(strss=args.model, batchsize =args.batch_size, number_class = args.num_part, number_point=args.npoint, normal = args.normal)

    checkpoint = torch.load(os.path.join(exp_dir,  f'{args.model}.pth'))
    classifier.load_state_dict(checkpoint['model_state_dict'])
 
    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    # import json
    # with open(os.path.join(root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
    #     # print(f'{11}_{d.split('/')[2]}')
    #     data_name = set([str(d.split('/')[1]) + '_' + str(d.split('/')[2]) for d in json.load(f)]) 
  
    # data_name = list(data_name) 
    # print(getfns[0])
    # print(data_name[0])
    # # exit()

    classifier = classifier.eval()
    
    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(args.num_part)]
        total_correct_class = [0 for _ in range(args.num_part)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        classifier = classifier.eval()

        for batch_id, (points, label, target, data_name ) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            cur_batch_size, NUM_POINT, get_test_paths = points.size()

            
            getpoints = points.data.numpy()
            points = torch.Tensor(getpoints[:, :, 0:3])
            points2 = points.cpu().data.numpy().squeeze(0).copy()
            points_feature = torch.Tensor(getpoints[:, :, 3:6])

            points, label, target, points_feature = points.float().cuda(), label.long().cuda(), target.long().cuda(), points_feature.cuda()


            if args.model in ['PointNet', 'PointNetpp']:
                
                args.normal = False
                points = points.transpose(2, 1)
                points_feature = points_feature.transpose(2, 1)
                pointsss = torch.cat((points, points_feature), dim = 1)
                seg_pred, trans_feat = classifier(pointsss, to_categorical(label, args.num_classes))
                points2 = points2.transpose(1, 0)
                # points = points.transpose(2, 1)
                # seg_pred, _ = classifier(points, to_categorical(label, args.num_classes))
            elif args.model in ['GCANet', 'PointNext']: 
                points = points.transpose(2, 1)
                points_feature = points_feature.transpose(2, 1)
                seg_pred = classifier(points, points_feature)
            elif args.model in ['PointCNN', 'Point_TransformerV1']: 
                seg_pred = classifier(points)
                # points = points.transpose(2, 1)
            elif args.model in ['pointnet2_seg_DQ']: 
                points = points.transpose(2, 1)
                points_feature = points_feature.transpose(2, 1)
                pointsss = torch.cat((points, points_feature), dim = 1)
                seg_pred, _= classifier(pointsss,  to_categorical(label, args.num_classes))
            elif args.model in ['PointMLP']: 
                points = points.transpose(2, 1)
                points_feature = points_feature.transpose(2, 1)
                seg_pred = classifier(points, points_feature,  to_categorical(label, args.num_classes)) 
            elif args.model in ['pointnetKAN']:
                points = points.transpose(2, 1)
                seg_pred = classifier(points,  to_categorical(label, args.num_classes))
            else:
                
                points = points.transpose(2, 1)
                seg_pred = classifier(points, to_categorical(label, args.num_classes))
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = target.cpu().data.numpy()

            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
            # save_data = np.c_[points.cpu().data.numpy().squeeze(0).transpose(0, 1), cur_pred_val.reshape(args.npoint, 
                
            if args.model in ['PointCNN', 'PointMLP', 'PointMamba', 'PoinTramba', 'Point_TransformerV1','PointNext','pointnetKAN', 'PCT','pointnet2_seg_DQ']:
                points2 = points2.T

            data_1 = points2
            data_2 = cur_pred_val.reshape(1, args.npoint)
            # print(data_1.shape)
            # print(data_2.shape)
            # exit()
            # print(list(data_name)[0])
            # exit()
            get_name = list(data_name)[0]
            save_data = np.concatenate([data_1, data_2], axis=0).T 
            os.makedirs(os.path.join(save_txt, get_name .split('_')[0]), exist_ok=True)
            np.savetxt(os.path.join(save_txt, get_name .split("_")[0], f'{get_name .split("_")[1]}.txt'), save_data)


            correct = np.sum(cur_pred_val == target)
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)

            for l in range(args.num_part):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=float))
        for cat in sorted(shape_ious.keys()):
            print('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['class_avg_iou'] > best_class_avg_iou:
            best_class_avg_iou = test_metrics['class_avg_iou']
        if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
            best_inctance_avg_iou = test_metrics['inctance_avg_iou']

        print('test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['inctance_avg_iou']))
        
        torch.cuda.empty_cache()
            


def set_seed(seed=1):
    print('Using random seed', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



if __name__ == '__main__': 
#     model_list = ['PointNet','PointNetpp','PointCNN', 'Point_TransformerV1', 'DGCNN','PointMLP','PointNext','PointMamba','PoinTramba', 'pointnetKAN',]
    # PointNetpp normal == True # 'PointMLP',
    set_seed(seed=42)
    random.seed = 42
    # model_list = ['PointNet','PointNetpp', 'PointCNN', 'Point_TransformerV1',  'PointMLP','PointNext','PointMamba','PoinTramba', 'pointnetKAN',]
    model_list = [   'PointMLP','PointNext','PointMamba','PoinTramba', ]
    model_list2 = [ 'PointNext', 'pointnet2_seg_DQ', 'PointMLP']
    for ml in model_list:
        print(ml)
        args = parse_args(ml)
        
        main(args)
