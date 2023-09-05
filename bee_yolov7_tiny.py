import torch
import torch.nn as nn
import torch.optim as optim
#from model.googlenet import Inception
from utils.ABCPruner_options import args
import utils.ABCPruner_common as utils

import os
import time
from datetime import datetime
import copy
import sys
import random
import numpy as np
import heapq
import yaml
from importlib import import_module
from copy import deepcopy
from torch.cuda import amp
#from data import cifar10, cifar100, imagenet
import torch.optim.lr_scheduler as lr_scheduler

import test
from models.yolo import Model
from utils.datasets import create_dataloader
from utils.general import colorstr
from utils.torch_utils import intersect_dicts, ModelEMA, is_parallel
from utils.general import one_cycle, init_seeds, check_img_size, \
    labels_to_class_weights, fitness
from utils.loss import ComputeLossOTA, ComputeLoss
from utils.autoanchor import check_anchors

# ------------------------------------------------ #

checkpoint = utils.checkpoint(args)
device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
logger = utils.get_logger(os.path.join(args.job_dir + '/' + 'logger.log'))
#loss_func = ComputeLossOTA()

conv_num_cfg = {
    'yolov7-tiny': 55
    }
food_dimension = conv_num_cfg[args.arch]

# Hyperparameters
with open(args.hyp) as f:
    hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

# Data
print('==> Loading Data..')
trainLoader = None
trainDataset = None

validLoader = None

# Model
print('==> Loading Model..')

ckpt = torch.load(args.honey_model, map_location=device)  # load checkpoint
origin_model = Model(args.cfg or ckpt['model'].yaml, ch=3, nc=1, anchors=hyp.get('anchors')).to(device)  # create

if args.honey_model is None or not os.path.exists(args.honey_model):
    raise ('Honey_model path should be exist!')

state_dict = ckpt['model'].float().state_dict()  # to FP32
state_dict = intersect_dicts(state_dict, origin_model.state_dict())  # intersect
origin_model.load_state_dict(state_dict, strict=False)  # load
oristate_dict = origin_model.state_dict()



#Define BeeGroup 
class BeeGroup():
    """docstring for BeeGroup"""
    def __init__(self):
        super(BeeGroup, self).__init__() 
        self.code = [] #size : num of conv layers  value:{1,2,3,4,5,6,7,8,9,10}
        self.fitness = 0
        self.rfitness = 0 
        self.trail = 0

#Initilize global element
best_honey = BeeGroup()
NectraSource = []
EmployedBee = []
OnLooker = []
#best_honey_state = {}

def load_yolov7_tiny_honey_model(model, random_rule):
    #print(ckpt['state_dict'])
    global oristate_dict
    state_dict = model.state_dict()
    last_select_index = None #Conv index selected in the previous layer

    for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d):

            oriweight = oristate_dict[name + '.weight']
            curweight = state_dict[name + '.weight']

            ori_output, ori_input, _, _ = oriweight.shape
            cur_output, cur_input, _, _ = curweight.shape

            if ori_output != cur_output and ori_input != cur_input:
                if random_rule == 'random_pretrain':
                    sampled_output_index = random.sample(range(0, ori_output-1), cur_output)
                    sampled_output_index.sort()
                    sampled_input_index = random.sample(range(0, ori_input-1), cur_input)
                    sampled_input_index.sort()

                elif random_rule == 'l1_pretrain':
                    l1_sum = list(torch.sum(torch.abs(oriweight), [3, 2, 1]))
                    sampled_output_index = list(map(l1_sum.index, heapq.nlargest(cur_output, l1_sum)))
                    sampled_output_index.sort()

                    l1_sum = list(torch.sum(torch.abs(oriweight), [3, 2, 0]))
                    sampled_input_index = list(map(l1_sum.index, heapq.nlargest(cur_input, l1_sum)))
                    sampled_input_index.sort()

                for cur_index_out, ori_index_out in enumerate(sampled_output_index):
                        for cur_index_in, ori_index_in in enumerate(sampled_input_index):
                            state_dict[name + '.weight'][cur_index_out, cur_index_in, :, :] = \
                                oristate_dict[name + '.weight'][ori_index_out, ori_index_in, :, :]
            
            elif ori_output == cur_output and ori_input != cur_input:
                if random_rule == 'random_pretrain':
                    sampled_input_index = random.sample(range(0, ori_input-1), cur_input)
                    sampled_input_index.sort()

                elif random_rule == 'l1_pretrain':
                    l1_sum = list(torch.sum(torch.abs(oriweight), [3, 2, 0]))
                    sampled_input_index = list(map(l1_sum.index, heapq.nlargest(cur_input, l1_sum)))
                    sampled_input_index.sort()

                for cur_index_in, ori_index_in in enumerate(sampled_input_index):
                    state_dict[name + '.weight'][:, cur_index_in, :, :] = \
                            oristate_dict[name + '.weight'][:, ori_index_in, :, :]
                        
            elif ori_output != cur_output and ori_input == cur_input:
                if random_rule == 'random_pretrain':
                    sampled_output_index = random.sample(range(0, ori_output-1), cur_output)
                    sampled_output_index.sort()
                    
                elif random_rule == 'l1_pretrain':
                    l1_sum = list(torch.sum(torch.abs(oriweight), [3, 2, 1]))
                    sampled_output_index = list(map(l1_sum.index, heapq.nlargest(cur_output, l1_sum)))
                    sampled_output_index.sort()

                for cur_index_out, ori_index_out in enumerate(sampled_output_index):
                    state_dict[name + '.weight'][cur_index_out, :, :, :] = \
                            oristate_dict[name + '.weight'][ori_index_out, :, :, :]
            
            else:
                state_dict[name + '.weight'] = oriweight

            # orifilter_num = oriweight.size(0)
            # currentfilter_num = curweight.size(0)

            # if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

            #     select_num = currentfilter_num
            #     if random_rule == 'random_pretrain':
            #         select_index = random.sample(range(0, orifilter_num-1), select_num)
            #         select_index.sort()
            #     else:
            #         l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
            #         select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
            #         select_index.sort()
            #     if last_select_index is not None:
            #         for index_i, i in enumerate(select_index):
            #             for index_j, j in enumerate(last_select_index):
            #                 state_dict[name + '.weight'][index_i][index_j] = \
            #                     oristate_dict[name + '.weight'][i][j]
            #     else:
            #         for index_i, i in enumerate(select_index):
            #             state_dict[name + '.weight'][index_i] = \
            #                 oristate_dict[name + '.weight'][i]

            #     last_select_index = select_index

            # else:
            #     state_dict[name + '.weight'] = oriweight
            #     last_select_index = None

    model.load_state_dict(state_dict)

# Create Optimizer
def createOptimizer(model, args):
    global hyp
    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / args.train_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= args.train_batch_size * accumulate / nbs  # scale weight_decay
    #logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
        if hasattr(v, 'im'):
            if hasattr(v.im, 'implicit'):           
                pg0.append(v.im.implicit)
            else:
                for iv in v.im:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imc'):
            if hasattr(v.imc, 'implicit'):           
                pg0.append(v.imc.implicit)
            else:
                for iv in v.imc:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imb'):
            if hasattr(v.imb, 'implicit'):           
                pg0.append(v.imb.implicit)
            else:
                for iv in v.imb:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imo'):
            if hasattr(v.imo, 'implicit'):           
                pg0.append(v.imo.implicit)
            else:
                for iv in v.imo:
                    pg0.append(iv.implicit)
        if hasattr(v, 'ia'):
            if hasattr(v.ia, 'implicit'):           
                pg0.append(v.ia.implicit)
            else:
                for iv in v.ia:
                    pg0.append(iv.implicit)
        if hasattr(v, 'attn'):
            if hasattr(v.attn, 'logit_scale'):   
                pg0.append(v.attn.logit_scale)
            if hasattr(v.attn, 'q_bias'):   
                pg0.append(v.attn.q_bias)
            if hasattr(v.attn, 'v_bias'):  
                pg0.append(v.attn.v_bias)
            if hasattr(v.attn, 'relative_position_bias_table'):  
                pg0.append(v.attn.relative_position_bias_table)
        if hasattr(v, 'rbr_dense'):
            if hasattr(v.rbr_dense, 'weight_rbr_origin'):  
                pg0.append(v.rbr_dense.weight_rbr_origin)
            if hasattr(v.rbr_dense, 'weight_rbr_avg_conv'): 
                pg0.append(v.rbr_dense.weight_rbr_avg_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_pfir_conv'):  
                pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'): 
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):   
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_dw'):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_pw'):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
            if hasattr(v.rbr_dense, 'vector'):   
                pg0.append(v.rbr_dense.vector)

    
    optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    #logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    return optimizer

# Create Model
def createModelwithHoney(args, honey=None):

    if honey is not None:
        with open(os.getcwd() + '/' + args.cfg) as file:
            data = yaml.load(file, Loader=yaml.FullLoader)

            honeyIndex = 0
            for layer in range(len(data['backbone'])):

                if data['backbone'][layer][2] == 'Conv':
                    data['backbone'][layer][3][0] = data['backbone'][layer][3][0] * (honey[honeyIndex] / 10)
                    honeyIndex += 1
            
            for layer in range(len(data['head'])):

                if data['head'][layer][2] == 'Conv':
                    data['head'][layer][3][0] = data['head'][layer][3][0] * (honey[honeyIndex] / 10)
                    honeyIndex += 1

            now = datetime.now()
            filename = now.strftime('%Y-%m-%d %H:%M:%S')

            with open(os.getcwd() + '/' + args.job_dir + '/' + filename + '.yaml', 'w') as f:
                yaml.dump(data, f)

            honeyCfg = os.getcwd() + '/' + args.job_dir + '/' + filename + '.yaml'
        
        createdModel = Model(honeyCfg, ch=3, nc=1, anchors=hyp.get('anchors')).to(device)  # create
    
    else:
        createdModel = Model(args.cfg, ch=3, nc=1, anchors=hyp.get('anchors')).to(device)  # create

    return createdModel

def train_yolo(model, hyp, args, isSave, epochs):
    """
    yolo 모델을 학습합니다.

    Parameters
    ----------
    model (YOLO model with weight loaded)
        weight가 load되어 있는 YOLO클래스의 Model함수의 리턴값을 입력으로 받는 인자.

    hyp (loaded yaml)
        YOLO모델에 사용되는 하이퍼파라미터 파일.

    args
        프로그램 실행 시 입력받은 인자를 입력받아야 합니다.

    isSave (bool)
        학습시킨 모델의 저장 여부를 결정하는 flag입니다.

    epochs (int)
        모델을 몇 epoch동안 학습시킬지 결정하는 인자입니다.

    Returns
    --------
    fitness value
        학습된 모델의 fitness값을 반환한다.

    Note
    ----
    fitness calculation
        Model fitness as a weighted combination of metrics
        w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
        return (input[:, :4] * w).sum(1)
    """
    # Hardcoding Parameters.. future update needed
    rank = -1
    single_cls = False

    # Configure
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)

    with open(args.data_path) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict

    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, args.data_path)  # check

    # This function expects to receive complete weight-loaded YOLOv7 Model

    # Optimizer
    optimizer = createOptimizer(model, args)

    # Scheduler
    lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA
    ema = ModelEMA(model)

    # Resume
    global ckpt
    # Optimizer
    if ckpt['optimizer'] is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
        best_fitness = ckpt['best_fitness']

    # EMA
    if ema and ckpt.get('ema'):
        ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
        ema.updates = ckpt['updates']

    # Epochs
    start_epoch = ckpt['epoch'] + 1
    best_fitness = 0.0

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in [args.img_size, args.img_size]]  # verify imgsz are gs-multiples

    # Use Custom Loss
    if args.loss_change:
        model.loss_change = True
        model.loss_ratio = args.loss_ratio
    else:
        model.loss_change = False

    # Dataset
    global trainLoader, trainDataset
    global validLoader

    with open(args.data_path) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    train_path = data_dict['train']
    valid_path = data_dict['val']
    
    if trainLoader is None:
        trainLoader, trainDataset = create_dataloader(train_path, imgsz, args.train_batch_size, stride=gs, opt=args,
                                            hyp=hyp, augment=True, cache=False, rect=False, rank=rank,
                                            world_size=1, workers=6,
                                            image_weights=False, quad=False, prefix=colorstr('train: '))

    mlc = np.concatenate(trainDataset.labels, 0)[:, 0].max()  # max label class
    nb = len(trainLoader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, args.data, nc - 1)

    # Process 0
    if rank in [-1, 0]:
        if validLoader is None:
            validLoader = create_dataloader(valid_path, imgsz_test, args.train_batch_size * 2, stride=gs,  # testloader
                                       opt=args, hyp=hyp, cache=False, rect=True, rank=rank,
                                       world_size=1, workers=6,
                                       pad=0.5, prefix=colorstr('val: '))[0]

        labels = np.concatenate(trainDataset.labels, 0)
        c = torch.tensor(labels[:, 0])  # classes
        # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
        # model._initialize_biases(cf.to(device))

        # Anchors
        check_anchors(trainDataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
        model.half().float()  # pre-reduce anchor precision
    
    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(trainDataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss_ota = ComputeLossOTA(model)  # init loss class
    compute_loss = ComputeLoss(model)  # init loss class
    
    for epoch in range(start_epoch, epochs):

        model.train()

        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            trainLoader.sampler.set_epoch(epoch)

        optimizer.zero_grad()

        #batch_counter = 0
        for i, (inputs, targets, paths, _) in enumerate(trainLoader):
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = inputs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, 64 / args.train_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                    
                if 'loss_ota' not in hyp or hyp['loss_ota'] == 1:
                    loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)  # loss scaled by batch_size
                else:
                    loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if rank != -1:
                    loss *= args.world_size  # gradient averaged between devices in DDP mode
            
            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Count
            #batch_counter += 1
        

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            results, maps, times = test.test(data_dict,
                                                batch_size=args.train_batch_size * 2,
                                                imgsz=imgsz_test,
                                                model=ema.ema,
                                                single_cls=False,
                                                dataloader=validLoader,
                                                save_dir=args.job_dir,
                                                verbose=nc < 50 and final_epoch,
                                                plots=False,
                                                compute_loss=compute_loss,
                                                is_coco=False,
                                                v5_metric=False)

            # Update best mAP
            # fitness calculation
            #   Model fitness as a weighted combination of metrics
            #   w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
            #   return (input[:, :4] * w).sum(1)
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi

            # Save model
            if isSave:  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model': deepcopy(model.module if is_parallel(model) else model).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict()}

                # Save last, best and delete
                torch.save(ckpt, args.job_dir +'/'+ 'last.pt')
                if best_fitness == fi:
                    torch.save(ckpt, args.job_dir +'/'+ 'best.pt')
                if (best_fitness == fi) and (epoch >= 200):
                    torch.save(ckpt, args.job_dir +'/'+ 'best_{:03d}.pt'.format(epoch))
                if epoch == 0:
                    torch.save(ckpt, args.job_dir +'/'+ 'epoch_{:03d}.pt'.format(epoch))
                elif ((epoch+1) % 25) == 0:
                    torch.save(ckpt, args.job_dir +'/'+ 'epoch_{:03d}.pt'.format(epoch))
                elif epoch >= (epochs-5):
                    torch.save(ckpt, args.job_dir +'/'+ 'epoch_{:03d}.pt'.format(epoch))
                
                del ckpt
    torch.cuda.empty_cache()
    return best_fitness

#Calculate fitness of a honey source
def calculationFitness(honey, args):
    global best_honey
    #global best_honey_state

    # create model with honey source
    new_model = createModelwithHoney(args, honey)
    load_yolov7_tiny_honey_model(new_model, args.random_rule)

    data = utils.AverageMeter()

    data.update(train_yolo(new_model, hyp, args, isSave=False, epochs=args.calfitness_epoch))


    if data.fitness > best_honey.fitness:
        #best_honey_state = copy.deepcopy(model.module.state_dict() if len(args.gpus) > 1 else model.state_dict())
        best_honey.code = copy.deepcopy(honey)
        best_honey.fitness = data.fitness

    return data.fitness


#Initilize Bee-Pruning
def initilize():
    print('==> Initilizing Honey_model..')
    global best_honey, NectraSource, EmployedBee, OnLooker

    for i in range(args.food_number):
        
        NectraSource.append(copy.deepcopy(BeeGroup()))
        EmployedBee.append(copy.deepcopy(BeeGroup()))
        OnLooker.append(copy.deepcopy(BeeGroup()))
        for j in range(food_dimension):
            if i > 0:
                rand_num = random.randint(1, args.max_preserve)
                while 1:
                    if rand_num == NectraSource[i-1].code[j]:
                        rand_num = random.randint(1, args.max_preserve)
                    else:
                        break
            else:
                rand_num = random.randint(1, args.max_preserve)
            NectraSource[i].code.append(copy.deepcopy(rand_num))

        #initilize honey souce
        bee_start_time = time.time()
        NectraSource[i].fitness = calculationFitness(NectraSource[i].code, args)
        bee_end_time = time.time()
        logger.info(
                'Source Number [{}] initialized with honey source {}. \tFitness : {}, Time : {:2f}s'.format(i, NectraSource[i].code, NectraSource[i].fitness, (bee_end_time - bee_start_time))
            )
        NectraSource[i].rfitness = 0
        NectraSource[i].trail = 0

        #initilize employed bee  
        EmployedBee[i].code = copy.deepcopy(NectraSource[i].code)
        EmployedBee[i].fitness=NectraSource[i].fitness 
        EmployedBee[i].rfitness=NectraSource[i].rfitness 
        EmployedBee[i].trail=NectraSource[i].trail

        #initilize onlooker 
        OnLooker[i].code = copy.deepcopy(NectraSource[i].code)
        OnLooker[i].fitness=NectraSource[i].fitness 
        OnLooker[i].rfitness=NectraSource[i].rfitness 
        OnLooker[i].trail=NectraSource[i].trail

    #initilize best honey
    best_honey.code = copy.deepcopy(NectraSource[0].code)
    best_honey.fitness = NectraSource[0].fitness
    best_honey.rfitness = NectraSource[0].rfitness
    best_honey.trail = NectraSource[0].trail

#Send employed bees to find better honey source
def sendEmployedBees():
    global NectraSource, EmployedBee
    for i in range(args.food_number):
        
        while 1:
            k = random.randint(0, args.food_number-1)
            if k != i:
                break

        EmployedBee[i].code = copy.deepcopy(NectraSource[i].code)

        param2change = np.random.randint(0, food_dimension-1, args.honeychange_num)
        R = np.random.uniform(-1, 1, args.honeychange_num)
        for j in range(args.honeychange_num):
            EmployedBee[i].code[param2change[j]] = int(NectraSource[i].code[param2change[j]]+ R[j]*(NectraSource[i].code[param2change[j]]-NectraSource[k].code[param2change[j]]))
            if EmployedBee[i].code[param2change[j]] < 1:
                EmployedBee[i].code[param2change[j]] = 1
            if EmployedBee[i].code[param2change[j]] > args.max_preserve:
                EmployedBee[i].code[param2change[j]] = args.max_preserve

        bee_start_time = time.time()
        EmployedBee[i].fitness = calculationFitness(EmployedBee[i].code, args)
        bee_end_time = time.time()
        logger.info(
                'EmployedBee Number [{}] finded honey source {}. \tFitness : {}, Time : {:2f}s'.format(i, EmployedBee[i].code, EmployedBee[i].fitness, (bee_end_time - bee_start_time))
            )

        if EmployedBee[i].fitness > NectraSource[i].fitness:                
            NectraSource[i].code = copy.deepcopy(EmployedBee[i].code)              
            NectraSource[i].trail = 0  
            NectraSource[i].fitness = EmployedBee[i].fitness 
            
        else:          
            NectraSource[i].trail = NectraSource[i].trail + 1

#Calculate whether a Onlooker to update a honey source
def calculateProbabilities():
    global NectraSource
    
    maxfit = NectraSource[0].fitness

    for i in range(1, args.food_number):
        if NectraSource[i].fitness > maxfit:
            maxfit = NectraSource[i].fitness

    for i in range(args.food_number):
        NectraSource[i].rfitness = (0.9 * (NectraSource[i].fitness / maxfit)) + 0.1

#Send Onlooker bees to find better honey source
def sendOnlookerBees():
    global NectraSource, EmployedBee, OnLooker
    i = 0
    t = 0
    while t < args.food_number:
        R_choosed = random.uniform(0,1)
        if(R_choosed < NectraSource[i].rfitness):
            t += 1

            while 1:
                k = random.randint(0, args.food_number-1)
                if k != i:
                    break
            OnLooker[i].code = copy.deepcopy(NectraSource[i].code)

            param2change = np.random.randint(0, food_dimension-1, args.honeychange_num)
            R = np.random.uniform(-1, 1, args.honeychange_num)
            for j in range(args.honeychange_num):
                OnLooker[i].code[param2change[j]] = int(NectraSource[i].code[param2change[j]]+ R[j]*(NectraSource[i].code[param2change[j]]-NectraSource[k].code[param2change[j]]))
                if OnLooker[i].code[param2change[j]] < 1:
                    OnLooker[i].code[param2change[j]] = 1
                if OnLooker[i].code[param2change[j]] > args.max_preserve:
                    OnLooker[i].code[param2change[j]] = args.max_preserve

            bee_start_time = time.time()
            OnLooker[i].fitness = calculationFitness(OnLooker[i].code, args)
            bee_end_time = time.time()
            logger.info(
                    'OnLookerBee Number [{}] finded honey source {}. \tFitness : {}, Time : {:2f}s'.format(i, OnLooker[i].code, OnLooker[i].fitness, (bee_end_time - bee_start_time))
                )

            if OnLooker[i].fitness > NectraSource[i].fitness:                
                NectraSource[i].code = copy.deepcopy(OnLooker[i].code)              
                NectraSource[i].trail = 0  
                NectraSource[i].fitness = OnLooker[i].fitness 
            else:          
                NectraSource[i].trail = NectraSource[i].trail + 1
        i += 1
        if i == args.food_number:
            i = 0

#If a honey source has not been update for args.food_limiet times, send a scout bee to regenerate it
def sendScoutBees():
    global  NectraSource, EmployedBee, OnLooker
    maxtrailindex = 0
    for i in range(args.food_number):
        if NectraSource[i].trail > NectraSource[maxtrailindex].trail:
            maxtrailindex = i
    if NectraSource[maxtrailindex].trail >= args.food_limit:
        for j in range(food_dimension):
            
            R = random.uniform(0,1)
            NectraSource[maxtrailindex].code[j] = int(R * args.max_preserve)
            if NectraSource[maxtrailindex].code[j] == 0:
                NectraSource[maxtrailindex].code[j] += 1
        NectraSource[maxtrailindex].trail = 0
        NectraSource[maxtrailindex].fitness = calculationFitness(NectraSource[maxtrailindex].code, args)
 
 #Memorize best honey source
def memorizeBestSource():
    global best_honey, NectraSource
    for i in range(args.food_number):
        if NectraSource[i].fitness > best_honey.fitness:
            #print(NectraSource[i].fitness, NectraSource[i].code)
            #print(best_honey.fitness, best_honey.code)
            best_honey.code = copy.deepcopy(NectraSource[i].code)
            best_honey.fitness = NectraSource[i].fitness


def main():

    if args.best_honey == None:

        start_time = time.time()
        
        bee_start_time = time.time()
        
        print('==> Start BeePruning..')

        initilize()

        #memorizeBestSource()

        for cycle in range(args.max_cycle):

            current_time = time.time()
            logger.info(
                'Search Cycle [{}]\t Best Honey Source {}\tBest Honey Source fitness {}\tTime {:.2f}s\n'
                .format(cycle, best_honey.code, best_honey.fitness, (current_time - start_time))
            )
            start_time = time.time()

            sendEmployedBees() 
                
            calculateProbabilities() 
                
            sendOnlookerBees()  
                
            #memorizeBestSource() 
                
            sendScoutBees() 
                
            #memorizeBestSource() 

        print('==> BeePruning Complete!')
        bee_end_time = time.time()
        logger.info(
            'Best Honey Source {}\tBest Honey Source fitness {}\tTime Used {:.2f}s\n'
            .format(best_honey.code, float(best_honey.fitness), (bee_end_time - bee_start_time))
        )
        #checkpoint.save_honey_model(state)
    else:
        best_honey.code = args.best_honey

    # Model
    print('==> Start Trainig with Best honey code..')
    
    # create model with honey source
    final_model = createModelwithHoney(args, best_honey.code)
    load_yolov7_tiny_honey_model(final_model, args.random_rule)

    print(args.random_rule + ' Done!')

    final_fitness = train_yolo(final_model, hyp, args, isSave=True, epochs=args.num_epochs)

    logger.info('Final Train fitness: {:.3f}'.format(float(final_fitness)))
    print('best honey code : ', best_honey.code)

if __name__ == '__main__':
    main()
