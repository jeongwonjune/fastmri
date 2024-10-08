import shutil
import numpy as np
import torch
import torch.nn as nn
import time
import requests
from tqdm import tqdm
from pathlib import Path
import copy

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss, SSIML1Loss
from utils.model.varnet import VarNet
from utils.model.nafnet import NAFNet
from utils.data.mraugment.data_augment import DataAugmentor

from torch.nn.utils import clip_grad_norm_
#from torch.cuda.amp import autocast, GradScaler
import os

class ConcatenateModels(nn.Module): 
    def __init__(self, varnet_model, nafnet_model):
        super(ConcatenateModels, self).__init__()
        self.varnet_model = varnet_model
        self.nafnet_model = nafnet_model

    def forward(self, kspace, mask):
        varnet_output = self.varnet_model(kspace, mask)
        nafnet_output = self.nafnet_model(varnet_output)
        
        return nafnet_output
    

def train_epoch(args, epoch, model, data_loader, optimizer, loss_type):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    
    for iter, data in enumerate(data_loader):
       
        mask, kspace, target, maximum, _, _ = data
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        output = model(kspace, mask)
        
        accum = 16
        loss = loss_type(output, target, maximum)
        loss = loss/ accum
        loss.backward()
        
        if (iter+1) % accum == 0 or iter + 1 == len_loader: 
            clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * accum  

        if iter % args.report_interval == 0 or iter + 1 == len_loader or iter == 50:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item()* accum :.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, target, _, fnames, slices = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def download_model(url, fname):
    response = requests.get(url, timeout=10, stream=True)

    chunk_size = 8 * 1024 * 1024  # 8 MB chunks
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(
        desc="Downloading state_dict",
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
    )

    with open(fname, "wb") as fh:
        for chunk in response.iter_content(chunk_size):
            progress_bar.update(len(chunk))
            fh.write(chunk)


        
def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    varnetmodel = VarNet(num_cascades=args.cascade, 
                   chans=args.chans, 
                   sens_chans=args.sens_chans)
    #nafnetmodel = NAFNet(img_channel=1, width=16, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    #model = ConcatenateModels(varnetmodel, nafnetmodel)
    
    model = varnetmodel
    model.to(device=device)
    

    """
    # using pretrained parameter
    VARNET_FOLDER = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/"
    MODEL_FNAMES = "brain_leaderboard_state_dict.pt"
    if not Path(MODEL_FNAMES).exists():
        url_root = VARNET_FOLDER
        download_model(url_root + MODEL_FNAMES, MODEL_FNAMES)
    
    pretrained = torch.load(MODEL_FNAMES)
    pretrained_copy = copy.deepcopy(pretrained)
    for layer in pretrained_copy.keys():
        if layer.split('.',2)[1].isdigit() and (args.cascade <= int(layer.split('.',2)[1]) <=11):
            del pretrained[layer]
    model.load_state_dict(pretrained)
    """

    #loss_type = SSIMLoss().to(device=device)
    loss_type = SSIML1Loss().to(device=device)
    
    optimizer = torch.optim.RAdam(model.parameters(), args.lr)
    
    best_val_loss = 1.
    start_epoch = 0
    
    loaded_epoch = 0
    loaded_best_val_loss = -1
    if args.load is 1:
        checkpoint = torch.load(args.exp_dir / 'best_model.pt', map_location='cpu')
        
        model.load_state_dict(checkpoint["model"])
        model.cuda()
        optimizer.load_state_dict(checkpoint["optimizer"])
        loaded_epoch = checkpoint["epoch"]
        loaded_best_val_loss = checkpoint["best_val_loss"]
        print(checkpoint['epoch'], checkpoint['best_val_loss'].item())
        
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20 ,  gamma=0.5, last_epoch=- 1, verbose=True)
    
    best_val_loss = 1 if loaded_best_val_loss == -1 else loaded_best_val_loss    
    start_epoch = 0 if loaded_epoch == 0 else loaded_epoch 
    
    #train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle=True)
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args)
    
    val_loss_log = np.empty((0, 2))
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        
        #mraugment
        current_epoch_fn = epoch
        augmentor = DataAugmentor(args, current_epoch_fn)
        
        train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle = True, isforward=False , augmentor=augmentor , use_seed = False)
        
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type)
        scheduler.step()
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)
        
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = os.path.join(args.val_loss_dir, "val_loss_log")
        np.save(file_path, val_loss_log)
        print(f"loss file saved! {file_path}")

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )
