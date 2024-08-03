import shutil
import numpy as np
import torch
import torch.nn as nn
import time

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss

from utils.model.nafnet import NAFNet
from torchvision.models import *


def train_epoch(args, epoch, model, data_loader, optimizer, loss_type):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in enumerate(data_loader):
        input, varnet, target, maximum, _, _ = data
        input = input.cuda(non_blocking=True)
      
        varnet = varnet.cuda(non_blocking=True)

        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)
        
        input = torch.stack((input, varnet), dim=1)
        output = model(input)
        
        accum = 64
        loss = loss_type(output, target, maximum)
        loss = loss/ accum
        loss.backward()
        
        if (iter+1) % accum == 0 or iter + 1 == len_loader: 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1/16)
            optimizer.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * accum  

        if iter % args.report_interval == 0 or iter + 1 == len_loader or iter == 50:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item()* accum:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    inputs = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            input, varnet, target, _, fnames, slices = data
            input = input.cuda(non_blocking=True)
            
            varnet = varnet.cuda(non_blocking=True)
           
            stacked_input = torch.stack((input,  varnet), dim=1)
        
            output = model(stacked_input)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()
                inputs[fnames[i]][int(slices[i])] = input[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    for fname in inputs:
        inputs[fname] = np.stack(
            [out for _, out in sorted(inputs[fname].items())]
        )
        metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, inputs, time.perf_counter() - start


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


        
def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())
    


    img_channel = 2
    width = 32

    enc_blks = [4, 8, 8, 12]
    middle_blk_num = 16
    dec_blks = [12, 8, 8, 4]

    model = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks).cuda()

    model.to(device=device)
    loss_type = SSIMLoss().to(device=device)

    optimizer = torch.optim.RAdam(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 20, gamma= 0.5)
    
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
        
    best_val_loss = 1 if loaded_best_val_loss == -1 else loaded_best_val_loss    
    start_epoch = 0 if loaded_epoch == 0 else loaded_epoch 
    
    
    train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle=True)
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args)

    val_loss_log = np.empty((0, 2))
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type)
        scheduler.step()
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)

        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = args.val_loss_dir / "val_loss_log"
        np.save(file_path, val_loss_log)
        print(f"loss file saved! {file_path}")

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
