import torch
import time
from tqdm import tqdm
from net.GTC_3DEMv4 import MeshCodec
import torch.utils.data.dataloader as DataLoader
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
from pathlib import Path
from net.utils import increment_path, EDataset, MultiEMRCSDataset, get_logger, get_model_memory, psnr, ssim, find_matching_files, process_files, WrappedModel, savefigdata
from NNval_GTC import valmain, plotstatistic2, plot4D_E_RealImage
from pytictoc import TicToc
t = TicToc()
t.tic()
import random
import numpy as np
import argparse
from thop import profile
import copy

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.backends.cudnn.benchmark = False 
     torch.backends.cudnn.deterministic = True
     np.random.seed(seed)
     random.seed(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Script with customizable parameters using argparse.")
    parser.add_argument('--epoch', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='batchsize')
    parser.add_argument('--valbatch', type=int, default=16, help='valbatchsize')
    parser.add_argument('--draw', type=bool, default=True, help='Whether to enable drawing')

    parser.add_argument('--trainname', type=str, default='v4.0', help='logname')
    parser.add_argument('--savedir', type=str, default='testtrain', help='exp output folder name')
    parser.add_argument('--mode', type=str, default='fasttest', help='10train 50fine 100fine fasttest')
    parser.add_argument('--loss', type=str, default='L1', help='L1 best, mse 2nd')
    # parser.add_argument('--rcsdir', type=str, default='/mnt/Disk/jiangxiaotian/datasets/Datasets_3DEM/allplanes/mie/b943_mie_val', help='Path to rcs directory')
    # parser.add_argument('--valdir', type=str, default='/mnt/Disk/jiangxiaotian/datasets/Datasets_3DEM/allplanes/mie/b943_mie_val', help='Path to validation directory') #3090red
    # parser.add_argument('--rcsdir', type=str, default='/mnt/truenas_jiangxiaotian/allplanes/mie/b943_mie_val', help='Path to rcs directory')
    # parser.add_argument('--valdir', type=str, default='/mnt/truenas_jiangxiaotian/allplanes/mie/b943_mie_val', help='Path to validation directory') #3090liang
    parser.add_argument('--rcsdir', type=str, default='/mnt/truenas_jiangxiaotian/Edataset/complexE_mie_RealImage/testtrain', help='Path to rcs directory')
    parser.add_argument('--valdir', type=str, default='/mnt/truenas_jiangxiaotian/Edataset/complexE_mie_RealImage/testtrain', help='Path to validation directory') #3090liang
    parser.add_argument('--pretrainweight', type=str, default=None, help='Path to pretrained weights')

    parser.add_argument('--seed', type=int, default=7, help='Random seed for reproducibility')
    parser.add_argument('--attn', type=int, default=0, help='Transformer layers')
    parser.add_argument('--lr', type=float, default=0.0001, help='Loss threshold or gamma parameter')
    parser.add_argument('--cuda', type=str, default='cuda:0', help='CUDA device to use(cpu cuda:0 cuda:1...)')
    parser.add_argument('--fold', type=str, default=None, help='Fold to use for validation (None fold1 fold2 fold3 fold4)')

    parser.add_argument('--lam_max', type=float, default=0.001, help='control max loss, i love 0.001')
    parser.add_argument('--lam_hel', type=float, default=0, help='control helmholtz loss, i love 0.001')
    parser.add_argument('--lam_fft', type=float, default=0, help='control fft loss, i love 0.001')
    parser.add_argument('--lam_rec', type=float, default=0, help='control receprocity loss, i love 0.001')
    parser.add_argument('--pinnepoch', type=int, default=-1, help='Number of pinn loss adding epochs, if epochnow > pinnepoch, start to add pinn loss. 0 or -1 start from beginning, >200 means never add pinn loss')


    return parser.parse_args()

tic0 = time.time()
tic = time.time()
print('code start time:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))  

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) 
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

args = parse_args()

epoch = args.epoch
draw = args.draw
pretrainweight = args.pretrainweight
seed = args.seed
attnlayer = args.attn
learning_rate = args.lr
cudadevice = args.cuda
name = args.trainname
folder = args.savedir
mode = args.mode
batchsize = args.batch
valbatch = args.valbatch
loss_type = args.loss

lambda_max = args.lam_max
lambda_helmholtz = args.lam_hel
lambda_bandlimit = args.lam_fft
lambda_reciprocity = args.lam_rec

if args.fold: 
    # datafolder = '/mnt/d/datasets/Dataset_3DEM/mie' # 305simu
    # datafolder = '/mnt/Disk/jiangxiaotian/datasets/Datasets_3DEM/allplanes/mie' # 3090red
    datafolder = '/mnt/truenas_jiangxiaotian/allplanes/mie' #3090liang
    Fold1 = ['b871','bb7d','b827','b905','bbc6']
    Fold2 = ['b80b','ba0f','b7c1','b9e6','bb7c']
    Fold3 = ['b943','b97b','b812','bc2c','b974']
    Fold4 = ['bb26','b7fd','baa9','b979','b8ed']
    fold_mapping = {
        'fold1': Fold1,
        'fold2': Fold2,
        'fold3': Fold3,
        'fold4': Fold4,
    }
    val_planes = fold_mapping[args.fold]
    train_planes = [files for fold in [Fold1, Fold2, Fold3, Fold4] if fold != val_planes for files in fold]
    valdir = None
    rcsdir = None

else: 
    rcsdir = args.rcsdir
    valdir = args.valdir

# setup_seed(seed)
if args.seed is not None:
    seed = args.seed
    setup_seed(args.seed)
    print(f"use provided seed: {args.seed}")
else:
    random_seed = torch.randint(0, 10000, (1,)).item()
    setup_seed(random_seed)
    print(f"not provide seed, use random seed: {random_seed}")
    seed = random_seed

accumulation_step = 8
threshold = 20
bestloss = 1
epoch_mean_loss = 0.0
maxpsnr = 1.0
# minmse = 1.0
valmse = 1.0
in_ems = []
rcss = []
cnt = 0
losses, psnrs, ssims, mses = [], [], [], []
nmses, rmses, l1s, percentage_errors = [], [], [], []
mainLs, maxLs, helmholtzLs, bandlimitLs, reciprocityLs = [], [], [], [], []
lastpsnr, lastmse, lastssim, lasttime = 0.0, 0.0, 0.0, 0.0
corrupted_files = []
lgrcs = False
shuffle = True
multigpu = False
alpha = 0.0
lr_time = epoch

encoder_layer = 6
decoder_outdim = 12  # 3S 6M 12L
cpucore = 8
oneplane = args.rcsdir.split('/')[-1][0:4]

from datetime import datetime
date = datetime.today().strftime("%m%d")
save_dir = str(increment_path(Path(ROOT / "output" / f"{folder}" / f'{date}_{name}_{mode}{loss_type}_{args.fold if args.fold else oneplane}_b{batchsize}e{epoch}lr{learning_rate}sd{seed}Tr{attnlayer}_lm{lambda_max}_{cudadevice}_'), exist_ok=False))
# save_dir = str(increment_path(Path(ROOT / "output" / f"{folder}" / f'{date}_{name}_{mode}{loss_type}_{args.fold if args.fold else oneplane}_b{batchsize}e{epoch}ep{args.pinnepoch}Tr{attnlayer}_lh{lambda_helmholtz}lf{lambda_bandlimit}lc{lambda_reciprocity}_{cudadevice}_'), exist_ok=False))

lastsavedir = os.path.join(save_dir,'last.pt')
bestsavedir = os.path.join(save_dir,'best.pt')

global logger
logger = get_logger(os.path.join(save_dir,'log.txt'))
logger.info(args)
logger.info(f'seed:{seed}')




if args.fold:
    logger.info(f'dataset setting:{args.fold} ,val on {val_planes}, train on {train_planes}, mode={mode}')
    val_mse_per_plane = {plane: [] for plane in val_planes}
    val_psnr_per_plane = {plane: [] for plane in val_planes}
    val_ssim_per_plane = {plane: [] for plane in val_planes}

    
    if mode=='10train' or 'fasttest': #10train 50fine 100fine
        train_files = [plane + '_mie_10train' for plane in train_planes]
    elif mode=='50fine':
        train_files = [plane + '_mie_50train' for plane in train_planes]
    elif mode=='100fine':
        train_files = [plane + '_mie_train' for plane in train_planes]
    
    val_files = [plane + '_mie_val' for plane in val_planes]

    dataset = MultiEMRCSDataset(train_files, datafolder)
    dataloader = DataLoader.DataLoader(dataset, batch_size=batchsize, shuffle=shuffle, num_workers=cpucore, pin_memory=True)
    val_dataloaders = {} 
    for valfile1 in val_files:
        valdataset = MultiEMRCSDataset([valfile1], datafolder)
        plane1 = valfile1[:4]
        val_dataloaders[plane1] = DataLoader.DataLoader(valdataset, batch_size=valbatch, shuffle=False, num_workers=cpucore, pin_memory=True)

    logger.info(f'train set samples:{dataset.__len__()}，single val set samples:{valdataset.__len__()}，val set count:{len(val_dataloaders)}，tatal val set samples:{valdataset.__len__()*len(val_dataloaders)}')

else:
    logger.info(f'train set is{rcsdir}')
    # filelist = os.listdir(rcsdir)
    dataset = EDataset(rcsdir)
    # dataset = EMRCSDataset(filelist, rcsdir)
    dataloader = DataLoader.DataLoader(dataset, batch_size=batchsize, shuffle=shuffle, num_workers=cpucore, pin_memory=True) #这里调用的是getitem 在上面datasets里还是numpy,从dataloader出来就成torch了

    # valfilelist = os.listdir(valdir)
    valdataset = EDataset(valdir) #这里进的是init
    # valdataset = EMRCSDataset(valfilelist, valdir) #这里进的是init
    valdataloader = DataLoader.DataLoader(valdataset, batch_size=valbatch, shuffle=shuffle, num_workers=cpucore, pin_memory=True) #transformer的话40才行？20.。 纯GNN的话60都可以
    logger.info(f'train set samples:{dataset.__len__()}，val set samples:{valdataset.__len__()}')

logger.info(f'saved to {lastsavedir}')

device = torch.device(cudadevice if torch.cuda.is_available() else "cpu")
# device = 'cpu'
logger.info(f'device:{device}')

autoencoder = MeshCodec(
    device = device,
    attn_encoder_depth = attnlayer,
)
get_model_memory(autoencoder,logger)
total_params = sum(p.numel() for p in autoencoder.parameters())
logger.info(f"Total parameters: {total_params}")

if pretrainweight != None:
    autoencoder.load_state_dict(torch.load(pretrainweight), strict=True)
    logger.info(f'successfully load pretrain_weight:{pretrainweight}')
else:
    logger.info('not use pretrain_weight, starting new train')

autoencoder = autoencoder.to(device)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=lr_time)

allavemses = []
allavepsnrs = []
allavessims = []
flag = 1
GTflag = 1
flopflag = 1
for i in range(epoch):
    epoch_flag = 1
    valallpsnrs = []
    valallssims = []
    valallmses = []
    psnr_list, ssim_list, mse_list, nmse_list, rmse_list, l1_list, percentage_error_list = [], [], [], [], [], [], []
    mainLlist, maxLlist, helmholtzLlist, bandlimitLlist, reciprocityLlist = [], [], [], [], [] 
    jj=0
    logger.info('\n')
    epoch_loss = []
    timeepoch = time.time()
    for in_em1,rcs1 in tqdm(dataloader,desc=f'epoch:{i},lr={scheduler.get_last_lr()[0]:.5f}',ncols=100,postfix=f'loss:{(epoch_mean_loss):.4f}'):
        jj=jj+1
        in_em0 = in_em1.copy()
        # optimizer.zero_grad()
        # objlist , ptlist = find_matching_files(in_em1[0], "./testplane")
        objlist , ptlist = find_matching_files(in_em1[0], "./planes")
        planesur_faces, planesur_verts, planesur_faceedges, geoinfo = process_files(objlist, device)

        loss, outrcs, psnr_mean, _, ssim_mean, _, mse, nmse, rmse, l1, percentage_error, _ , metrics= autoencoder(
            vertices = planesur_verts,
            faces = planesur_faces, #torch.Size([batchsize, 33564, 3])
            face_edges = planesur_faceedges,
            in_em = in_em1,#.to(device)
            GT = rcs1.to(device),
            logger = logger,
            device = device,
            loss_type=loss_type,
            epochnow = i,
            pinnepoch= args.pinnepoch,
            epoch_flag = epoch_flag,
            lambda_max=lambda_max,
            lambda_helmholtz=lambda_helmholtz,
            lambda_bandlimit=lambda_bandlimit,
            lambda_reciprocity=lambda_reciprocity,
        )

        if epoch_flag == 1:
            logger.info(f'\n{metrics}')
            epoch_flag = 0
        if flopflag == 1:
            temp_model = copy.deepcopy(autoencoder)
            wrapped_model = WrappedModel(temp_model)
            flops, params = profile(wrapped_model, (planesur_verts, planesur_faces, planesur_faceedges, in_em1, rcs1.to(device),device))
            logger.info(f' params:{params / 1000000.0:.2f}M, Gflops:{flops / 1000000000.0:.2f}G')
            flopflag = 0
            del temp_model 

        if lgrcs == True:
            outrcslg = outrcs
            outrcs = torch.pow(10, outrcs)
        if batchsize > 1:
            lossback=loss.mean() / accumulation_step 
            lossback.backward() 
        else:
            outem = [int(in_em1[1]), int(in_em1[2]), float(f'{in_em1[3].item():.3f}')]
            tqdm.write(f'em:{outem},loss:{loss.item():.4f}')
            lossback=loss / accumulation_step
            lossback.backward()
        epoch_loss.append(loss.item())

        torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=threshold)
        if (jj) % accumulation_step == 0 or (jj) == len(dataloader):
            optimizer.step() 
            optimizer.zero_grad()
        
        # 将一个batch的指标append到列表中，用于后续计算epoch指标
        psnr_list.append(psnr_mean)
        ssim_list.append(ssim_mean)
        mse_list.append(mse)
        nmse_list.append(nmse)
        rmse_list.append(rmse)
        l1_list.append(l1)
        percentage_error_list.append(percentage_error)
        mainLlist.append(metrics['main_loss'])
        maxLlist.append(metrics['max_loss'])
        helmholtzLlist.append(metrics['helmholtz_loss'])
        bandlimitLlist.append(metrics['bandlimit_loss'])
        reciprocityLlist.append(metrics['reciprocity_loss'])

                # metrics = {
        #         'total_loss': total_loss,
        #         'main_loss': mainloss,
        #         'max_loss': maxloss,
        #         'helmholtz_loss': helmholtz_loss,
        #         'bandlimit_loss': bandlimit_loss,
        #         'reciprocity_loss': reciprocity_loss,
        #     }
        

        in_em0[1:] = [tensor.to(device) for tensor in in_em0[1:]]
        if flag == 1:
            drawrcs = outrcs[0].unsqueeze(0)
            drawem = torch.stack(in_em0[1:]).t()[0]
            drawGT = rcs1[0].unsqueeze(0)
            drawplane = in_em0[0][0]
            flag = 0
        for j in range(torch.stack(in_em0[1:]).t().shape[0]):
            if flag == 0 and torch.equal(torch.stack(in_em0[1:]).t()[j], drawem):
                drawrcs = outrcs[j].unsqueeze(0)
                break
    logger.info(save_dir)

    p = psnr(drawrcs.to(device), drawGT.to(device))
    s = ssim(drawrcs.to(device), drawGT.to(device))
    m = torch.nn.functional.mse_loss(drawrcs.to(device), drawGT.to(device))
    if GTflag == 1:
        outGTpngpath = os.path.join(save_dir,f'{drawplane}theta{drawem[0]}phi{drawem[1]}freq{drawem[2]}_GT.png')
        out2DGTpngpath = os.path.join(save_dir,f'{drawplane}theta{drawem[0]}phi{drawem[1]}freq{drawem[2]}_2DGT.png')
        plot4D_E_RealImage(drawGT.squeeze(), out2DGTpngpath, logger) #作图模块全线崩溃
        GTflag = 0
        logger.info('drawed GT map')
    if i == 0 or (i+1) % 20 == 0: 
        outrcspngpath = os.path.join(save_dir,f'{drawplane}theta{drawem[0]}phi{drawem[1]}freq{drawem[2]}_epoch{i}.png')
        out2Drcspngpath = os.path.join(save_dir,f'{drawplane}theta{drawem[0]}phi{drawem[1]}freq{drawem[2]}_epoch{i}_psnr{p.item():.2f}_ssim{s.item():.4f}_mse{m:.4f}_2D.png')
        plot4D_E_RealImage(drawrcs.squeeze(), out2Drcspngpath, logger)
        logger.info(f'drawed {i+1} epoch map')

    # 将batch指标list 计算为epoch指标
    epoch_mean_loss = sum(epoch_loss)/len(epoch_loss)
    epoch_psnr = sum(psnr_list)/len(psnr_list) 
    epoch_ssim = sum(ssim_list)/len(ssim_list)
    epoch_mse = sum(mse_list)/len(mse_list)
    epoch_nmse = sum(nmse_list)/len(nmse_list)
    epoch_rmse = sum(rmse_list)/len(rmse_list)
    epoch_l1 = sum(l1_list)/len(l1_list)
    epoch_percentage_error = sum(percentage_error_list)/len(percentage_error_list)
    epoch_main_loss = sum(mainLlist)/len(mainLlist)
    epoch_max_loss = sum(maxLlist)/len(maxLlist)
    epoch_helmholtz_loss = sum(helmholtzLlist)/len(helmholtzLlist)
    epoch_bandlimit_loss = sum(bandlimitLlist)/len(bandlimitLlist)
    epoch_reciprocity_loss = sum(reciprocityLlist)/len(reciprocityLlist)

    # 将epoch指标append到list中，用于后续绘图
    losses.append(epoch_mean_loss)
    psnrs.append(epoch_psnr.detach().cpu())
    ssims.append(epoch_ssim.detach().cpu())
    mses.append(epoch_mse.detach().cpu())
    nmses.append(epoch_nmse.detach().cpu())
    rmses.append(epoch_rmse.detach().cpu())
    l1s.append(epoch_l1.detach().cpu())
    percentage_errors.append(epoch_percentage_error.detach().cpu())
    mainLs.append(epoch_main_loss.detach().cpu())
    # maxLs.append(epoch_max_loss.detach().cpu())
    maxLs.append(epoch_max_loss)
    helmholtzLs.append(epoch_helmholtz_loss)
    bandlimitLs.append(epoch_bandlimit_loss)
    reciprocityLs.append(epoch_reciprocity_loss)
    logger.info('用于绘图的epoch metrics 计算完成')

    if bestloss > epoch_mean_loss:
        bestloss = epoch_mean_loss
        if os.path.exists(bestsavedir):
            os.remove(bestsavedir)
        torch.save(autoencoder.to('cpu').state_dict(), bestsavedir)
    if os.path.exists(lastsavedir):
        os.remove(lastsavedir)
    torch.save(autoencoder.to('cpu').state_dict(), lastsavedir)
    logger.info('model weight saved')
    autoencoder.to(device)

    scheduler.step()
    logger.info('lr scheduled')

    logger.info(f'↓-----------------------this epoch time consume：{time.strftime("%H:%M:%S", time.gmtime(time.time()-timeepoch))}-----------------------↓')
    logger.info(f'↑----epoch:{i+1}(lr:{scheduler.get_last_lr()[0]:.4f}),loss:{epoch_mean_loss:.4f},psnr:{epoch_psnr:.2f},ssim:{epoch_ssim:.4f},mse:{epoch_mse:.4f}----↑')

    def draw2dcurve(curve, savedir, ylabel, title):
        if not os.path.exists(os.path.dirname(savedir)):
            os.makedirs(os.path.dirname(savedir))
        plt.clf()
        plt.figure(figsize=(7, 4.5))
        plt.plot(range(0, i+1), curve)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(savedir)
        savefigdata(curve,img_path=savedir)
        plt.close()
    draw2dcurve(losses, os.path.join(save_dir,'fig/loss.png'), 'Loss', 'Training Loss Curve')
    draw2dcurve(psnrs, os.path.join(save_dir,'fig/psnr.png'), 'PSNR', 'Training PSNR Curve')
    draw2dcurve(ssims, os.path.join(save_dir,'fig/ssim.png'), 'SSIM', 'Training SSIM Curve')
    draw2dcurve(mses, os.path.join(save_dir,'fig/mse.png'), 'MSE', 'Training MSE Curve')
    draw2dcurve(nmses, os.path.join(save_dir,'fig/nmse.png'), 'NMSE', 'Training NMSE Curve')
    draw2dcurve(rmses, os.path.join(save_dir,'fig/rmse.png'), 'RMSE', 'Training RMSE Curve')
    draw2dcurve(l1s, os.path.join(save_dir,'fig/l1.png'), 'L1', 'Training L1 Curve')
    draw2dcurve(percentage_errors, os.path.join(save_dir,'fig/percentage_error.png'), 'Percentage Error', 'Training Percentage Error Curve')
    draw2dcurve(mainLs, os.path.join(save_dir,'fig/mainloss.png'), 'Main Loss', 'Training Main Loss Curve')
    draw2dcurve(maxLs, os.path.join(save_dir,'fig/maxloss.png'), 'Max Loss', 'Training Max Loss Curve')
    draw2dcurve(helmholtzLs, os.path.join(save_dir,'fig/helmholtzloss.png'), 'Helmholtz Loss', 'Training Helmholtz Loss Curve')
    draw2dcurve(bandlimitLs, os.path.join(save_dir,'fig/bandlimitloss.png'), 'Bandlimit Loss', 'Training Bandlimit Loss Curve')
    draw2dcurve(reciprocityLs, os.path.join(save_dir,'fig/reciprocityloss.png'), 'Reciprocity Loss', 'Training Reciprocity Loss Curve')


    plt.clf() 
    plt.plot(range(0, i+1), losses, label='total_Loss')
    plt.plot(range(0, i+1), mainLs, label='main_Loss')
    plt.plot(range(0, i+1), maxLs, label='max_Loss')
    plt.plot(range(0, i+1), helmholtzLs, label='helmholtz_Loss')
    plt.plot(range(0, i+1), bandlimitLs, label='bandlimit_Loss')
    plt.plot(range(0, i+1), reciprocityLs, label='reciprocity_Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.savefig(os.path.join(save_dir,'lossallinone.png'))
    plt.close()

    if args.fold:
        for plane, valdataloader in val_dataloaders.items():
            logger.info(f"val on aircraft{plane}")
            valplanedir=os.path.join(save_dir,plane)
            if not os.path.exists(valplanedir):
                os.makedirs(valplanedir)
            if mode == "10train":
                if (i+1) % 1 == 0 or i == -1: 
                    if (i+1) % 100 == 0 or i+1==epoch: 
                        valmse, valpsnr, valssim, valpsnrs, valssims, valmses =valmain(draw=True, device=device, weight=lastsavedir, rcsdir=valdir, save_dir=valplanedir, logger=logger, epoch=i, trainval=True, draw3d=False, valdataloader=valdataloader, attnlayer=attnlayer, batchsize=valbatch)
                    else:
                        valmse, valpsnr, valssim, valpsnrs, valssims, valmses =valmain(draw=False, device=device, weight=lastsavedir, rcsdir=valdir, save_dir=valplanedir, logger=logger, epoch=i, trainval=True, draw3d=False, valdataloader=valdataloader, attnlayer=attnlayer, batchsize=valbatch)
            elif mode == "fasttest":
                if (i+1) % 1 == 0 or i == -1: 
                    if i+1==epoch:
                        valmse, valpsnr, valssim, valpsnrs, valssims, valmses =valmain(draw=True, device=device, weight=lastsavedir, rcsdir=valdir, save_dir=valplanedir, logger=logger, epoch=i, trainval=True, draw3d=False, valdataloader=valdataloader, attnlayer=attnlayer, batchsize=valbatch)
                    else:
                        valmse, valpsnr, valssim, valpsnrs, valssims, valmses =valmain(draw=False, device=device, weight=lastsavedir, rcsdir=valdir, save_dir=valplanedir, logger=logger, epoch=i, trainval=True, draw3d=False, valdataloader=valdataloader, attnlayer=attnlayer, batchsize=valbatch)
            else :
                if (i+1) % 1 == 0 or i == -1:
                    if (i+1) % 2 == 0 or i+1==epoch:
                        valmse, valpsnr, valssim, valpsnrs, valssims, valmses =valmain(draw=True, device=device, weight=lastsavedir, rcsdir=valdir, save_dir=valplanedir, logger=logger, epoch=i, trainval=True, draw3d=False, valdataloader=valdataloader, attnlayer=attnlayer, batchsize=valbatch)
                    else:
                        valmse, valpsnr, valssim, valpsnrs, valssims, valmses =valmain(draw=False, device=device, weight=lastsavedir, rcsdir=valdir, save_dir=valplanedir, logger=logger, epoch=i, trainval=True, draw3d=False, valdataloader=valdataloader, attnlayer=attnlayer, batchsize=valbatch)
            val_mse_per_plane[plane].append(valmse.item())
            val_psnr_per_plane[plane].append(valpsnr.item())
            val_ssim_per_plane[plane].append(valssim.item())

            valallpsnrs.extend(valpsnrs) #这里用是因为val是单飞机，但是指标要总的
            valallssims.extend(valssims)
            valallmses.extend(valmses) 
        ave_psnr = sum(valallpsnrs)/len(valallpsnrs)
        ave_ssim = sum(valallssims)/len(valallssims)
        ave_mse = sum(valallmses)/len(valallmses)
        allavemses.append(ave_mse)
        allavepsnrs.append(ave_psnr)
        allavessims.append(ave_ssim)

        statisdir = os.path.join(save_dir,f'sta/statisticAll_epoch{i}_PSNR{ave_psnr:.2f}dB_SSIM{ave_ssim:.4f}_MSE:{ave_mse:.4f}.png')
        if not os.path.exists(os.path.dirname(statisdir)):
            os.makedirs(os.path.dirname(statisdir))
        plotstatistic2(valallpsnrs,valallssims,valallmses,statisdir)
        savefigdata(valallpsnrs,img_path=os.path.join(save_dir,f'sta/valall_epoch{i}psnrs{ave_psnr:.2f}.png'))
        savefigdata(valallssims,img_path=os.path.join(save_dir,f'sta/valall_epoch{i}ssims{ave_ssim:.4f}.png'))
        savefigdata(valallmses,img_path=os.path.join(save_dir,f'sta/valall_epoch{i}mses{ave_mse:.4f}.png'))
        valmse = ave_mse

        #只画val的
        plt.clf()
        for plane, mse_values in val_mse_per_plane.items():
            plt.plot(range(0, i+1), mse_values, label=plane)
            savefigdata(mse_values,img_path=os.path.join(save_dir,f'{plane}_valmse.png'))
        plt.plot(range(0, i+1),allavemses, label='ave', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('Val MSE Curve')
        plt.legend()
        plt.savefig(os.path.join(save_dir,'valmse.png'))
        plt.close()
        savefigdata(allavemses,img_path=os.path.join(save_dir,'valmse.png'))

        plt.clf()
        for plane, psnr_values in val_psnr_per_plane.items():
            plt.plot(range(0, i+1), psnr_values, label=plane)
            savefigdata(psnr_values,img_path=os.path.join(save_dir,f'{plane}_valpsnr.png'))
        plt.plot(range(0, i+1),allavepsnrs, label='ave', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR')
        plt.title('Val PSNR Curve')
        plt.legend()
        plt.savefig(os.path.join(save_dir,'valpsnr.png'))
        plt.close()
        savefigdata(allavepsnrs,img_path=os.path.join(save_dir,'valpsnr.png'))


        plt.clf()
        for plane, ssim_values in val_ssim_per_plane.items():
            plt.plot(range(0, i+1), ssim_values, label=plane)
            savefigdata(ssim_values,img_path=os.path.join(save_dir,f'{plane}_valssim.png'))
        plt.plot(range(0, i+1),allavessims, label='ave', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        plt.title('Val SSIM Curve')
        plt.legend()
        plt.savefig(os.path.join(save_dir,'valssim.png'))
        plt.close()
        savefigdata(allavessims,img_path=os.path.join(save_dir,'valssim.png'))

        lastmse = {k: v[-1] for k, v in val_mse_per_plane.items() if v}
        lastpsnr = {k: v[-1] for k, v in val_psnr_per_plane.items() if v}
        lastssim = {k: v[-1] for k, v in val_ssim_per_plane.items() if v}
        logger.info(f'epoch{i} every aircraft val mse:{lastmse},\npsnr:{lastpsnr},\nssim:{lastssim}')
        logger.info(f'total average val mse:{ave_mse:.4f},psnr:{ave_psnr:.2f},ssim:{ave_ssim:.4f}')

        #画val和train在一起的
        plt.clf()
        for plane, mse_values in val_mse_per_plane.items():
            plt.plot(range(0, i+1), mse_values, label=plane)
        plt.plot(range(0, i+1),allavemses, label='val ave', linestyle='--')
        plt.plot(range(0, i+1),mses, label='train ave', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('Train+Val MSE Curve')
        plt.legend()
        plt.savefig(os.path.join(save_dir,'Trainvalmse.png'))
        plt.close()

        plt.clf()
        for plane, psnr_values in val_psnr_per_plane.items():
            plt.plot(range(0, i+1), psnr_values, label=plane)
        plt.plot(range(0, i+1),allavepsnrs, label='val ave', linestyle='--')
        plt.plot(range(0, i+1),psnrs, label='train ave', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR')
        plt.title('Train+Val PSNR Curve')
        plt.legend()
        plt.savefig(os.path.join(save_dir,'Trainvalpsnr.png'))
        plt.close()

        plt.clf()
        for plane, ssim_values in val_ssim_per_plane.items():
            plt.plot(range(0, i+1), ssim_values, label=plane)
        plt.plot(range(0, i+1),allavessims, label='val ave', linestyle='--')
        plt.plot(range(0, i+1),ssims, label='train ave', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        plt.title('Train+Val SSIM Curve')
        plt.legend()
        plt.savefig(os.path.join(save_dir,'Trainvalssim.png'))
        plt.close()

    else: #ID实验
        if mode == "10train":
            if (i+1) % 1 == 0 or i == -1: 
                logger.info('every epoch val，every 100 epoch draw')
                if (i+1) % 100 == 0:
                    valmse, valpsnr, valssim, valpsnrs, valssims, valmses =valmain(draw=True, device=device, weight=lastsavedir, rcsdir=valdir, save_dir=save_dir, logger=logger, epoch=i, trainval=True, draw3d=False, valdataloader=valdataloader, attnlayer=attnlayer)
                else:
                    valmse, valpsnr, valssim, valpsnrs, valssims, valmses =valmain(draw=False, device=device, weight=lastsavedir, rcsdir=valdir, save_dir=save_dir, logger=logger, epoch=i, trainval=True, draw3d=False, valdataloader=valdataloader, attnlayer=attnlayer)
                
        elif mode == "fasttest":
            if (i+1) % 1 == 0 or i == -1: 
                logger.info('every epoch val，last epoch draw')
                if i+1==epoch:
                    valmse, valpsnr, valssim, valpsnrs, valssims, valmses =valmain(draw=True, device=device, weight=lastsavedir, rcsdir=valdir, save_dir=save_dir, logger=logger, epoch=i, trainval=True, draw3d=False, valdataloader=valdataloader, attnlayer=attnlayer)
                else:
                    valmse, valpsnr, valssim, valpsnrs, valssims, valmses =valmain(draw=False, device=device, weight=lastsavedir, rcsdir=valdir, save_dir=save_dir, logger=logger, epoch=i, trainval=True, draw3d=False, valdataloader=valdataloader, attnlayer=attnlayer)
        else :
            if (i+1) % 1 == 0 or i == -1:
                logger.info('ID 50/100fine, every epoch val，every 50 epoch draw')
                if (i+1) % 50 == 0 or i+1==epoch:
                    valmse, valpsnr, valssim, valpsnrs, valssims, valmses =valmain(draw=True, device=device, weight=lastsavedir, rcsdir=valdir, save_dir=save_dir, logger=logger, epoch=i, trainval=True, draw3d=False, valdataloader=valdataloader, attnlayer=attnlayer)
                else:
                    valmse, valpsnr, valssim, valpsnrs, valssims, valmses =valmain(draw=False, device=device, weight=lastsavedir, rcsdir=valdir, save_dir=save_dir, logger=logger, epoch=i, trainval=True, draw3d=False, valdataloader=valdataloader, attnlayer=attnlayer)

        allavemses.append(valmse)
        allavepsnrs.append(valpsnr)
        allavessims.append(valssim)
        lastmse = valmse
        lastpsnr = valpsnr
        lastssim = valssim


        def draw2dcurve2(curve, savedir, ylabel, title):
            if not os.path.exists(os.path.dirname(savedir)):
                os.makedirs(os.path.dirname(savedir))
            plt.clf()
            plt.figure(figsize=(7, 4.5))
            plt.plot(range(0, i+1), curve, label='ave', linestyle='--')
            plt.xlabel('Epoch')
            plt.ylabel(ylabel)
            plt.title(title)
            plt.savefig(savedir)
            savefigdata(curve,img_path=savedir)
            plt.close()
        draw2dcurve2(allavemses, os.path.join(save_dir,'fig/valmse.png'), 'MSE', 'Val MSE Curve')
        draw2dcurve2(allavepsnrs, os.path.join(save_dir,'fig/valpsnr.png'), 'PSNR', 'Val PSNR Curve')
        draw2dcurve2(allavessims, os.path.join(save_dir,'fig/valssim.png'), 'SSIM', 'Val SSIM Curve')

        def draw2dcurve3(curve1, curve2, savedir, ylabel, title):
            plt.clf()
            plt.figure(figsize=(7, 4.5))
            plt.plot(range(0, i+1), curve1, label='val ave', linestyle='--')
            plt.plot(range(0, i+1), curve2, label='train ave', linestyle='--')
            plt.xlabel('Epoch')
            plt.ylabel(ylabel)
            plt.title(title)
            plt.savefig(savedir)
            plt.close()
        draw2dcurve3(allavemses, mses, os.path.join(save_dir,'Trainvalmse.png'), 'MSE', 'Train+Val MSE Curve')
        draw2dcurve3(allavepsnrs, psnrs, os.path.join(save_dir,'Trainvalpsnr.png'), 'PSNR', 'Train+Val PSNR Curve')
        draw2dcurve3(allavessims, ssims, os.path.join(save_dir,'Trainvalssim.png'), 'SSIM', 'Train+Val SSIM Curve')

    if maxpsnr < valpsnr:
        maxpsnr = valpsnr
    # if minmse > valmse:
    #     minmse = valmse
        # if os.path.exists(maxsavedir):
        #     os.remove(maxsavedir)
        # torch.save(autoencoder.state_dict(), maxsavedir)

lasttime = (time.time()-tic0)/3600
logger.info(f"damaged files：{corrupted_files}")
logger.info(f'train finished time：{time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))}')
logger.info(f'train time consume：{lasttime:.2f}小时')

renamedir = save_dir+f'p{lastpsnr:.2f}-'+f's{lastssim:.4f}-'+f'm{lastmse:.4f}-'+ f't{lasttime:.2f}h'
os.rename(save_dir,renamedir)



