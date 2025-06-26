import torch
import time
from net.GTC_3DEMv4 import MeshCodec
from net.utils import increment_path, EMRCSDataset, get_logger, find_matching_files, process_files,savefigdata
import torch.utils.data.dataloader as DataLoader
# import trimesh
from pathlib import Path
import sys
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from matplotlib.ticker import FuncFormatter
import argparse

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) 
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

def plotRCS3d(rcs,savedir,logger):
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    tic = time.time()
    rcs_np = rcs.detach().cpu().numpy()
    npmax = np.max(rcs_np)
    npmin = np.min(rcs_np)
    theta = np.linspace(0, 2 * np.pi, rcs_np.shape[1])
    phi = np.linspace(0, np.pi, rcs_np.shape[0])
    theta, phi = np.meshgrid(theta, phi)

    x = rcs_np * np.sin(phi) * np.cos(theta)
    y = rcs_np * np.sin(phi) * np.sin(theta)
    z = rcs_np * np.cos(phi)

    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, cmin = 0, cmax = npmax,  surfacecolor=rcs_np, colorscale='Jet', colorbar=dict(exponentformat='E',title=dict(side='top',text="RCS/m²"), showexponent='all', showticklabels=True, thickness = 30,tick0 = 0, dtick = npmax))])

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
            aspectratio=dict(x=1, y=1, z=0.8),
            aspectmode="manual",
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        )
    )
    # pio.show(fig)
    pio.write_image(fig, savedir)

def plot2DRCS(rcs, savedir,logger,cutmax,cutmin=None):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    if cutmin==None:
        cutmin=torch.min(rcs).item()
    tic = time.time()
    # print(rcs.shape)
    vmin = torch.min(rcs)
    vmax = torch.max(rcs)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.jet

    plt.figure()
    plt.imshow(rcs.detach().cpu().numpy(), cmap=cmap, norm=norm, origin='lower')
    plt.colorbar(label='RCS/m²')
    if cutmax != None:
        plt.clim(cutmin, cutmax)
    plt.xlabel("Theta")
    plt.ylabel("Phi")
    plt.savefig(savedir)
    plt.close()
    if logger!=None:
        # logger.info(f'画图用时：{time.time()-tic:.4f}s')
        1
    else:
        print(f'draw time consume：{time.time()-tic:.4f}s')
     
# def plot4D_E_RealImage(ri_tensor, savedir, use_same_max=False):
def plot4D_E_RealImage(ri_tensor, savedir, logger=None, use_same_max=False):
    """
    可视化Real-Image (实部-虚部) 4D张量，并算出总场同步绘制。输入四维须为Real(E_theta) Imag(E_theta) Real(E_phi) Imag(E_phi)的格式。

    Args:
        ri_tensor_path (str): 包含实部虚部信息的 .pt 文件路径。
        savedir (str): 图像保存的完整路径。
        use_same_max (bool): 如果为True，所有子图使用统一的颜色刻度范围。
        logger: 可选的日志记录器。
    """
    # tic = time.time()

    if ri_tensor.ndim != 3:
        logger.info(f"警告: 输入张量维度不正确，期望3D张量，但实际维度为{ri_tensor.ndim}D。尝试将其转换为3D张量。")
        ri_tensor = ri_tensor.squeeze()  # 确保是3D张量

    # E_theta_real = ri_tensor[:, :, 0] #原始数据channel在最后一个维度
    # E_theta_imagine = ri_tensor[:, :, 1]
    # E_phi_real = ri_tensor[:, :, 2]
    # E_phi_imagine = ri_tensor[:, :, 3]
    E_theta_real = ri_tensor[0, :, :] #网络运行的数据channel维在最前面
    E_theta_imagine = ri_tensor[1, :, :]
    E_phi_real = ri_tensor[2, :, :]
    E_phi_imagine = ri_tensor[3, :, :]

    # # --- 2. 动态加载总场强的真值 (Ground Truth) ---
    # amphase_path = ri_tensor_path.replace('_RealImage/', '_Amphase/').replace('_RI.pt', '.pt')    
    # try:
    #     amphase_tensor_gt = torch.load(amphase_path, map_location=torch.device('cpu'))
    #     E_total_abs_gt = amphase_tensor_gt[:, :, 0]
    # except FileNotFoundError:
    #     print(f"警告: 未找到对应的真值文件: {amphase_path}。总场强GT将无法绘制。")
    #     E_total_abs_gt = torch.zeros_like(E_theta_real)
    # # E_total_abs_gt = torch.load('/mnt/truenas_jiangxiaotian/Edataset/complexE_mie_Amphase/b7fd_E_mie_train/b7fd_theta60phi30f0.39.pt')[:, :, 0] # 直接加载总场强的实部

    # --- 3. 从实部虚部正确计算总场强（极化椭圆长半轴）---
    E_abs_theta = torch.sqrt(E_theta_real**2 + E_theta_imagine**2)
    E_abs_phi = torch.sqrt(E_phi_real**2 + E_phi_imagine**2)
    E_phase_theta_rad = torch.atan2(E_theta_imagine, E_theta_real)
    E_phase_phi_rad = torch.atan2(E_phi_imagine, E_phi_real)
    delta_phi_rad = E_phase_theta_rad - E_phase_phi_rad
    E_abs_theta_sq = E_abs_theta**2
    E_abs_phi_sq = E_abs_phi**2
    term1 = E_abs_theta_sq + E_abs_phi_sq
    term2_inner_sqrt = torch.sqrt((E_abs_theta_sq - E_abs_phi_sq)**2 + 4 * E_abs_theta_sq * E_abs_phi_sq * (torch.cos(delta_phi_rad))**2)
    E_total_abs_sq = 0.5 * (term1 + term2_inner_sqrt)
    E_total_abs_compute = torch.sqrt(E_total_abs_sq)

    # --- 4. 准备绘图数据 ---
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle('E-Field Real/Imaginary Components', fontsize=16)

    components_data = [
        (E_theta_real, 'Real(E_theta)', 'V/m'),
        (E_theta_imagine, 'Imag(E_theta)', 'V/m'),
        (E_phi_real, 'Real(E_phi)', 'V/m'),
        (E_phi_imagine, 'Imag(E_phi)', 'V/m'),
        # (E_total_abs_gt, 'Total E-Field GT', 'V/m'),
        (E_total_abs_compute, 'Total E-Field Computed', 'V/m')
    ]
    
    # --- 5. 新增功能：计算全局颜色范围 ---
    global_min, global_max = None, None
    if use_same_max:
        # 将所有6个张量的数据堆叠起来，高效地计算全局最大最小值
        all_tensors = [d[0] for d in components_data]
        stacked_data = torch.stack(all_tensors)
        global_min = stacked_data.min().item()
        global_max = stacked_data.max().item()
        print(f"启用全局颜色刻度: Min={global_min:.4f}, Max={global_max:.4f}")

    # --- 6. 绘制 3x2 六子图 ---
    ax_flat = axes.flatten()

    for i, (data, title, label) in enumerate(components_data):
        ax = ax_flat[i]
        
        # 根据 use_same_max 参数决定 vmin 和 vmax
        if use_same_max:
            vmin, vmax = global_min, global_max
        else:
            # 默认行为：只统一最后两个总场图的颜色范围 也不要了
            vmin, vmax = None, None
            # if 'Total E-Field' in title:
            #     vmin = min(E_total_abs_gt.min(), E_total_abs_compute.min()).item()
            #     vmax = max(E_total_abs_gt.max(), E_total_abs_compute.max()).item()

        im = ax.imshow(data.detach().cpu().numpy(), cmap='jet', origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Theta Index")
        ax.set_ylabel("Phi Index")
        fig.colorbar(im, ax=ax, label=label)

    if len(components_data)==5:
        axes[2, 1].axis('off')

    # 调整布局防止重叠
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # 保存和显示图像
    plt.savefig(savedir) #这一步好慢
    # plt.show()
    plt.close(fig)
    # logger.info(f'画图并保存到 {os.path.basename(savedir)} 用时：{time.time()-tic:.2f}s')



def plotstatistic2(psnr_list, ssim_list, mse_list, statisticdir):
    def to_percent(y,position):
        return str(int((100*y)))
    binss = 20

    plt.clf()
    plt.figure(figsize=(12, 6))

    #-----------------------------------mse-------------------------------------------
    print(len(mse_list))
    plt.subplot(3, 3, 1)
    counts, bins, patches = plt.hist(mse_list, bins=binss, edgecolor='black', density=True)

    mu, std = norm.fit(mse_list)
    # x = np.linspace(-5, 15, 1000)
    x = np.linspace(min(mse_list), max(mse_list), 1000)
    plt.plot(x, norm.pdf(x, mu, std), 'r-', linewidth=2, label='Normal fit')
    plt.xlabel('MSE')
    # plt.ylabel('Probability of samples')
    plt.ylabel('Probability of samples (%)')
    plt.title('MSE Histogram and Normal Fit')
    plt.legend()


    #-----------------------------------PSNR-------------------------------------------
    plt.subplot(3, 3, 2)
    counts, bins, patches = plt.hist(psnr_list, bins=binss, edgecolor='black', density=True)
    fomatter=FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(fomatter)
    mu, std = norm.fit(psnr_list)
    # x = np.linspace(15,45, 1000)
    x = np.linspace(min(psnr_list), max(psnr_list), 1000)
    plt.plot(x, norm.pdf(x, mu, std), 'r-', linewidth=2, label='Normal fit')
    plt.xlabel('PSNR')
    # plt.ylabel('Probability of samples')
    plt.ylabel('Probability of samples (%)')
    plt.title('PSNR Histogram and Normal Fit')
    plt.legend()

    #-----------------------------------SSIM-------------------------------------------
    plt.subplot(3, 3, 3)
    # counts, bins, patches = plt.hist(ssim_list, bins=binss, edgecolor='black', density=True, stacked=True)
    counts, bins, patches = plt.hist(ssim_list, bins=binss, edgecolor='black', density=True)
    mu, std = norm.fit(ssim_list)
    # x = np.linspace(0.6,1.1, 1000)
    x = np.linspace(min(ssim_list), max(ssim_list), 1000)
    plt.plot(x, norm.pdf(x, mu, std), 'r-', linewidth=2, label='Normal fit')
    plt.xlabel('SSIM')
    # plt.ylabel('Probability of samples')
    plt.ylabel('Probability of samples (%)')
    plt.title('SSIM Histogram and Normal Fit')
    plt.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig(statisticdir)
    plt.close()


def valmain(draw, device, weight, rcsdir, save_dir, logger, epoch, trainval=False, draw3d=False, n=4, middim=64,attnlayer=0, valdataloader=None):
    tic = time.time()
    # logger.info(f'val batchsize={batchsize}')

    in_ems = []
    rcss = []
    psnrs = []
    ssims = []
    mses = []
    losses = []
    inftimes = []
    corrupted_files = []
    dataloader=valdataloader
    #-------------------------------------------------------------------------------------
    if trainval == False:
        logger.info(f'device:{device}')

    autoencoder = MeshCodec(device= device,
                            attn_encoder_depth = attnlayer,
                            ).to(device)
    autoencoder.load_state_dict(torch.load(weight,weights_only=False), strict=True)
    # autoencoder = autoencoder.to(device)
    #-------------------------------------------------------------------------------------
    with torch.no_grad():
        for in_em1,rcs1 in tqdm(dataloader,desc=f'val process',ncols=70,postfix=f''):
            in_em0 = in_em1.copy()
            objlist , _ = find_matching_files(in_em1[0], "./planes")
            planesur_faces, planesur_verts, planesur_faceedges, geoinfo = process_files(objlist, device)

            start_time0 = time.time()
            loss, outrcs, psnr_mean, psnrlist, ssim_mean, ssimlist, mse, nmse, rmse, l1, percentage_error, mselist, metrics= autoencoder( #这里使用网络，是进去跑了forward
                vertices = planesur_verts,
                faces = planesur_faces, #torch.Size([batchsize, 33564, 3])
                face_edges = planesur_faceedges,
                in_em = in_em1,#.to(device)
                GT = rcs1.to(device), 
                logger = logger,
                device = device,
            )
            inftime = time.time()-start_time0

            if trainval == False:
                logger.info(f'one batch inference：{time.time()-start_time0:.4f}s，average one sample inference：{(time.time()-start_time0)/rcs1.shape[0]:.4f}s')
            # torch.cuda.empty_cache()
            if draw == True:
                for i in range(outrcs.shape[0]): 
                    single_outrcs = outrcs[i].squeeze().to(device) #这里i是batch索引
                    single_rcs1 = rcs1[i].squeeze().to(device)
                    single_diff = single_rcs1-single_outrcs

                    eminfo = [int(in_em0[1][i]), int(in_em0[2][i]), float(in_em0[3][i])]
                    plane = in_em0[0][i]
                    psnr1 = psnrlist[i].item()
                    ssim1 = ssimlist[i].item()
                    mse1 = mselist[i].item()

                    save_dir2 = os.path.join(save_dir,f'epoch{epoch}')
                    Path(save_dir2).mkdir(exist_ok=True)
                    # out3dpngpath = os.path.join(save_dir2,f'epoch{epoch}_{plane}_theta{eminfo[0]}phi{eminfo[1]}freq{eminfo[2]:.3f}.png')
                    # out3dGTpngpath = os.path.join(save_dir2,f'epoch{epoch}_{plane}_theta{eminfo[0]}phi{eminfo[1]}freq{eminfo[2]:.3f}_GT.png')

                    out2DGTpngpath = os.path.join(save_dir2,f'epoch{epoch}_{plane}_theta{eminfo[0]}phi{eminfo[1]}freq{eminfo[2]:.3f}_2DGT.png')
                    out2Drcspngpath = os.path.join(save_dir2,f'epoch{epoch}_{plane}_theta{eminfo[0]}phi{eminfo[1]}freq{eminfo[2]:.3f}_psnr{psnr1:.2f}_ssim{ssim1:.4f}_mse{mse1:.4f}_2D.png')
                    # out2Drcspngpathcut = os.path.join(save_dir2,f'epoch{epoch}_{plane}_theta{eminfo[0]}phi{eminfo[1]}freq{eminfo[2]:.3f}_psnr{psnr1:.2f}_ssim{ssim1:.4f}_mse{mse1:.4f}_2Dcut.png')
                    # out2Drcspngpathdiff = os.path.join(save_dir2,f'epoch{epoch}_{plane}_theta{eminfo[0]}phi{eminfo[1]}freq{eminfo[2]:.3f}_psnr{psnr1:.2f}_ssim{ssim1:.4f}_mse{mse1:.4f}_diff{(torch.max(torch.abs(torch.max(single_diff)),torch.abs(torch.min(single_diff)))).item():.4f}_2Ddiff.png')
                    plot4D_E_RealImage(single_rcs1, savedir=out2DGTpngpath, logger=logger)
                    plot4D_E_RealImage(single_outrcs, savedir=out2Drcspngpath, logger=logger)
                    # plot4D_E_RealImage(rcs=single_outrcs, savedir=out2Drcspngpathcut, logger=logger,cutmax=torch.max(single_rcs1).item())
                    # plot4D_E_RealImage(single_diff, savedir=out2Drcspngpathdiff, logger=logger)

                    # if draw3d == True:
                    #     plotRCS3d(rcs=single_rcs1, savedir=out3dGTpngpath, logger=logger)
                    #     plotRCS3d(rcs=single_outrcs, savedir=out3dpngpath, logger=logger)
            
            torch.cuda.empty_cache()
            losses.append(loss.detach().cpu())
            psnrs.extend(psnrlist.detach().cpu())
            ssims.extend(ssimlist.detach().cpu())
            mses.extend(mselist.detach().cpu())
            inftimes.append(inftime)

        ave_loss = sum(losses)/len(losses)
        ave_psnr = sum(psnrs)/len(psnrs)
        ave_ssim = sum(ssims)/len(ssims)
        ave_mse = sum(mses)/len(mses)
        ave_inftime = sum(inftimes)/len(inftimes)
        if trainval == False:
            logger.info(f"using model weight{weight} test {len(losses)} samples, Mean Loss: {ave_loss:.4f}, Mean PSNR: {ave_psnr:.2f}dB, Mean SSIM: {ave_ssim:.4f}, Mean MSE:{ave_mse:.4f}")
            logger.info(f'test set dir:{rcsdir}, total time consume:{time.strftime("%H:%M:%S", time.gmtime(time.time()-tic))}')
            logger.info(f"damaged files：{corrupted_files}")
        logger.info(f'val set:{rcsdir}, total time consume:{time.strftime("%H:%M:%S", time.gmtime(time.time()-tic))}')
        logger.info(f'↑----val psnr:{ave_psnr:.2f},ssim:{ave_ssim:.4f},mse:{ave_mse:.4f},inftime:{ave_inftime:.4f}s----↑') #这个val loss没用
        # logger.info(f'↑----val loss:{ave_loss:.4f},psnr:{ave_psnr:.2f},ssim:{ave_ssim:.4f},mse:{ave_mse:.4f},inftime:{ave_inftime:.4f}s----↑') #这个val loss没用

        statisdir = os.path.join(save_dir,f'sta/statistic_epoch{epoch}_PSNR{ave_psnr:.2f}dB_SSIM{ave_ssim:.4f}_MSE:{ave_mse:.4f}_Loss{ave_loss:.4f}.png')
        if not os.path.exists(os.path.dirname(statisdir)):
            os.makedirs(os.path.dirname(statisdir))
        plotstatistic2(psnrs,ssims,mses,statisdir) #这里直接用psnrs画统计图和统计指标
        savefigdata(psnrs,img_path=os.path.join(save_dir,f'sta/valall_epoch{epoch}psnrs{ave_psnr:.2f}.png'))
        savefigdata(ssims,img_path=os.path.join(save_dir,f'sta/valall_epoch{epoch}ssims{ave_ssim:.4f}.png'))
        savefigdata(mses,img_path=os.path.join(save_dir,f'sta/valall_epoch{epoch}mses{ave_mse:.4f}.png'))
    return ave_mse, ave_psnr, ave_ssim, psnrs, ssims, mses  #ave_psnr, 

def parse_args():
    parser = argparse.ArgumentParser(description="Script with customizable parameters using argparse.")
    parser.add_argument('--valdir', type=str, default='testrcs', help='Path to validation directory')
    parser.add_argument('--pretrainweight', type=str, default=None, help='Path to pretrained weights')
    parser.add_argument('--attn', type=int, default=1, help='Transformer layers')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    weight = args.pretrainweight
    valdir = args.valdir
    attnlayer = args.attn
    
    trainval = False
    # cuda = 'cuda:0'
    cuda = 'cpu'
    draw = True
    # draw = False
    draw3d = False
    lgrcs = False
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    batchsize = 40
    
    from datetime import datetime
    date = datetime.today().strftime("%m%d")
    save_dir = str(increment_path(Path(ROOT / "output" / "inference" /f'{date}_{valdir}'), exist_ok=False))
    logdir = os.path.join(save_dir,'alog.txt')
    logger = get_logger(logdir)
    epoch = -1
    
    # valfilelist = os.listdir(valdir)
    valdataset = EMRCSDataset(valdir)
    # valdataset = EMRCSDataset(valfilelist, valdir)
    valdataloader = DataLoader.DataLoader(valdataset, batch_size=batchsize, shuffle=False, num_workers=16, pin_memory=True)
    if trainval == False:
        logger.info(f'using model weight{weight} inference test set{valdir} and draw')
    valmain(draw, device, weight, valdir, save_dir, logger, epoch, trainval, draw3d, attnlayer=attnlayer, valdataloader=valdataloader, batchsize=batchsize)