# GTC-3DEMv4.4
#
# 主要改动:
# 1. 引入 complexPyTorch 库，这是一个现代且维护良好的复数网络库。
# 2. 将 Decoder 中的实数层替换为 complexPyTorch.complexLayers 中的复数版本。
# 3. 为保持参数量近似，我们将进入Decoder的`middim`个实数通道分为两半，
#    分别作为`middim/2`个复数通道的实部和虚部。后续复数层的通道数也相应减半。
# 4. 修改了 decode 方法，增加了实数到复数的转换逻辑，并使用复数激活函数。
# 5. (v4.3) 将原有的单解码器结构修改为两个独立的、不共享权重的解码器头。
#    一个头专门用于预测 E_theta 复数场，另一个专门用于预测 E_phi 复数场。
# 6. 最终将两个解码器头的复数输出分解为四个实数通道，以保持与原有代码和GT的兼容性。

from torch.nn import Module, ModuleList
import torch 
from torch import nn
from torch_geometric.nn.conv import SAGEConv
import torch.nn.functional as F
# import numpy as np
from net.utils import derive_face_edges_from_faces, transform_to_log_coordinates, psnr, batch_mse
from net.utils import ssim as myssim
from net.utils_pinn import bandlimit_energy_ratio, helmholtz_consistency, helmholtz_loss_4d, bandlimit_loss_4d, WeightedFieldLoss, reciprocity_loss_fn, maxloss_4d
from einops import rearrange, pack
from math import pi
from complexPyTorch.complexLayers import ComplexConvTranspose2d, ComplexConv2d, ComplexBatchNorm2d
from complexPyTorch.complexFunctions import complex_relu

def get_angle_embedding(theta_degrees, phi_degrees):
    """
    将物理角度（theta, phi）转换为4D周期性嵌入向量。
    
    参数:
        theta_degrees (Tensor): 天顶角 (polar angle), 范围 0-180.
        phi_degrees (Tensor): 方位角 (azimuthal angle), 范围 0-360.
        
    返回:
        Tensor: 形状为 [N, 1, 4] 的嵌入向量.
    """
    # 1. 将角度从度(degrees)转换为弧度(radians)
    theta_rad = theta_degrees * (torch.pi / 180.0)
    phi_rad = phi_degrees * (torch.pi / 180.0)

    # 2. 计算每个角度的sin和cos值
    theta_cos = torch.cos(theta_rad)
    theta_sin = torch.sin(theta_rad)
    phi_cos = torch.cos(phi_rad)
    phi_sin = torch.sin(phi_rad)
    
    # 3. 堆叠成 [N, 4] 的张量
    # 约定顺序: [theta_cos, theta_sin, phi_cos, phi_sin]
    embedding = torch.stack([theta_cos, theta_sin, phi_cos, phi_sin], dim=1)
    
    # 4. 增加一个维度以匹配后续层的期望输入形状 [N, 1, 4]
    return embedding.unsqueeze(1)

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

def jxtget_face_coords(vertices, face_indices):
    batch_size, num_faces, num_vertices_per_face = face_indices.shape
    reshaped_face_indices = face_indices.reshape(batch_size, -1).to(dtype=torch.int64) 
    face_coords = torch.gather(vertices, 1, reshaped_face_indices.unsqueeze(-1).expand(-1, -1, vertices.shape[-1]))
    face_coords = face_coords.reshape(batch_size, num_faces, num_vertices_per_face, -1)
    return face_coords

def coords_interanglejxt2(x, y, eps=1e-5):
    edge_vector = x - y
    normv = l2norm(edge_vector)
    normdot = -(normv * torch.cat((normv[..., -1:], normv[..., :-1]), dim=3)).sum(dim=2)
    normdot = torch.clamp(normdot, -1 + eps, 1 - eps)
    radians = torch.acos(normdot)
    angle = torch.rad2deg(radians)
    return radians, angle

def polar_to_cartesian2(theta, phi):
    theta_rad = torch.deg2rad(theta)
    phi_rad = torch.deg2rad(phi)
    x = torch.sin(phi_rad) * torch.cos(theta_rad)
    y = torch.sin(phi_rad) * torch.sin(theta_rad)
    z = torch.cos(phi_rad)
    return torch.stack([x, y, z], dim=1)

def vector_anglejxt2(x, y, eps=1e-5):
    normdot = -(l2norm(x) * l2norm(y)).sum(dim=-1)
    normdot = torch.clamp(normdot, -1 + eps, 1 - eps)
    radians = normdot.acos()
    angle = torch.rad2deg(radians)
    return radians, angle

def get_derived_face_featuresjxt(face_coords, in_em, device):
    shifted_face_coords = torch.cat((face_coords[:, :, -1:], face_coords[:, :, :-1]), dim = 2).to(device)
    angles, _  = coords_interanglejxt2(face_coords, shifted_face_coords)
    edge1, edge2, *_ = (face_coords - shifted_face_coords).unbind(dim = 2)
    normals = l2norm(torch.cross(edge1, edge2, dim = -1))
    area = torch.cross(edge1, edge2, dim = -1).norm(dim = -1, keepdim = True) * 0.5
    incident_angle_vec = polar_to_cartesian2(in_em[1],in_em[2]).float()
    incident_angle_mtx = incident_angle_vec.unsqueeze(1).repeat(1, area.shape[1], 1).to(device)
    incident_freq_mtx = in_em[3].float().unsqueeze(1).unsqueeze(2).repeat(1, area.shape[1], 1).to(device)
    incident_mesh_anglehudu, _ = vector_anglejxt2(normals, incident_angle_mtx)
    return dict(
        angles = angles, area = area, normals = normals,
        emnoangle = incident_mesh_anglehudu.unsqueeze(-1),
        emangle = incident_angle_mtx, emfreq = incident_freq_mtx
    )

class MeshCodec(Module):
    def __init__(
            self,
            device,
            attn_encoder_depth=0,
            middim=64,
            attn_dropout=0.,
            dim_coor_embed = 64,      
            dim_area_embed = 16,      
            dim_normal_embed = 64,    
            dim_angle_embed = 16,     
            dim_emnoangle_embed = 16, 
            dim_emangle_embed = 64,       
            dim_emfreq_embed = 16,        
            encoder_dims_through_depth = (64, 128, 256, 256, 576),    
            ):
        super().__init__()

        self.condfreqlayers = ModuleList([
            nn.Linear(1, 64), nn.Linear(1, 128), nn.Linear(1, 256), nn.Linear(1, 256),
        ])
        self.condanglelayers = ModuleList([
            nn.Linear(4, 64), nn.Linear(4, 128), nn.Linear(4, 256), nn.Linear(4, 256),
        ])#sincos方案后 将输入的维度从 2 改为 4

        # self.condanglelayers = ModuleList([ nn.Linear(2, 64), nn.Linear(2, 128), nn.Linear(2, 256), nn.Linear(2, 256),])
        

        self.angle_embed = nn.Linear(3, 3*dim_angle_embed)
        self.area_embed = nn.Linear(1, dim_area_embed)
        self.normal_embed = nn.Linear(3, 3*dim_normal_embed)
        self.emnoangle_embed = nn.Linear(1, dim_emnoangle_embed)
        self.emangle_embed = nn.Linear(3, 3*dim_emangle_embed)
        self.emfreq_embed = nn.Linear(1, dim_emfreq_embed)
        self.coor_embed = nn.Linear(9, 9*dim_coor_embed) 

        init_dim = dim_coor_embed * (3 * 3) + dim_angle_embed * 3 + dim_normal_embed * 3 + dim_area_embed + dim_emangle_embed * 3 + dim_emnoangle_embed + dim_emfreq_embed

        sageconv_kwargs = dict(normalize = True, project = True)
        init_encoder_dim, *encoder_dims_through_depth = encoder_dims_through_depth
        curr_dim = init_encoder_dim

        self.init_sage_conv = SAGEConv(init_dim, init_encoder_dim, **sageconv_kwargs)
        self.init_encoder_act_and_norm = nn.Sequential(nn.SiLU(), nn.LayerNorm(init_encoder_dim)) 
        
        self.encoders = ModuleList([])
        self.encoder_act_and_norm = ModuleList([])

        for dim_layer in encoder_dims_through_depth:
            self.encoders.append(SAGEConv(curr_dim, dim_layer, **sageconv_kwargs))
            self.encoder_act_and_norm.append(nn.Sequential(nn.SiLU(), nn.LayerNorm(dim_layer)))
            curr_dim = dim_layer

        self.encoder_attn_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=curr_dim, nhead=8, dropout=attn_dropout)
            for _ in range(attn_encoder_depth)
        ])

        #--- Adaptation Module (保持不变) ---
        self.conv1d1 = nn.Conv1d(576, middim*2, kernel_size=10, stride=10, dilation=1 ,padding=0)
        self.fc1d1 = nn.Linear(2250, 45*90)

        # --- Complex Decoder (重大修改 - 两个独立的复数解码器头) ---
        assert middim % 2 == 0, "middim 必须是偶数才能转换为复数通道"
        complex_in_channels = middim #不减半
        # complex_in_channels = middim // 2
        
        # --- Decoder for E_theta ---
        # 第一次上采样
        self.upconv1_complex_theta = ComplexConvTranspose2d(complex_in_channels, complex_in_channels // 2, kernel_size=2, stride=2)
        self.bn1_complex_theta = ComplexBatchNorm2d(complex_in_channels // 2)
        self.conv1_1_complex_theta = ComplexConv2d(complex_in_channels // 2, complex_in_channels // 2, kernel_size=3, stride=1, padding=1)
        self.bn1_1_complex_theta = ComplexBatchNorm2d(complex_in_channels // 2)
        self.conv1_2_complex_theta = ComplexConv2d(complex_in_channels // 2, complex_in_channels // 2, kernel_size=3, stride=1, padding=1)
        self.bn1_2_complex_theta = ComplexBatchNorm2d(complex_in_channels // 2)
        # 第二次上采样
        self.upconv2_complex_theta = ComplexConvTranspose2d(complex_in_channels // 2, complex_in_channels // 4, kernel_size=2, stride=2)
        self.bn2_complex_theta = ComplexBatchNorm2d(complex_in_channels // 4)
        self.conv2_1_complex_theta = ComplexConv2d(complex_in_channels // 4, complex_in_channels // 4, kernel_size=3, stride=1, padding=1)
        self.bn2_1_complex_theta = ComplexBatchNorm2d(complex_in_channels // 4)
        self.conv2_2_complex_theta = ComplexConv2d(complex_in_channels // 4, complex_in_channels // 4, kernel_size=3, stride=1, padding=1)
        self.bn2_2_complex_theta = ComplexBatchNorm2d(complex_in_channels // 4)
        # 第三次上采样
        self.upconv3_complex_theta = ComplexConvTranspose2d(complex_in_channels // 4, complex_in_channels // 8, kernel_size=2, stride=2)
        self.bn3_complex_theta = ComplexBatchNorm2d(complex_in_channels // 8)
        self.conv3_1_complex_theta = ComplexConv2d(complex_in_channels // 8, complex_in_channels // 8, kernel_size=3, stride=1, padding=1)
        self.bn3_1_complex_theta = ComplexBatchNorm2d(complex_in_channels // 8)
        self.conv3_2_complex_theta = ComplexConv2d(complex_in_channels // 8, complex_in_channels // 8, kernel_size=3, stride=1, padding=1)
        self.bn3_2_complex_theta = ComplexBatchNorm2d(complex_in_channels // 8)
        # E_theta 输出头
        self.head_complex_theta = ComplexConv2d(complex_in_channels // 8, 1, kernel_size=1, stride=1, padding=0)

        # --- Decoder for E_phi ---
        # 第一次上采样
        self.upconv1_complex_phi = ComplexConvTranspose2d(complex_in_channels, complex_in_channels // 2, kernel_size=2, stride=2)
        self.bn1_complex_phi = ComplexBatchNorm2d(complex_in_channels // 2)
        self.conv1_1_complex_phi = ComplexConv2d(complex_in_channels // 2, complex_in_channels // 2, kernel_size=3, stride=1, padding=1)
        self.bn1_1_complex_phi = ComplexBatchNorm2d(complex_in_channels // 2)
        self.conv1_2_complex_phi = ComplexConv2d(complex_in_channels // 2, complex_in_channels // 2, kernel_size=3, stride=1, padding=1)
        self.bn1_2_complex_phi = ComplexBatchNorm2d(complex_in_channels // 2)
        # 第二次上采样
        self.upconv2_complex_phi = ComplexConvTranspose2d(complex_in_channels // 2, complex_in_channels // 4, kernel_size=2, stride=2)
        self.bn2_complex_phi = ComplexBatchNorm2d(complex_in_channels // 4)
        self.conv2_1_complex_phi = ComplexConv2d(complex_in_channels // 4, complex_in_channels // 4, kernel_size=3, stride=1, padding=1)
        self.bn2_1_complex_phi = ComplexBatchNorm2d(complex_in_channels // 4)
        self.conv2_2_complex_phi = ComplexConv2d(complex_in_channels // 4, complex_in_channels // 4, kernel_size=3, stride=1, padding=1)
        self.bn2_2_complex_phi = ComplexBatchNorm2d(complex_in_channels // 4)
        # 第三次上采样
        self.upconv3_complex_phi = ComplexConvTranspose2d(complex_in_channels // 4, complex_in_channels // 8, kernel_size=2, stride=2)
        self.bn3_complex_phi = ComplexBatchNorm2d(complex_in_channels // 8)
        self.conv3_1_complex_phi = ComplexConv2d(complex_in_channels // 8, complex_in_channels // 8, kernel_size=3, stride=1, padding=1)
        self.bn3_1_complex_phi = ComplexBatchNorm2d(complex_in_channels // 8)
        self.conv3_2_complex_phi = ComplexConv2d(complex_in_channels // 8, complex_in_channels // 8, kernel_size=3, stride=1, padding=1)
        self.bn3_2_complex_phi = ComplexBatchNorm2d(complex_in_channels // 8)
        # E_phi 输出头
        self.head_complex_phi = ComplexConv2d(complex_in_channels // 8, 1, kernel_size=1, stride=1, padding=0)

        
    def encode(self, *, vertices, faces, face_edges, in_em):
        device = vertices.device 
        face_coords = jxtget_face_coords(vertices, faces) 
        in_em[3] = transform_to_log_coordinates(in_em[3])
        in_em = [in_em[0], in_em[1].float(), in_em[2].float(), in_em[3].float()]
        derived_features = get_derived_face_featuresjxt(face_coords, in_em, device)

        angle_embed = self.angle_embed(derived_features['angles'])
        area_embed = self.area_embed(derived_features['area'])
        normal_embed = self.normal_embed(derived_features['normals'])
        emnoangle_embed = self.emnoangle_embed(derived_features['emnoangle'])
        emangle_embed = self.emangle_embed(derived_features['emangle'])
        emfreq_embed = self.emfreq_embed(derived_features['emfreq'])
        face_coords = rearrange(face_coords, 'b nf nv c -> b nf (nv c)')
        face_coor_embed = self.coor_embed(face_coords)

        face_embed, _ = pack([face_coor_embed, angle_embed, area_embed, normal_embed, emnoangle_embed, emangle_embed, emfreq_embed], 'b nf *') 

        face_edges = face_edges.reshape(2, -1).to(device)
        orig_face_embed_shape = face_embed.shape[:2]
        face_embed = face_embed.reshape(-1, face_embed.shape[-1])
        
        face_embed = self.init_sage_conv(face_embed, face_edges)
        face_embed = self.init_encoder_act_and_norm(face_embed)
        face_embed = face_embed.reshape(orig_face_embed_shape[0], orig_face_embed_shape[1], -1)

        # in_angle = torch.stack([in_em[1]/180, in_em[2]/360]).t().float().unsqueeze(1).to(device)
        in_angle = get_angle_embedding(in_em[1].float(), in_em[2].float()).to(device) #sincos方案后
        in_freq = in_em[3].float().unsqueeze(1).unsqueeze(1).to(device)

        for i, (conv, act_norm) in enumerate(zip(self.encoders, self.encoder_act_and_norm)):
            condfreq = self.condfreqlayers[i](in_freq)
            condangle = self.condanglelayers[i](in_angle)
            face_embed = face_embed + condangle + condfreq
            face_embed = face_embed.reshape(-1, face_embed.shape[-1])
            face_embed = conv(face_embed, face_edges)
            face_embed = act_norm(face_embed)
            face_embed = face_embed.reshape(orig_face_embed_shape[0], orig_face_embed_shape[1], -1)
          
        for attn_layer in self.encoder_attn_blocks:
            face_embed = face_embed.permute(1, 0, 2)
            face_embed = attn_layer(face_embed) + face_embed
            face_embed = face_embed.permute(1, 0, 2)

        return face_embed, in_angle, in_freq
    
    def decode(self, x, in_angle, in_freq, device):
        # Adaptation Module (保持不变)
        pad_size = 22500 - x.size(1)
        x = F.pad(x, (0, 0, 0, pad_size)) 
        x = x.view(x.size(0), -1, 22500) 
        
        x = self.conv1d1(x) 
        x = F.relu(x)

        x = self.fc1d1(x)
        x = x.reshape(x.size(0), -1, 45*90) 
        x = x.reshape(x.size(0), -1, 45, 90) 
        
        # --- Complex Decoder Forward Pass ---
        # 将实数张量 x 转换为复数张量
        # 前一半通道作为实部，后一半通道作为虚部
        middim_half = x.shape[1] // 2
        x_re = x[:, :middim_half, :, :]
        x_im = x[:, middim_half:, :, :]
        x_complex = torch.complex(x_re, x_im)

        # --- E_theta Decoder Path ---
        x_theta = self.upconv1_complex_theta(x_complex)
        x_theta = self.bn1_complex_theta(x_theta)
        x_theta = complex_relu(x_theta)
        x_theta = self.conv1_1_complex_theta(x_theta)
        x_theta = self.bn1_1_complex_theta(x_theta)
        x_theta = complex_relu(x_theta)
        x_theta = self.conv1_2_complex_theta(x_theta)
        x_theta = self.bn1_2_complex_theta(x_theta)
        x_theta = complex_relu(x_theta)
        
        x_theta = self.upconv2_complex_theta(x_theta)
        x_theta = self.bn2_complex_theta(x_theta)
        x_theta = complex_relu(x_theta)
        x_theta = self.conv2_1_complex_theta(x_theta)
        x_theta = self.bn2_1_complex_theta(x_theta)
        x_theta = complex_relu(x_theta)
        x_theta = self.conv2_2_complex_theta(x_theta)
        x_theta = self.bn2_2_complex_theta(x_theta)
        x_theta = complex_relu(x_theta)

        x_theta = self.upconv3_complex_theta(x_theta)
        x_theta = self.bn3_complex_theta(x_theta)
        x_theta = complex_relu(x_theta)
        x_theta = self.conv3_1_complex_theta(x_theta)
        x_theta = self.bn3_1_complex_theta(x_theta)
        x_theta = complex_relu(x_theta)
        x_theta = self.conv3_2_complex_theta(x_theta)
        x_theta = self.bn3_2_complex_theta(x_theta)
        x_theta = complex_relu(x_theta)
        
        e_theta_complex = self.head_complex_theta(x_theta)

        # --- E_phi Decoder Path ---
        x_phi = self.upconv1_complex_phi(x_complex)
        x_phi = self.bn1_complex_phi(x_phi)
        x_phi = complex_relu(x_phi)
        x_phi = self.conv1_1_complex_phi(x_phi)
        x_phi = self.bn1_1_complex_phi(x_phi)
        x_phi = complex_relu(x_phi)
        x_phi = self.conv1_2_complex_phi(x_phi)
        x_phi = self.bn1_2_complex_phi(x_phi)
        x_phi = complex_relu(x_phi)

        x_phi = self.upconv2_complex_phi(x_phi)
        x_phi = self.bn2_complex_phi(x_phi)
        x_phi = complex_relu(x_phi)
        x_phi = self.conv2_1_complex_phi(x_phi)
        x_phi = self.bn2_1_complex_phi(x_phi)
        x_phi = complex_relu(x_phi)
        x_phi = self.conv2_2_complex_phi(x_phi)
        x_phi = self.bn2_2_complex_phi(x_phi)
        x_phi = complex_relu(x_phi)

        x_phi = self.upconv3_complex_phi(x_phi)
        x_phi = self.bn3_complex_phi(x_phi)
        x_phi = complex_relu(x_phi)
        x_phi = self.conv3_1_complex_phi(x_phi)
        x_phi = self.bn3_1_complex_phi(x_phi)
        x_phi = complex_relu(x_phi)
        x_phi = self.conv3_2_complex_phi(x_phi)
        x_phi = self.bn3_2_complex_phi(x_phi)
        x_phi = complex_relu(x_phi)

        e_phi_complex = self.head_complex_phi(x_phi)

        # 将复数输出分解为 4 个实数通道以匹配 GT 格式
        out_re_etheta = e_theta_complex.real
        out_im_etheta = e_theta_complex.imag
        out_re_ephi   = e_phi_complex.real
        out_im_ephi   = e_phi_complex.imag
        
        return torch.cat([out_re_etheta, out_im_etheta, out_re_ephi, out_im_ephi], dim=1)
    

    def forward(self, *, vertices, faces, face_edges=None, in_em, GT=None, logger=None, device='cpu', loss_type='L1', **kwargs):
        original_in_em = [in_em[0], in_em[1].clone(), in_em[2].clone(), in_em[3].clone()]
        original_freqs_ghz = in_em[3].clone()

        loss_fn = WeightedFieldLoss(
            lambda_main=kwargs.get('lambda_main', 10.0), 
            loss_type=loss_type,
        )
        lambda_max = kwargs.get('lambda_max', 0.0001)
        lambda_helmholtz = kwargs.get('lambda_helmholtz', 0)
        lambda_bandlimit = kwargs.get('lambda_bandlimit', 0)
        lambda_reciprocity = kwargs.get('lambda_reciprocity', 0)
        
        epochnow = kwargs.get('epochnow', 0)
        pinnepoch = kwargs.get('pinnepoch', 0)
                
        if face_edges is None:
            face_edges = derive_face_edges_from_faces(faces, pad_id=-1)

        encoded, in_angle, in_freq = self.encode(
            vertices=vertices, faces=faces, face_edges=face_edges, in_em=in_em)

        decoded = self.decode(encoded, in_angle, in_freq, device)

        if GT is None:
            return decoded
        
        else:
            if GT.dim() > 1 and GT.shape[2:4] == (361, 720):
                GT = GT[:, :, :-1, :]
            
            mainloss = loss_fn(decoded, GT)
            total_loss = mainloss.clone()
            
            maxloss, helmholtz_loss, bandlimit_loss, reciprocity_loss, kk_loss, freq_smooth_loss = 0.,0.,0.,0.,0.,0.

            if lambda_max > 0:
                maxloss = maxloss_4d(decoded, GT)
                total_loss += lambda_max * maxloss

            if (pinnepoch >= 0 and epochnow > pinnepoch) or pinnepoch < 0 :
                c = 299792458.0
                freqs_hz = original_freqs_ghz * 1e9
                k0 = (2 * pi * freqs_hz / c).view(-1, 1, 1, 1).to(device)

                if lambda_helmholtz > 0:
                    helmholtz_loss = helmholtz_loss_4d(decoded, k0)
                    total_loss += lambda_helmholtz * helmholtz_loss

                if lambda_bandlimit > 0:
                    bandlimit_loss = bandlimit_loss_4d(decoded, k0)
                    total_loss += lambda_bandlimit * bandlimit_loss

                if lambda_reciprocity > 0:
                    reciprocity_loss = reciprocity_loss_fn(self, decoded, vertices, faces, face_edges, original_in_em, device)
                    total_loss += lambda_reciprocity * reciprocity_loss
                
            helm_metric, band_metric, reciprocity_metric, kk_metric, freq_smooth_metric = 0.,0.,0.,0.,0.
            with torch.no_grad():
                psnr_list = psnr(decoded, GT)
                ssim_list = myssim(decoded, GT)
                mse_list = batch_mse(decoded, GT)
                mean_psnr = psnr_list.mean()
                mean_ssim = ssim_list.mean()
                minus = decoded - GT
                mse = ((minus) ** 2).mean()
                nmse = mse / torch.var(GT)
                rmse = torch.sqrt(mse)
                l1 = (decoded-GT).abs().mean()
                percentage_error = (minus / (GT + 1e-4)).abs().mean() * 100
                
                # PINN物理指标
                c = 299792458.0
                freqs_hz = original_freqs_ghz * 1e9
                k0_metric = (2 * pi * freqs_hz / c).to(device) # .item() for single value
                helm_metric = helmholtz_consistency(decoded, k0_metric)
                band_metric = bandlimit_energy_ratio(decoded, k0_metric)
                # kk_metric = kramers_kronig_consistency(...)
                # freq_smooth_metric = frequency_smoothness(...)


            # 封装所有损失和指标以便返回和记录
            metrics = {
                'psnr': mean_psnr,
                'ssim': mean_ssim,
                'mse': mse,
                'psnrlist': psnr_list,
                'ssimlist': ssim_list,
                'mselist': mse_list,

                'nmse': nmse,
                'rmse': rmse,
                'l1': l1,
                'percentage_error': percentage_error,
                'pinn_helmholtz': helm_metric,
                'pinn_bandlimit': band_metric,
                'pinn_reciprocity': reciprocity_metric,
                'pinn_kramers_kronig': kk_metric, 
                'pinn_frequency_smoothness': freq_smooth_metric, 

                'total_loss': total_loss,
                'main_loss': mainloss,
                'max_loss': maxloss,
                'helmholtz_loss': helmholtz_loss,
                'bandlimit_loss': bandlimit_loss,
                'reciprocity_loss': reciprocity_loss,
                'kramers_kronig_loss': kk_loss,
                'freq_smooth_loss': freq_smooth_loss,
            }

            return decoded, metrics
