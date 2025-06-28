# GTC-3DEMv3.3.py
# 版本 3.3: 加入pinnepoch和lossfn优化，但是有bug。。。

from torch.nn import Module, ModuleList
import torch 
from torch import nn
from torch_geometric.nn.conv import SAGEConv
import torch.nn.functional as F
import numpy as np
# from functools import partial
from net.utils import derive_face_edges_from_faces, transform_to_log_coordinates, psnr, batch_mse
from net.utils import ssim as myssim
from net.utils_pinn import bandlimit_energy_ratio, helmholtz_consistency, helmholtz_loss_4d, bandlimit_loss_4d, WeightedFieldLoss, reciprocity_loss_fn, maxloss_4d
# from pytorch_msssim import ms_ssim, ssim
from einops import rearrange, pack
from math import pi

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
    incident_angle_vec = polar_to_cartesian2(in_em[1],in_em[2]).float() #这里成float64了。。
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
            dim_coor_embed = 64,        #坐标embedding维度
            dim_area_embed = 16,        #面积embedding维度
            dim_normal_embed = 64,      #法线矢量embedding维度
            dim_angle_embed = 16,       #角度值embedding维度
            dim_emnoangle_embed = 16,       #法线与入射夹角值embedding维度
            dim_emangle_embed = 64,         #入射矢量embedding维度
            dim_emfreq_embed = 16,          #频率embedding维度
            encoder_dims_through_depth = (64, 128, 256, 256, 576),    
            ):
        super().__init__()

        #---Conditioning
        self.condfreqlayers = ModuleList([ #长度不定没关系，我可以变成固定的维度让他在长度上广播！这样的加法是加在每一根token上，而不是在特征上
            nn.Linear(1, 64),
            nn.Linear(1, 128), 
            nn.Linear(1, 256), 
            nn.Linear(1, 256),]
        )
        self.condanglelayers = ModuleList([
            nn.Linear(2, 64),
            nn.Linear(2, 128),
            nn.Linear(2, 256),
            nn.Linear(2, 256),]
        )
        self.incident_angle_linear1 = nn.Linear(2, 2250) #这样的加法是展平了加在每一个dim上，不是每一个特征上
        self.emfreq_embed1 = nn.Linear(1, 2250)
        self.incident_angle_linear2 = nn.Linear(2, 4050)
        self.emfreq_embed2 = nn.Linear(1, 4050)

        self.incident_angle_linear3 = nn.Linear(2, 90*180)
        self.emfreq_embed3 = nn.Linear(1, 90*180)
        self.incident_angle_linear4 = nn.Linear(2, 180*360)
        self.emfreq_embed4 = nn.Linear(1, 180*360)
        self.incident_angle_linear5 = nn.Linear(2, 360*720)
        self.emfreq_embed5 = nn.Linear(1, 360*720)

        #---Encoder
        self.angle_embed = nn.Linear(3, 3*dim_angle_embed)
        self.area_embed = nn.Linear(1, dim_area_embed)
        self.normal_embed = nn.Linear(3, 3*dim_normal_embed)
        self.emnoangle_embed = nn.Linear(1, dim_emnoangle_embed) #jxt
        self.emangle_embed = nn.Linear(3, 3*dim_emangle_embed) #jxt
        self.emfreq_embed = nn.Linear(1, dim_emfreq_embed) #jxt
        self.coor_embed = nn.Linear(9, 9*dim_coor_embed) 


        init_dim = dim_coor_embed * (3 * 3) + dim_angle_embed * 3 + dim_normal_embed * 3 + dim_area_embed + dim_emangle_embed * 3 + dim_emnoangle_embed + dim_emfreq_embed

        sageconv_kwargs = dict(   #SAGEconv参数
            normalize = True,
            project = True
        )
        # sageconv_kwargs = {**sageconv_kwargs }
        init_encoder_dim, *encoder_dims_through_depth = encoder_dims_through_depth #64, 128, 256, 256, 576
        curr_dim = init_encoder_dim

        self.init_sage_conv = SAGEConv(init_dim, init_encoder_dim, **sageconv_kwargs)
        self.init_encoder_act_and_norm = nn.Sequential(
            nn.SiLU(),
            nn.LayerNorm(init_encoder_dim)
        ) 
        self.encoders = ModuleList([])
        self.encoder_act_and_norm = ModuleList([])  # 新增的激活和归一化层列表

        for dim_layer in encoder_dims_through_depth:
            sage_conv = SAGEConv(
                curr_dim,
                dim_layer,
                **sageconv_kwargs
            )
            self.encoders.append(sage_conv)
            self.encoder_act_and_norm.append(nn.Sequential(
                nn.SiLU(),
                nn.LayerNorm(dim_layer)
            ))
            curr_dim = dim_layer

        self.encoder_attn_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=curr_dim, nhead=8, dropout=attn_dropout)
            for _ in range(attn_encoder_depth)
        ])

        #---Adaptation Module
        self.conv1d1 = nn.Conv1d(576, middim, kernel_size=10, stride=10, dilation=1 ,padding=0)
        self.fc1d1 = nn.Linear(2250, 45*90)


        #---Decoder
        # Decoder3
        self.upconv1 = nn.ConvTranspose2d(middim, int(middim/2), kernel_size=2, stride=2, groups=4)
        self.bn1 = nn.BatchNorm2d(int(middim/2))  # 添加的批量归一化层1
        self.conv1_1 = nn.Conv2d(int(middim/2), int(middim/2), kernel_size=3, stride=1, padding=1, groups=4)  # 添加的卷积层1
        self.conv1_2 = nn.Conv2d(int(middim/2), int(middim/2), kernel_size=3, stride=1, padding=1, groups=4)  # 添加的卷积层2
        self.bn1_1 = nn.BatchNorm2d(int(middim/2))  # 添加的批量归一化层1
        self.bn1_2 = nn.BatchNorm2d(int(middim/2))  # 添加的批量归一化层2
        self.upconv2 = nn.ConvTranspose2d(int(middim/2), int(middim/4), kernel_size=2, stride=2, groups=4)
        self.bn2 = nn.BatchNorm2d(int(middim/4))  # 添加的批量归一化层1
        self.conv2_1 = nn.Conv2d(int(middim/4), int(middim/4), kernel_size=3, stride=1, padding=1, groups=4)  # 添加的卷积层1
        self.conv2_2 = nn.Conv2d(int(middim/4), int(middim/4), kernel_size=3, stride=1, padding=1, groups=4)  # 添加的卷积层2
        self.bn2_1 = nn.BatchNorm2d(int(middim/4))  # 添加的批量归一化层1
        self.bn2_2 = nn.BatchNorm2d(int(middim/4))  # 添加的批量归一化层2
        self.upconv3 = nn.ConvTranspose2d(int(middim/4), int(middim/8), kernel_size=2, stride=2, groups=4)
        # self.upconv3 = nn.ConvTranspose2d(int(middim/4), int(middim/8), kernel_size=2, stride=2, output_padding=1)
        self.bn3 = nn.BatchNorm2d(int(middim/8))
        self.conv3_1 = nn.Conv2d(int(middim/8), int(middim/8), kernel_size=3, stride=1, padding=1, groups=4)  # 添加的卷积层1
        self.conv3_2 = nn.Conv2d(int(middim/8), int(middim/8), kernel_size=3, stride=1, padding=1, groups=4)  # 添加的卷积层1
        self.bn3_1 = nn.BatchNorm2d(int(middim/8))  # 添加的批量归一化层1
        self.bn3_2 = nn.BatchNorm2d(int(middim/8))  # 添加的批量归一化层2
        # self.conv1x1 = nn.Conv2d(int(middim/8), 1, kernel_size=1, stride=1, padding=0)
        # self.conv1x1 = nn.Conv2d(int(middim/8), 4, kernel_size=1, stride=1, padding=0) #在这里运行复电场预测 四维就行
        self.head_re_etheta = nn.Conv2d(int(middim/8), 1, kernel_size=1, stride=1, padding=0)
        self.head_im_etheta = nn.Conv2d(int(middim/8), 1, kernel_size=1, stride=1, padding=0)
        self.head_re_ephi = nn.Conv2d(int(middim/8), 1, kernel_size=1, stride=1, padding=0)
        self.head_im_ephi = nn.Conv2d(int(middim/8), 1, kernel_size=1, stride=1, padding=0)

        # self.upconv1 = nn.ConvTranspose2d(middim, int(middim/2), kernel_size=2, stride=2)
        # self.bn1 = nn.BatchNorm2d(int(middim/2))  # 添加的批量归一化层1
        # self.conv1_1 = nn.Conv2d(int(middim/2), int(middim/2), kernel_size=3, stride=1, padding=1)  # 添加的卷积层1
        # self.conv1_2 = nn.Conv2d(int(middim/2), int(middim/2), kernel_size=3, stride=1, padding=1)  # 添加的卷积层2
        # self.bn1_1 = nn.BatchNorm2d(int(middim/2))  # 添加的批量归一化层1
        # self.bn1_2 = nn.BatchNorm2d(int(middim/2))  # 添加的批量归一化层2
        # self.upconv2 = nn.ConvTranspose2d(int(middim/2), int(middim/4), kernel_size=2, stride=2)
        # self.bn2 = nn.BatchNorm2d(int(middim/4))  # 添加的批量归一化层1
        # self.conv2_1 = nn.Conv2d(int(middim/4), int(middim/4), kernel_size=3, stride=1, padding=1)  # 添加的卷积层1
        # self.conv2_2 = nn.Conv2d(int(middim/4), int(middim/4), kernel_size=3, stride=1, padding=1)  # 添加的卷积层2
        # self.bn2_1 = nn.BatchNorm2d(int(middim/4))  # 添加的批量归一化层1
        # self.bn2_2 = nn.BatchNorm2d(int(middim/4))  # 添加的批量归一化层2
        # self.upconv3 = nn.ConvTranspose2d(int(middim/4), int(middim/8), kernel_size=2, stride=2)
        # # self.upconv3 = nn.ConvTranspose2d(int(middim/4), int(middim/8), kernel_size=2, stride=2, output_padding=1)
        # self.bn3 = nn.BatchNorm2d(int(middim/8))
        # self.conv3_1 = nn.Conv2d(int(middim/8), int(middim/8), kernel_size=3, stride=1, padding=1)  # 添加的卷积层1
        # self.conv3_2 = nn.Conv2d(int(middim/8), int(middim/8), kernel_size=3, stride=1, padding=1)  # 添加的卷积层1
        # self.bn3_1 = nn.BatchNorm2d(int(middim/8))  # 添加的批量归一化层1
        # self.bn3_2 = nn.BatchNorm2d(int(middim/8))  # 添加的批量归一化层2
        # # self.conv1x1 = nn.Conv2d(int(middim/8), 1, kernel_size=1, stride=1, padding=0)
        # # self.conv1x1 = nn.Conv2d(int(middim/8), 4, kernel_size=1, stride=1, padding=0) #在这里运行复电场预测 四维就行
        # self.head_re_etheta = nn.Conv2d(int(middim/8), 1, kernel_size=1, stride=1, padding=0)
        # self.head_im_etheta = nn.Conv2d(int(middim/8), 1, kernel_size=1, stride=1, padding=0)
        # self.head_re_ephi = nn.Conv2d(int(middim/8), 1, kernel_size=1, stride=1, padding=0)
        # self.head_im_ephi = nn.Conv2d(int(middim/8), 1, kernel_size=1, stride=1, padding=0)

        
    def encode(
        self,
        *,
        vertices,
        faces,
        face_edges,
        in_em
        ):
        device =vertices.device 
        face_coords = jxtget_face_coords(vertices, faces) 
        in_em[3]=transform_to_log_coordinates(in_em[3]) #频率转换为对数坐标 加在encoder里！
        in_em = [in_em[0], in_em[1].float(), in_em[2].float(), in_em[3].float()] #将角度和频率转换为float32
        derived_features = get_derived_face_featuresjxt(face_coords, in_em, device) #这一步用了2s

        
        angle_embed = self.angle_embed(derived_features['angles'])
        area_embed = self.area_embed(derived_features['area'])
        normal_embed = self.normal_embed(derived_features['normals'])#这里都是float32

        emnoangle_embed = self.emnoangle_embed(derived_features['emnoangle']) #jxt 这个怎么变成float64了
        emangle_embed = self.emangle_embed(derived_features['emangle']) #jxt torch.Size([2, 20804, 3])
        emfreq_embed = self.emfreq_embed(derived_features['emfreq'])
        face_coords = rearrange(face_coords, 'b nf nv c -> b nf (nv c)') # 9 or 12 coordinates per face #重新排布
        face_coor_embed = self.coor_embed(face_coords) #在这里把face做成embedding


        face_embed, _ = pack([face_coor_embed, angle_embed, area_embed, normal_embed, emnoangle_embed, emangle_embed, emfreq_embed], 'b nf *') 


        face_edges = face_edges.reshape(2, -1).to(device)
        orig_face_embed_shape = face_embed.shape[:2]#本来就记下了分批了
        face_embed = face_embed.reshape(-1, face_embed.shape[-1])#torch.Size([129960, 192])
        face_embed = self.init_sage_conv(face_embed, face_edges)#torch.Size([129960, 64])
        face_embed = self.init_encoder_act_and_norm(face_embed)#torch.Size([129960, 64])
        face_embed = face_embed.reshape(orig_face_embed_shape[0], orig_face_embed_shape[1], -1)#回复分批

        in_angle = torch.stack([in_em[1]/180, in_em[2]/360]).t().float().unsqueeze(1).to(device)
        in_freq = in_em[3].float().unsqueeze(1).unsqueeze(1).to(device)

        for i, (conv, act_norm) in enumerate(zip(self.encoders, self.encoder_act_and_norm)):
            condfreq = self.condfreqlayers[i](in_freq)
            condangle = self.condanglelayers[i](in_angle)
            face_embed = face_embed + condangle + condfreq  # 自带广播操作
            face_embed = face_embed.reshape(-1, face_embed.shape[-1])  # 再次合并批次
            face_embed = conv(face_embed, face_edges)  # 图卷积操作
            face_embed = act_norm(face_embed)  # 应用激活函数和LayerNorm
            face_embed = face_embed.reshape(orig_face_embed_shape[0], orig_face_embed_shape[1], -1)  # 重新分割批次
          
        for attn_layer in self.encoder_attn_blocks:
            face_embed = face_embed.permute(1, 0, 2)  # (nf, b, d) torch.Size([10, 12996, 576])
            face_embed = attn_layer(face_embed) + face_embed  # (nf, b, d) torch.Size([12996, 10, 576]) 残差骚操作
            # face_embed = attn_layer(face_embed)  # (nf, b, d) torch.Size([12996, 10, 576]) 无残差
            face_embed = face_embed.permute(1, 0, 2)  # (b, nf, d)

        return face_embed, in_angle, in_freq
    
    def decode( 
        self,
        x, 
        in_angle,
        in_freq,
        device,
    ):
        
        condangle1 = self.incident_angle_linear1(in_angle)
        condangle2 = self.incident_angle_linear2(in_angle)
        condangle3 = self.incident_angle_linear3(in_angle).reshape(in_angle.shape[0],-1,90,180)#这样的加法是展平了加在每一个dim上，不是每一根特征上
        condangle4 = self.incident_angle_linear4(in_angle).reshape(in_angle.shape[0],-1,180,360)
        condangle5 = self.incident_angle_linear5(in_angle).reshape(in_angle.shape[0],-1,360,720)

        condfreq1 = self.emfreq_embed1(in_freq)
        condfreq2 = self.emfreq_embed2(in_freq)
        condfreq3 = self.emfreq_embed3(in_freq).reshape(in_angle.shape[0],-1,90,180)
        condfreq4 = self.emfreq_embed4(in_freq).reshape(in_angle.shape[0],-1,180,360)
        condfreq5 = self.emfreq_embed5(in_freq).reshape(in_angle.shape[0],-1,360,720)

        pad_size = 22500 - x.size(1) #在这里完成了padding。。。前面变长无所谓。。。
        x = F.pad(x, (0, 0, 0, pad_size)) 
        x = x.view(x.size(0), -1, 22500) 
        
        # # ------------------------1D Conv+FC-----------------------------
        # torch.Size([10, 784, 22500])
        x = self.conv1d1(x) 
        x = F.relu(x) #非线性变换
        # torch.Size([10, 64(middim), 2250])
        # x = x + condangle1 
        # x = x + condfreq1


        x = self.fc1d1(x)
        x = x.reshape(x.size(0), -1, 45*90) 
        # torch.Size([10, 64, 4050])
        # x = x + condangle2 
        # x = x + condfreq2
        x = x.reshape(x.size(0), -1, 45, 90) 
        # torch.Size([10, 64, 45, 90])

        # ------------------------2D upConv------------------------------
        x = self.upconv1(x)
        # torch.Size([10, 32, 90, 180])
        x = self.bn1(x)
        x = F.relu(x)
        # x = x + condangle3
        # x = x + condfreq3
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = F.relu(x)
        # x = x + condangle3
        # x = x + condfreq3
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = F.relu(x)
        # x = x + condangle3
        # x = x + condfreq3

        x = self.upconv2(x) 
        #torch.Size([10, 16, 180, 360])
        x = self.bn2(x)
        x = F.relu(x)
        # x = x + condangle4
        # x = x + condfreq4
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = F.relu(x)
        # x = x + condangle4
        # x = x + condfreq4
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = F.relu(x)
        # x = x + condangle4
        # x = x + condfreq4

        x = self.upconv3(x) 
        #torch.Size([10, 8, 360, 720])
        x = self.bn3(x)
        x = F.relu(x)
        # x = x + condangle5
        # x = x + condfreq5
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = F.relu(x)
        # x = x + condangle5
        # x = x + condfreq5
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = F.relu(x)
        # x = x + condangle5
        # x = x + condfreq5

        # x = self.conv1x1(x)
        # torch.Size([10, 1, 360, 720])
        out_re_etheta = self.head_re_etheta(x)
        out_im_etheta = self.head_im_etheta(x)
        out_re_ephi = self.head_re_ephi(x)
        out_im_ephi = self.head_im_ephi(x)

        return torch.cat([out_re_etheta, out_im_etheta, out_re_ephi, out_im_ephi], dim=1)
        # return x
        # return x.permute(0, 2, 3, 1)  # (b, h, w, c) -> (b, c, h, w)

    def forward(self, *, vertices, faces, face_edges=None, in_em, GT=None, logger=None, device='cpu', loss_type='L1', **kwargs):
        # =========================================================================
        #               Part 5: forward() 方法重大升级
        # =========================================================================
        original_in_em = [in_em[0], in_em[1].clone(), in_em[2].clone(), in_em[3].clone()]
        original_freqs_ghz = in_em[3].clone()


        # 1. 初始化损失函数和权重
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
        
        # 3. 如果是训练或验证，计算所有loss和指标
        else:
            if GT.shape[2:4] == (361, 720):
                GT = GT[:, :, :-1, :]
            
            # # --- 核心Loss计算 ---
            # 1. mainloss, weighted 
            mainloss = loss_fn(decoded, GT)
            total_loss = mainloss.clone()
            # 1. mainloss
            # if loss_type == 'L1':
            #     mainloss = F.l1_loss(decoded, GT)
            # elif loss_type == 'mse':
            #     mainloss = F.mse_loss(decoded, GT)
            # else: # 默认为L1
            #     mainloss = F.l1_loss(decoded, GT)
            # total_loss = mainloss.clone()

            # # 3.2 PINN 物理损失 (仅在pinnepoch之后激活)
            maxloss, helmholtz_loss, bandlimit_loss, reciprocity_loss, kk_loss, freq_smooth_loss = 0.,0.,0.,0.,0.,0.

            # 2. maxloss
            if lambda_max > 0:
                maxloss = maxloss_4d(decoded, GT)
                total_loss += lambda_max * maxloss

            if (pinnepoch >= 0 and epochnow > pinnepoch) or pinnepoch < 0 :
                # 计算k0
                c = 299792458.0
                freqs_hz = original_freqs_ghz * 1e9
                k0 = (2 * pi * freqs_hz / c).view(-1, 1, 1, 1).to(device)

                # 计算亥姆霍兹损失
                if lambda_helmholtz > 0:
                    helmholtz_loss = helmholtz_loss_4d(decoded, k0)
                    total_loss += lambda_helmholtz * helmholtz_loss

                # 计算带限损失
                if lambda_bandlimit > 0:
                    bandlimit_loss = bandlimit_loss_4d(decoded, k0)
                    total_loss += lambda_bandlimit * bandlimit_loss

                # 5. reciprocity_loss
                if lambda_reciprocity > 0:
                    reciprocity_loss = reciprocity_loss_fn(self, decoded, vertices, faces, face_edges, original_in_em, device)
                    total_loss += lambda_reciprocity * reciprocity_loss
                
                # 6.计算Kramers-Kronig一致性损失（如果需要）
                # if lambda_kramers_kronig > 0:
                #     kk_loss = kramers_kronig_consistency(decoded)
                

            # --- 性能指标计算 ---
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
            # return total_loss, decoded, mean_psnr, psnr_list, mean_ssim, ssim_list, mse, nmse, rmse, l1, percentage_error, mse_list, metrics
        
    # def forward(self, *, vertices, faces, face_edges=None, in_em, GT=None, logger=None, device='cpu', loss_type='L1', **kwargs):
    #     # 更新物理损失的权重
        
    #     epochnow = kwargs.get('epochnow', 0)
    #     pinnepoch = kwargs.get('pinnepoch', 0)
    #     lambda_max = kwargs.get('lambda_max', 0.0001)
    #     lambda_helmholtz = kwargs.get('lambda_helmholtz', 0)
    #     lambda_bandlimit = kwargs.get('lambda_bandlimit', 0)
    #     lambda_reciprocity = kwargs.get('lambda_reciprocity', 0) # (新增)
        
        
    #     # 保存原始输入，以供互易性损失计算使用
    #     original_in_em = [in_em[0], in_em[1].clone(), in_em[2].clone(), in_em[3].clone()]
        
    #     if face_edges is None:
    #         face_edges = derive_face_edges_from_faces(faces, pad_id=-1)

    #     encoded, in_angle, in_freq = self.encode(
    #         vertices=vertices, faces=faces, face_edges=face_edges, in_em=in_em)

    #     decoded = self.decode(encoded, in_angle, in_freq, device)

    #     if GT is None:
    #         return decoded
    #     else:
    #         if GT.shape[2:4] == (361, 720):
    #             GT = GT[:, :, :-1, :]
            
    #         mainloss = 0.
    #         maxloss = 0.
    #         helmholtz_loss = 0.
    #         bandlimit_loss = 0.
    #         reciprocity_loss = 0.

    #         #-----------------------------------------------loss计算----------------------------------------------
    #         # 1. mainloss
    #         if loss_type == 'L1':
    #             mainloss = F.l1_loss(decoded, GT)
    #         elif loss_type == 'mse':
    #             mainloss = F.mse_loss(decoded, GT)
    #         else: # 默认为L1
    #             mainloss = F.l1_loss(decoded, GT)
    #         total_loss = mainloss.clone()

    #         # 2. maxloss
    #         if lambda_max > 0:
    #             maxloss = torch.mean(torch.abs(torch.amax(decoded, dim=(1, 2)) - torch.amax(GT, dim=(1, 2))))
    #             total_loss += lambda_max * maxloss

    #         if (pinnepoch >= 0 and epochnow > pinnepoch) or pinnepoch < 0 :#若有pinnepoch>=0，则仅当当前轮数大于pinnepoch时计算pinnloss；如果pinnloss=-1则从头开始
    #             # 3. helmholtz_loss
    #             if lambda_helmholtz > 0:
    #                 helmholtz_loss = calculate_helmholtz_loss(decoded, original_in_em[3], device)
    #                 total_loss += lambda_helmholtz * helmholtz_loss

    #             # 4. bandlimit_loss
    #             if lambda_bandlimit > 0:
    #                 bandlimit_loss = calculate_bandlimit_loss(decoded, original_in_em[3], device)
    #                 total_loss += lambda_bandlimit * bandlimit_loss

    #             # 5. reciprocity_loss
    #             if lambda_reciprocity > 0:
    #                 reciprocity_loss = reciprocity_loss_fn(self, decoded, vertices, faces, face_edges, original_in_em, device)
    #                 total_loss += lambda_reciprocity * reciprocity_loss
    #         #-----------------------------------------------loss计算----------------------------------------------
            

    #         with torch.no_grad():
    #             psnr_list = psnr(decoded, GT)
    #             ssim_list = myssim(decoded, GT)
    #             mse_list = batch_mse(decoded, GT)
    #             mean_psnr = psnr_list.mean()
    #             mean_ssim = ssim_list.mean()
    #             minus = decoded - GT
    #             mse = ((minus) ** 2).mean()
    #             nmse = mse / torch.var(GT)
    #             rmse = torch.sqrt(mse)
    #             l1 = (decoded-GT).abs().mean()
    #             percentage_error = (minus / (GT + 1e-4)).abs().mean() * 100

    #         metrics = {
    #             'total_loss': total_loss,
    #             'main_loss': mainloss,
    #             'max_loss': maxloss,
    #             'helmholtz_loss': helmholtz_loss,
    #             'bandlimit_loss': bandlimit_loss,
    #             'reciprocity_loss': reciprocity_loss,
    #         }

    #         return total_loss, decoded, mean_psnr, psnr_list, mean_ssim, ssim_list, mse, nmse, rmse, l1, percentage_error, mse_list, metrics
