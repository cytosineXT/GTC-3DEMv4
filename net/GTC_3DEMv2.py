# 全流程可微 embedding全用linear代替，lg频率范围修正，adaptation module nonlinear
from torch.nn import Module, ModuleList
import torch 
from torch import nn
from torch_geometric.nn.conv import SAGEConv
import torch.nn.functional as F
from functools import partial
from net.utils import derive_face_edges_from_faces, transform_to_log_coordinates, psnr, batch_mse
from net.utils import ssim as myssim
from pytorch_msssim import ms_ssim, ssim
from einops import rearrange, pack
from math import pi



def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

def jxtget_face_coords(vertices, face_indices):
    batch_size, num_faces, num_vertices_per_face = face_indices.shape
    reshaped_face_indices = face_indices.reshape(batch_size, -1).to(dtype=torch.int64) 
    face_coords = torch.gather(vertices, 1, reshaped_face_indices.unsqueeze(-1).expand(-1, -1, vertices.shape[-1])) # 使用索引张量获取具有坐标的面
    face_coords = face_coords.reshape(batch_size, num_faces, num_vertices_per_face, -1)# 还原形状
    return face_coords

def coords_interanglejxt2(x, y, eps=1e-5): #不要用爱恩斯坦求和 会变得不幸
    edge_vector = x - y
    normv = l2norm(edge_vector) #torch.Size([2, 20804, 3, 3])
    normdot = -(normv * torch.cat((normv[..., -1:], normv[..., :-1]), dim=3)).sum(dim=2) #应为torch.Size([2, 20804])
    normdot = torch.clamp(normdot, -1 + eps, 1 - eps)
    radians = torch.acos(normdot) #tensor([1.1088, 0.8747, 1.1511], device='cuda:0')
    angle = torch.rad2deg(radians) #tensor([63.5302, 50.1188, 65.9518], device='cuda:0')
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
    radians = normdot.acos() #tensor(1.4104, device='cuda:0')
    angle = torch.rad2deg(radians) #tensor(80.8117, device='cuda:0')
    return radians, angle

def get_derived_face_featuresjxt(
    face_coords,
    in_em,
    device
):
    shifted_face_coords = torch.cat((face_coords[:, :, -1:], face_coords[:, :, :-1]), dim = 2).to(device) #这是对face_coords循环移位，face_coords[:, :, -1:]取最后一个切片，
    angles, _  = coords_interanglejxt2(face_coords, shifted_face_coords) #得到了每个三角形face的三个内角，弧度形式的，如果要角度形式的要用_的(angle2)
    edge1, edge2, *_ = (face_coords - shifted_face_coords).unbind(dim = 2) #这里是坐标相减得到边
    normals = l2norm(torch.cross(edge1, edge2, dim = -1)) #然后用边叉乘得到法向量
    area = torch.cross(edge1, edge2, dim = -1).norm(dim = -1, keepdim = True) * 0.5 #两边矢量叉乘模/2得到面积
    incident_angle_vec = polar_to_cartesian2(in_em[1],in_em[2]) #得到入射方向的xyz矢量
    incident_angle_mtx = incident_angle_vec.unsqueeze(1).repeat(1, area.shape[1], 1).to(device) #得到入射方向的矢量矩阵dim3
    incident_freq_mtx = in_em[3].float().unsqueeze(1).unsqueeze(2).repeat(1, area.shape[1], 1).to(device) #得到入射波频率的矩阵dim1
    incident_mesh_anglehudu, _ = vector_anglejxt2(normals, incident_angle_mtx) #得到入射方向和每个mesh法向的夹角,是在0到180度的
    return dict(
        angles = angles,
        area = area,
        normals = normals,
        emnoangle = incident_mesh_anglehudu.unsqueeze(-1),
        emangle = incident_angle_mtx,
        emfreq = incident_freq_mtx
    )

def loss_fn(decoded, GT, loss_type='L1', gama=0.01, delta=0.5):
    maxloss = torch.mean(torch.abs(torch.amax(decoded, dim=(1, 2)) - torch.amax(GT, dim=(1, 2))))
    minus = decoded - GT
    mse = ((minus) ** 2).mean() #111 就是一样的
    nmse = mse / torch.var(GT)
    rmse = torch.sqrt(mse)
    l1 = (decoded-GT).abs().mean()
    # percentage_error = (minus / (GT + 1e-2)).abs().mean()
    percentage_error = (minus / (GT + 1e-8)).abs().mean()

    if loss_type == 'mse':
        # loss = F.mse_loss(decoded, GT)
        loss = mse
    elif loss_type == 'L1':
        # loss = F.l1_loss(pred, target)
        loss = l1
    elif loss_type == 'rmse':
        loss = rmse
    elif loss_type == 'nmse':
        loss = nmse
    elif loss_type == 'per':
        loss = percentage_error
    elif loss_type == 'ssim':
        loss = 1 - torch.stack([ssim(decoded[i].unsqueeze(0).unsqueeze(0), GT[i].unsqueeze(0).unsqueeze(0), data_range=max(decoded[i].max().item(), GT[i].max().item()), size_average=False) for i in range(decoded.size(0))]).squeeze().mean()
    elif loss_type == 'msssim':
        loss = 1 - torch.stack([ms_ssim(decoded[i].unsqueeze(0).unsqueeze(0), GT[i].unsqueeze(0).unsqueeze(0), data_range=max(decoded[i].max().item(), GT[i].max().item()), size_average=False) for i in range(decoded.size(0))]).squeeze().mean()

    elif loss_type == 'mse_l1':
        loss = delta * mse + (1 - delta) * l1
    elif loss_type == 'mse_nmse':
        loss = delta * mse + (1 - delta) * nmse
    elif loss_type == 'l1_nmse':
        loss = delta * l1 + (1 - delta) * nmse
    elif loss_type == 'mse_ssim':
        ssim_val = torch.stack([ssim(decoded[i].unsqueeze(0).unsqueeze(0), GT[i].unsqueeze(0).unsqueeze(0), data_range=max(decoded[i].max().item(), GT[i].max().item()), size_average=False) for i in range(decoded.size(0))]).squeeze().mean()
        loss = delta * mse + (1 - delta) * (1 - ssim_val)
    elif loss_type == 'mse_msssim':
        msssim_val = torch.stack([ms_ssim(decoded[i].unsqueeze(0).unsqueeze(0), GT[i].unsqueeze(0).unsqueeze(0), data_range=max(decoded[i].max().item(), GT[i].max().item()), size_average=False) for i in range(decoded.size(0))]).squeeze().mean()
        loss = delta * mse + (1 - delta) * (1 - msssim_val)
    elif loss_type == 'l1_ssim':
        ssim_val = torch.stack([ssim(decoded[i].unsqueeze(0).unsqueeze(0), GT[i].unsqueeze(0).unsqueeze(0), data_range=max(decoded[i].max().item(), GT[i].max().item()), size_average=False) for i in range(decoded.size(0))]).squeeze().mean()
        loss = delta * l1 + (1 - delta) * (1 - ssim_val)
    elif loss_type == 'l1_msssim':
        msssim_val = torch.stack([ms_ssim(decoded[i].unsqueeze(0).unsqueeze(0), GT[i].unsqueeze(0).unsqueeze(0), data_range=max(decoded[i].max().item(), GT[i].max().item()), size_average=False) for i in range(decoded.size(0))]).squeeze().mean()
        loss = delta * l1 + (1 - delta) * (1 - msssim_val)
    else:
        print(f"Unsupported loss type: {loss_type}, will use l1 loss")
        loss = l1

    total_loss = loss + gama * maxloss
    return total_loss

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

            # num_discrete_coors = 512,   #坐标离散量改成512
            # coor_continuous_range = (-1., 1.), #连续坐标范围
            # num_discrete_area = 128,    #面积离散量
            # num_discrete_normals = 128, #法线离散量
            # num_discrete_angle = 128,   #角度离散量
            # num_discrete_emnoangle = 128,   #法线、入射夹角离散量 jxt
            # num_discrete_emangle = 128,     #入射角离散量 jxt
            # num_discrete_emfreq = 512,
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
        self.upconv1 = nn.ConvTranspose2d(middim, int(middim/2), kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(int(middim/2))  # 添加的批量归一化层1
        self.conv1_1 = nn.Conv2d(int(middim/2), int(middim/2), kernel_size=3, stride=1, padding=1)  # 添加的卷积层1
        self.conv1_2 = nn.Conv2d(int(middim/2), int(middim/2), kernel_size=3, stride=1, padding=1)  # 添加的卷积层2
        self.bn1_1 = nn.BatchNorm2d(int(middim/2))  # 添加的批量归一化层1
        self.bn1_2 = nn.BatchNorm2d(int(middim/2))  # 添加的批量归一化层2
        self.upconv2 = nn.ConvTranspose2d(int(middim/2), int(middim/4), kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(int(middim/4))  # 添加的批量归一化层1
        self.conv2_1 = nn.Conv2d(int(middim/4), int(middim/4), kernel_size=3, stride=1, padding=1)  # 添加的卷积层1
        self.conv2_2 = nn.Conv2d(int(middim/4), int(middim/4), kernel_size=3, stride=1, padding=1)  # 添加的卷积层2
        self.bn2_1 = nn.BatchNorm2d(int(middim/4))  # 添加的批量归一化层1
        self.bn2_2 = nn.BatchNorm2d(int(middim/4))  # 添加的批量归一化层2
        self.upconv3 = nn.ConvTranspose2d(int(middim/4), int(middim/8), kernel_size=2, stride=2)
        # self.upconv3 = nn.ConvTranspose2d(int(middim/4), int(middim/8), kernel_size=2, stride=2, output_padding=1)
        self.bn3 = nn.BatchNorm2d(int(middim/8))
        self.conv3_1 = nn.Conv2d(int(middim/8), int(middim/8), kernel_size=3, stride=1, padding=1)  # 添加的卷积层1
        self.conv3_2 = nn.Conv2d(int(middim/8), int(middim/8), kernel_size=3, stride=1, padding=1)  # 添加的卷积层1
        self.bn3_1 = nn.BatchNorm2d(int(middim/8))  # 添加的批量归一化层1
        self.bn3_2 = nn.BatchNorm2d(int(middim/8))  # 添加的批量归一化层2
        self.conv1x1 = nn.Conv2d(int(middim/8), 1, kernel_size=1, stride=1, padding=0)

        
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
        derived_features = get_derived_face_featuresjxt(face_coords, in_em, device) #这一步用了2s

        
        angle_embed = self.angle_embed(derived_features['angles'])
        area_embed = self.area_embed(derived_features['area'])
        normal_embed = self.normal_embed(derived_features['normals'])

        emnoangle_embed = self.emnoangle_embed(derived_features['emnoangle']) #jxt
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

        x = self.conv1x1(x)
        # torch.Size([10, 1, 360, 720])

        return x.squeeze(dim=1)


    def forward(
        self,
        *,
        vertices,
        faces,
        face_edges=None,
        in_em,
        GT=None,
        logger=None,
        device='cpu',
        gama=0.001,
        loss_type='L1',
    ):
        if face_edges==None:
            face_edges = derive_face_edges_from_faces(faces, pad_id = self.pad_id)


        encoded, in_angle, in_freq = self.encode( 
            vertices = vertices, #顶点
            faces = faces, #面
            face_edges = face_edges, #图论边
            in_em = in_em,
        )

        decoded = self.decode(
            encoded, 
            in_angle,
            in_freq,
            device,
        )

        if GT == None:
            return decoded
        else:
            GT = GT[:,:-1,:] #361*720变360*720
            loss = loss_fn(decoded, GT, loss_type=loss_type, gama=gama)

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

            return loss, decoded, mean_psnr, psnr_list, mean_ssim, ssim_list, mse, nmse, rmse, l1, percentage_error, mse_list