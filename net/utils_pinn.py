# PINN_metrics.py
import torch
import torch.fft
import torch.nn.functional as F
import torch.nn as nn

# --- 辅助函数 ---

def _laplacian(field_component):
    """ 计算单个2D场分量的拉普拉斯算子 """
    # 定义拉普拉斯算子核
    laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 
                                    dtype=field_component.dtype, 
                                    device=field_component.device).view(1, 1, 3, 3)
    # 增加批次和通道维度以符合conv2d要求
    field_component = field_component.unsqueeze(1) # [B, 1, H, W]
    # 使用卷积计算拉普拉斯
    laplacian_out = F.conv2d(field_component, laplacian_kernel, padding=1)
    return laplacian_out.squeeze(1) # [B, H, W]

class WeightedFieldLoss(nn.Module):
    """
    加权的电场损失函数，强制模型关注高能量/高绝对值的区域。
    根据每个通道的绝对值进行独立加权。
    """
    def __init__(self, lambda_main=10.0, loss_type='L1'):
        """
        初始化加权损失函数。

        Args:
            alpha (float): 权重增强因子。alpha越大，模型越关注高绝对值的区域。
            l1_weight (float): L1损失部分的权重。
            mse_weight (float): MSE损失部分的权重。(默认关闭MSE部分)
        """
        super().__init__()
        self.lam_main = lambda_main
        self.loss_type = loss_type
        if self.lam_main > 0:
            print(f"初始化独立加权损失函数: lam_main={self.lam_main}, 损失类型={self.loss_type}")
        else:
            print(f"初始化标准损失函数 (lam_main=0): 损失类型={self.loss_type}")

    def forward(self, decoded, gt):
        """
        计算加权损失。

        Args:
            decoded (torch.Tensor): 模型的输出, shape [B, 4, H, W]
            gt (torch.Tensor): 真实值 (Ground Truth), shape [B, 4, H, W]

        Returns:
            torch.Tensor: 一个标量损失值。
        """
        if self.alpha <= 0:
            # 如果alpha为0或负，则退化为标准loss
            l1_loss = F.l1_loss(decoded, gt) if self.loss_type == 'l1' else 0.
            mse_loss = F.mse_loss(decoded, gt) if self.loss_type == 'mse' else 0.
            return l1_loss + mse_loss

        with torch.no_grad():
            # --- 根据GT每个通道的绝对值创建权重矩阵 ---
            # weights shape: [B, 4, H, W]
            weights = 1.0 + self.lam_main * torch.abs(gt)
        # --- 计算加权Loss ---
        mainloss = 0.
        if self.loss_type == 'L1':
            l1_diff = torch.abs(decoded - gt)
            weighted_l1_loss = torch.mean(weights * l1_diff)
            mainloss += weighted_l1_loss
        if self.loss_type == 'mse':
            mse_diff_sq = (decoded - gt)**2
            weighted_mse_loss = torch.mean(weights * mse_diff_sq)
            mainloss += weighted_mse_loss
        else:
            l1_diff = torch.abs(decoded - gt)
            weighted_l1_loss = torch.mean(weights * l1_diff)
            mainloss += weighted_l1_loss
        return mainloss



# --- 指标计算函数 (修正版) ---
def helmholtz_consistency(E_field_4d, k0):
    """
    亥姆霍兹波动性检验。
    计算 ||∇²E + k₀²E|| / ||k₀²E||，值越小表示物理一致性越好。
    
    Args:
        E_field_4d (torch.Tensor): 四维复电场 [B, 4, H, W]
        k0 (torch.Tensor): 自由空间波数，形状应为 [B]

    Returns:
        torch.Tensor: 一个标量，表示平均的亥姆霍兹方程残差。
    """
    with torch.no_grad():
        # 分离 E_theta 和 E_phi 的复数场
        E_theta = E_field_4d[:, 0, :, :] + 1j * E_field_4d[:, 1, :, :]
        E_phi = E_field_4d[:, 2, :, :] + 1j * E_field_4d[:, 3, :, :]

        # 计算复数拉普拉斯
        lap_E_theta = _laplacian(E_theta.real) + 1j * _laplacian(E_theta.imag)
        lap_E_phi = _laplacian(E_phi.real) + 1j * _laplacian(E_phi.imag)

        # --- 核心修正 ---
        # k0 的原始形状是 [B]，为了能和 [B, H, W] 的场进行广播乘法，
        # 需要将其形状变为 [B, 1, 1]。
        if k0.dim() == 1:
            k0 = k0.view(-1, 1, 1)

        # 现在可以安全地进行计算
        # k0**2 的形状是 [B, 1, 1]
        # E_theta 的形状是 [B, 360, 720]
        # 广播操作会自动将k0**2扩展到 [B, 360, 720]
        residual_theta = lap_E_theta + k0**2 * E_theta
        residual_phi = lap_E_phi + k0**2 * E_phi

        # --- 范数归一化解释 ---
        # 我们计算残差的能量(范数)，并用原始场的能量(范数)来归一化它。
        # 这就像计算一个信噪比或相对误差，使得结果与电场的绝对强度无关，
        # 从而可以在不同样本间进行公平比较。
        residual_norm = torch.norm(residual_theta) + torch.norm(residual_phi)
        field_norm = torch.norm(k0**2 * E_theta) + torch.norm(k0**2 * E_phi)

        # 避免除以零的保护措施
        consistency = residual_norm / (field_norm + 1e-8)
        return consistency


def bandlimit_energy_ratio(E_field_4d, k0, radius_factor=1.5): #这里怎么也是用的比例啊 好像比例确实合理一点？
    """
    带限能量检验。
    计算在k空间中，半径 k0*radius_factor 范围外的能量占总能量的比例。
    一个好的物理场，其能量应主要集中在k0附近。值越小越好。

    Args:
        E_field_4d (torch.Tensor): 四维复电场 [B, 4, H, W]
        k0 (float): 自由空间波数 (2π/λ)
        radius_factor (float): 定义带内区域的半径因子。

    Returns:
        torch.Tensor: 一个标量，表示带外能量的平均比例。
    """
    with torch.no_grad():
        B, _, H, W = E_field_4d.shape
    if k0.dim() == 1:
        k0 = k0.view(-1, 1, 1)
    
    # 对每个通道进行2D FFT
        fft_result = torch.fft.fftshift(torch.fft.fft2(E_field_4d, dim=(-2, -1)))
        
        # 创建k空间坐标网格
        kx = torch.linspace(-W//2, W//2 - 1, W, device=E_field_4d.device)
        ky = torch.linspace(-H//2, H//2 - 1, H, device=E_field_4d.device)
        k_grid_y, k_grid_x = torch.meshgrid(ky, kx, indexing='ij')
        k_radius = torch.sqrt(k_grid_x**2 + k_grid_y**2)

        # 创建一个mask，标记出带外区域
        band_out_mask = (k_radius > k0 * radius_factor).float().unsqueeze(1)
        
        # 计算能量（幅度的平方）
        energy = (fft_result.real**2 + fft_result.imag**2)
        
        # 计算带外能量和总能量
        out_of_band_energy = torch.sum(energy * band_out_mask)
        total_energy = torch.sum(energy)
        
        ratio = out_of_band_energy / (total_energy + 1e-8)
        return ratio

def frequency_smoothness(E_field_f1, E_field_f2, delta_f): #这个感觉不够好，才两个 能有多大意义 有问题
    """
    频域平滑性检验。
    计算场对频率的差分（导数的近似）。值越小表示随频率变化越平滑。

    Args:
        E_field_f1 (torch.Tensor): 在频率 f1 处的场 [B, 4, H, W]
        E_field_f2 (torch.Tensor): 在频率 f2 处的场 [B, 4, H, W]
        delta_f (float): 频率差 f2 - f1

    Returns:
        torch.Tensor: 一个标量，表示归一化的场对频率的导数范数。
    """
    # 计算差分
    field_diff = E_field_f2 - E_field_f1
    # 计算导数的近似
    derivative = field_diff / (delta_f + 1e-8)
    # 计算范数进行评估
    derivative_norm = torch.norm(derivative)
    field_norm = (torch.norm(E_field_f1) + torch.norm(E_field_f2)) / 2.0
    
    smoothness = derivative_norm / (field_norm + 1e-8)
    return smoothness


def kramers_kronig_consistency(E_fields_vs_freq): #这个和频率没关系吧，
    """
    Kramers-Kronig 关系检验。
    通过希尔伯特变换，从实部推导虚部，并与网络输出的虚部进行比较。
    值越小表示越符合因果律。

    Args:
        E_fields_vs_freq (torch.Tensor): 按频率排序的场 [Num_Freqs, 4, H, W]
    
    Returns:
        torch.Tensor: 一个标量，表示平均的KK残差。
    """
    num_freqs = E_fields_vs_freq.shape[0]
    if num_freqs < 2:
        return torch.tensor(0.0) # 无法计算

    # --- 对 E_theta 进行KK检验 ---
    re_theta = E_fields_vs_freq[:, 0, :, :] # [N_f, H, W]
    im_theta_pred = E_fields_vs_freq[:, 1, :, :]

    # 通过FFT实现希尔伯特变换
    re_theta_fft = torch.fft.fft(re_theta, dim=0)
    freqs = torch.fft.fftfreq(num_freqs, device=E_fields_vs_freq.device)
    h = -1j * torch.sign(freqs).view(-1, 1, 1) # 希尔伯特变换的频域表示
    im_theta_physical = torch.fft.ifft(re_theta_fft * h, dim=0).imag
    
    kk_residual_theta = torch.norm(im_theta_pred - im_theta_physical) / (torch.norm(im_theta_pred) + 1e-8)

    # --- 对 E_phi 进行KK检验 ---
    re_phi = E_fields_vs_freq[:, 2, :, :]
    im_phi_pred = E_fields_vs_freq[:, 3, :, :]
    
    re_phi_fft = torch.fft.fft(re_phi, dim=0)
    im_phi_physical = torch.fft.ifft(re_phi_fft * h, dim=0).imag

    kk_residual_phi = torch.norm(im_phi_pred - im_phi_physical) / (torch.norm(im_phi_pred) + 1e-8)
    
    return (kk_residual_theta + kk_residual_phi) / 2.0


def maxloss_4d(E_field_4d, gt):
    """
    计算最大损失，用于评估模型输出与真实值之间的最大差异。不光要最大值还要最小值
    
    Args:
        E_field_4d (torch.Tensor): 网络的输出 [B, 4, H, W]
        gt (torch.Tensor): 真实值 (Ground Truth) [B, 4, H, W]

    Returns:
        torch.Tensor: 一个标量，表示最大损失。
    """
    max_loss=0.
    # 计算每个通道的最大值的绝对误差，然后求和
    for i in range(gt.shape[1]):
        max_loss += torch.abs(torch.max(E_field_4d[:, i, :, :]) - torch.max(gt[:, i, :, :]))
        max_loss += torch.abs(torch.min(E_field_4d[:, i, :, :]) - torch.min(gt[:, i, :, :]))
    return max_loss/gt.shape[1] 

def helmholtz_loss_4d(E_field_4d, k0):
    """
    亥姆霍兹方程Loss，用于反向传播。
    
    Args:
        E_field_4d (torch.Tensor): 网络的输出 [B, 4, H, W]
        k0 (float): 自由空间波数 (2π/λ)

    Returns:
        torch.Tensor: 一个可用于反向传播的标量loss。
    """
    # 分离 E_theta 和 E_phi 的实部虚部
    E_theta_re, E_theta_im = E_field_4d[:, 0, :, :], E_field_4d[:, 1, :, :]
    E_phi_re, E_phi_im = E_field_4d[:, 2, :, :], E_field_4d[:, 3, :, :]

    # 计算拉普拉斯
    lap_E_theta_re, lap_E_theta_im = _laplacian(E_theta_re), _laplacian(E_theta_im)
    lap_E_phi_re, lap_E_phi_im = _laplacian(E_phi_re), _laplacian(E_phi_im)

    # 计算亥姆霍兹方程的残差 (∇²E + k₀²E)
    residual_theta_re = lap_E_theta_re + k0**2 * E_theta_re
    residual_theta_im = lap_E_theta_im + k0**2 * E_theta_im
    residual_phi_re = lap_E_phi_re + k0**2 * E_phi_re
    residual_phi_im = lap_E_phi_im + k0**2 * E_phi_im

    # 使用MSE计算loss
    loss = F.mse_loss(residual_theta_re, torch.zeros_like(residual_theta_re)) + \
           F.mse_loss(residual_theta_im, torch.zeros_like(residual_theta_im)) + \
           F.mse_loss(residual_phi_re, torch.zeros_like(residual_phi_re)) + \
           F.mse_loss(residual_phi_im, torch.zeros_like(residual_phi_im))
           
    return loss / 4.0

def bandlimit_loss_4d(E_field_4d, k0, radius_factor=1.5):
    """
    带限能量Loss，惩罚在k空间中远离k0的能量分量。
    
    Args:
        E_field_4d (torch.Tensor): 网络的输出 [B, 4, H, W]
        k0 (float): 自由空间波数 (2π/λ)
        radius_factor (float): 定义带内区域的半径因子。

    Returns:
        torch.Tensor: 一个可用于反向传播的标量loss。
    """
    B, _, H, W = E_field_4d.shape
    
    # 对每个通道进行2D FFT
    fft_result = torch.fft.fftshift(torch.fft.fft2(E_field_4d, dim=(-2, -1)))
    
    # 创建k空间坐标网格
    kx = torch.linspace(-W//2, W//2 - 1, W, device=E_field_4d.device)
    ky = torch.linspace(-H//2, H//2 - 1, H, device=E_field_4d.device)
    k_grid_y, k_grid_x = torch.meshgrid(ky, kx, indexing='ij')
    k_radius = torch.sqrt(k_grid_x**2 + k_grid_y**2)

    # 创建一个mask，标记出带外区域
    band_out_mask = (k_radius > k0 * radius_factor).float().unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
    
    # 计算带外能量（幅度的平方）
    out_of_band_energy = (fft_result.real**2 + fft_result.imag**2) * band_out_mask
    
    # loss是带外能量的均值
    loss = torch.mean(out_of_band_energy)
    return loss


def reciprocity_loss_fn(model, decoded_rcs, vertices, faces, face_edges, original_in_em, device):
    """
    (新增) 计算实时互易性损失。
    通过一次额外的网络推理来验证互易定理。
    """
    batch_size, h, w = decoded_rcs.shape
    plane_name, theta_in, phi_in, freq_in = original_in_em

    # 1. 随机选择一个出射角 p_out，作为新的入射角 p_in'
    rand_h = torch.randint(0, h, (batch_size,), device=device)
    rand_w = torch.randint(0, w, (batch_size,), device=device)
    
    # 将像素坐标转换为角度
    theta_out = rand_h * (180.0 / h)
    phi_out = rand_w * (360.0 / w)
    
    # 2. 从第一次的预测结果中，获取p_out对应的值 value_1
    value_1 = decoded_rcs[torch.arange(batch_size), rand_h, rand_w]

    # 3. 将 p_out 作为新的输入，进行第二次前向传播
    reciprocal_in_em = [plane_name, theta_out, phi_out, freq_in]
    
    # 这里的推理也需要梯度，所以正常调用
    with torch.enable_grad():
        encoded_rec, in_angle_rec, in_freq_rec = model.encode(
            vertices=vertices, faces=faces, face_edges=face_edges, in_em=reciprocal_in_em
        )
        decoded_reciprocal = model.decode(
            encoded_rec, in_angle_rec, in_freq_rec, device
        )

    # 4. 从第二次的预测结果中，获取对应原始入射角 p_in 的值 value_2
    h_in_idx = (theta_in / 180.0 * h).long().clamp(0, h-1)
    w_in_idx = (phi_in / 360.0 * w).long().clamp(0, w-1)
    value_2 = decoded_reciprocal[torch.arange(batch_size), h_in_idx, w_in_idx]
    
    # 5. 计算互易性损失
    loss = F.l1_loss(value_1, value_2) #mse=0.2511 l1=0.4264
    return loss

