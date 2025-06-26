# v4
2025年6月24日09:02:05 计划实现复电场预测，复数网络，重点是PINN loss和新的物理指标。
PINN loss:
helmholtz loss
fft bandlimit loss
reciprocity loss
total E frequency consistency partial diffrencial loss
real-imaginary hilbert transform loss
本征模感应电场？

2025年6月24日21:17:37 4维复电场预测，网络输出从1改为4维，已跑通，差作图，计划做四张图+一张abs(E)这样子
2025年6月26日15:16:15 完成作图，确定AmPhase / RealImage的数据处理，且转化为总场的计算公式分别为：
 - AmPhase为6维     Abs(E )[V/m] Abs(Theta)[V/m] Phase(Theta)[deg.] Abs(Phi )[V/m] Phase(Phi )[deg.] Ax.Ratio[ ]
 ```
delta_phi_rad = (E_phase_theta - E_phase_phi) * (torch.pi / 180.0)
E_abs_theta_sq = E_abs_theta**2 #预计算振幅的平方
E_abs_phi_sq = E_abs_phi**2
term1 = E_abs_theta_sq + E_abs_phi_sq #应用计算椭圆长半轴(A_maj)的精确公式: A_maj^2 = 0.5 * [|Eθ|²+|Eφ|² + sqrt( (|Eθ|²-|Eφ|²)² + 4|Eθ|²|Eφ|²cos²(Δφ) )]
term2_inner_sqrt = torch.sqrt((E_abs_theta_sq - E_abs_phi_sq)**2 + 4 * E_abs_theta_sq * E_abs_phi_sq * (torch.cos(delta_phi_rad))**2)
E_total_abs_sq = 0.5 * (term1 + term2_inner_sqrt)
E_total_abs_compute = torch.sqrt(E_total_abs_sq) # 开方得到最终的总场强（椭圆长半轴）
 ```
 - RealImage为4维   Real(E_theta)[V/m] Imag(E_theta)[V/m] Real(E_phi)[V/m] Imag(E_phi)[V/m]
 ```
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
 ```

## v4.0
2025年6月26日16:31:46 成功跑通 开始训练原生GTC-3DEMv4.0 无复数网络 无PINNloss 无任何先验