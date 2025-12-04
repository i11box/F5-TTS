import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_mel_spectrogram(tensor_3d: torch.Tensor, title: str = "Mel Spectrogram"):
    """
    接收一个三维张量，检查其第一维是否为1，压缩后将其绘制为梅尔频谱图。

    Args:
        tensor_3d (torch.Tensor): 形状为 (1, H, W) 或 (H, W) 的张量。
        title (str): 频谱图的标题。
    
    Returns:
        None: 直接显示或保存频谱图。
    """
    
    # 1. 检查张量维度
    if tensor_3d.dim() not in [2, 3]:
        print(f"❌ 错误：输入张量的维度必须是 2 或 3，但实际是 {tensor_3d.dim()}。函数退出。")
        return

    # 2. 检查并压缩第一维
    if tensor_3d.dim() == 3:
        if tensor_3d.shape[0] != 1:
            print(f"❌ 错误：三维张量的第一维必须是 1 进行压缩，但实际是 {tensor_3d.shape[0]}。函数退出。")
            return
        
        # 压缩第一维 (1, H, W) -> (H, W)
        tensor_2d = tensor_3d.squeeze(0)
        print(f"✅ 张量形状已从 {tuple(tensor_3d.shape)} 压缩为 {tuple(tensor_2d.shape)}。")
    elif tensor_3d.dim() == 2:
        tensor_2d = tensor_3d
        print(f"✅ 输入张量已经是二维的，形状为 {tuple(tensor_2d.shape)}。")

    # 3. 转换为 NumPy 数组并进行可视化
    try:
        # 确保数据在 CPU 上并转换为 NumPy 
        spectrogram = tensor_2d.cpu().detach().numpy()
        
        # 绘制频谱图
        plt.figure(figsize=(10, 5))
        
        # 使用 pcolormesh 或 imshow 绘制，这里使用 imshow 
        # 'origin='lower'' 让低频在底部
        # 'aspect='auto'' 调整宽高比
        img = plt.imshow(
            spectrogram, 
            aspect='auto', 
            origin='lower', 
            interpolation='none'
        )
        
        # 添加颜色条
        plt.colorbar(img, format="%+2.0f dB")
        
        # 设置坐标轴标签和标题
        plt.title(title)
        plt.xlabel("Time Frames")
        plt.ylabel("Mel Filter Banks")
        
        # 显示图表
        plt.savefig('mel_spectrogram_output.png', bbox_inches='tight', dpi=300)
        
    except Exception as e:
        print(f"❌ 绘制过程中发生错误: {e}")