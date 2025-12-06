"""
测试 DTM Head 的维度正确性
"""
import torch
from src.f5_tts.model.dtm_heads.matcha_head import MatchaDTMHead

def test_matcha_dtm_head():
    """测试 MatchaDTMHead 的维度传递"""
    print("=" * 60)
    print("测试 MatchaDTMHead 维度传递")
    print("=" * 60)
    
    # 测试配置
    batch_size = 2
    seq_len = 100
    backbone_dim = 1024
    mel_dim = 100
    hidden_dim = 256
    
    # 创建模型
    head = MatchaDTMHead(
        backbone_dim=backbone_dim,
        mel_dim=mel_dim,
        hidden_dim=hidden_dim,
    )
    head.eval()
    
    print(f"\n模型配置:")
    print(f"  backbone_dim: {backbone_dim}")
    print(f"  mel_dim: {mel_dim}")
    print(f"  hidden_dim: {hidden_dim}")
    
    # 测试输入
    h_t = torch.randn(batch_size, seq_len, backbone_dim)
    Y_s = torch.randn(batch_size, seq_len, mel_dim)
    s = torch.rand(batch_size)
    
    print(f"\n输入维度:")
    print(f"  h_t: {h_t.shape} (应为 {(batch_size, seq_len, backbone_dim)})")
    print(f"  Y_s: {Y_s.shape} (应为 {(batch_size, seq_len, mel_dim)})")
    print(f"  s: {s.shape} (应为 {(batch_size,)})")
    
    # 前向传播
    with torch.no_grad():
        try:
            v_pred = head(h_t, Y_s, s)
            print(f"\n输出维度:")
            print(f"  v_pred: {v_pred.shape} (应为 {(batch_size, seq_len, mel_dim)})")
            
            # 验证输出维度
            expected_shape = (batch_size, seq_len, mel_dim)
            if v_pred.shape == expected_shape:
                print(f"\n✓ 维度验证通过！")
                return True
            else:
                print(f"\n✗ 维度验证失败！")
                print(f"  期望: {expected_shape}")
                print(f"  实际: {v_pred.shape}")
                return False
        except Exception as e:
            print(f"\n✗ 前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_different_hidden_dims():
    """测试不同 hidden_dim 配置"""
    print("\n" + "=" * 60)
    print("测试不同 hidden_dim 配置")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 50
    backbone_dim = 1024
    mel_dim = 100
    
    hidden_dims = [128, 256, 512, 1024]
    
    for hidden_dim in hidden_dims:
        print(f"\n测试 hidden_dim={hidden_dim}...")
        
        head = MatchaDTMHead(
            backbone_dim=backbone_dim,
            mel_dim=mel_dim,
            hidden_dim=hidden_dim,
        )
        head.eval()
        
        h_t = torch.randn(batch_size, seq_len, backbone_dim)
        Y_s = torch.randn(batch_size, seq_len, mel_dim)
        s = torch.rand(batch_size)
        
        with torch.no_grad():
            try:
                v_pred = head(h_t, Y_s, s)
                expected_shape = (batch_size, seq_len, mel_dim)
                if v_pred.shape == expected_shape:
                    print(f"  ✓ hidden_dim={hidden_dim} 通过")
                else:
                    print(f"  ✗ hidden_dim={hidden_dim} 失败: {v_pred.shape}")
                    return False
            except Exception as e:
                print(f"  ✗ hidden_dim={hidden_dim} 出错: {e}")
                return False
    
    print(f"\n✓ 所有配置测试通过！")
    return True


def test_different_seq_lens():
    """测试不同序列长度"""
    print("\n" + "=" * 60)
    print("测试不同序列长度")
    print("=" * 60)
    
    batch_size = 2
    backbone_dim = 1024
    mel_dim = 100
    hidden_dim = 256
    
    seq_lens = [32, 64, 128, 256]
    
    head = MatchaDTMHead(
        backbone_dim=backbone_dim,
        mel_dim=mel_dim,
        hidden_dim=hidden_dim,
    )
    head.eval()
    
    for seq_len in seq_lens:
        print(f"\n测试 seq_len={seq_len}...")
        
        h_t = torch.randn(batch_size, seq_len, backbone_dim)
        Y_s = torch.randn(batch_size, seq_len, mel_dim)
        s = torch.rand(batch_size)
        
        with torch.no_grad():
            try:
                v_pred = head(h_t, Y_s, s)
                expected_shape = (batch_size, seq_len, mel_dim)
                if v_pred.shape == expected_shape:
                    print(f"  ✓ seq_len={seq_len} 通过")
                else:
                    print(f"  ✗ seq_len={seq_len} 失败: {v_pred.shape}")
                    return False
            except Exception as e:
                print(f"  ✗ seq_len={seq_len} 出错: {e}")
                import traceback
                traceback.print_exc()
                return False
    
    print(f"\n✓ 所有序列长度测试通过！")
    return True


if __name__ == "__main__":
    print("\n开始测试 DTM Head 维度...")
    
    success = True
    
    # 运行所有测试
    success &= test_matcha_dtm_head()
    success &= test_different_hidden_dims()
    success &= test_different_seq_lens()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ 所有测试通过！")
    else:
        print("✗ 部分测试失败")
    print("=" * 60)

