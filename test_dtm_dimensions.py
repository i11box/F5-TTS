"""
测试 DTM 和 MatchaDTMHead 的维度流动
"""
import torch
import sys
sys.path.insert(0, 'src')

from f5_tts.model.dtm_heads.matcha_head import MatchaDTMHead

def test_matcha_dtm_head_3d():
    """测试 MatchaDTMHead 在 3D 输入下的维度"""
    print("=" * 60)
    print("测试 MatchaDTMHead 3D 输入")
    print("=" * 60)
    
    # 参数设置
    batch_size = 2
    seq_len = 100
    backbone_dim = 1024
    mel_dim = 100
    hidden_dim = 256
    
    # 创建模型
    head = MatchaDTMHead(
        backbone_dim=backbone_dim,
        mel_dim=mel_dim,
        hidden_dim=hidden_dim
    )
    
    # 创建输入（3D）
    h_t = torch.randn(batch_size, seq_len, backbone_dim)
    y_s = torch.randn(batch_size, seq_len, mel_dim)
    s = torch.rand(batch_size)
    
    print(f"\n输入维度：")
    print(f"  h_t: {h_t.shape} (期望: [{batch_size}, {seq_len}, {backbone_dim}])")
    print(f"  y_s: {y_s.shape} (期望: [{batch_size}, {seq_len}, {mel_dim}])")
    print(f"  s: {s.shape} (期望: [{batch_size}])")
    
    # Forward
    try:
        v_pred = head(h_t, y_s, s)
        print(f"\n✓ Forward 成功!")
        print(f"  v_pred: {v_pred.shape} (期望: [{batch_size}, {seq_len}, {mel_dim}])")
        
        # 验证输出维度
        assert v_pred.shape == (batch_size, seq_len, mel_dim), \
            f"输出维度不正确: 期望 ({batch_size}, {seq_len}, {mel_dim}), 实际 {v_pred.shape}"
        
        print(f"\n✓ 维度验证通过!")
        return True
        
    except Exception as e:
        print(f"\n✗ Forward 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_matcha_dtm_head_2d():
    """测试 MatchaDTMHead 在 2D 输入下的维度（向后兼容）"""
    print("\n" + "=" * 60)
    print("测试 MatchaDTMHead 2D 输入（向后兼容）")
    print("=" * 60)
    
    # 参数设置
    num_tokens = 200  # batch_size * seq_len
    backbone_dim = 1024
    mel_dim = 100
    hidden_dim = 256
    
    # 创建模型
    head = MatchaDTMHead(
        backbone_dim=backbone_dim,
        mel_dim=mel_dim,
        hidden_dim=hidden_dim
    )
    
    # 创建输入（2D）
    h_t = torch.randn(num_tokens, backbone_dim)
    y_s = torch.randn(num_tokens, mel_dim)
    s = torch.rand(num_tokens)
    
    print(f"\n输入维度：")
    print(f"  h_t: {h_t.shape} (期望: [{num_tokens}, {backbone_dim}])")
    print(f"  y_s: {y_s.shape} (期望: [{num_tokens}, {mel_dim}])")
    print(f"  s: {s.shape} (期望: [{num_tokens}])")
    
    # Forward
    try:
        v_pred = head(h_t, y_s, s)
        print(f"\n✗ 不应该支持 2D 输入（MatchaDTMHead 需要 3D）")
        print(f"  v_pred: {v_pred.shape}")
        return False
        
    except ValueError as e:
        if "requires 3D input" in str(e):
            print(f"\n✓ 正确拒绝了 2D 输入: {e}")
            return True
        else:
            print(f"\n✗ 错误类型不正确: {e}")
            return False
    except Exception as e:
        print(f"\n✗ 未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_scalar_time():
    """测试标量时间输入"""
    print("\n" + "=" * 60)
    print("测试标量时间输入")
    print("=" * 60)
    
    # 参数设置
    batch_size = 2
    seq_len = 50
    backbone_dim = 1024
    mel_dim = 100
    hidden_dim = 256
    
    # 创建模型
    head = MatchaDTMHead(
        backbone_dim=backbone_dim,
        mel_dim=mel_dim,
        hidden_dim=hidden_dim
    )
    
    # 创建输入
    h_t = torch.randn(batch_size, seq_len, backbone_dim)
    y_s = torch.randn(batch_size, seq_len, mel_dim)
    s = torch.tensor(0.5)  # 标量
    
    print(f"\n输入维度：")
    print(f"  h_t: {h_t.shape}")
    print(f"  y_s: {y_s.shape}")
    print(f"  s: {s.shape} (标量)")
    
    # Forward
    try:
        v_pred = head(h_t, y_s, s)
        print(f"\n✓ Forward 成功!")
        print(f"  v_pred: {v_pred.shape} (期望: [{batch_size}, {seq_len}, {mel_dim}])")
        
        # 验证输出维度
        assert v_pred.shape == (batch_size, seq_len, mel_dim), \
            f"输出维度不正确"
        
        print(f"\n✓ 标量时间处理正确!")
        return True
        
    except Exception as e:
        print(f"\n✗ Forward 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("开始测试 DTM 维度修复")
    print("=" * 60)
    
    results = []
    
    # 运行测试
    results.append(("3D 输入测试", test_matcha_dtm_head_3d()))
    results.append(("2D 输入测试（向后兼容）", test_matcha_dtm_head_2d()))
    results.append(("标量时间测试", test_scalar_time()))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    if all_passed:
        print(f"\n{'=' * 60}")
        print("所有测试通过! ✓")
        print("=" * 60)
    else:
        print(f"\n{'=' * 60}")
        print("部分测试失败! ✗")
        print("=" * 60)
        sys.exit(1)

