"""
测试 ONNX 导出脚本
用于验证模型是否能成功导出为 opset_version=12
"""
import torch
import onnx
import os
from nets import Net
from configs import Config

def test_export(project_name, checkpoint_name):
    """
    测试指定项目的模型导出
    
    Args:
        project_name: 项目名称
        checkpoint_name: checkpoint 文件名（例如：checkpoint_test_3_23_6000.tar）
    """
    print(f"\n{'='*60}")
    print(f"测试项目: {project_name}")
    print(f"Checkpoint: {checkpoint_name}")
    print(f"{'='*60}\n")
    
    # 加载配置
    config = Config(project_name)
    conf = config.load_config()
    
    # 构建模型
    print("加载模型...")
    net = Net(conf)
    
    # 加载 checkpoint
    project_path = os.path.join("projects", project_name)
    checkpoint_path = os.path.join(project_path, "checkpoints", checkpoint_name)
    
    if not os.path.exists(checkpoint_path):
        print(f"错误: Checkpoint 文件不存在: {checkpoint_path}")
        return False
    
    param, state_dict, optimizer = Net.load_checkpoint(checkpoint_path, torch.device('cpu'))
    net.load_state_dict(state_dict)
    net = net.eval().cpu()
    
    # 准备导出
    print("准备导出参数...")
    dummy_input = net.get_random_tensor()
    input_names = ["input1"]
    output_names = ["output"]
    dynamic_ax = {'input1': {3: 'image_width'}, "output": {1: 'seq'}}
    
    # 输出路径
    output_path = os.path.join(project_path, "models", f"{project_name}_test_opset12.onnx")
    
    # 导出模型
    print(f"导出模型到: {output_path}")
    print("使用 opset_version=12...\n")
    
    try:
        net.export_onnx(net, dummy_input, output_path, input_names, output_names, dynamic_ax)
        print("\n✓ 导出成功!")
        
        # 验证 ONNX 模型
        print("\n验证 ONNX 模型...")
        model = onnx.load(output_path)
        onnx.checker.check_model(model)
        
        # 获取 opset 版本
        opset_version = model.opset_import[0].version
        print(f"✓ ONNX 模型验证通过")
        print(f"✓ Opset 版本: {opset_version}")
        
        if opset_version == 12:
            print("✓ 成功导出 opset_version=12 模型!")
            return True
        else:
            print(f"⚠ 警告: 导出的模型 opset 版本是 {opset_version}，而不是 12")
            return False
            
    except Exception as e:
        print(f"\n✗ 导出失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 测试可用的项目
    test_cases = [
        ("test_3", "checkpoint_test_3_23_6000.tar"),
        # 如果需要测试其他项目，可以添加更多
        # ("test_2", "checkpoint_test_2_23_6000.tar"),
        # ("test_1", "checkpoint_test_1_23_6000.tar"),
    ]
    
    results = []
    for project, checkpoint in test_cases:
        result = test_export(project, checkpoint)
        results.append((project, result))
        print()
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    for project, result in results:
        status = "✓ 成功" if result else "✗ 失败"
        print(f"{project}: {status}")

