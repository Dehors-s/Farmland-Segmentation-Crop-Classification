import subprocess
import sys
import os

def launch_labelme(directory=None):
    """
    启动 labelme 标注工具。
    
    Args:
        directory (str, optional): 即使启动时自动打开的图像目录路径。
    """
    cmd = ["labelme"]
    
    if directory:
        if os.path.exists(directory):
            cmd.append(directory)
            print(f"将在目录中启动 labelme: {directory}")
        else:
            print(f"警告: 目录 '{directory}' 不存在。将启动默认 labelme。")
    
    print(f"正在执行命令: {' '.join(cmd)}")
    
    try:
        # 使用 subprocess 调用 labelme 命令行工具
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("\n错误: 未找到 'labelme' 命令。")
        print("请确保已安装 labelme。")
        print("安装命令推荐:")
        print("  pip install labelme")
        print("  或")
        print("  conda install -c conda-forge labelme")
    except KeyboardInterrupt:
        print("\nLabelme 已由用户终止。")
    except Exception as e:
        print(f"\n发生未知错误: {e}")

if __name__ == "__main__":
    # ================= 参数配置区域 =================
    # 手动设置默认打开的图像目录
    # 如果想打开其他目录，请修改这里的路径
    DEFAULT_IMAGE_DIR = r"dataset/images"  
    # ==============================================

    if len(sys.argv) > 1:
        # 如果提供了命令行参数，则将其作为目录路径
        target_dir = sys.argv[1]
    else:
        # 使用配置的默认目录
        if os.path.exists(DEFAULT_IMAGE_DIR):
            target_dir = DEFAULT_IMAGE_DIR
            print(f"没有提供参数，默认使用配置目录: {DEFAULT_IMAGE_DIR}")
        else:
            print(f"配置的默认目录 '{DEFAULT_IMAGE_DIR}' 不存在。")
            target_dir = None
    
    # 尝试启动 labelme
    launch_labelme(target_dir)
