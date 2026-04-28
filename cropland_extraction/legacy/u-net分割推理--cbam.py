# u-net分割推理.py
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import argparse
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import warnings

# ==================== 来自训练脚本的模型定义 ====================
import torch.nn.functional as F  # 确保导入了 F


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x_channel = self.channel_attention(x) * x
        x_spatial = self.spatial_attention(x_channel) * x_channel
        return x_spatial


class CBAMDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_cbam=True):
        super(CBAMDecoderBlock, self).__init__()
        self.use_cbam = use_cbam
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        if use_cbam:
            self.cbam = CBAM(out_channels + skip_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        if self.use_cbam:
            x = self.cbam(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class CBAMUNet(nn.Module):
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet',
                 in_channels=3, classes=1, decoder_channels=[256, 128, 64, 32],
                 use_cbam=True):
        super(CBAMUNet, self).__init__()
        self.use_cbam = use_cbam
        if encoder_name.startswith('resnet'):
            import torchvision.models as models
            if encoder_name == 'resnet34':
                base_model = models.resnet34(weights=None)  # 推理时不需要下载 ImageNet 权重
            elif encoder_name == 'resnet50':
                base_model = models.resnet50(weights=None)
            else:
                base_model = models.resnet18(weights=None)

            self.encoder1 = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool)
            self.encoder2 = base_model.layer1
            self.encoder3 = base_model.layer2
            self.encoder4 = base_model.layer3
            self.encoder5 = base_model.layer4

            if encoder_name in ['resnet18', 'resnet34']:
                encoder_channels = [64, 64, 128, 256, 512]
            else:
                encoder_channels = [64, 256, 512, 1024, 2048]

        self.bridge = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], decoder_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[0], decoder_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True)
        )

        self.decoder4 = CBAMDecoderBlock(decoder_channels[0], encoder_channels[-2], decoder_channels[1], use_cbam)
        self.decoder3 = CBAMDecoderBlock(decoder_channels[1], encoder_channels[-3], decoder_channels[2], use_cbam)
        self.decoder2 = CBAMDecoderBlock(decoder_channels[2], encoder_channels[-4], decoder_channels[3], use_cbam)
        self.decoder1 = CBAMDecoderBlock(decoder_channels[3], encoder_channels[-5], decoder_channels[3], use_cbam)

        self.final_up = nn.Sequential(
            nn.Conv2d(decoder_channels[3], decoder_channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(decoder_channels[3], decoder_channels[3] // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(decoder_channels[3] // 2),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(decoder_channels[3] // 2, classes, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        b = self.bridge(e5)
        d4 = self.decoder4(b, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)
        up = self.final_up(d1)
        if up.shape[2:] != x.shape[2:]:
            up = F.interpolate(up, size=x.shape[2:], mode='bilinear', align_corners=True)
        output = self.final_conv(up)
        return output


warnings.filterwarnings('ignore')


class FarmlandPredictor:
    """
    耕地分割预测器
    使用训练好的模型进行预测
    """

    def __init__(self, config_path, model_path):
        """
        初始化预测器

        参数:
        - config_path: 配置文件路径
        - model_path: 模型权重文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 初始化模型
        self.model = self._load_model(model_path)

        # 数据预处理变换
        self.transform = self._get_transforms()

    def _load_model(self, model_path):
        """加载训练好的模型"""
        print(f"加载模型: {model_path}")

        # 修改这里：使用 CBAMUNet 而不是 smp.Unet
        # 还要注意检查 config 中是否有 use_cbam 参数，没有则默认为 True
        use_cbam = self.config.get('use_cbam', True)

        model = CBAMUNet(
            encoder_name=self.config['encoder_name'],
            encoder_weights=None,
            in_channels=3,
            classes=1,
            use_cbam=use_cbam
        ).to(self.device)

        # 加载训练好的权重
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # 设置模型为评估模式
        model.eval()

        print(f"模型加载完成，最佳IoU: {checkpoint.get('best_iou', 'N/A')}")
        return model

    def _get_transforms(self):
        """获取数据预处理变换（与验证集相同）"""
        img_size = self.config.get('img_size', 512)

        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),
        ])

    def read_image_with_chinese_path(self, image_path):
        """读取中文路径的图像"""
        # 方法1: 使用PIL读取中文路径图像
        try:
            pil_image = Image.open(str(image_path))
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return image
        except Exception as e:
            # 方法2: 使用OpenCV的imdecode
            try:
                with open(str(image_path), 'rb') as f:
                    image_data = np.frombuffer(f.read(), dtype=np.uint8)
                    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                return image
            except Exception as e2:
                raise ValueError(f"无法读取图像 {image_path}: {e2}")

    def preprocess_image(self, image_path):
        """预处理单张图像"""
        # 读取图像（支持中文路径）
        image = self.read_image_with_chinese_path(image_path)

        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 保存原始尺寸
        original_size = image.shape[:2]  # (H, W)

        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 应用变换
        transformed = self.transform(image=image_rgb)
        image_tensor = transformed['image']

        # 添加batch维度
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        return image_tensor, original_size, image_rgb

    def predict_single(self, image_path, threshold=0.5):
        """
        预测单张图像

        参数:
        - image_path: 图像路径
        - threshold: 阈值，大于阈值的像素被认为是耕地

        返回:
        - mask: 二值分割掩膜 (0: 背景, 1: 耕地)
        - probability: 概率图 (0-1之间)
        - original_image: 原始图像
        """
        # 预处理图像
        image_tensor, original_size, original_image = self.preprocess_image(image_path)

        # 预测
        with torch.no_grad():
            output = self.model(image_tensor)
            probability_map = torch.sigmoid(output).cpu().numpy()[0, 0]  # 获取概率图

        # 将概率图调整回原始尺寸
        probability_map_resized = cv2.resize(
            probability_map,
            (original_size[1], original_size[0]),  # (W, H)
            interpolation=cv2.INTER_LINEAR
        )

        # 应用阈值得到二值掩膜
        binary_mask = (probability_map_resized > threshold).astype(np.uint8) * 255

        return binary_mask, probability_map_resized, original_image

    def predict_batch(self, image_paths, threshold=0.5):
        """批量预测多张图像"""
        results = []
        for image_path in tqdm(image_paths, desc="批量预测"):
            try:
                binary_mask, probability_map, original_image = self.predict_single(
                    image_path, threshold
                )
                results.append({
                    'path': image_path,
                    'binary_mask': binary_mask,
                    'probability_map': probability_map,
                    'original_image': original_image
                })
            except Exception as e:
                print(f"预测图像 {image_path} 时出错: {e}")
                results.append({
                    'path': image_path,
                    'error': str(e)
                })

        return results

    def save_image_with_chinese_path(self, image, output_path):
        """保存图像到中文路径"""
        # 使用PIL保存到中文路径
        try:
            # 将OpenCV图像转换为PIL图像
            if len(image.shape) == 3 and image.shape[2] == 3:
                # BGR转RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
            else:
                # 灰度图像
                pil_image = Image.fromarray(image)

            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 使用PIL保存
            pil_image.save(str(output_path))
        except Exception as e:
            print(f"保存图像到 {output_path} 时出错: {e}")
            # 尝试使用OpenCV的imencode
            try:
                ext = output_path.suffix.lower()
                if ext == '.jpg' or ext == '.jpeg':
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                elif ext == '.png':
                    encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
                else:
                    encode_param = []

                success, encoded_image = cv2.imencode(ext, image, encode_param)
                if success:
                    with open(str(output_path), 'wb') as f:
                        f.write(encoded_image)
                else:
                    raise ValueError("图像编码失败")
            except Exception as e2:
                raise ValueError(f"两种方法都无法保存图像: {e2}")

    def predict_directory(self, input_dir, output_dir, threshold=0.5,
                          save_binary=True, save_probability=False,
                          save_overlay=False, file_extensions=['.png', '.jpg', '.jpeg', '.tif', '.tiff']):
        """
        预测整个目录中的图像

        参数:
        - input_dir: 输入图像目录
        - output_dir: 输出目录
        - threshold: 阈值
        - save_binary: 是否保存二值掩膜
        - save_probability: 是否保存概率图
        - save_overlay: 是否保存叠加可视化图
        - file_extensions: 支持的文件扩展名
        """
        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 获取所有图像文件（解决中文路径问题）
        input_dir = Path(input_dir)
        image_paths = []

        # 使用os.walk来处理中文路径
        for root, dirs, files in os.walk(str(input_dir)):
            for file in files:
                file_path = Path(root) / file
                # 检查文件扩展名
                if any(file_path.suffix.lower() == ext.lower() for ext in file_extensions):
                    image_paths.append(file_path)

        print(f"找到 {len(image_paths)} 张图像")

        if len(image_paths) == 0:
            print(f"在目录 {input_dir} 中未找到图像文件")
            return

        # 批量预测
        results = self.predict_batch(image_paths, threshold)

        # 保存结果
        for result in tqdm(results, desc="保存结果"):
            if 'error' in result:
                continue

            image_path = result['path']
            stem = image_path.stem

            # 保存二值掩膜
            if save_binary:
                binary_output_path = output_dir / f'{stem}_binary.png'
                self.save_image_with_chinese_path(result['binary_mask'], binary_output_path)

            # 保存概率图（归一化到0-255）
            if save_probability:
                prob_map = result['probability_map']
                prob_map_normalized = (prob_map * 255).astype(np.uint8)
                prob_output_path = output_dir / f'{stem}_probability.png'
                self.save_image_with_chinese_path(prob_map_normalized, prob_output_path)

            # 保存叠加可视化图
            if save_overlay:
                overlay = self.create_overlay_image(result['original_image'], result['binary_mask'])
                overlay_output_path = output_dir / f'{stem}_overlay.png'
                self.save_image_with_chinese_path(overlay, overlay_output_path)

        print(f"预测完成，结果保存在: {output_dir}")

    def create_overlay_image(self, original_image, binary_mask):
        """创建原始图像和分割结果的叠加图"""
        # 创建彩色掩膜（红色表示耕地）
        colored_mask = np.zeros_like(original_image)
        colored_mask[binary_mask == 255] = [255, 0, 0]  # 红色

        # 创建叠加图像
        overlay = cv2.addWeighted(original_image, 0.7, colored_mask, 0.3, 0)
        return cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    def visualize_prediction(self, image_path, threshold=0.5, save_path=None):
        """
        可视化单张图像的预测结果

        参数:
        - image_path: 图像路径
        - threshold: 阈值
        - save_path: 保存路径（如果为None则显示图像）
        """
        # 预测
        binary_mask, probability_map, original_image = self.predict_single(image_path, threshold)

        # 创建可视化图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 原始图像
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('原始图像')
        axes[0, 0].axis('off')

        # 概率图
        prob_im = axes[0, 1].imshow(probability_map, cmap='hot')
        axes[0, 1].set_title('概率图 (耕地概率)')
        axes[0, 1].axis('off')
        plt.colorbar(prob_im, ax=axes[0, 1])

        # 二值掩膜
        axes[0, 2].imshow(binary_mask, cmap='gray')
        axes[0, 2].set_title('二值分割结果')
        axes[0, 2].axis('off')

        # 叠加图
        overlay = self.create_overlay_image(original_image, binary_mask)
        axes[1, 0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('叠加显示')
        axes[1, 0].axis('off')

        # 耕地面积统计
        total_pixels = binary_mask.size
        farmland_pixels = np.sum(binary_mask == 255)
        farmland_ratio = farmland_pixels / total_pixels * 100

        axes[1, 1].bar(['背景', '耕地'], [total_pixels - farmland_pixels, farmland_pixels])
        axes[1, 1].set_title(f'像素统计 (耕地比例: {farmland_ratio:.2f}%)')
        axes[1, 1].set_ylabel('像素数量')

        # 概率分布直方图
        axes[1, 2].hist(probability_map.flatten(), bins=50, range=(0, 1))
        axes[1, 2].set_title('概率分布直方图')
        axes[1, 2].set_xlabel('概率')
        axes[1, 2].set_ylabel('频数')
        axes[1, 2].axvline(x=threshold, color='r', linestyle='--', label=f'阈值={threshold}')
        axes[1, 2].legend()

        plt.suptitle(f'耕地分割结果 - {Path(image_path).name}', fontsize=16)
        plt.tight_layout()

        if save_path:
            # 确保目录存在
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
            print(f"可视化结果已保存: {save_path}")
        else:
            plt.show()

        plt.close()


def create_config_from_checkpoint(checkpoint_path, output_config_path):
    """从检查点文件创建配置文件"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']

    # 确保输出目录存在
    output_config_path = Path(output_config_path)
    output_config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(str(output_config_path), 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"配置文件已保存: {output_config_path}")
    return config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='耕地分割预测脚本')
    parser.add_argument('--model', type=str, required=True, help='模型权重文件路径 (.pth)')
    parser.add_argument('--input', type=str, required=True, help='输入图像或目录路径')
    parser.add_argument('--output', type=str, default='./predictions', help='输出目录路径')
    parser.add_argument('--threshold', type=float, default=0.5, help='分割阈值 (0-1)')
    parser.add_argument('--config', type=str, help='配置文件路径 (.yaml)，如不提供则从模型提取')
    parser.add_argument('--visualize', action='store_true', help='可视化预测结果')
    parser.add_argument('--save_all', action='store_true', help='保存所有输出类型')

    args = parser.parse_args()

    # 检查输入路径
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入路径不存在: {input_path}")
        return

    # 处理配置文件
    if args.config and Path(args.config).exists():
        config_path = args.config
    else:
        # 从模型文件提取配置
        config_path = Path(args.model).with_suffix('.yaml')
        if not config_path.exists():
            print(f"从模型文件提取配置: {config_path}")
            try:
                create_config_from_checkpoint(args.model, config_path)
            except Exception as e:
                print(f"提取配置失败: {e}")
                # 使用默认配置
                config_path = None
                print("使用默认配置")

    if config_path is None:
        # 使用默认配置
        config = {
            'encoder_name': 'resnet34',
            'img_size': 512
        }
        # 临时创建配置文件
        import tempfile
        import json
        temp_dir = tempfile.gettempdir()
        config_path = Path(temp_dir) / 'temp_config.yaml'
        with open(str(config_path), 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        print(f"使用临时配置文件: {config_path}")

    # 创建预测器
    predictor = FarmlandPredictor(str(config_path), args.model)

    # 判断输入是文件还是目录
    if input_path.is_file():
        # 单张图像预测
        if args.visualize:
            # 可视化单张图像
            save_path = Path(args.output) / f'{input_path.stem}_visualization.png'
            predictor.visualize_prediction(str(input_path), args.threshold, str(save_path))
        else:
            # 只保存预测结果
            binary_mask, probability_map, original_image = predictor.predict_single(
                str(input_path), args.threshold
            )

            # 创建输出目录
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)

            # 保存结果
            stem = input_path.stem
            binary_output_path = output_dir / f'{stem}_binary.png'
            predictor.save_image_with_chinese_path(binary_mask, binary_output_path)

            prob_map_normalized = (probability_map * 255).astype(np.uint8)
            prob_output_path = output_dir / f'{stem}_probability.png'
            predictor.save_image_with_chinese_path(prob_map_normalized, prob_output_path)

            overlay = predictor.create_overlay_image(original_image, binary_mask)
            overlay_output_path = output_dir / f'{stem}_overlay.png'
            predictor.save_image_with_chinese_path(overlay, overlay_output_path)

            print(f"预测完成，结果保存在: {output_dir}")
    else:
        # 目录批量预测
        predictor.predict_directory(
            str(input_path),
            args.output,
            threshold=args.threshold,
            save_binary=True,
            save_probability=args.save_all,
            save_overlay=args.save_all or args.visualize
        )


if __name__ == "__main__":
    # python u-net分割推理.py --model ./farmland_segmentation_results_u-net/best_model.pth --input ./test_image.png --output ./predictions --visualize
    # python u-net分割推理.py --model ./farmland_segmentation_results_u-net/best_model.pth --input ./遥感图像 --output ./predictions --threshold 0.5
    #（二值掩膜、概率图、叠加图）
    # python u-net分割推理.py --model ./farmland_segmentation_results_u-net/best_model.pth --input ./遥感图像 --output ./predictions --save_all
    main()