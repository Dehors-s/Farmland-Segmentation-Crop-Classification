import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class CropCNN(nn.Module):
    """
    用于作物分类的1D-CNN模型
    """
    
    def __init__(self, input_dim, num_classes):
        """
        初始化模型
        
        Args:
            input_dim (int): 输入特征维度
            num_classes (int): 分类类别数
        """
        super(CropCNN, self).__init__()
        
        # 卷积块
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # 分类头
        self.fc = nn.Linear(128, num_classes)
        
        # 保存输入维度
        self.input_dim = input_dim
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (N, input_dim)
        
        Returns:
            torch.Tensor: 输出张量，形状为 (N, num_classes)
        """
        # 调整输入形状为 (N, C, L)，其中 C=1, L=input_dim
        x = x.unsqueeze(1)  # (N, input_dim) -> (N, 1, input_dim)
        
        # 卷积块
        x = self.conv_block(x)
        
        # 全局平均池化
        x = self.gap(x).squeeze(2)  # (N, 128, 1) -> (N, 128)
        
        # 分类
        x = self.fc(x)
        
        return x


class CNNTrainer:
    """
    CNN训练器类，封装训练和预测方法，使其接口与sklearn保持一致
    """
    
    def __init__(self, input_dim, num_classes, **kwargs):
        """
        初始化训练器
        
        Args:
            input_dim (int): 输入特征维度
            num_classes (int): 分类类别数
            **kwargs: 训练参数
        """
        # 创建模型
        self.model = CropCNN(input_dim, num_classes)
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 训练参数
        self.epochs = kwargs.get('epochs', 100)
        self.batch_size = kwargs.get('batch_size', 32)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def fit(self, X, y):
        """
        训练模型
        
        Args:
            X (numpy array): 训练特征，形状为 (N, input_dim)
            y (numpy array): 训练标签，形状为 (N,)
        
        Returns:
            self: 训练器实例
        """
        # 转换为张量
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
        
        # 数据集和数据加载器
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        # 训练循环
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for batch_X, batch_y in dataloader:
                # 清零梯度
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # 反向传播和优化
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item() * batch_X.size(0)
            
            # 计算 epoch 损失
            epoch_loss = running_loss / len(dataset)
            
            # 每10个epoch打印一次
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}')
        
        return self
    
    def predict(self, X):
        """
        预测
        
        Args:
            X (numpy array): 预测特征，形状为 (N, input_dim)
        
        Returns:
            numpy array: 预测标签，形状为 (N,)
        """
        # 转换为张量
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        
        # 转换为numpy数组
        return predicted.cpu().numpy()
    
    def predict_proba(self, X):
        """
        预测概率
        
        Args:
            X (numpy array): 预测特征，形状为 (N, input_dim)
        
        Returns:
            numpy array: 预测概率，形状为 (N, num_classes)
        """
        # 转换为张量
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        # 转换为numpy数组
        return probabilities.cpu().numpy()
