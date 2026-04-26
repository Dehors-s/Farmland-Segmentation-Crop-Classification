@echo off
setlocal enabledelayedexpansion
:: 设置UTF-8编码
chcp 65001 >nul
color 0A

echo =======================================================
echo     农田分割程序 - 自动化训练与矢量化流水线
echo =======================================================

:: =======================================================
:: 0. 基础路径配置区
:: =======================================================
:: 请根据实际情况修改数据集路径
set DATA_ROOT=D:\Work space\DeepLearning\farm\dataset
:: 请根据实际情况修改输出路径
set OUTPUT_DIR=D:\Work space\DeepLearning\farm\results_v7_ultimate
:: 验证集图像路径，供矢量化阶段使用
set TEST_IMG_DIR=%DATA_ROOT%\val\img
:: 编码器选择,需与训练脚本中一致resnet50,resnet34,efficientnet-b0等
set ENCODER=resnet50
:: 训练轮数
set EPOCHS=20
:: 学习率
set LR=0.0001
:: 矢量化 epsilon 参数
set EPSILON=0.02
:: 形态学核大小
set MORPH_KERNEL=5
:: 模型路径 (推理模式下必填)
set MODEL_PATH=%OUTPUT_DIR%\best_model-resnet-32.pth

:: =======================================================
:: 1. 环境侦察与性能自检 (核心逻辑)
:: =======================================================
echo.
echo [INFO] 正在侦察本地硬件环境与依赖库...

:: 检查 Python 和 PyTorch
python -c "import torch; print('PyTorch Version:', torch.__version__)" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    color 0C
    echo [ERROR] 找不到 PyTorch！请先激活您的 Conda/Python 虚拟环境。
    pause
    exit /b 1
)

:: 检查 CUDA 与显卡
set HAS_GPU=0
set VRAM_MB=0
for /f "tokens=*" %%i in ('python -c "import torch; print(1 if torch.cuda.is_available() else 0)"') do set HAS_GPU=%%i

if "%HAS_GPU%"=="0" (
    color 0E
    echo [WARN] 未检测到可用的 NVIDIA 显卡，将使用 CPU 进行极慢速训练！
    set BATCH_SIZE=2
    set NUM_WORKERS=0
) else (
    REM 获取显存大小 (MB)
    for /f "tokens=*" %%v in ('python -c "import torch; print(int(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024))"') do set VRAM_MB=%%v
    echo [INFO] 成功点亮 GPU! 检测到显存: !VRAM_MB! MB
    
    REM 智能设置 Batch Size 和 Workers (基于显存阶梯)
    if !VRAM_MB! GEQ 22000 (
        echo [INFO] 检测到旗舰级显卡 ^(24GB+，如 RTX 3090/4090^)
        set BATCH_SIZE=32
        set NUM_WORKERS=8
    ) else if !VRAM_MB! GEQ 15000 (
        echo [INFO] 检测到高端显卡 ^(16GB+，如 RTX 4080^)
        set BATCH_SIZE=16
        set NUM_WORKERS=4
    ) else if !VRAM_MB! GEQ 7000 (
        echo [INFO] 检测到中端显卡 ^(8GB+，如 RTX 3060/4060^)
        set BATCH_SIZE=8
        set NUM_WORKERS=2
    ) else (
        echo [WARN] 显存较低 ^(^<8GB^)，已切换至保守训练模式。
        set BATCH_SIZE=4
        set NUM_WORKERS=0
    )
)

echo -------------------------------------------------------
echo [自适应配置] Batch Size 自动设定为: %BATCH_SIZE%
echo [自适应配置] Num Workers 自动设定为: %NUM_WORKERS%
echo -------------------------------------------------------

:: =======================================================
:: 交互式选择运行模式
:: =======================================================
echo.
echo 请选择运行模式 / Select Run Mode:
echo [1] 全流程 (训练 + 矢量化) / Train + Vectorize [Default]
echo [2] 仅进行模型训练 / Train Only
echo [3] 仅进行矢量化推理 / Vectorize Only
set /p RUN_MODE="Please input (1/2/3): "

if "%RUN_MODE%"=="" set RUN_MODE=1
if "%RUN_MODE%"=="3" goto STAGE_INFERENCE

:STAGE_TRAIN
:: =======================================================
:: 2. 阶段一：模型训练
:: =======================================================
echo.
echo 🚀 [阶段 1/2]: 开始模型训练...

:: 这里假设您的 u-net--CBAMV7.py 已经支持了 --num_workers 参数
python u-net--CBAMV7.py ^
    --data_root "%DATA_ROOT%" ^
    --output_dir "%OUTPUT_DIR%" ^
    --encoder_name "%ENCODER%" ^
    --batch_size %BATCH_SIZE% ^
    --epochs %EPOCHS% ^
    --lr %LR%

if %ERRORLEVEL% neq 0 (
    color 0C
    echo.
    echo ❌ 训练阶段崩溃！请向上翻看报错信息。
    pause
    exit /b %ERRORLEVEL%
)

if "%RUN_MODE%"=="2" goto FINAL_END

:STAGE_INFERENCE
:: =======================================================
:: 3. 阶段二：矢量化推理
:: =======================================================
echo.
echo 🚀 [阶段 2/2]: 开始进行验证集推理与多边形矢量化提取...

:: 检查模型文件是否存在 (增强的容错逻辑)
if not exist "%MODEL_PATH%" (
    if exist "%OUTPUT_DIR%\best_boundary_model.pth" (
       echo [WARN] 未找到 best_model.pth，将尝试使用 best_boundary_model.pth ...
       set MODEL_PATH=%OUTPUT_DIR%\best_boundary_model.pth
    ) else (
        if exist "%OUTPUT_DIR%\final_model.pth" (
            echo [WARN] 未找到 best_model.pth，将尝试使用 final_model.pth ...
            set MODEL_PATH=%OUTPUT_DIR%\final_model.pth
        ) else (
            color 0C
            echo.
            echo ❌ 错误：在 %OUTPUT_DIR% 中找不到任何模型文件！
            echo 请先运行训练阶段 ^(模式 1 或 2^)，或手动指定模型路径。
            pause
            goto FINAL_END
        )
    )
)

python ".\u-net矢量化.py" ^
    --model "%MODEL_PATH%" ^
    --input "%TEST_IMG_DIR%" ^
    --output "%OUTPUT_DIR%\vectorized" ^
    --encoder_name "%ENCODER%" ^
    --epsilon %EPSILON% ^
    --morph_kernel %MORPH_KERNEL%

if %ERRORLEVEL% neq 0 (
    color 0C
    echo.
    echo ❌ 矢量化阶段报错！
    pause
    exit /b %ERRORLEVEL%
)

:FINAL_END
:: =======================================================
:: 4. 完美收官
:: =======================================================
color 0B
echo.
echo =======================================================
echo 🎉 全部流水线执行完毕！
echo 📂 输出结果已保存至: %OUTPUT_DIR%
echo =======================================================
pause