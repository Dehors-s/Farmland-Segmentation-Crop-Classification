import os
from pathlib import Path
from markitdown import MarkItDown

def batch_convert_pdfs_to_md(input_dir: str, output_dir: str):
    """
    批量将指定文件夹内的所有 PDF 转换为 Markdown
    """
    # 1. 实例化转换器
    md = MarkItDown()
    
    # 2. 规范化路径并创建输出文件夹
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 3. 递归查找所有 .pdf 文件（包括子文件夹里的）
    pdf_files = list(input_path.rglob("*.pdf"))
    
    if not pdf_files:
        print(f"⚠️ 在目录 [{input_path.absolute()}] 中没有找到任何 PDF 文件！")
        return
        
    print(f"🔍 共找到 {len(pdf_files)} 个 PDF 文件，准备开始批量转换...\n")
    print("-" * 50)
    
    success_count = 0
    fail_count = 0
    
    # 4. 遍历并转换
    for index, pdf_file in enumerate(pdf_files, start=1):
        # 构造输出文件名 (原文件名.md)
        md_filename = pdf_file.stem + ".md"
        md_filepath = output_path / md_filename
        
        print(f"[{index}/{len(pdf_files)}] 正在转换: {pdf_file.name} ...")
        
        try:
            # 执行核心转换
            result = md.convert(str(pdf_file))
            
            # 将转换结果写入 MD 文件
            with open(md_filepath, "w", encoding="utf-8") as f:
                f.write(result.text_content)
                
            print(f"  └─ ✅ 成功! 已保存至: {md_filename}")
            success_count += 1
            
        except Exception as e:
            # 捕获异常，防止某篇损坏的 PDF 导致整个程序崩溃
            print(f"  └─ ❌ 失败! 错误信息: {e}")
            fail_count += 1
            
    # 5. 打印最终统计信息
    print("\n" + "=" * 50)
    print("🎉 批量转换任务完成！")
    print(f"✅ 成功: {success_count} 篇")
    print(f"❌ 失败: {fail_count} 篇")
    print(f"📂 输出目录: {output_path.absolute()}")
    print("=" * 50)


if __name__ == "__main__":
    # ==========================================
    # 在这里配置您的输入和输出文件夹路径
    # ==========================================
    
    # 存放待转换 PDF 论文的文件夹
    INPUT_FOLDER = r"./paper" 
    
    # 转换后 MD 文件保存的文件夹
    OUTPUT_FOLDER = r"./papers_md" 
    
    batch_convert_pdfs_to_md(INPUT_FOLDER, OUTPUT_FOLDER)