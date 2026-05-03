"""Extract text from PDF paper"""
import sys
pdf_path = r"D:\Work space\DeepLearning\farm\cropland_extraction\paper\结合空间注意力机制与多任务学习的耕地地块分割模型_NormalPdf.pdf"

# Try pypdf first
try:
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        text += f"\n=== Page {i+1} ===\n"
        text += page.extract_text()
    print(f"Extracted {len(reader.pages)} pages, {len(text)} chars")
    # Print first 5000 chars
    print(text[:5000])
except ImportError:
    try:
        import pdfminer
        print("pdfminer available but not used")
    except:
        print("No PDF library available")
