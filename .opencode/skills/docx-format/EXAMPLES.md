# Word 文档格式处理示例集

本文档提供 `docx-format` skill 的详细示例代码，涵盖多级标题、正文、表格、图片等各类元素的格式处理。

## 目录

1. [多级标题处理](#多级标题处理)
2. [正文格式处理](#正文格式处理)
3. [表格格式处理](#表格格式处理)
4. [图片说明处理](#图片说明处理)
5. [页眉页脚处理](#页眉页脚处理)
6. [综合示例](#综合示例)

---

## 多级标题处理

### 示例 1：识别并格式化多级标题

```python
from docx import Document
from docx.shared import Pt
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document('input.docx')

def format_heading(para, level):
    """格式化标题"""
    if level == 1:  # 一级标题
        para.alignment = WD_ALIGN_PARAGRAPH.LEFT
        para.paragraph_format.space_before = Pt(12)
        para.paragraph_format.space_after = Pt(6)
        para.paragraph_format.line_spacing = 1.5
        for run in para.runs:
            run.font.name = 'Arial'
            run._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
            run.font.size = Pt(16)
            run.font.bold = True

    elif level == 2:  # 二级标题
        para.alignment = WD_ALIGN_PARAGRAPH.LEFT
        para.paragraph_format.space_before = Pt(6)
        para.paragraph_format.space_after = Pt(6)
        para.paragraph_format.line_spacing = 1.5
        for run in para.runs:
            run.font.name = 'Arial'
            run._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
            run.font.size = Pt(15)
            run.font.bold = True

    elif level == 3:  # 三级标题
        para.alignment = WD_ALIGN_PARAGRAPH.LEFT
        para.paragraph_format.space_before = Pt(6)
        para.paragraph_format.space_after = Pt(6)
        para.paragraph_format.line_spacing = 1.5
        for run in para.runs:
            run.font.name = 'Arial'
            run._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
            run.font.size = Pt(14)
            run.font.bold = True

# 智能识别标题级别
for para in doc.paragraphs:
    text = para.text.strip()
    if not text:
        continue

    # 根据标题编号识别级别
    if text.startswith('一、') or text.startswith('第一章'):
        format_heading(para, 1)
    elif text.startswith('（一）') or text.startswith('1.'):
        format_heading(para, 2)
    elif text.startswith('1）') or text.startswith('①'):
        format_heading(para, 3)

doc.save('output.docx')
```

### 示例 2：使用内置样式设置标题

```python
from docx import Document

doc = Document('input.docx')

# 应用内置标题样式
for para in doc.paragraphs:
    text = para.text.strip()
    if text.startswith('一、'):
        para.style = 'Heading 1'
    elif text.startswith('（一）'):
        para.style = 'Heading 2'
    elif text.startswith('1）'):
        para.style = 'Heading 3'

doc.save('output.docx')
```

---

## 正文格式处理

### 示例 3：公文正文格式（仿宋三号固定行距28磅）

```python
from docx import Document
from docx.shared import Pt
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document('input.docx')

def format_body_official(para):
    """公文正文格式"""
    para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY  # 两端对齐
    para.paragraph_format.first_line_indent = Pt(32)  # 首行缩进2字符（三号字）
    para.paragraph_format.line_spacing = Pt(28)  # 固定行距28磅
    para.paragraph_format.space_before = Pt(0)
    para.paragraph_format.space_after = Pt(0)

    for run in para.runs:
        run.font.name = 'Times New Roman'
        run._element.rPr.rFonts.set(qn('w:eastAsia'), '仿宋_GB2312')
        run.font.size = Pt(16)  # 三号

# 处理所有正文段落
for para in doc.paragraphs:
    if len(para.text.strip()) > 30:  # 过滤短段落和标题
        format_body_official(para)

# 处理表格内正文
for table in doc.tables:
    for row in table.rows:
        for cell in row.cells:
            for para in cell.paragraphs:
                if len(para.text.strip()) > 30:
                    format_body_official(para)

doc.save('output.docx')
```

### 示例 4：学术论文正文格式（宋体小四1.5倍行距）

```python
from docx import Document
from docx.shared import Pt
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document('input.docx')

def format_body_academic(para):
    """学术论文正文格式"""
    para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    para.paragraph_format.first_line_indent = Pt(24)  # 首行缩进2字符（小四字）
    para.paragraph_format.line_spacing = 1.5  # 1.5倍行距
    para.paragraph_format.space_before = Pt(0)
    para.paragraph_format.space_after = Pt(0)

    for run in para.runs:
        run.font.name = 'Times New Roman'
        run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
        run.font.size = Pt(12)  # 小四

for para in doc.paragraphs:
    if len(para.text.strip()) > 30:
        format_body_academic(para)

for table in doc.tables:
    for row in table.rows:
        for cell in row.cells:
            for para in cell.paragraphs:
                if len(para.text.strip()) > 30:
                    format_body_academic(para)

doc.save('output.docx')
```

### 示例 5：处理段前段后间距

```python
from docx import Document
from docx.shared import Pt

doc = Document('input.docx')

for para in doc.paragraphs:
    # 设置段前段后间距
    para.paragraph_format.space_before = Pt(6)   # 段前6磅
    para.paragraph_format.space_after = Pt(6)    # 段后6磅

    # 或使用行数（1行 = 12pt）
    # para.paragraph_format.space_before = Pt(12)  # 段前1行

doc.save('output.docx')
```

---

## 表格格式处理

### 示例 6：统一表格内容格式

```python
from docx import Document
from docx.shared import Pt
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document('input.docx')

for table in doc.tables:
    # 处理表头（第一行）
    header_row = table.rows[0]
    for cell in header_row.cells:
        for para in cell.paragraphs:
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in para.runs:
                run.font.name = 'Arial'
                run._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
                run.font.size = Pt(10.5)  # 五号
                run.font.bold = True

    # 处理表格内容（其余行）
    for row in table.rows[1:]:
        for cell in row.cells:
            for para in cell.paragraphs:
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in para.runs:
                    run.font.name = 'Times New Roman'
                    run._element.rPr.rFonts.set(qn('w:eastAsia'), '仿宋_GB2312')
                    run.font.size = Pt(10.5)

doc.save('output.docx')
```

### 示例 7：设置表格边框和单元格间距

```python
from docx import Document
from docx.shared import Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

doc = Document('input.docx')

def set_cell_border(cell, **kwargs):
    """设置单元格边框"""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()

    # 创建边框元素
    tcBorders = OxmlElement('w:tcBorders')
    for edge in ('top', 'left', 'bottom', 'right'):
        edge_element = OxmlElement(f'w:{edge}')
        edge_element.set(qn('w:val'), 'single')
        edge_element.set(qn('w:sz'), '4')  # 边框宽度
        edge_element.set(qn('w:space'), '0')
        edge_element.set(qn('w:color'), '000000')  # 黑色
        tcBorders.append(edge_element)

    tcPr.append(tcBorders)

for table in doc.tables:
    for row in table.rows:
        for cell in row.cells:
            set_cell_border(cell)

doc.save('output.docx')
```

---

## 图片说明处理

### 示例 8：格式化图片说明

```python
from docx import Document
from docx.shared import Pt
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document('input.docx')

for para in doc.paragraphs:
    text = para.text.strip()

    # 识别图片说明（以"图"或"Figure"开头）
    if text.startswith('图') or text.startswith('Figure'):
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_before = Pt(6)
        para.paragraph_format.space_after = Pt(12)

        for run in para.runs:
            run.font.name = 'Times New Roman'
            run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
            run.font.size = Pt(10.5)  # 五号

doc.save('output.docx')
```

### 示例 9：批量调整图片大小

```python
from docx import Document
from docx.shared import Inches

doc = Document('input.docx')

# 遍历所有图片
for shape in doc.inline_shapes:
    # 设置图片宽度（保持纵横比）
    if shape.width > Inches(6):  # 如果宽度超过6英寸
        shape.width = Inches(6)

doc.save('output.docx')
```

---

## 页眉页脚处理

### 示例 10：设置页眉页脚格式

```python
from docx import Document
from docx.shared import Pt
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document('input.docx')

# 获取第一个节的页眉
section = doc.sections[0]
header = section.header

# 格式化页眉
for para in header.paragraphs:
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in para.runs:
        run.font.name = 'Times New Roman'
        run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
        run.font.size = Pt(9)  # 小五

# 格式化页脚
footer = section.footer
for para in footer.paragraphs:
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in para.runs:
        run.font.name = 'Times New Roman'
        run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
        run.font.size = Pt(9)

doc.save('output.docx')
```

### 示例 11：添加页码

```python
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

doc = Document('input.docx')

def add_page_number(paragraph):
    """在段落中添加页码"""
    run = paragraph.add_run()
    fldChar1 = OxmlElement('w:fldChar')
    fldChar1.set(qn('w:fldCharType'), 'begin')

    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')
    instrText.text = "PAGE"

    fldChar2 = OxmlElement('w:fldChar')
    fldChar2.set(qn('w:fldCharType'), 'end')

    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)

# 在页脚添加页码
section = doc.sections[0]
footer = section.footer
para = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
add_page_number(para)

doc.save('output.docx')
```

---

## 综合示例

### 示例 12：完整的公文格式化

```python
from docx import Document
from docx.shared import Pt
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document('input.docx')

def format_official_document(doc):
    """完整的公文格式化"""

    # 1. 处理标题
    if doc.paragraphs:
        title_para = doc.paragraphs[0]
        title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in title_para.runs:
            run.font.name = 'Arial'
            run._element.rPr.rFonts.set(qn('w:eastAsia'), '方正小标宋简体')
            run.font.size = Pt(22)  # 二号
            run.font.bold = True

    # 2. 处理正文
    for i, para in enumerate(doc.paragraphs[1:], 1):
        text = para.text.strip()
        if not text:
            continue

        # 一级标题
        if text.startswith('一、') or text.startswith('第一'):
            para.alignment = WD_ALIGN_PARAGRAPH.LEFT
            para.paragraph_format.space_before = Pt(12)
            para.paragraph_format.space_after = Pt(6)
            for run in para.runs:
                run.font.name = 'Arial'
                run._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
                run.font.size = Pt(16)  # 三号
                run.font.bold = True

        # 正文
        elif len(text) > 30:
            para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            para.paragraph_format.first_line_indent = Pt(32)
            para.paragraph_format.line_spacing = Pt(28)
            for run in para.runs:
                run.font.name = 'Times New Roman'
                run._element.rPr.rFonts.set(qn('w:eastAsia'), '仿宋_GB2312')
                run.font.size = Pt(16)

    # 3. 处理表格
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in para.runs:
                        run.font.name = 'Times New Roman'
                        run._element.rPr.rFonts.set(qn('w:eastAsia'), '仿宋_GB2312')
                        run.font.size = Pt(10.5)

    # 4. 设置页面
    section = doc.sections[0]
    section.page_height = Pt(845)  # A4纸高度
    section.page_width = Pt(598)   # A4纸宽度
    section.top_margin = Pt(72)    # 上边距2.54cm
    section.bottom_margin = Pt(72)
    section.left_margin = Pt(90)   # 左边距3.17cm
    section.right_margin = Pt(90)

format_official_document(doc)
doc.save('output.docx')
```

### 示例 13：完整的学术论文格式化

```python
from docx import Document
from docx.shared import Pt
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
import re

doc = Document('input.docx')

def format_academic_paper(doc):
    """完整的学术论文格式化"""

    ref_pattern = re.compile(r'\[\d+\]')

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # 论文标题
        if len(text) < 50 and para == doc.paragraphs[0]:
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            para.paragraph_format.space_after = Pt(18)
            for run in para.runs:
                run.font.name = 'Times New Roman'
                run._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
                run.font.size = Pt(18)  # 小二
                run.font.bold = True

        # 一级标题
        elif text.startswith('第') and '章' in text:
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            para.paragraph_format.space_before = Pt(12)
            para.paragraph_format.space_after = Pt(6)
            for run in para.runs:
                run.font.name = 'Times New Roman'
                run._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
                run.font.size = Pt(15)  # 小三
                run.font.bold = True

        # 二级标题
        elif re.match(r'^\d+\.\d+', text):
            para.alignment = WD_ALIGN_PARAGRAPH.LEFT
            para.paragraph_format.space_before = Pt(6)
            para.paragraph_format.space_after = Pt(6)
            for run in para.runs:
                run.font.name = 'Times New Roman'
                run._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
                run.font.size = Pt(14)  # 四号
                run.font.bold = True

        # 参考文献
        elif text.startswith('[') and ']' in text[:5]:
            para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            para.paragraph_format.first_line_indent = Pt(-21)
            para.paragraph_format.left_indent = Pt(21)
            para.paragraph_format.line_spacing = 1.0
            for run in para.runs:
                run.font.name = 'Times New Roman'
                run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
                run.font.size = Pt(10.5)

        # 正文
        elif len(text) > 30:
            para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            para.paragraph_format.first_line_indent = Pt(24)
            para.paragraph_format.line_spacing = 1.5
            for run in para.runs:
                run.font.name = 'Times New Roman'
                run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
                run.font.size = Pt(12)

                # 引用编号变蓝色
                if ref_pattern.search(run.text):
                    run.font.color.rgb = RGBColor(0, 0, 255)

    # 处理表格
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in para.runs:
                        run.font.name = 'Times New Roman'
                        run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
                        run.font.size = Pt(10.5)

format_academic_paper(doc)
doc.save('output.docx')
```

---

## 常见问题

### Q1: 如何判断段落是否在表格内？

```python
# 方法1：遍历时区分
for para in doc.paragraphs:
    print("普通段落:", para.text)

for table in doc.tables:
    for row in table.rows:
        for cell in row.cells:
            for para in cell.paragraphs:
                print("表格内段落:", para.text)

# 方法2：检查父元素
def is_in_table(para):
    parent = para._element.getparent()
    while parent is not None:
        if parent.tag.endswith('tc'):  # tc = table cell
            return True
        parent = parent.getparent()
    return False
```

### Q2: 如何清除现有格式？

```python
from docx import Document

doc = Document('input.docx')

for para in doc.paragraphs:
    # 清除段落格式
    para.paragraph_format.first_line_indent = None
    para.paragraph_format.line_spacing = None
    para.paragraph_format.space_before = None
    para.paragraph_format.space_after = None

    # 清除字体格式
    for run in para.runs:
        run.font.name = None
        run.font.size = None
        run.font.bold = None
        run.font.italic = None

doc.save('output.docx')
```

### Q3: 如何处理嵌套表格？

```python
from docx import Document

doc = Document('input.docx')

def process_table(table, level=0):
    """递归处理表格（包括嵌套表格）"""
    print(f"{'  ' * level}处理表格，级别 {level}")

    for row in table.rows:
        for cell in row.cells:
            # 处理单元格内的段落
            for para in cell.paragraphs:
                print(f"{'  ' * level}段落: {para.text[:30]}")

            # 处理单元格内的嵌套表格
            for nested_table in cell.tables:
                process_table(nested_table, level + 1)

for table in doc.tables:
    process_table(table)

doc.save('output.docx')
```

---

## 参数速查表

### 字号对照

| 中文名称 | 磅值(pt) | 像素(px) | 英寸(in) |
|---------|---------|---------|---------|
| 初号 | 42 | 56 | 0.58 |
| 小初 | 36 | 48 | 0.50 |
| 一号 | 26 | 35 | 0.36 |
| 小一 | 24 | 32 | 0.33 |
| 二号 | 22 | 29 | 0.31 |
| 小二 | 18 | 24 | 0.25 |
| 三号 | 16 | 21 | 0.22 |
| 小三 | 15 | 20 | 0.21 |
| 四号 | 14 | 19 | 0.19 |
| 小四 | 12 | 16 | 0.17 |
| 五号 | 10.5 | 14 | 0.15 |
| 小五 | 9 | 12 | 0.13 |

### 行距对照

| 行距类型 | 代码 | 说明 |
|---------|------|------|
| 单倍行距 | `para.paragraph_format.line_spacing = 1.0` | 默认 |
| 1.5倍行距 | `para.paragraph_format.line_spacing = 1.5` | 常用 |
| 2倍行距 | `para.paragraph_format.line_spacing = 2.0` | |
| 固定值 | `para.paragraph_format.line_spacing = Pt(28)` | 公文标准 |
| 最小值 | `para.paragraph_format.line_spacing_rule = WD_LINE_SPACING.AT_LEAST` | |

### 对齐方式

| 对齐方式 | 代码 | 数值 |
|---------|------|------|
| 左对齐 | `WD_ALIGN_PARAGRAPH.LEFT` | 0 |
| 居中 | `WD_ALIGN_PARAGRAPH.CENTER` | 1 |
| 右对齐 | `WD_ALIGN_PARAGRAPH.RIGHT` | 2 |
| 两端对齐 | `WD_ALIGN_PARAGRAPH.JUSTIFY` | 3 |
| 分散对齐 | `WD_ALIGN_PARAGRAPH.DISTRIBUTE` | 4 |

---

**文档版本**: 1.0
**最后更新**: 2026-01-12
**相关文档**: `docx-format.md`
