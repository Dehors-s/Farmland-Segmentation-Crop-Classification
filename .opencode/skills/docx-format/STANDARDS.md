# 默认格式标准

本文档定义了 Word 文档的默认格式标准，包括中国公文格式和学术论文格式。

## 中国公文格式标准（GB/T 9704-2012）

| 元素 | 中文字体 | 英文字体 | 字号 | 行距 | 段前 | 段后 | 首行缩进 | 对齐 |
|------|---------|---------|------|------|------|------|---------|------|
| 文档标题 | 方正小标宋简体 | Arial Black | 二号(22pt) | 1.5倍 | 0 | 18pt | 无 | 居中 |
| 一级标题 | 黑体 | Arial | 三号(16pt) | 1.5倍 | 12pt | 6pt | 无 | 左对齐 |
| 二级标题 | 黑体 | Arial | 小三(15pt) | 1.5倍 | 6pt | 6pt | 无 | 左对齐 |
| 三级标题 | 黑体 | Arial | 四号(14pt) | 1.5倍 | 6pt | 6pt | 无 | 左对齐 |
| 正文 | 仿宋_GB2312 | Times New Roman | 三号(16pt) | 固定28pt | 0 | 0 | 2字符(32pt) | 两端对齐 |
| 表格标题 | 黑体 | Arial | 五号(10.5pt) | 单倍 | 6pt | 6pt | 无 | 居中 |
| 表格内容 | 仿宋_GB2312 | Times New Roman | 五号(10.5pt) | 单倍 | 0 | 0 | 无 | 居中 |

**关键要求**：
- 正文必须使用仿宋_GB2312，三号字
- 行距必须是固定值28磅（不是1.5倍）
- 首行缩进2字符（三号字为32pt）

## 学术论文格式标准

| 元素 | 中文字体 | 英文字体 | 字号 | 行距 | 段前 | 段后 | 首行缩进 | 对齐 |
|------|---------|---------|------|------|------|------|---------|------|
| 论文标题 | 黑体 | Arial Black | 小二(18pt) | 1.5倍 | 0 | 18pt | 无 | 居中 |
| 一级标题 | 黑体 | Arial | 小三(15pt) | 1.5倍 | 12pt | 6pt | 无 | 居中 |
| 二级标题 | 黑体 | Arial | 四号(14pt) | 1.5倍 | 6pt | 6pt | 无 | 左对齐 |
| 三级标题 | 黑体 | Arial | 小四(12pt) | 1.5倍 | 6pt | 6pt | 无 | 左对齐 |
| 正文 | 宋体 | Times New Roman | 小四(12pt) | 1.5倍 | 0 | 0 | 2字符(24pt) | 两端对齐 |
| 表格标题 | 黑体 | Arial | 五号(10.5pt) | 单倍 | 6pt | 6pt | 无 | 居中 |
| 表格内容 | 宋体 | Times New Roman | 五号(10.5pt) | 单倍 | 0 | 0 | 无 | 居中 |
| 图片说明 | 宋体 | Times New Roman | 五号(10.5pt) | 单倍 | 6pt | 12pt | 无 | 居中 |
| 参考文献 | 宋体 | Times New Roman | 五号(10.5pt) | 单倍 | 0 | 0 | 悬挂(-21pt) | 两端对齐 |

**关键要求**：
- 正文使用宋体，小四字（12pt）
- 行距1.5倍
- 首行缩进2字符（小四字为24pt）
- 参考文献使用悬挂缩进

## 参数速查表

### 字号对照

| 中文名称 | 磅值(pt) | 代码 | 常用场景 |
|---------|---------|------|---------|
| 初号 | 42 | `Pt(42)` | 封面大标题 |
| 小初 | 36 | `Pt(36)` | 封面标题 |
| 一号 | 26 | `Pt(26)` | 章标题 |
| 小一 | 24 | `Pt(24)` | 章标题 |
| 二号 | 22 | `Pt(22)` | 文档标题 |
| 小二 | 18 | `Pt(18)` | 论文标题 |
| 三号 | 16 | `Pt(16)` | 公文正文、一级标题 |
| 小三 | 15 | `Pt(15)` | 二级标题 |
| 四号 | 14 | `Pt(14)` | 三级标题 |
| 小四 | 12 | `Pt(12)` | 学术论文正文 |
| 五号 | 10.5 | `Pt(10.5)` | 表格内容、脚注 |
| 小五 | 9 | `Pt(9)` | 页眉页脚 |

### 首行缩进对照

| 字号 | 2字符缩进 | 代码 |
|------|----------|------|
| 三号(16pt) | 32pt | `Pt(32)` |
| 小三(15pt) | 30pt | `Pt(30)` |
| 四号(14pt) | 28pt | `Pt(28)` |
| 小四(12pt) | 24pt | `Pt(24)` |
| 五号(10.5pt) | 21pt | `Pt(21)` |

**计算公式**：2字符缩进 = 字号(pt) × 2

### 行距设置

| 行距类型 | 代码 | 说明 |
|---------|------|------|
| 单倍行距 | `para.paragraph_format.line_spacing = 1.0` | 默认 |
| 1.5倍行距 | `para.paragraph_format.line_spacing = 1.5` | 学术论文常用 |
| 2倍行距 | `para.paragraph_format.line_spacing = 2.0` | 草稿 |
| 固定值28磅 | `para.paragraph_format.line_spacing = Pt(28)` | 公文标准 |

**注意**：公文必须使用固定值28磅，不能使用倍数行距。

### 对齐方式

| 对齐方式 | 代码 | 数值 | 常用场景 |
|---------|------|------|---------|
| 左对齐 | `WD_ALIGN_PARAGRAPH.LEFT` | 0 | 标题、列表 |
| 居中 | `WD_ALIGN_PARAGRAPH.CENTER` | 1 | 文档标题、表格标题 |
| 右对齐 | `WD_ALIGN_PARAGRAPH.RIGHT` | 2 | 日期、签名 |
| 两端对齐 | `WD_ALIGN_PARAGRAPH.JUSTIFY` | 3 | 正文（推荐） |
| 分散对齐 | `WD_ALIGN_PARAGRAPH.DISTRIBUTE` | 4 | 特殊需求 |

### 段前段后间距

| 间距 | 代码 | 说明 |
|------|------|------|
| 无间距 | `Pt(0)` | 正文段落 |
| 6磅 | `Pt(6)` | 标题段后 |
| 12磅 | `Pt(12)` | 一级标题段前 |
| 18磅 | `Pt(18)` | 文档标题段后 |
| 1行(12pt) | `Pt(12)` | 相当于1行间距 |

### 常用颜色

| 颜色 | RGB值 | 代码 | 用途 |
|------|-------|------|------|
| 黑色 | (0, 0, 0) | `RGBColor(0, 0, 0)` | 正文 |
| 蓝色 | (0, 0, 255) | `RGBColor(0, 0, 255)` | 引用编号 |
| 红色 | (255, 0, 0) | `RGBColor(255, 0, 0)` | 强调 |
| 灰色 | (128, 128, 128) | `RGBColor(128, 128, 128)` | 注释 |

## 代码示例

### 应用公文格式

```python
from docx import Document
from docx.shared import Pt
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document('input.docx')

for para in doc.paragraphs:
    if len(para.text.strip()) > 30:  # 正文
        para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        para.paragraph_format.first_line_indent = Pt(32)
        para.paragraph_format.line_spacing = Pt(28)  # 固定28磅
        for run in para.runs:
            run.font.name = 'Times New Roman'
            run._element.rPr.rFonts.set(qn('w:eastAsia'), '仿宋_GB2312')
            run.font.size = Pt(16)

doc.save('output.docx')
```

### 应用学术论文格式

```python
from docx import Document
from docx.shared import Pt
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document('input.docx')

for para in doc.paragraphs:
    if len(para.text.strip()) > 30:  # 正文
        para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        para.paragraph_format.first_line_indent = Pt(24)
        para.paragraph_format.line_spacing = 1.5
        for run in para.runs:
            run.font.name = 'Times New Roman'
            run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
            run.font.size = Pt(12)

doc.save('output.docx')
```

## 单位转换

### EMU 转 pt

```python
def emu_to_pt(emu):
    """EMU 转 pt"""
    return round(emu / 914400 * 72, 1) if emu else None

# 使用示例
size_in_emu = run.font.size  # 返回 EMU 单位
size_in_pt = emu_to_pt(size_in_emu)  # 转换为 pt
```

### pt 转字符数（首行缩进）

```python
def pt_to_chars(pt, font_size_pt):
    """pt 转字符数"""
    return round(pt / font_size_pt, 1)

# 示例：21pt 缩进在五号字(10.5pt)下是多少字符？
chars = pt_to_chars(21, 10.5)  # 结果：2.0字符
```

## 常见字体

### 中文字体

| 字体名称 | 用途 | 备注 |
|---------|------|------|
| 仿宋_GB2312 | 公文正文 | 必须使用此字体 |
| 宋体 | 学术论文正文 | 最常用 |
| 黑体 | 标题 | 加粗效果 |
| 楷体_GB2312 | 引用、注释 | 公文二级标题 |
| 方正小标宋简体 | 公文标题 | 特殊字体 |

### 英文字体

| 字体名称 | 用途 | 备注 |
|---------|------|------|
| Times New Roman | 正文 | 最常用 |
| Arial | 标题 | 无衬线字体 |
| Arial Black | 文档标题 | 加粗效果 |
| Calibri | 现代文档 | Office 默认 |

## 页面设置

### A4 纸张标准

```python
from docx.shared import Pt

section = doc.sections[0]
section.page_height = Pt(845)  # 29.7cm
section.page_width = Pt(598)   # 21cm
section.top_margin = Pt(72)    # 2.54cm (1英寸)
section.bottom_margin = Pt(72)
section.left_margin = Pt(90)   # 3.17cm
section.right_margin = Pt(90)
```

### 公文页面设置

```python
section.top_margin = Pt(106)    # 3.7cm
section.bottom_margin = Pt(85)  # 3.0cm
section.left_margin = Pt(99)    # 3.5cm
section.right_margin = Pt(85)   # 3.0cm
```

## 参考资料

- GB/T 9704-2012《党政机关公文格式》
- 学位论文格式规范（各高校标准）
- python-docx 官方文档：https://python-docx.readthedocs.io/
