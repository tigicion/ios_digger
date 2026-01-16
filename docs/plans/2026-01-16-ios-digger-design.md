# iOS Digger 设计文档

App Store 评论挖掘工具 —— 从用户评论中提取痛点、好评与需求洞察。

## 概述

**输入**：应用类别（如效率类）、关键词（如备忘录）、或直接指定 App ID
**输出**：PDF 报告，包含用户痛点、好评亮点、需求洞察
**覆盖地区**：中国、美国、日本、韩国、新加坡、泰国、越南、印尼

## 技术选型

| 组件 | 选择 | 理由 |
|------|------|------|
| 语言 | Python | 生态丰富，爬虫/NLP/PDF 库成熟 |
| LLM | 阿里云 Qwen (通义千问) | OpenAI 兼容接口，多语言理解强 |
| 数据源 | iTunes RSS Feed + Search API | 免费、稳定、无需认证 |
| PDF 生成 | WeasyPrint | 支持中文、样式灵活 |
| CLI 框架 | Typer | 现代、易用 |

## 项目结构

```
ios_digger/
├── main.py              # CLI 入口
├── config.py            # 配置
├── scraper/
│   ├── __init__.py
│   ├── search.py        # App 搜索
│   ├── reviews.py       # 评论抓取
│   └── models.py        # 数据模型
├── analyzer/
│   ├── __init__.py
│   ├── llm_client.py    # Qwen API 封装
│   └── insights.py      # 分析逻辑
├── reporter/
│   ├── __init__.py
│   ├── generator.py     # PDF 生成
│   └── templates/
│       ├── base.html
│       ├── report_brief.html
│       ├── report_standard.html
│       ├── report_full.html
│       └── styles.css
├── requirements.txt
├── .env.example
└── .gitignore
```

## 核心流程

```
输入 → 搜索App → 抓取评论 → LLM分析 → 生成PDF
```

## 模块设计

### 1. 数据抓取模块 (scraper)

#### App 搜索 (`search.py`)

使用 iTunes Search API：
```
GET https://itunes.apple.com/search?term={关键词}&country={地区}&entity=software&limit=10
```

三种输入模式：
- **关键词搜索**：调用 Search API，返回各地区 Top 10 相关 App
- **类别浏览**：调用类别排行榜 RSS，获取指定类别 Top N
- **直接指定**：用户提供 App ID 列表，跳过搜索

#### 评论抓取 (`reviews.py`)

使用 App Store RSS Feed：
```
GET https://itunes.apple.com/{地区}/rss/customerreviews/id={appId}/sortBy=mostRecent/json
```

- 每个 App 每个地区最多 500 条最新评论
- 包含：评分、标题、内容、用户名、日期、版本号
- 支持并发抓取（多地区同时进行）

#### 数据模型 (`models.py`)

```python
@dataclass
class App:
    id: str
    name: str
    developer: str
    category: str

@dataclass
class Review:
    app_id: str
    region: str
    rating: int      # 1-5
    title: str
    content: str
    version: str
    date: datetime
```

### 2. LLM 分析模块 (analyzer)

#### LLM 客户端 (`llm_client.py`)

使用阿里云 Qwen 通过 OpenAI 兼容接口：

```python
import os
from openai import OpenAI

def get_llm_client():
    """Get configured OpenAI client for Qwen API."""
    return OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

def analyze_reviews(prompt: str, model: str = "qwen-plus") -> str:
    client = get_llm_client()
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content
```

可用模型：
| 模型 | 用途 |
|------|------|
| `qwen-plus` | 默认，性价比高（推荐） |
| `qwen-turbo` | 更快更便宜 |
| `qwen-max` | 最强能力，复杂分析用 |

#### 分析策略 (`insights.py`)

采用分批 + 汇总策略（控制 token 成本）：

```
第一轮：分批分析（每批 50 条评论）
  ├── 批次1 → 提取痛点/好评/需求
  ├── 批次2 → 提取痛点/好评/需求
  └── ...

第二轮：汇总分析
  └── 合并所有批次结果 → 去重、排序、归类
```

#### 输出结构

```python
@dataclass
class AnalysisResult:
    pain_points: list[Insight]      # 痛点列表（按频率排序）
    positive_feedback: list[Insight] # 好评亮点
    user_needs: list[Insight]        # 挖掘出的用户需求
    regional_diff: dict              # 各地区差异（如有）

@dataclass
class Insight:
    summary: str           # 一句话总结
    frequency: int         # 出现频率
    severity: str          # 严重程度：high/medium/low
    sample_reviews: list   # 代表性原评论引用
```

### 3. PDF 报告模块 (reporter)

#### 技术方案

使用 WeasyPrint（HTML → PDF），搭配 Jinja2 模板 + CSS 样式。

#### 模板结构

```
templates/
├── base.html          # 基础布局
├── report_brief.html  # 精简版（2-3页）
├── report_standard.html # 标准版（5-8页）
├── report_full.html   # 完整版（10+页）
└── styles.css         # 报告样式
```

#### 报告内容

```
┌─────────────────────────────────────┐
│ 封面：App名称 / 分析日期 / 地区范围   │
├─────────────────────────────────────┤
│ 概览：评论总数 / 平均评分 / 评分分布   │
├─────────────────────────────────────┤
│ 痛点排名（按频率）                    │
├─────────────────────────────────────┤
│ 好评亮点                             │
├─────────────────────────────────────┤
│ 用户需求洞察                         │
├─────────────────────────────────────┤
│ 地区差异分析（标准版+）               │
├─────────────────────────────────────┤
│ 产品建议（完整版）                    │
└─────────────────────────────────────┘
```

#### 生成接口

```python
def generate_report(
    analysis: AnalysisResult,
    apps: list[App],
    level: str = "standard"  # brief / standard / full
) -> bytes:  # 返回 PDF 字节
```

### 4. CLI 命令行接口

使用 Typer 库：

```bash
# 关键词搜索模式
ios-digger search "备忘录" --regions cn,us,jp --level standard

# 类别浏览模式
ios-digger category "效率" --top 10 --regions cn,us --level brief

# 直接指定 App ID
ios-digger analyze 123456789,987654321 --regions all --level full

# 查看支持的类别列表
ios-digger categories

# 查看支持的地区列表
ios-digger regions
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--regions` | 地区列表，逗号分隔，或 `all` | `cn,us` |
| `--level` | 报告级别：brief/standard/full | `standard` |
| `--top` | 搜索/类别模式下取 Top N | `10` |
| `--output` | 输出文件路径 | `./report_{timestamp}.pdf` |
| `--model` | LLM 模型选择 | `qwen-plus` |

## 配置

### 支持的地区

| 代码 | 地区 |
|------|------|
| cn | 中国 |
| us | 美国 |
| jp | 日本 |
| kr | 韩国 |
| sg | 新加坡 |
| th | 泰国 |
| vn | 越南 |
| id | 印尼 |

### 环境变量

```bash
# 必需：阿里云 DashScope API Key
DASHSCOPE_API_KEY=sk-xxx
```

### 依赖

```
openai>=1.0.0      # LLM API（Qwen 兼容）
typer>=0.9.0       # CLI 框架
httpx>=0.25.0      # HTTP 请求（异步支持）
weasyprint>=60.0   # PDF 生成
jinja2>=3.0.0      # 模板引擎
pydantic>=2.0.0    # 数据模型
python-dotenv>=1.0 # 环境变量
```
