# iOS Digger

App Store 评论挖掘工具 - 从用户评论中提取痛点、好评与需求洞察，生成 PDF 分析报告。

## 功能特性

- **多地区支持**：覆盖中国、美国、日本、韩国、新加坡、泰国、越南、印尼 8 个 App Store 地区
- **多种搜索方式**：关键词搜索、类别浏览、指定 App ID
- **智能分析**：基于 LLM 提取用户痛点、好评亮点、需求洞察
- **PDF 报告**：支持简要 (2-3页)、标准 (5-8页)、完整 (10+页) 三种报告级别
- **地区差异分析**：对比不同地区用户反馈差异

## 安装

### 1. 克隆项目

```bash
git clone <repository-url>
cd ios_digger
```

### 2. 创建虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# 或 .venv\Scripts\activate  # Windows
```

### 3. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 4. 安装系统依赖 (PDF 生成需要)

**macOS:**
```bash
brew install pango gdk-pixbuf libffi
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0
```

**其他系统:** 参考 [WeasyPrint 安装文档](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#installation)

### 5. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的阿里云 DashScope API Key
```

获取 API Key: [阿里云 DashScope](https://dashscope.console.aliyun.com/)

## 使用方法

### 查看帮助

```bash
python main.py --help
```

### 查看支持的地区

```bash
python main.py regions
```

输出:
```
┌────────────────┐
│ 支持的地区     │
├──────┬─────────┤
│ 代码 │ 名称    │
├──────┼─────────┤
│ cn   │ 中国    │
│ us   │ 美国    │
│ jp   │ 日本    │
│ ...  │ ...     │
└──────┴─────────┘
```

### 查看支持的类别

```bash
python main.py categories
```

### 关键词搜索模式

搜索相关 App 并分析评论：

```bash
python main.py search "备忘录" --regions cn,us,jp --level standard
```

参数说明：
- `--regions, -r`: 地区列表，逗号分隔，或 `all` 表示全部地区 (默认: `cn,us`)
- `--level, -l`: 报告级别 `brief` / `standard` / `full` (默认: `standard`)
- `--top, -t`: 每个地区搜索 Top N 个 App (默认: `10`)
- `--model, -m`: LLM 模型 `qwen-turbo` / `qwen-plus` / `qwen-max` (默认: `qwen-plus`)
- `--output, -o`: 输出文件路径 (默认自动生成)

### 类别浏览模式

获取指定类别 Top App 并分析：

```bash
python main.py category "效率" --regions cn,us --top 10 --level brief
```

支持的类别：效率、健康健身、生活、工具、社交、财务、教育、娱乐、摄影与录像、音乐

### 指定 App ID 模式

直接分析指定的 App：

```bash
python main.py analyze "544007664,425424353" --regions all --level full
```

## 输出示例

运行后会在当前目录生成 PDF 报告：

```
report_2026-01-17_143052_备忘录 相关 App.pdf
```

报告包含：
- 概览统计（评论总数、平均评分、评分分布）
- 痛点排名（按严重程度和频率排序）
- 好评亮点
- 用户需求洞察
- 地区差异分析（标准版及以上）
- 产品建议（完整版）

## 项目结构

```
ios_digger/
├── main.py              # CLI 入口
├── config.py            # 配置（地区、类别、API 地址）
├── requirements.txt     # Python 依赖
├── .env.example         # 环境变量模板
│
├── scraper/             # 数据抓取模块
│   ├── models.py        # 数据模型 (App, Review)
│   ├── search.py        # App 搜索 (iTunes API)
│   └── reviews.py       # 评论抓取 (RSS Feed)
│
├── analyzer/            # LLM 分析模块
│   ├── llm_client.py    # Qwen API 客户端
│   └── insights.py      # 洞察提取逻辑
│
├── reporter/            # PDF 报告模块
│   ├── generator.py     # PDF 生成器
│   └── templates/       # HTML 模板
│       ├── base.html
│       ├── report_brief.html
│       ├── report_standard.html
│       ├── report_full.html
│       └── styles.css
│
└── tests/               # 测试
    ├── test_models.py
    ├── test_search.py
    ├── test_reviews.py
    ├── test_llm_client.py
    ├── test_insights.py
    ├── test_generator.py
    ├── test_cli.py
    └── test_integration.py
```

## 开发

### 运行测试

```bash
python -m pytest tests/ -v
```

### 代码结构

| 模块 | 职责 |
|------|------|
| `scraper` | 数据抓取：App 搜索、评论获取 |
| `analyzer` | LLM 分析：批量处理、洞察提取 |
| `reporter` | 报告生成：PDF 渲染、模板管理 |

### API 说明

数据源使用 iTunes 公开 API：
- **搜索 API**: `https://itunes.apple.com/search`
- **评论 RSS**: `https://itunes.apple.com/{region}/rss/customerreviews/id={app_id}/json`
- **排行榜 RSS**: `https://itunes.apple.com/{region}/rss/topfreeapplications/genre={genre_id}/json`

LLM 使用阿里云 Qwen 通过 OpenAI 兼容接口调用。

## 配置说明

### 环境变量

| 变量名 | 必需 | 说明 |
|--------|------|------|
| `DASHSCOPE_API_KEY` | 是 | 阿里云 DashScope API Key |

### LLM 模型选择

| 模型 | 特点 | 适用场景 |
|------|------|----------|
| `qwen-turbo` | 速度快、成本低 | 快速测试、大批量处理 |
| `qwen-plus` | 平衡性价比 | 日常使用（推荐） |
| `qwen-max` | 能力最强 | 复杂分析、高质量报告 |

## 许可证

MIT License
