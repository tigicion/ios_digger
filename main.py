#!/usr/bin/env python3
"""iOS Digger - App Store Review Mining Tool."""
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from config import REGIONS, CATEGORIES, DEFAULT_REGIONS, DEFAULT_LEVEL, DEFAULT_MODEL, DEFAULT_TOP
from scraper.search import AppSearcher
from scraper.reviews import ReviewScraper
from analyzer.insights import InsightsAnalyzer
from reporter.generator import PDFGenerator

app = typer.Typer(
    name="ios-digger",
    help="App Store 评论挖掘工具 - 从用户评论中提取痛点、好评与需求洞察",
)
console = Console()


@app.command()
def regions():
    """显示支持的地区列表"""
    table = Table(title="支持的地区")
    table.add_column("代码", style="cyan")
    table.add_column("名称", style="green")

    for code, name in REGIONS.items():
        table.add_row(code, name)

    console.print(table)


@app.command()
def categories():
    """显示支持的 App 类别"""
    table = Table(title="支持的类别")
    table.add_column("类别名称", style="cyan")
    table.add_column("ID", style="dim")

    for name, genre_id in CATEGORIES.items():
        table.add_row(name, str(genre_id))

    console.print(table)


@app.command()
def search(
    keyword: str = typer.Argument(..., help="搜索关键词，如：备忘录、健身"),
    regions: str = typer.Option(
        ",".join(DEFAULT_REGIONS),
        "--regions", "-r",
        help="地区列表，逗号分隔，或 'all'",
    ),
    level: str = typer.Option(
        DEFAULT_LEVEL,
        "--level", "-l",
        help="报告级别：brief / standard / full",
    ),
    top: int = typer.Option(
        DEFAULT_TOP,
        "--top", "-t",
        help="每个地区搜索 Top N 个 App",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model", "-m",
        help="LLM 模型：qwen-plus / qwen-turbo / qwen-max",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="输出文件路径（默认自动生成）",
    ),
):
    """按关键词搜索 App 并分析评论"""
    region_list = list(REGIONS.keys()) if regions == "all" else regions.split(",")

    # Validate regions
    for r in region_list:
        if r not in REGIONS:
            console.print(f"[red]错误：未知地区 '{r}'[/red]")
            console.print(f"支持的地区：{', '.join(REGIONS.keys())}")
            raise typer.Exit(1)

    asyncio.run(_search_and_analyze(keyword, region_list, level, top, model, output))


@app.command()
def category(
    category_name: str = typer.Argument(..., help="类别名称，如：效率、健康健身"),
    regions: str = typer.Option(
        ",".join(DEFAULT_REGIONS),
        "--regions", "-r",
        help="地区列表，逗号分隔，或 'all'",
    ),
    level: str = typer.Option(
        DEFAULT_LEVEL,
        "--level", "-l",
        help="报告级别：brief / standard / full",
    ),
    top: int = typer.Option(
        DEFAULT_TOP,
        "--top", "-t",
        help="每个地区获取 Top N 个 App",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model", "-m",
        help="LLM 模型",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="输出文件路径",
    ),
):
    """按类别获取 Top App 并分析评论"""
    if category_name not in CATEGORIES:
        console.print(f"[red]错误：未知类别 '{category_name}'[/red]")
        console.print(f"支持的类别：{', '.join(CATEGORIES.keys())}")
        raise typer.Exit(1)

    region_list = list(REGIONS.keys()) if regions == "all" else regions.split(",")

    asyncio.run(_category_and_analyze(category_name, region_list, level, top, model, output))


@app.command()
def analyze(
    app_ids: str = typer.Argument(..., help="App ID 列表，逗号分隔"),
    regions: str = typer.Option(
        ",".join(DEFAULT_REGIONS),
        "--regions", "-r",
        help="地区列表，逗号分隔，或 'all'",
    ),
    level: str = typer.Option(
        DEFAULT_LEVEL,
        "--level", "-l",
        help="报告级别：brief / standard / full",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model", "-m",
        help="LLM 模型",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="输出文件路径",
    ),
):
    """直接分析指定 App ID 的评论"""
    id_list = [id.strip() for id in app_ids.split(",")]
    region_list = list(REGIONS.keys()) if regions == "all" else regions.split(",")

    asyncio.run(_analyze_apps(id_list, region_list, level, model, output))


async def _search_and_analyze(
    keyword: str,
    regions: list[str],
    level: str,
    top: int,
    model: str,
    output: Optional[str],
):
    """Search apps and analyze reviews."""
    searcher = AppSearcher()

    # Search apps
    console.print(f"\n[bold]搜索 '{keyword}' 相关 App...[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("搜索中...", total=None)
        region_apps = await searcher.search_by_keyword_multi_region(keyword, regions, top)
        progress.update(task, completed=True)

    # Collect unique app IDs
    all_apps = {}
    for region, apps in region_apps.items():
        console.print(f"  {REGIONS[region]}：找到 {len(apps)} 个 App")
        for app in apps:
            if app.id not in all_apps:
                all_apps[app.id] = app

    if not all_apps:
        console.print("[yellow]未找到任何 App[/yellow]")
        return

    app_ids = list(all_apps.keys())
    await _fetch_analyze_report(app_ids, regions, level, model, output, f"{keyword} 相关 App")


async def _category_and_analyze(
    category_name: str,
    regions: list[str],
    level: str,
    top: int,
    model: str,
    output: Optional[str],
):
    """Get top apps in category and analyze."""
    searcher = AppSearcher()

    console.print(f"\n[bold]获取 '{category_name}' 类别 Top {top} App...[/bold]")

    all_apps = {}
    for region in regions:
        try:
            apps = await searcher.get_top_apps_by_category(category_name, region, top)
            console.print(f"  {REGIONS[region]}：找到 {len(apps)} 个 App")
            for app in apps:
                if app.id not in all_apps:
                    all_apps[app.id] = app
        except Exception as e:
            console.print(f"  [yellow]{REGIONS[region]}：获取失败 - {e}[/yellow]")

    if not all_apps:
        console.print("[yellow]未找到任何 App[/yellow]")
        return

    app_ids = list(all_apps.keys())
    await _fetch_analyze_report(app_ids, regions, level, model, output, f"{category_name} 类 Top App")


async def _analyze_apps(
    app_ids: list[str],
    regions: list[str],
    level: str,
    model: str,
    output: Optional[str],
):
    """Analyze specified app IDs."""
    await _fetch_analyze_report(app_ids, regions, level, model, output, f"App 分析")


async def _fetch_analyze_report(
    app_ids: list[str],
    regions: list[str],
    level: str,
    model: str,
    output: Optional[str],
    title: str,
):
    """Fetch reviews, analyze, and generate report."""
    scraper = ReviewScraper()

    # Fetch reviews
    console.print(f"\n[bold]抓取评论中...[/bold]")

    all_reviews = []
    reviews_by_region = {r: [] for r in regions}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("抓取中...", total=len(app_ids) * len(regions))

        for app_id in app_ids:
            region_reviews = await scraper.get_reviews_multi_region(app_id, regions)
            for region, reviews in region_reviews.items():
                all_reviews.extend(reviews)
                reviews_by_region[region].extend(reviews)
            progress.advance(task, len(regions))

    console.print(f"  共抓取 {len(all_reviews)} 条评论")

    if not all_reviews:
        console.print("[yellow]未抓取到任何评论[/yellow]")
        return

    # Analyze
    console.print(f"\n[bold]LLM 分析中...[/bold]")

    analyzer = InsightsAnalyzer(model=model)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("分析中...", total=None)
        result, _ = analyzer.analyze_by_region(reviews_by_region)
        progress.update(task, completed=True)

    # Generate report
    console.print(f"\n[bold]生成报告...[/bold]")

    generator = PDFGenerator()
    output_path = generator.generate_to_file(
        title=title,
        result=result,
        regions=regions,
        level=level,
        output_dir="." if not output else str(Path(output).parent),
    )

    if output:
        import shutil
        shutil.move(output_path, output)
        output_path = output

    console.print(f"  [green]报告已保存: {output_path}[/green]")


if __name__ == "__main__":
    app()
