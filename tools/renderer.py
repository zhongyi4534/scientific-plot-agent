"""
B线工具：PlotRenderer
接收 PlotSpec 和数据源，输出 PNG 图表文件。
当前为 Mock 实现，B线完成后替换各子渲染器函数体，render_plot 接口签名不变。
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tools.loader import get_dataframe, load_data
from tools.themes import ThemeConfig, apply_theme

# 中文字体缺失会触发此 warning，属正常 fallback 行为，过滤掉
warnings.filterwarnings("ignore", message=r"Glyph \d+ .* missing from font")

OUTPUT_DIR = Path("output")


class RenderError(Exception):
    """图表渲染失败时抛出。"""


# ---------------------------------------------------------------------------
# 公共辅助函数
# ---------------------------------------------------------------------------

def _prepare_axes(spec: dict, theme: ThemeConfig) -> tuple[plt.Figure, plt.Axes]:
    """创建 Figure/Axes 并设置标题、轴标签。"""
    fig, ax = plt.subplots(figsize=(theme.figure_width, theme.figure_height))
    ax.set_title(spec.get("label_title", ""), fontsize=theme.font_size + 1)
    ax.set_xlabel(spec.get("label_x", ""), fontsize=theme.font_size)
    ax.set_ylabel(spec.get("label_y", ""), fontsize=theme.font_size)
    return fig, ax


def _apply_axis_limits(ax: plt.Axes, spec: dict, horizontal: bool = False) -> None:
    """应用坐标轴范围限制。"""
    y_min = spec.get("axes_y_min")
    y_max = spec.get("axes_y_max")
    if horizontal:
        ax.set_xlim(left=y_min, right=y_max)
    else:
        ax.set_ylim(bottom=y_min, top=y_max)


def _add_bar_values(ax: plt.Axes, bars, horizontal: bool, theme: ThemeConfig) -> None:
    """为柱状图添加数值标签。"""
    for bar in bars:
        if horizontal:
            value = bar.get_width()
            ax.text(value, bar.get_y() + bar.get_height() / 2,
                    f"{value:.1f}", va="center", ha="left", fontsize=theme.font_size)
        else:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value,
                    f"{value:.1f}", ha="center", va="bottom", fontsize=theme.font_size)


def _apply_theme_to_fig(fig: plt.Figure, ax: plt.Axes, theme: ThemeConfig) -> None:
    """将 ThemeConfig 的排版参数应用到 Figure/Axes。所有子渲染器在 return fig 前必须调用。"""
    mpl.rcParams["font.size"] = theme.font_size
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(side in theme.spines)
        if side in theme.spines:
            ax.spines[side].set_linewidth(theme.line_width)
    if theme.grid:
        ax.grid(True, linestyle=theme.grid_style, linewidth=theme.line_width * 0.5, alpha=0.7)
    else:
        ax.grid(False)
    fig.set_size_inches(theme.figure_width, theme.figure_height)
    ax.tick_params(direction="in", width=theme.line_width, labelsize=theme.font_size)
    fig.patch.set_facecolor(theme.bg_color)
    ax.set_facecolor(theme.bg_color)
    if theme.bg_color != "white":
        for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
            item.set_color(theme.text_color)
        ax.tick_params(colors=theme.text_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(theme.text_color)


# ---------------------------------------------------------------------------
# 主渲染入口
# ---------------------------------------------------------------------------

def render_plot(spec: dict, data_source: str) -> str:
    """
    主渲染入口。

    Args:
        spec:        已通过校验且 optional 字段已填充默认值的 PlotSpec dict。
        data_source: CSV 文件路径（首次加载）或 cache_key（已缓存）。

    Returns:
        生成图表的 PNG 文件路径字符串。

    Raises:
        RenderError: 渲染过程中出现任何错误时。
    """
    # Mock实现，B线替换
    try:
        df = _resolve_dataframe(data_source, spec)
        theme = apply_theme(spec["style_theme"], spec.get("style_palette_override"))
        # Microsoft YaHei 放首位：同时覆盖中文和西文，不依赖 fallback 机制
        # 主题字体（Arial/Times New Roman）排后，作为在 YaHei 缺字时的备用
        mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", theme.font_family, "SimHei", "DejaVu Sans"]
        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["axes.unicode_minus"] = False
        chart_type = spec["chart_type"]
        if chart_type not in RENDERERS:
            raise RenderError(f"不支持的图表类型：{chart_type}")
        fig = RENDERERS[chart_type](df=df, spec=spec, theme=theme)
        out_path = _output_path(chart_type)
        fig.savefig(out_path, dpi=theme.dpi, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)
    except RenderError:
        raise
    except Exception as exc:
        raise RenderError(f"渲染失败：{exc}") from exc


# ---------------------------------------------------------------------------
# 子渲染器
# ---------------------------------------------------------------------------

def _render_bar(df: pd.DataFrame, spec: dict, theme: ThemeConfig) -> plt.Figure:
    """渲染柱状图。Mock实现，B线替换"""
    x_col = spec["data_x"]
    y_col = spec["data_y"]
    group_col = spec.get("data_group_by")
    err_col = spec.get("data_error")
    horizontal = spec.get("params_orientation", "vertical") == "horizontal"
    stacked = spec.get("params_stacked", False)
    sort_mode = spec.get("params_sort")
    show_values = spec.get("params_show_values", False)
    hatch = spec.get("params_hatch")
    edgecolor = spec.get("params_edgecolor") or "none"
    if hatch:
        mpl.rcParams["hatch.linewidth"] = spec.get("params_hatch_linewidth", 0.5)

    # data_y 是列表时 melt 为长表，复用分组逻辑；多指标下排序和误差棒无意义
    if isinstance(y_col, list):
        df = df.melt(id_vars=[x_col], value_vars=y_col,
                     var_name="_metric", value_name="_value")
        group_col, y_col, err_col, sort_mode = "_metric", "_value", None, None

    if sort_mode == "asc":
        df = df.sort_values(y_col)
    elif sort_mode == "desc":
        df = df.sort_values(y_col, ascending=False)

    fig, ax = _prepare_axes(spec, theme)
    bars = None

    if group_col is None:
        kw = dict(color=theme.palette[0], linewidth=theme.line_width,
                  capsize=3, hatch=hatch, edgecolor=edgecolor)
        if horizontal:
            bars = ax.barh(df[x_col], df[y_col],
                           xerr=df[err_col] if err_col else None, **kw)
        else:
            bars = ax.bar(df[x_col], df[y_col],
                          yerr=df[err_col] if err_col else None, **kw)
    else:
        pivot = df.pivot(index=x_col, columns=group_col, values=y_col)
        x = np.arange(len(pivot.index))
        groups = list(pivot.columns)

        if stacked:
            acc = np.zeros(len(pivot.index))
            for i, g in enumerate(groups):
                values = pivot[g].values
                color = theme.palette[i % len(theme.palette)]
                kw = dict(color=color, linewidth=theme.line_width,
                          label=str(g), hatch=hatch, edgecolor=edgecolor)
                bars = ax.barh(pivot.index, values, left=acc, **kw) if horizontal \
                    else ax.bar(x, values, bottom=acc, **kw)
                acc += values
        else:
            width = 0.8 / len(groups)
            for i, g in enumerate(groups):
                values = pivot[g].values
                offset = (i - len(groups) / 2) * width + width / 2
                color = theme.palette[i % len(theme.palette)]
                kw = dict(color=color, linewidth=theme.line_width,
                          label=str(g), hatch=hatch, edgecolor=edgecolor)
                if horizontal:
                    bars = ax.barh(x + offset, values, height=width, **kw)
                    ax.set_yticks(x)
                    ax.set_yticklabels(pivot.index)
                else:
                    bars = ax.bar(x + offset, values, width=width, **kw)
                    ax.set_xticks(x)
                    ax.set_xticklabels(pivot.index)

        ax.legend(frameon=theme.legend_frameon, fontsize=theme.legend_fontsize)

    if show_values and bars is not None:
        _add_bar_values(ax=ax, bars=bars, horizontal=horizontal, theme=theme)

    _apply_axis_limits(ax, spec, horizontal)
    plt.xticks(rotation=spec.get("axes_x_tick_rotation", 0))
    _apply_theme_to_fig(fig, ax, theme)
    return fig


def _render_line(df: pd.DataFrame, spec: dict, theme: ThemeConfig) -> plt.Figure:
    """渲染折线图。Mock实现，B线替换"""
    x_col = spec["data_x"]
    y_spec = spec["data_y"]
    group_col = spec.get("data_group_by")
    err_col = spec.get("data_error")
    linestyle = spec.get("params_linestyle", "solid")
    line_colors = spec.get("params_line_colors")
    palette = line_colors if line_colors else theme.palette
    use_markers = spec.get("params_markers", True)
    marker = (spec.get("params_marker_style") or "o") if use_markers else None
    msize = spec.get("params_marker_size", 4)
    smooth = spec.get("params_smooth", False)

    def _smooth(x, y):
        if not smooth:
            return x, y
        try:
            from scipy.interpolate import make_interp_spline
            xa, ya = np.asarray(x), np.asarray(y)
            xn = np.linspace(xa.min(), xa.max(), 300)
            return xn, make_interp_spline(xa, ya)(xn)
        except Exception:
            return x, y

    fig, ax = _prepare_axes(spec, theme)

    plot_kw = dict(linestyle=linestyle, linewidth=theme.line_width,
                   marker=marker, markersize=msize)

    if isinstance(y_spec, list):
        for i, y_col in enumerate(y_spec):
            xd, yd = _smooth(df[x_col], df[y_col])
            ax.plot(xd, yd, label=y_col, color=palette[i % len(palette)], **plot_kw)
        ax.legend(frameon=theme.legend_frameon, fontsize=theme.legend_fontsize)
    else:
        y_col = y_spec
        if group_col:
            for i, (gval, gdf) in enumerate(df.groupby(group_col)):
                xd, yd = _smooth(gdf[x_col], gdf[y_col])
                ax.plot(xd, yd, label=str(gval),
                        color=palette[i % len(palette)], **plot_kw)
            ax.legend(frameon=theme.legend_frameon, fontsize=theme.legend_fontsize)
        else:
            xd, yd = _smooth(df[x_col], df[y_col])
            ax.plot(xd, yd, color=palette[0], **plot_kw)
            if err_col:
                ax.fill_between(df[x_col],
                                df[y_col] - df[err_col],
                                df[y_col] + df[err_col],
                                alpha=0.2, color=palette[0])

    _apply_axis_limits(ax, spec)
    _apply_theme_to_fig(fig, ax, theme)
    return fig


def _render_scatter(df: pd.DataFrame, spec: dict, theme: ThemeConfig) -> plt.Figure:
    """渲染散点图。Mock实现，B线替换"""
    x_col = spec["data_x"]
    y_col = spec["data_y"]
    group_col = spec.get("data_group_by")
    alpha = spec.get("params_alpha", 0.8)
    show_regression = spec.get("params_show_regression", False)
    marker = spec.get("params_marker_style") or "o"
    msize = spec.get("params_marker_size", 4)

    fig, ax = _prepare_axes(spec, theme)

    def _draw_regression(x_vals: np.ndarray, y_vals: np.ndarray, color: str) -> None:
        coef = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        ax.plot(x_line, np.poly1d(coef)(x_line), "--",
                color=color, linewidth=theme.line_width)

    if group_col is None:
        ax.scatter(df[x_col], df[y_col], marker=marker, s=msize ** 2,
                   alpha=alpha, color=theme.palette[0], linewidths=theme.line_width)
        if show_regression:
            _draw_regression(df[x_col].values, df[y_col].values,
                             theme.palette[1 % len(theme.palette)])
    else:
        for i, (gval, gdf) in enumerate(df.groupby(group_col)):
            color = theme.palette[i % len(theme.palette)]
            ax.scatter(gdf[x_col], gdf[y_col], marker=marker, s=msize ** 2,
                       alpha=alpha, color=color, linewidths=theme.line_width, label=str(gval))
            if show_regression and len(gdf) >= 2:
                _draw_regression(gdf[x_col].values, gdf[y_col].values, color)
        ax.legend(frameon=theme.legend_frameon, fontsize=theme.legend_fontsize)

    _apply_axis_limits(ax, spec)
    _apply_theme_to_fig(fig, ax, theme)
    return fig


def _render_box(df: pd.DataFrame, spec: dict, theme: ThemeConfig) -> plt.Figure:
    """渲染箱线图。Mock实现，B线替换"""
    x_col = spec["data_x"]
    y_col = spec["data_y"]
    show_points = spec.get("params_show_points", "outliers")
    notch = spec.get("params_notch", False)

    fig, ax = _prepare_axes(spec, theme)
    flierprops = {"marker": ""} if show_points == "none" else {}

    sns.boxplot(data=df, x=x_col, y=y_col, notch=notch,
                palette=theme.palette, linewidth=theme.line_width,
                flierprops=flierprops, ax=ax)

    if show_points == "all":
        sns.stripplot(data=df, x=x_col, y=y_col,
                      color="black", alpha=0.4, size=2, ax=ax)

    _apply_axis_limits(ax, spec)
    _apply_theme_to_fig(fig, ax, theme)
    return fig


def _render_heatmap(df: pd.DataFrame, spec: dict, theme: ThemeConfig) -> plt.Figure:
    """渲染热力图。Mock实现，B线替换"""
    x_col = spec["data_x"]
    y_col = spec["data_y"]

    filter_expr = spec.get("data_filter")
    if filter_expr:
        df = df.query(filter_expr)

    num_cols = [c for c in df.columns
                if c not in (x_col, y_col) and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise RenderError("heatmap 需要至少一个数值列作为热力值")

    pivot = df.pivot(index=y_col, columns=x_col, values=num_cols[0])
    override = spec.get("style_palette_override")
    cmap = "coolwarm" if override == "coolwarm" else "Blues"

    fig, ax = _prepare_axes(spec, theme)
    sns.heatmap(pivot, annot=spec.get("params_annot", True),
                fmt=spec.get("params_fmt", ".2f"),
                cmap=cmap, linewidths=theme.line_width * 0.4, ax=ax)

    _apply_theme_to_fig(fig, ax, theme)
    return fig


# ---------------------------------------------------------------------------
# 渲染器注册表与路径工具
# ---------------------------------------------------------------------------

RENDERERS: dict[str, object] = {
    "bar":     _render_bar,
    "line":    _render_line,
    "scatter": _render_scatter,
    "box":     _render_box,
    "heatmap": _render_heatmap,
}


def _ensure_output_dir() -> None:
    """首次调用时自动创建 output/ 目录。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _output_path(chart_type: str) -> Path:
    """生成带时间戳的输出文件路径。格式：output/plot_20240101_120000_bar.png"""
    _ensure_output_dir()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"plot_{timestamp}_{chart_type}.png"


def _resolve_dataframe(data_source: str, spec: dict) -> pd.DataFrame:
    """根据 data_source 取回 DataFrame（缓存键或 CSV 路径均支持）。"""
    if data_source.startswith("cache://"):
        return get_dataframe(data_source)
    _, cache_key = load_data(data_source)
    return get_dataframe(cache_key)
