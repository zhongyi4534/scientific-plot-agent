"""
B线工具：主题与调色板注册表。
提供 ThemeConfig dataclass、THEMES 字典、PALETTES 字典及 apply_theme() 函数。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import matplotlib.pyplot as plt

from schema import STYLE_THEMES, PALETTE_OVERRIDES


@dataclass
class ThemeConfig:
    """单个主题的完整排版与配色配置（静态，数据无关）。"""

    font_family: str
    font_size: int          # 标准图幅下的基准字号（磅），LayoutEngine 按图幅缩放后得实际字号
    line_width: float
    figure_width: float     # 最小基准图幅宽度（英寸），LayoutEngine 只扩大不缩小
    aspect_ratio: float     # 宽高比：figure_height = figure_width × aspect_ratio
    dpi: int
    spines: list[str]
    grid: bool
    grid_style: str
    legend_frameon: bool
    palette: list[str]
    bg_color: str = "white"         # 背景色；dark 主题使用深色
    text_color: str = "black"       # 文字/刻度/轴脊颜色；dark 主题使用浅色
    hatch: list[str] | None = None  # 柱子纹理序列（None=不使用）；多组时按索引轮换；仅 bar 图生效
    edgecolor: str | None = None    # 柱子/纹理边框颜色（None=matplotlib默认）；仅 bar 图生效
    hatch_linewidth: float = 0.5    # 纹理线宽；仅 hatch 不为 None 时生效


def _tab10_colors() -> list[str]:
    """从 matplotlib tab10 colormap 提取十六进制颜色列表。"""
    cmap = plt.cm.tab10
    return [
        "#{:02x}{:02x}{:02x}".format(
            int(cmap(i)[0] * 255),
            int(cmap(i)[1] * 255),
            int(cmap(i)[2] * 255),
        )
        for i in range(10)
    ]


THEMES: dict[str, ThemeConfig] = {
    "nature": ThemeConfig(
        font_family="Arial",
        font_size=7,
        line_width=0.75,
        figure_width=3.5,
        aspect_ratio=0.75,      # 原 3.5 × 2.625
        dpi=300,
        spines=["left", "bottom"],
        grid=False,
        grid_style="--",
        legend_frameon=False,
        palette=["#E64B35", "#F39B7F", "#4DBBD5", "#91D1C2", "#00A087", "#8491B4", "#3C5488"],
    ),
    "ieee": ThemeConfig(
        font_family="Times New Roman",
        font_size=8,
        line_width=0.5,
        figure_width=3.5,
        aspect_ratio=0.75,      # 原 3.5 × 2.625
        dpi=300,
        spines=["left", "bottom"],
        grid=True,
        grid_style="--",
        legend_frameon=False,
        palette=["#d62728", "#e377c2", "#9467bd", "#1f77b4", "#2ca02c", "#ff7f0e", "#8c564b"],
    ),
    "vivid": ThemeConfig(
        font_family="DejaVu Sans",
        font_size=10,
        line_width=1.5,
        figure_width=6.0,
        aspect_ratio=0.667,     # 原 6.0 × 4.0
        dpi=150,
        spines=["left", "bottom"],
        grid=True,
        grid_style="--",
        legend_frameon=True,
        palette=["#E63946", "#FF5722", "#FF9800", "#4CAF50", "#00BCD4", "#2196F3", "#9C27B0"],
    ),
    "morandi": ThemeConfig(
        font_family="Arial",
        font_size=11,
        line_width=1.0,
        figure_width=5.5,
        aspect_ratio=4.0 / 5.5,        # ≈0.7273
        dpi=300,
        spines=["left", "bottom"],
        grid=False,
        grid_style="--",
        legend_frameon=True,
        palette=["#C4A882", "#A89888", "#9CAF88", "#88A0A8", "#8B9BAB", "#B89BAD", "#C4B8A8"],
        bg_color="#FFFFFF",
        text_color="#333333",
    ),
    "clean": ThemeConfig(
        font_family="DejaVu Sans",
        font_size=10,
        line_width=1.0,
        figure_width=6.0,
        aspect_ratio=0.667,     # 原 6.0 × 4.0
        dpi=150,
        spines=["left", "bottom"],
        grid=False,
        grid_style="--",
        legend_frameon=True,
        palette=["#A45C7A", "#7C5CA4", "#5C85A4", "#5CA48C", "#7AA45C", "#A4A45C", "#A4785C"],
    ),
    "dark": ThemeConfig(
        font_family="DejaVu Sans",
        font_size=11,
        line_width=1.5,
        figure_width=7.0,
        aspect_ratio=0.667,     # 原 7.0 × 4.5，按设计规范取 0.667
        dpi=150,
        spines=["left", "bottom"],
        grid=True,
        grid_style="--",
        legend_frameon=True,
        palette=["#FB7185", "#F97316", "#F7B731", "#A3E635", "#34D399", "#61DAFB", "#C084FC"],
        bg_color="#1e1e2e",
        text_color="#cdd6f4",
    ),
    "macaron": ThemeConfig(
        font_family="DejaVu Sans",
        font_size=11,
        line_width=1.2,
        figure_width=5.0,
        aspect_ratio=4.0 / 5.0,        # 旧 figure_height=4.0 ÷ 5.0 = 0.8
        dpi=300,
        spines=["top", "bottom", "left", "right"],
        grid=True,
        grid_style=":",
        legend_frameon=True,
        palette=["#FCC2C6", "#FCE07C", "#F9E4BC", "#CBE7B9", "#B1E0E9", "#D6C9E0"],
        bg_color="#FDFBF7",
        text_color="#4A3B32",
    ),
    "mondrian": ThemeConfig(
        font_family="Times New Roman",
        font_size=12,
        line_width=1.5,
        figure_width=6.0,
        aspect_ratio=4.5 / 6.0,        # = 0.75
        dpi=300,
        spines=["left", "bottom"],
        grid=False,
        grid_style="--",
        legend_frameon=False,
        palette=["#E42D27", "#FFD100", "#F5F5F0", "#0F478C", "#1E1E1E"]
,
        bg_color="#FFFFFF",
        text_color="#000000",
    ),
        # 孟菲斯：高饱和撞色
    "memphis": ThemeConfig(
        font_family="DejaVu Sans",
        font_size=10,
        line_width=1.2,
        figure_width=6.0,
        aspect_ratio=0.667,          # 6×4
        dpi=150,
        spines=["left", "bottom"],
        grid=True,
        grid_style="-",
        legend_frameon=True,
        palette=["#FF71CE", "#B967FF", "#00C2BA", "#84E5D2", "#FFCE5C", "#FF8C42"],
        bg_color="#FFFFFF",
        text_color="#000000",
    ),

    # 洛可可：柔和浅色 + 金边（沿用一般图幅）
    "rococo": ThemeConfig(
        font_family="Arial",
        font_size=10,
        line_width=0.9,
        figure_width=5.5,
        aspect_ratio=0.75,           # 4:3
        dpi=300,
        spines=["left", "bottom"],
        grid=False,
        grid_style="--",
        legend_frameon=True,
        palette=["#F5DDD6", "#E5C7A1", "#F9F6EE", "#B9D4CF", "#B4CFD4", "#D3CCDC"],
        bg_color="#FFFFFF",
        text_color="#3E3A39",
    ),

    # 敦煌：浓郁矿物色
    "dunhuang": ThemeConfig(
        font_family="Times New Roman",
        font_size=10,
        line_width=1.0,
        figure_width=5.0,
        aspect_ratio=0.75,
        dpi=300,
        spines=["left", "bottom"],
        grid=False,
        grid_style="--",
        legend_frameon=False,
        palette=["#B44138", "#863A35", "#A6643A", "#C28B55", "#52958B", "#2F5D8C"],
        bg_color="#FEF8E7",
        text_color="#2C2B28",
    ),

    # Lo-Fi：高对比复古色
    "lofi": ThemeConfig(
        font_family="DejaVu Sans",
        font_size=11,
        line_width=1.5,
        figure_width=6.0,
        aspect_ratio=0.667,
        dpi=150,
        spines=["left", "bottom", "top", "right"],
        grid=True,
        grid_style=":",
        legend_frameon=True,
        palette=["#8D0246", "#700962", "#00356F", "#D7FFFF", "#006F46", "#DDCA8D"],
        bg_color="#FAFAFA",
        text_color="#1A1A1A",
    ),

    # Science 蓝调：5 色科学风格
    "science_blue": ThemeConfig(
        font_family="Arial",
        font_size=9,
        line_width=0.8,
        figure_width=4.5,
        aspect_ratio=0.75,
        dpi=300,
        spines=["left", "bottom"],
        grid=False,
        grid_style="--",
        legend_frameon=True,
        palette=["#0D497F", "#4193C5", "#6AADD7", "#9DCAE1", "#E0EAF6"],
        bg_color="#FFFFFF",
        text_color="#111111",
    ),

    # Nature 散色（nature 变种，活泼散点配色）
    "nature_scatter": ThemeConfig(
        font_family="Arial",
        font_size=8,
        line_width=0.8,
        figure_width=4.0,
        aspect_ratio=0.75,
        dpi=300,
        spines=["left", "bottom"],
        grid=False,
        grid_style="--",
        legend_frameon=False,
        palette=["#F7C651", "#F69A29", "#EAAEB1", "#F74043", "#75717C", "#5A7892", "#44A98B"],
        bg_color="#FFFFFF",
        text_color="#222222",
    ),


    "morandi_light": ThemeConfig(
        font_family="Arial",
        font_size=10,
        line_width=1.0,
        figure_width=5.5,
        aspect_ratio=0.75,
        dpi=300,
        spines=["left", "bottom"],
        grid=False,
        grid_style="--",
        legend_frameon=True,
        palette=["#E36E5D", "#E9B198", "#FBE8D5", "#F4C756", "#9EC4BE", "#4D9D95", "#ABD0F1", "#DCE9F4"],
        bg_color="#FEF9F0",
        text_color="#3C2E2A",
    ),
}

# 配色覆盖注册表；"coolwarm" 是字符串，供 heatmap cmap 参数使用
PALETTES: dict[str, list[str] | str] = {
    "morandi":  ["#8B9BAB", "#C4A882", "#9CAF88", "#B89BAD", "#A89888"],
    "nature_d": ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F"],
    "tab10":    _tab10_colors(),
    "coolwarm": "coolwarm",
}


def apply_theme(
    theme_name: str,
    palette_override: str | None = None,
) -> ThemeConfig:
    """
    返回指定主题的 ThemeConfig；若有 palette_override 则替换调色板字段。

    Args:
        theme_name:       STYLE_THEMES 中的主题名称。
        palette_override: PALETTE_OVERRIDES 中的覆盖名称，或 None。

    Returns:
        ThemeConfig 实例（新对象，不修改注册表中的原始配置）。

    Raises:
        ValueError: 主题名或覆盖名不在合法枚举内时。
    """
    if theme_name not in STYLE_THEMES:
        raise ValueError(f"未知主题：{theme_name}，合法值：{STYLE_THEMES}")

    base = THEMES[theme_name]

    if palette_override is None:
        return dataclasses.replace(base)

    if palette_override not in PALETTE_OVERRIDES:
        raise ValueError(f"未知配色覆盖：{palette_override}，合法值：{PALETTE_OVERRIDES}")

    override_palette = PALETTES[palette_override]
    # coolwarm 是字符串（heatmap 专用），不能直接赋给 palette list
    new_palette = override_palette if isinstance(override_palette, list) else base.palette

    return dataclasses.replace(base, palette=new_palette)


# spec 字段名 → ThemeConfig 字段名的映射（style_* 覆写通道）
_STYLE_OVERRIDE_MAP: dict[str, str] = {
    "style_grid":           "grid",
    "style_line_width":     "line_width",
    "style_font_size":      "font_size",
    "style_hatch":          "hatch",
    "style_edgecolor":      "edgecolor",
    "style_hatch_linewidth": "hatch_linewidth",
    "style_dpi":            "dpi",
    "style_legend_frameon": "legend_frameon",
    "style_bg_color":       "bg_color",
    "style_text_color":     "text_color",
    "style_aspect_ratio":   "aspect_ratio",
    "style_figure_width":   "figure_width",
    "style_font_family":    "font_family",
    "style_spines":         "spines",
}


def apply_style_overrides(theme: ThemeConfig, spec: dict) -> ThemeConfig:
    """
    将 spec 中的 style_* 覆写字段应用到 ThemeConfig，返回新实例。
    值为 None 的字段跳过（保留主题默认）。
    在 apply_theme() 之后调用。
    style_hatch 支持单个字符串或字符串列表，统一转为 list[str] 存入 ThemeConfig。
    """
    overrides: dict = {}
    for spec_key, theme_key in _STYLE_OVERRIDE_MAP.items():
        val = spec.get(spec_key)
        if val is None:
            continue
        if theme_key == "hatch":
            overrides[theme_key] = [val] if isinstance(val, str) else list(val)
        else:
            overrides[theme_key] = val
    return dataclasses.replace(theme, **overrides) if overrides else theme
