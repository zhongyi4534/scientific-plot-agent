# Scientific Plot Agent

自然语言驱动的科研绘图 Agent 系统。用户用自然语言描述绘图需求，系统通过大语言模型将需求转化为结构化的 PlotSpec JSON，再由工具层渲染为 publication-ready 的科学图表，支持 Nature / IEEE 等主流期刊风格及多种视觉风格。

---

## 目录结构

```
scientific-plot-agent/
├── schema.py              # 全局 Schema 定义（唯一共享契约）
├── model/
│   └── generator.py       # A线：generate_spec()，当前为 Plan B（DeepSeek API）
├── tools/
│   ├── loader.py          # B线：DataLoader，CSV 解析 + 数据摘要
│   ├── themes.py          # B线：ThemeConfig 静态风格注册表
│   ├── layout.py          # B线：LayoutEngine，数据驱动的动态布局计算
│   └── renderer.py        # B线：PlotRenderer，5 种图表类型完整实现
├── system/
│   ├── validator.py       # C线：PlotSpec 合法性校验
│   ├── merger.py          # C线：多轮 delta 合并 + 默认值填充
│   └── agent.py           # C线：PlotAgent 主循环
├── ui/
│   └── app.py             # C线：Gradio 界面
├── data/
│   ├── example_bar.csv    # 示例数据：6 模型 × 4 数据集准确率
│   ├── example_line.csv   # 示例数据：训练曲线
│   └── placeholder.png    # 占位图（历史保留）
├── tests/                 # 单元测试 + 集成测试
├── conftest.py            # pytest 全局配置
├── .env                   # API Key 配置（不提交，见 .gitignore）
├── output/                # 生成图表（.gitignore 忽略）
└── requirements.txt
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

在项目根目录创建 `.env` 文件，填入 DeepSeek API Key：

```
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. 启动

```bash
python ui/app.py
```

浏览器打开 `http://localhost:7860`，上传 `data/example_bar.csv`，然后输入：

> 画一张柱状图，对比各模型在不同数据集上的准确率，使用 nature 风格，按数据集分组

---

## 三条开发线接口说明

### A线（`model/generator.py`）

**当前状态**：Plan B 实现——通过 DeepSeek API（OpenAI 兼容接口）将用户意图转化为 PlotSpec JSON。未来可替换为本地 LoRA 微调模型，接口签名不变。

```python
def generate_spec(
    user_input: str,
    data_context: str,
    current_spec: dict | None = None,
) -> dict:
    ...
```

- 首轮（`current_spec=None`）：返回包含所有 `REQUIRED_FIELDS` 的完整 PlotSpec dict。
- 修改轮：返回仅含变更字段的 delta dict，由 `merger.py` 合并。
- **只替换函数体，不修改签名。**
- 调试：设置 `DEBUG = True`（默认开启）可在终端打印每次 API 的原始响应。

### B线（`tools/renderer.py`）

**当前状态**：5 种图表类型（bar / line / scatter / box / heatmap）已完整实现。

```python
def render_plot(spec: dict, data_source: str) -> str: ...
```

通过 `tools/themes.py` 的 `apply_theme()` 获取静态 `ThemeConfig`（风格），再由 `tools/layout.py` 的 `compute_layout()` 根据数据规模计算动态 `LayoutParams`（尺寸/图例/刻度旋转），最后按 spec 参数渲染对应图表，输出 PNG 到 `output/` 目录。

### C线（`system/agent.py` + `ui/app.py`）

**当前状态**：Agent 主循环、校验、delta 合并、Gradio UI 均已完整实现。

UI 功能：
- 上传 CSV → 展示数据摘要
- 自然语言输入 → 调用 LLM → 渲染图表
- 底部 PlotSpec 编辑区：**可直接修改 JSON 字段，点击「重新渲染」即时生效**（不经过 LLM）
- 重置按钮清空所有状态

---

## PlotSpec 字段说明

### 必填字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `chart_type` | str | 图表类型：`bar` / `line` / `scatter` / `box` / `heatmap` |
| `data_source` | str | 数据缓存 key（`cache://xxxxxxxx`），由 DataLoader 生成 |
| `data_x` | str | X 轴列名（字符串，非列的值，只能是单列） |
| `data_y` | str 或 list[str] | Y 轴列名，单列或多列名列表（非列的值） |
| `style_theme` | str | 风格主题：`clean` / `vivid` / `nature` / `ieee` / `morandi` / `dark` |

### 可选字段（含默认值）

| 字段 | 默认值 | 说明 | 适用 |
|------|--------|------|------|
| `data_group_by` | `null` | 分组列名，用于多系列图表 | 全部 |
| `data_error` | `null` | 误差棒列名 | bar, line |
| `data_filter` | `null` | 行过滤条件（pandas query 语法） | 全部 |
| `label_title` | `""` | 图表标题 | 全部 |
| `label_x` / `label_y` | `""` | 坐标轴标签 | 全部 |
| `axes_y_min` / `axes_y_max` | `null` | Y 轴范围 | 全部 |
| `axes_x_tick_rotation` | `0` | X 轴刻度旋转角度 | 全部 |
| `axes_y_scale` | `"linear"` | Y 轴缩放：`"linear"` / `"log"`（水平柱状图自动映射到 X 轴） | 全部（heatmap 除外） |
| `style_palette_override` | `null` | 配色覆盖：`morandi` / `nature_d` / `tab10` / `coolwarm` | 全部 |
| `params_orientation` | `"vertical"` | 柱方向：`"vertical"` / `"horizontal"` | bar |
| `params_stacked` | `false` | 堆叠柱状图 | bar |
| `params_sort` | `null` | 排序：`"asc"` / `"desc"` | bar |
| `params_show_values` | `false` | 柱顶显示数值 | bar |
| `params_hatch` | `null` | 柱子纹理：`"/"` `"\\"` `"\|"` `"-"` 等 | bar |
| `params_edgecolor` | `null` | 纹理边框颜色 | bar |
| `params_hatch_linewidth` | `0.5` | 纹理线宽 | bar |
| `params_show_markers` | `true` | 是否显示数据点标记（布尔值） | line |
| `params_smooth` | `false` | 平滑曲线 | line |
| `params_linestyle` | `"solid"` | 线型：`"solid"` / `"dashed"` / `"dotted"` / `"dashdot"` | line |
| `params_line_colors` | `null` | 自定义颜色列表，覆盖主题配色 | line |
| `params_marker_style` | `null` | 标记样式：`"o"` `"s"` `"^"` `"D"` 等 | line, scatter |
| `params_marker_size` | `4` | 标记大小 | line, scatter |
| `params_alpha` | `0.8` | 透明度 | scatter |
| `params_show_regression` | `false` | 显示回归线 | scatter |
| `params_show_points` | `"outliers"` | 数据点显示：`"all"` / `"outliers"` / `"none"` | box |
| `params_notch` | `false` | 缺口箱线图 | box |
| `params_annot` | `true` | 显示数值标注 | heatmap |
| `params_annot_fmt` | `".2f"` | 单元格数值格式字符串（与 `params_annot` 配套使用） | heatmap |

---

## 运行测试

```bash
# 全量测试（含真实 API 调用，需设置 DEEPSEEK_API_KEY）
pytest tests/ -v

# 跳过 API 测试（无 key 时）
pytest tests/ -v -k "not (first_round or delta_round or pipeline)"
```
