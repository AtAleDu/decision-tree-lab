"""PNG visualizations for trees and experiment metrics using Pillow."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from .decision_tree import TreeNode


def _font(size: int = 14) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _visible_nodes_by_depth(root: TreeNode, max_depth: int) -> dict[int, list[TreeNode]]:
    levels: dict[int, list[TreeNode]] = {}
    queue: list[TreeNode] = [root]
    while queue:
        node = queue.pop(0)
        relative_depth = node.depth - root.depth
        if relative_depth > max_depth:
            continue
        levels.setdefault(relative_depth, []).append(node)
        if node.left is not None:
            queue.append(node.left)
        if node.right is not None:
            queue.append(node.right)
    return levels


def visualize_tree(root: TreeNode, output_path: str | Path, max_depth: int = 4, title: str = "") -> None:
    """Save a readable PNG with the first max_depth levels of the tree."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    levels = _visible_nodes_by_depth(root, max_depth=max_depth)
    max_nodes = max(len(nodes) for nodes in levels.values())
    node_width = 260
    node_height = 128
    x_gap = 34
    y_gap = 74
    margin = 36
    title_height = 54
    width = max(1000, margin * 2 + max_nodes * node_width + (max_nodes - 1) * x_gap)
    height = title_height + margin * 2 + len(levels) * node_height + (len(levels) - 1) * y_gap

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    main_font = _font(14)
    small_font = _font(12)
    title_font = _font(20)

    if title:
        draw.text((margin, 18), title, fill=(30, 30, 30), font=title_font)

    positions: dict[int, tuple[int, int]] = {}
    for depth, nodes in levels.items():
        row_width = len(nodes) * node_width + (len(nodes) - 1) * x_gap
        start_x = (width - row_width) // 2
        y = title_height + margin + depth * (node_height + y_gap)
        for index, node in enumerate(nodes):
            x = start_x + index * (node_width + x_gap)
            positions[id(node)] = (x + node_width // 2, y + node_height // 2)

    for depth, nodes in levels.items():
        for node in nodes:
            if depth >= max_depth:
                continue
            for child in [node.left, node.right]:
                if child is not None and id(child) in positions:
                    x1, y1 = positions[id(node)]
                    x2, y2 = positions[id(child)]
                    draw.line((x1, y1 + node_height // 2, x2, y2 - node_height // 2), fill=(120, 120, 120), width=2)

    for nodes in levels.values():
        for node in nodes:
            cx, cy = positions[id(node)]
            x = cx - node_width // 2
            y = cy - node_height // 2
            fill = (234, 248, 240) if node.is_leaf else (235, 242, 255)
            outline = (76, 146, 99) if node.is_leaf else (78, 114, 184)
            draw.rounded_rectangle((x, y, x + node_width, y + node_height), radius=8, fill=fill, outline=outline, width=2)

            class_counts = ", ".join(f"{label}: {count}" for label, count in sorted(node.class_counts.items()))
            lines = []
            if node.is_leaf:
                lines.append(f"Leaf -> {node.predicted_class}")
            else:
                lines.extend(_wrap_text(node.condition_text(), 30))
                lines.append(f"score={node.criterion_value:.4f}")
            lines.append(f"n={node.samples_count}")
            lines.extend(_wrap_text(class_counts, 32))

            text_y = y + 12
            for line in lines[:6]:
                draw.text((x + 12, text_y), line, fill=(25, 25, 25), font=main_font if text_y == y + 12 else small_font)
                text_y += 19

    image.save(output_path)


def plot_metrics(metrics_rows: Iterable[dict[str, object]], output_path: str | Path) -> None:
    """Save line chart for accuracy, precision, recall and F1 versus train size."""

    rows = list(metrics_rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1100, 700
    margin_left, margin_right = 92, 42
    margin_top, margin_bottom = 72, 92
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(22)
    font = _font(14)
    small_font = _font(12)

    draw.text((margin_left, 26), "Classification quality vs train size", fill=(30, 30, 30), font=title_font)
    draw.rectangle((margin_left, margin_top, margin_left + plot_width, margin_top + plot_height), outline=(60, 60, 60), width=2)

    for tick in range(0, 11):
        value = tick / 10
        y = margin_top + plot_height - int(value * plot_height)
        draw.line((margin_left, y, margin_left + plot_width, y), fill=(230, 230, 230), width=1)
        draw.text((34, y - 8), f"{value:.1f}", fill=(50, 50, 50), font=small_font)

    train_sizes = [float(row["train_size"]) for row in rows]
    labels = [f"{int(size * 100)}%" for size in train_sizes]
    x_positions = []
    for i, label in enumerate(labels):
        x = margin_left + int(i * plot_width / max(1, len(labels) - 1))
        x_positions.append(x)
        draw.line((x, margin_top + plot_height, x, margin_top + plot_height + 7), fill=(60, 60, 60), width=2)
        draw.text((x - 15, margin_top + plot_height + 16), label, fill=(50, 50, 50), font=font)

    series = [
        ("accuracy", (44, 123, 182)),
        ("precision", (215, 95, 52)),
        ("recall", (76, 160, 83)),
        ("f1", (128, 84, 174)),
    ]
    for name, color in series:
        points = []
        for x, row in zip(x_positions, rows, strict=True):
            value = float(row[name])
            y = margin_top + plot_height - int(value * plot_height)
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=4)
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="white", width=2)

    legend_x = margin_left + plot_width - 170
    legend_y = margin_top + 18
    for index, (name, color) in enumerate(series):
        y = legend_y + index * 25
        draw.line((legend_x, y + 8, legend_x + 28, y + 8), fill=color, width=4)
        draw.text((legend_x + 38, y), name, fill=(30, 30, 30), font=font)

    draw.text((margin_left + plot_width // 2 - 48, height - 42), "Train size", fill=(30, 30, 30), font=font)
    draw.text((18, margin_top + plot_height // 2 - 10), "Metric", fill=(30, 30, 30), font=font)
    image.save(output_path)
