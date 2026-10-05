#!/usr/bin/env python3
"""Render a Step 8 schedule in the compact vertical trajectory style.

The solver's normal Matplotlib renderer is not available in every runtime, so
this small renderer writes a self-contained SVG.  It intentionally mirrors the
reference chart: bay index on the x-axis, time increasing downward, filled work
cells, dashed relocation arrows, and hollow idle points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def chart_data(payload: dict):
    """Use actual work completion for the chart, not a padded storage horizon."""
    all_slots = payload["slots"]
    completion = max(
        (int(slot["time"]) + 1 for slot in all_slots if slot["state"] == "work"),
        default=0,
    )
    schedule_horizon = int(payload.get(
        "schedule_horizon",
        max((int(slot["time"]) + 1 for slot in all_slots), default=0),
    ))
    move_time = int(payload.get("move_time", 1))
    visible_slots = [slot for slot in all_slots if int(slot["time"]) < completion]
    if move_time == 0:
        by_crane: dict[int, list[dict]] = {}
        for slot in all_slots:
            by_crane.setdefault(int(slot["crane"]), []).append(slot)
        directions: dict[int, list[int]] = {q: [] for q in by_crane}
        moves = 0
        for crane_slots in by_crane.values():
            ordered = sorted(crane_slots, key=lambda item: int(item["time"]))
            for previous, current in zip(ordered, ordered[1:]):
                if int(current["time"]) >= completion:
                    continue
                start, end = float(previous["end_bay"]), float(current["start_bay"])
                if start != end:
                    moves += 1
                    directions[int(current["crane"])].append(1 if end > start else -1)
        reversals = sum(
            left != right
            for values in directions.values()
            for left, right in zip(values, values[1:])
        )
    else:
        events: dict[tuple[int, int], list[dict]] = {}
        singles = []
        for slot in all_slots:
            if slot["state"] != "move":
                continue
            move_id = slot.get("move_id")
            if move_id is None:
                singles.append(slot)
            else:
                events.setdefault((int(slot["crane"]), int(move_id)), []).append(slot)
        starts = [
            min(int(slot["time"]) for slot in group)
            for group in events.values()
        ] + [int(slot["time"]) for slot in singles]
        moves = sum(start < completion for start in starts)
        reversals = int(payload.get("reversal_count", 0))
    cranes = sorted({int(slot["crane"]) for slot in all_slots})
    bays = max(
        (int(float(slot["start_bay"])) for slot in all_slots if slot["state"] != "offrail"),
        default=1,
    )
    source_horizon = int(payload.get(
        "normalization_source_horizon", schedule_horizon
    ))
    trimmed_slots = int(payload.get("normalization_trimmed_slots", 0))
    if schedule_horizon > completion:
        subtitle = (
            f"Step 8 trajectory — actual C={completion}; stored H={schedule_horizon}; "
            f"trailing idle={schedule_horizon - completion}"
        )
    elif source_horizon != schedule_horizon:
        subtitle = (
            f"Step 8 trajectory — actual C={completion}; stored H={schedule_horizon}; "
            f"source H={source_horizon}; trimmed={trimmed_slots}"
        )
    else:
        subtitle = f"Step 8 trajectory — actual completion C={completion}"
    return visible_slots, cranes, bays, completion, schedule_horizon, moves, reversals, subtitle


def esc(value: object) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render(payload: dict, output: Path, *, title: str | None = None) -> None:
    slots, cranes, bays, makespan, schedule_horizon, move_count, reversal_count, subtitle = chart_data(payload)

    # The dimensions preserve the tall, readable aspect of the supplied chart.
    width, height = 1320, 2250
    left, top = 72, 72
    plot_width, plot_height = 1000, 1900
    right = left + plot_width
    bottom = top + plot_height
    bay_step = plot_width / (bays + 1)
    time_step = plot_height / max(1, makespan)

    def x(bay: float) -> float:
        return left + (float(bay) - 0.5) * bay_step

    def y(time: float) -> float:
        return top + float(time) * time_step

    loads = {
        q: sum(1 for slot in slots if int(slot["crane"]) == q and slot["state"] == "work")
        for q in cranes
    }
    split_count = int(payload.get("split_bay_count", 0))
    heading = title or (
        f"CWP schedule — heuristic, completion C={makespan}, reversals {reversal_count}, "
        f"moves {move_count}, split bays {split_count}"
    )
    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" '
        'orient="auto" markerUnits="strokeWidth"><path d="M0,0 L7,3.5 L0,7 z" '
        'fill="context-stroke"/></marker></defs>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>'
        'text{font-family:Arial,"Arial Unicode MS",sans-serif;fill:#222}'
        '.tick{font-size:11px;fill:#555}.cell{font-size:8px;fill:#333}'
        '.axis{font-size:13px;fill:#333}.legend{font-size:12px}.title{font-size:17px;font-weight:600}'
        '</style>',
        f'<text x="{(left + right) / 2:.1f}" y="26" text-anchor="middle" class="title">{esc(heading)}</text>',
        f'<text x="{(left + right) / 2:.1f}" y="47" text-anchor="middle" class="axis">{esc(subtitle)}</text>',
    ]

    # Grid and axes.  Matplotlib's reference uses about twenty time intervals.
    for bay in range(1, bays + 1):
        xx = x(bay)
        lines.append(f'<line x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{bottom}" '
                     'stroke="#e5e5e5" stroke-width="1"/>')
    tick_step = max(1, (makespan + 19) // 20)
    ticks = list(range(0, makespan + 1, tick_step))
    if ticks[-1] != makespan:
        ticks.append(makespan)
    for tick in ticks:
        yy = y(tick)
        lines.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}" '
                     'stroke="#e5e5e5" stroke-width="1"/>')
        lines.append(f'<text x="{left - 10}" y="{yy + 4:.2f}" text-anchor="end" class="tick">{tick}</text>')
    lines.extend([
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#777"/>',
        f'<text x="{(left + right) / 2:.1f}" y="{bottom + 75}" text-anchor="middle" class="axis">贝位序号 / Bay index (1-based)</text>',
        f'<text x="18" y="{(top + bottom) / 2:.1f}" text-anchor="middle" class="axis" '
        f'transform="rotate(-90 18 {(top + bottom) / 2:.1f})">Time</text>',
    ])
    for bay in range(1, bays + 1):
        lines.append(f'<text x="{x(bay):.2f}" y="{bottom + 22}" text-anchor="middle" class="tick">{bay}</text>')

    # Work cells and idle dots.
    cell_width = 0.72 * bay_step
    for slot in sorted(slots, key=lambda item: (int(item["time"]), int(item["crane"]))):
        state = slot["state"]
        if state == "offrail":
            continue
        q = int(slot["crane"])
        color = COLORS[(q - 1) % len(COLORS)]
        xx = x(float(slot["start_bay"]))
        yy = y(int(slot["time"]))
        if state == "work":
            lines.append(f'<rect x="{xx - cell_width / 2:.2f}" y="{yy + 0.06 * time_step:.2f}" '
                         f'width="{cell_width:.2f}" height="{0.88 * time_step:.2f}" fill="{color}" fill-opacity="0.78"/>')
            lines.append(f'<text x="{xx:.2f}" y="{yy + 0.60 * time_step:.2f}" text-anchor="middle" class="cell">Q{q}</text>')
        elif state == "idle":
            lines.append(f'<circle cx="{xx:.2f}" cy="{yy + 0.5 * time_step:.2f}" r="3.1" '
                         f'fill="white" stroke="{color}" stroke-width="1.5"/>')

    # With move_time=0, relocations occur at a period boundary and have no
    # separate slot.  Draw the same dashed boundary arrows as the reference.
    for q in cranes:
        crane_slots = sorted((slot for slot in slots if int(slot["crane"]) == q), key=lambda item: int(item["time"]))
        color = COLORS[(q - 1) % len(COLORS)]
        for previous, current in zip(crane_slots, crane_slots[1:]):
            if previous["state"] == "offrail" or current["state"] == "offrail":
                continue
            if float(previous["end_bay"]) == float(current["start_bay"]):
                continue
            boundary = int(current["time"])
            x1, x2 = x(float(previous["end_bay"])), x(float(current["start_bay"]))
            yy = y(boundary)
            lines.append(f'<line x1="{x1:.2f}" y1="{yy - 1:.2f}" x2="{x2:.2f}" y2="{yy + 1:.2f}" '
                         f'stroke="{color}" stroke-width="1.8" stroke-dasharray="6,3" marker-end="url(#arrow)"/>')

    # Legends at the right, matching the reference placement.
    legend_x, legend_y, legend_w = right + 18, top + 8, 210
    crane_h = 28 + 24 * len(cranes)
    lines.extend([
        f'<rect x="{legend_x}" y="{legend_y}" width="{legend_w}" height="{crane_h}" fill="white" stroke="#d0d0d0"/>',
        f'<text x="{legend_x + legend_w / 2}" y="{legend_y + 20}" text-anchor="middle" class="legend">Cranes</text>',
    ])
    for index, q in enumerate(cranes):
        yy = legend_y + 43 + index * 24
        color = COLORS[(q - 1) % len(COLORS)]
        lines.append(f'<line x1="{legend_x + 10}" y1="{yy - 4}" x2="{legend_x + 42}" y2="{yy - 4}" stroke="{color}" stroke-width="5"/>')
        lines.append(f'<text x="{legend_x + 50}" y="{yy}" class="legend">Q{q} load={loads[q]}</text>')

    state_y = bottom - 164
    lines.extend([
        f'<rect x="{legend_x}" y="{state_y}" width="{legend_w}" height="156" fill="white" stroke="#d0d0d0"/>',
        f'<text x="{legend_x + legend_w / 2}" y="{state_y + 20}" text-anchor="middle" class="legend">States</text>',
        f'<rect x="{legend_x + 13}" y="{state_y + 36}" width="18" height="12" fill="#777" fill-opacity="0.78"/>',
        f'<text x="{legend_x + 40}" y="{state_y + 47}" class="legend">work: filled cell</text>',
        f'<line x1="{legend_x + 13}" y1="{state_y + 72}" x2="{legend_x + 35}" y2="{state_y + 72}" stroke="#555" stroke-width="1.8" stroke-dasharray="6,3" marker-end="url(#arrow)"/>',
        f'<text x="{legend_x + 40}" y="{state_y + 76}" class="legend">move: dashed arrow</text>',
        f'<circle cx="{legend_x + 22}" cy="{state_y + 108}" r="4" fill="white" stroke="#777" stroke-width="1.5"/>',
        f'<text x="{legend_x + 40}" y="{state_y + 112}" class="legend">idle: hollow dot</text>',
    ])

    lines.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def render_png(payload: dict, output: Path, *, title: str | None = None) -> None:
    """Raster counterpart for clients that do not preview local SVG files."""
    from PIL import Image, ImageDraw, ImageFont

    slots, cranes, bays, makespan, schedule_horizon, move_count, reversal_count, subtitle = chart_data(payload)
    width, height = 1320, 2250
    left, top = 72, 72
    plot_width, plot_height = 1000, 1900
    right, bottom = left + plot_width, top + plot_height
    bay_step = plot_width / (bays + 1)
    time_step = plot_height / max(1, makespan)

    def x(bay: float) -> float:
        return left + (float(bay) - 0.5) * bay_step

    def y(time: float) -> float:
        return top + float(time) * time_step

    font_path = "/Library/Fonts/Arial Unicode.ttf"
    try:
        title_font = ImageFont.truetype(font_path, 17)
        axis_font = ImageFont.truetype(font_path, 13)
        tick_font = ImageFont.truetype(font_path, 11)
        cell_font = ImageFont.truetype(font_path, 8)
        legend_font = ImageFont.truetype(font_path, 12)
    except OSError:
        title_font = axis_font = tick_font = cell_font = legend_font = ImageFont.load_default()

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    heading = title or (
        f"CWP schedule — heuristic, completion C={makespan}, reversals {reversal_count}, "
        f"moves {move_count}, split bays {payload.get('split_bay_count', 0)}"
    )

    def centered(text: str, xy: tuple[float, float], font: ImageFont.FreeTypeFont, fill: str = "#222") -> None:
        box = draw.textbbox((0, 0), text, font=font)
        draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2), text, font=font, fill=fill)

    centered(heading, ((left + right) / 2, 25), title_font)
    centered(subtitle, ((left + right) / 2, 47), axis_font)
    for bay in range(1, bays + 1):
        xx = int(round(x(bay)))
        draw.line((xx, top, xx, bottom), fill="#e5e5e5", width=1)
    tick_step = max(1, (makespan + 19) // 20)
    ticks = list(range(0, makespan + 1, tick_step))
    if ticks[-1] != makespan:
        ticks.append(makespan)
    for tick in ticks:
        yy = int(round(y(tick)))
        draw.line((left, yy, right, yy), fill="#e5e5e5", width=1)
        box = draw.textbbox((0, 0), str(tick), font=tick_font)
        draw.text((left - 10 - (box[2] - box[0]), yy - (box[3] - box[1]) / 2), str(tick), font=tick_font, fill="#555")
    draw.rectangle((left, top, right, bottom), outline="#777", width=1)
    centered("贝位序号 / Bay index (1-based)", ((left + right) / 2, bottom + 75), axis_font)
    # Rotate the y-axis label after drawing it on a small transparent layer.
    label_layer = Image.new("RGBA", (100, 100), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_layer)
    label_draw.text((50, 50), "Time", font=axis_font, fill="#333", anchor="mm")
    label_layer = label_layer.rotate(90, expand=True)
    image.paste(label_layer, (0, int((top + bottom) / 2 - label_layer.height / 2)), label_layer)
    for bay in range(1, bays + 1):
        centered(str(bay), (x(bay), bottom + 22), tick_font, "#555")

    cell_width = 0.72 * bay_step
    for slot in sorted(slots, key=lambda item: (int(item["time"]), int(item["crane"]))):
        state = slot["state"]
        if state == "offrail":
            continue
        q = int(slot["crane"])
        color = COLORS[(q - 1) % len(COLORS)]
        xx, yy = x(float(slot["start_bay"])), y(int(slot["time"]))
        if state == "work":
            draw.rectangle((xx - cell_width / 2, yy + 0.06 * time_step, xx + cell_width / 2, yy + 0.94 * time_step), fill=color)
            centered(f"Q{q}", (xx, yy + 0.51 * time_step), cell_font, "#333")
        elif state == "idle":
            draw.ellipse((xx - 3.1, yy + 0.5 * time_step - 3.1, xx + 3.1, yy + 0.5 * time_step + 3.1), outline=color, width=2)

    def dashed_arrow(x1: float, y1: float, x2: float, y2: float, color: str) -> None:
        # Draw a dashed segment and a small triangular arrow head.
        import math
        distance = math.hypot(x2 - x1, y2 - y1)
        if distance == 0:
            return
        ux, uy = (x2 - x1) / distance, (y2 - y1) / distance
        on, off, position = 7.0, 4.0, 0.0
        while position < distance:
            a, b = position, min(distance, position + on)
            draw.line((x1 + ux * a, y1 + uy * a, x1 + ux * b, y1 + uy * b), fill=color, width=2)
            position += on + off
        size = 6.0
        px, py = -uy, ux
        draw.polygon([(x2, y2), (x2 - ux * size + px * size * 0.55, y2 - uy * size + py * size * 0.55),
                      (x2 - ux * size - px * size * 0.55, y2 - uy * size - py * size * 0.55)], fill=color)

    for q in cranes:
        crane_slots = sorted((slot for slot in slots if int(slot["crane"]) == q), key=lambda item: int(item["time"]))
        color = COLORS[(q - 1) % len(COLORS)]
        for previous, current in zip(crane_slots, crane_slots[1:]):
            if previous["state"] == "offrail" or current["state"] == "offrail":
                continue
            if float(previous["end_bay"]) == float(current["start_bay"]):
                continue
            yy = y(int(current["time"]))
            dashed_arrow(x(float(previous["end_bay"])), yy - 1, x(float(current["start_bay"])), yy + 1, color)

    loads = {q: sum(1 for slot in slots if int(slot["crane"]) == q and slot["state"] == "work") for q in cranes}
    legend_x, legend_y, legend_w = right + 18, top + 8, 210
    crane_h = 28 + 24 * len(cranes)
    draw.rectangle((legend_x, legend_y, legend_x + legend_w, legend_y + crane_h), outline="#d0d0d0")
    centered("Cranes", (legend_x + legend_w / 2, legend_y + 20), legend_font)
    for index, q in enumerate(cranes):
        yy = legend_y + 43 + index * 24
        color = COLORS[(q - 1) % len(COLORS)]
        draw.line((legend_x + 10, yy - 4, legend_x + 42, yy - 4), fill=color, width=5)
        draw.text((legend_x + 50, yy - 8), f"Q{q} load={loads[q]}", font=legend_font, fill="#222")
    state_y = bottom - 164
    draw.rectangle((legend_x, state_y, legend_x + legend_w, state_y + 156), outline="#d0d0d0")
    centered("States", (legend_x + legend_w / 2, state_y + 20), legend_font)
    draw.rectangle((legend_x + 13, state_y + 36, legend_x + 31, state_y + 48), fill="#777")
    draw.text((legend_x + 40, state_y + 36), "work: filled cell", font=legend_font, fill="#222")
    dashed_arrow(legend_x + 13, state_y + 72, legend_x + 35, state_y + 72, "#555")
    draw.text((legend_x + 40, state_y + 64), "move: dashed arrow", font=legend_font, fill="#222")
    draw.ellipse((legend_x + 18, state_y + 104, legend_x + 26, state_y + 112), outline="#777", width=2)
    draw.text((legend_x + 40, state_y + 100), "idle: hollow dot", font=legend_font, fill="#222")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--title")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if not payload:
            raise ValueError("运行记录为空，无法绘图。")
        record = payload[0]
        updates = record.get("candidate_updates") or []
        if not updates or not updates[0].get("slots"):
            raise ValueError("运行记录中没有推荐候选的时间槽。")
        payload = {
            **(record.get("best") or {}),
            "slots": updates[0]["slots"],
        }
    if args.output.suffix.lower() == ".png":
        render_png(payload, args.output, title=args.title)
    else:
        render(payload, args.output, title=args.title)


if __name__ == "__main__":
    main()
