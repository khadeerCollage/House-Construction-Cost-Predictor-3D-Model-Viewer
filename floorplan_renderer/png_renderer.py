"""
Architectural-Grade PNG Renderer
=================================
High-resolution PNG renderer using Matplotlib and wall mask generator for 3D pipeline.

Features:
  - 90-degree door swing arcs with door leaf rectangles and hinge points
  - Main entrance steps and landing with 'UP' annotation
  - Double-glazing window symbols with sill projections
  - Room labels: Name + Dimensions (W x H) + Area at centroid
  - Dimension lines with 45-degree architectural tick marks
  - North compass arrow and ISO title block
  - Binary wall mask generation (preserved for 3D pipeline)
"""

import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from floorplan_generator.layout_engine import (
    GeneratedFloorPlan, RoomRect, WallSegment,
    DoorPlacement, WindowPlacement, DimLine
)


class PNGRenderer:
    """Renders floor plans to high-resolution PNG using matplotlib and creates wall masks using OpenCV."""

    def __init__(self):
        pass

    def render(self, plan: GeneratedFloorPlan, output_path: str):
        """Renders the colored architectural PNG and saves it at 300 DPI."""
        pad = 2.5
        fig_w = max(10, (plan.plot_width + pad * 2 + 5.0) * 0.7)
        fig_h = max(8, (plan.plot_height + pad * 2) * 0.7)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
        
        ax.set_xlim(-pad, plan.plot_width + pad + 5.0)
        ax.set_ylim(-pad, plan.plot_height + pad)
        ax.set_aspect('equal')
        ax.axis('off')

        # 1. Rooms fill & labels
        for r in plan.rooms:
            if r.w <= 0 or r.h <= 0:
                continue
            rect = patches.Rectangle(
                (r.x, r.y), r.w, r.h,
                linewidth=0.5, edgecolor='#94A3B8',
                facecolor=r.color, alpha=0.25
            )
            ax.add_patch(rect)

            # Room labels: Name, Width x Height, Area
            cx, cy = r.cx, r.cy
            area = r.area
            min_dim = min(r.w, r.h)
            font_size = max(6, min(10, 5 + math.sqrt(area) * 0.8))
            
            name = r.name
            if min_dim < 1.5 or area < 3.0:
                abbrevs = {
                    'Attached Bathroom': 'ATT. BATH',
                    'Bathroom': 'BATH',
                    'Balcony': 'BALC.',
                    'Corridor': 'CORR.',
                    'Utility': 'UTIL.',
                    'Store Room': 'STORE',
                    'Servant Quarter': 'SQ',
                }
                name = abbrevs.get(name, name)

            ax.text(cx, cy + 0.25, name.upper(), ha='center', va='center',
                    fontsize=font_size, weight='bold', color='#111827')
            ax.text(cx, cy - 0.05, f"{r.w:.2f} x {r.h:.2f} m", ha='center', va='center',
                    fontsize=max(5, font_size - 1.5), color='#4B5563')
            ax.text(cx, cy - 0.32, f"{area:.1f} m2", ha='center', va='center',
                    fontsize=max(4.5, font_size - 2.5), style='italic', color='#6B7280')

        # 2. Walls with thickness
        for w in plan.walls:
            line_w = 4.0 if w.is_external else 2.5
            ax.plot([w.x1, w.x2], [w.y1, w.y2], color='#1F2937',
                    linewidth=line_w, solid_capstyle='butt')

        # 3. Windows — Double glazing lines & sill
        for win in plan.windows:
            side = win.wall_side
            glass_gap = 0.04 / 2
            if side in ('N', 'S'):
                x1 = win.x - win.width / 2
                x2 = win.x + win.width / 2
                # Double glazing lines
                ax.plot([x1, x2], [win.y - glass_gap, win.y - glass_gap],
                        color='#0284C7', linewidth=1.5)
                ax.plot([x1, x2], [win.y + glass_gap, win.y + glass_gap],
                        color='#0284C7', linewidth=1.5)
                # Sill line
                sill_y = win.y + (0.08 if side == 'N' else -0.08)
                ax.plot([x1 - 0.03, x2 + 0.03], [sill_y, sill_y],
                        color='#64748B', linewidth=1.0)
            else:
                y1 = win.y - win.height / 2
                y2 = win.y + win.height / 2
                # Double glazing lines
                ax.plot([win.x - glass_gap, win.x - glass_gap], [y1, y2],
                        color='#0284C7', linewidth=1.5)
                ax.plot([win.x + glass_gap, win.x + glass_gap], [y1, y2],
                        color='#0284C7', linewidth=1.5)
                # Sill line
                sill_x = win.x + (0.08 if side == 'E' else -0.08)
                ax.plot([sill_x, sill_x], [y1 - 0.03, y2 + 0.03],
                        color='#64748B', linewidth=1.0)

        # 4. Doors — 90° Swing Arc & Leaf Panel
        for d in plan.doors:
            hx, hy = d.x, d.y
            r = d.width
            side = d.wall_side
            swing = d.swing_direction

            # Determine angle range and leaf position
            # Matplotlib Arc: theta1, theta2 in degrees (CCW from positive X)
            theta1, theta2 = 0, 90
            leaf_end_x, leaf_end_y = hx, hy

            if side == 'N':
                if swing == 'RIGHT':
                    theta1, theta2 = 270, 360
                    leaf_end_x, leaf_end_y = hx, hy - r
                else:
                    theta1, theta2 = 180, 270
                    leaf_end_x, leaf_end_y = hx, hy - r
            elif side == 'S':
                if swing == 'RIGHT':
                    theta1, theta2 = 0, 90
                    leaf_end_x, leaf_end_y = hx, hy + r
                else:
                    theta1, theta2 = 90, 180
                    leaf_end_x, leaf_end_y = hx, hy + r
            elif side == 'E':
                if swing == 'RIGHT':
                    theta1, theta2 = 90, 180
                    leaf_end_x, leaf_end_y = hx - r, hy
                else:
                    theta1, theta2 = 180, 270
                    leaf_end_x, leaf_end_y = hx - r, hy
            elif side == 'W':
                if swing == 'RIGHT':
                    theta1, theta2 = 0, 90
                    leaf_end_x, leaf_end_y = hx + r, hy
                else:
                    theta1, theta2 = 270, 360
                    leaf_end_x, leaf_end_y = hx + r, hy

            # Draw swing arc
            arc = patches.Arc((hx, hy), 2 * r, 2 * r, angle=0,
                              theta1=theta1, theta2=theta2,
                              color='#6B7280', linewidth=0.8, linestyle='--')
            ax.add_patch(arc)

            # Draw door leaf line
            ax.plot([hx, leaf_end_x], [hy, leaf_end_y],
                    color='#059669', linewidth=1.5)

            # Hinge point
            ax.plot(hx, hy, marker='o', markersize=3, color='#059669')

            # Main entrance steps
            if d.is_main_entrance:
                landing_depth = 1.2
                step_depth = 0.3
                side_ext = 0.3
                if side == 'S':
                    x_l = hx - r / 2 - side_ext
                    x_r = hx + r / 2 + side_ext
                    # Landing
                    ax.add_patch(patches.Rectangle(
                        (x_l, hy - landing_depth), x_r - x_l, landing_depth,
                        facecolor='#F1F5F9', edgecolor='#334155', linewidth=1.0
                    ))
                    # 3 Steps
                    for i in range(3):
                        sy = hy - landing_depth - (i + 1) * step_depth
                        ax.plot([x_l, x_r], [sy, sy], color='#334155', linewidth=1.0)
                    ax.plot([x_l, x_l], [hy - landing_depth, hy - landing_depth - 3 * step_depth], color='#334155', linewidth=1.0)
                    ax.plot([x_r, x_r], [hy - landing_depth, hy - landing_depth - 3 * step_depth], color='#334155', linewidth=1.0)
                    # UP annotation
                    mx = (x_l + x_r) / 2
                    ax.annotate('UP', xy=(mx, hy - landing_depth),
                                xytext=(mx, hy - landing_depth - 3 * step_depth - 0.2),
                                arrowprops=dict(arrowstyle="->", color='#334155', lw=1.0),
                                ha='center', va='center', fontsize=7, weight='bold', color='#334155')

        # 5. Dimension Lines with 45° Tick marks
        for dim in plan.dimensions:
            x1, y1 = dim.x1, dim.y1
            x2, y2 = dim.x2, dim.y2
            is_horizontal = abs(y2 - y1) < abs(x2 - x1)
            tier_offsets = {1: 0.8, 2: 1.6, 3: 2.4}
            offset = tier_offsets.get(dim.tier, dim.offset)
            
            dx = 0 if is_horizontal else offset
            dy = offset if is_horizontal else 0
            
            dim_x1, dim_y1 = x1 + dx, y1 + dy
            dim_x2, dim_y2 = x2 + dx, y2 + dy

            # Witness lines
            ax.plot([x1, dim_x1], [y1, dim_y1], color='#94A3B8', linewidth=0.5)
            ax.plot([x2, dim_x2], [y2, dim_y2], color='#94A3B8', linewidth=0.5)
            # Dim line
            ax.plot([dim_x1, dim_x2], [dim_y1, dim_y2], color='#4B5563', linewidth=0.8)
            # 45-degree ticks
            tick = 0.1
            ax.plot([dim_x1 - tick, dim_x1 + tick], [dim_y1 - tick, dim_y1 + tick], color='#1F2937', linewidth=1.2)
            ax.plot([dim_x2 - tick, dim_x2 + tick], [dim_y2 - tick, dim_y2 + tick], color='#1F2937', linewidth=1.2)
            # Label
            mid_x = (dim_x1 + dim_x2) / 2
            mid_y = (dim_y1 + dim_y2) / 2
            text_y = mid_y + (0.15 if is_horizontal else 0)
            text_x = mid_x + (0 if is_horizontal else -0.15)
            ax.text(text_x, text_y, dim.label, ha='center', va='center',
                    fontsize=6.5, color='#374151', weight='bold' if dim.tier == 3 else 'normal')

        # 6. North Arrow
        na_x = plan.plot_width + 1.8
        na_y = plan.plot_height - 0.8
        ax.annotate('N', xy=(na_x, na_y + 0.6), xytext=(na_x, na_y - 0.4),
                    arrowprops=dict(facecolor='#1F2937', edgecolor='#1F2937', width=2, headwidth=8),
                    ha='center', va='center', fontsize=10, weight='bold', color='#1F2937')

        # 7. Title Block
        tb_x = plan.plot_width + 0.8
        tb_y = 0.2
        tb_w = 4.0
        tb_h = 2.8
        ax.add_patch(patches.Rectangle(
            (tb_x, tb_y), tb_w, tb_h,
            fill=False, edgecolor='#1F2937', linewidth=1.5
        ))
        row_h = tb_h / 5
        for i in range(1, 5):
            ax.plot([tb_x, tb_x + tb_w], [tb_y + i * row_h, tb_y + i * row_h], color='#1F2937', linewidth=0.8)

        cents = (plan.total_built_up_area * 10.764) / 435.6
        tb_labels = [
            f"PROJECT: {plan.project_name}",
            f"TYPE: {plan.bhk_config}",
            f"BUILT AREA: {plan.total_built_up_area:.1f} m2 ({cents:.2f} Cents)",
            f"VASTU SCORE: {plan.vastu_score:.0f}/100",
            f"QUALITY: {plan.quality_level} | SCALE: 1:50",
        ]
        for idx, text_val in enumerate(reversed(tb_labels)):
            ax.text(tb_x + 0.15, tb_y + (idx + 0.45) * row_h, text_val,
                    ha='left', va='center', fontsize=6.5, color='#111827')

        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

    def render_wall_mask(self, plan: GeneratedFloorPlan, ppm: int = 50) -> np.ndarray:
        """Returns a binary wall mask numpy array (white walls on black background) for 3D pipeline."""
        w_px = int(plan.plot_width * ppm)
        h_px = int(plan.plot_height * ppm)
        
        if w_px == 0 or h_px == 0:
            return np.zeros((100, 100), dtype=np.uint8)
            
        if cv2 is not None:
            mask = np.zeros((h_px, w_px), dtype=np.uint8)
            for w in plan.walls:
                x1, y1 = int(w.x1 * ppm), int(w.y1 * ppm)
                x2, y2 = int(w.x2 * ppm), int(w.y2 * ppm)
                thickness = max(1, int(w.thickness * ppm))
                cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
            return mask
        else:
            img = Image.new("L", (w_px, h_px), 0)
            draw = ImageDraw.Draw(img)
            for w in plan.walls:
                x1, y1 = int(w.x1 * ppm), int(w.y1 * ppm)
                x2, y2 = int(w.x2 * ppm), int(w.y2 * ppm)
                thickness = max(1, int(w.thickness * ppm))
                draw.line([(x1, y1), (x2, y2)], fill=255, width=thickness)
            return np.array(img, dtype=np.uint8)
