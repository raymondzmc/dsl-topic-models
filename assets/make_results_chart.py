"""Draw the README results chart from assets/results_data.json.

    python3 assets/make_results_chart.py

Writes assets/results.svg (light) and assets/results-dark.svg. Standard library only.
The numbers in results_data.json are the paper's Table 1 (Purity) and Table 2
(Precision@5); nothing is typed into this script.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"

# Ink and surface follow GitHub's light and dark pages; the page shows through.
THEMES = {
    "results.svg": dict(ink="#1f2328", muted="#59636e", grid="#d8dee4", axis="#afb8c1",
                        baseline="#8c8b85", method="#2a78d6"),
    "results-dark.svg": dict(ink="#f0f6fc", muted="#9198a1", grid="#2a313c", axis="#3d444d",
                             baseline="#77766f", method="#3987e5"),
}

W = 960                      # viewBox width
PANEL_W, PANEL_GAP = 465, 30
LABEL_W, PLOT_W = 124, 272   # dataset label column, then the 0..1 axis
BAR_H, BAR_GAP, GROUP_GAP = 18, 2, 22
TOP = 92                     # y of the first bar (below legend and panel titles)


def fmt(v):
    return f"{v:.3f}".lstrip("0")


def bar(x, y, w, h, fill, r=4):
    """A bar that is square at the baseline and rounded at the data end."""
    r = min(r, w)
    return (f'<path d="M{x},{y} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} v{h - 2 * r} '
            f'a{r},{r} 0 0 1 -{r},{r} h-{w - r:.1f} z" fill="{fill}"/>')


def draw(data, c):
    groups_h = 3 * (2 * BAR_H + BAR_GAP) + 2 * GROUP_GAP
    axis_y = TOP + groups_h + 10
    height = axis_y + 34
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" width="{W}" height="{height}" '
        f'font-family="{FONT}" role="img" aria-labelledby="t d">',
        '<title id="t">ProdLDA + DSL compared with the best of eight baselines</title>',
        '<desc id="d">' + " ".join(
            f'{p["title"]} ({p["subtitle"]}): ' + "; ".join(
                f'{g["dataset"]} best baseline {g["best_baseline"]} {fmt(g["baseline"])}, DSL {fmt(g["dsl"])}'
                for g in p["groups"]) + "."
            for p in data["panels"]) + "</desc>",
    ]
    # Legend, one row above both panels.
    lx = LABEL_W
    out.append(f'<rect x="{lx}" y="14" width="14" height="14" rx="3" fill="{c["baseline"]}"/>')
    out.append(f'<text x="{lx + 21}" y="26" font-size="14" fill="{c["ink"]}">{data["baseline_label"]}</text>')
    lx2 = lx + 215
    out.append(f'<rect x="{lx2}" y="14" width="14" height="14" rx="3" fill="{c["method"]}"/>')
    out.append(f'<text x="{lx2 + 21}" y="26" font-size="14" fill="{c["ink"]}">{data["method_label"]}</text>')

    for i, panel in enumerate(data["panels"]):
        px = i * (PANEL_W + PANEL_GAP)
        x0 = px + LABEL_W
        out.append(f'<text x="{x0}" y="68" font-size="16" font-weight="600" fill="{c["ink"]}">{panel["title"]}'
                   f'<tspan dx="8" font-weight="400" fill="{c["muted"]}">· {panel["subtitle"]}</tspan></text>')
        # Hairline grid and axis ticks.
        for t in (0, 0.25, 0.5, 0.75, 1.0):
            gx = x0 + t * PLOT_W
            color = c["axis"] if t == 0 else c["grid"]
            out.append(f'<line x1="{gx:.1f}" y1="{TOP - 8}" x2="{gx:.1f}" y2="{axis_y}" stroke="{color}" stroke-width="1"/>')
            label = "0" if t == 0 else ("1" if t == 1 else f"{t:.2f}".lstrip("0"))
            out.append(f'<text x="{gx:.1f}" y="{axis_y + 18}" font-size="12" text-anchor="middle" fill="{c["muted"]}">{label}</text>')
        y = TOP
        for g in panel["groups"]:
            mid = y + BAR_H + BAR_GAP / 2
            out.append(f'<text x="{x0 - 10}" y="{mid + 5}" font-size="14" text-anchor="end" fill="{c["ink"]}">{g["dataset"]}</text>')
            wb, wm = g["baseline"] * PLOT_W, g["dsl"] * PLOT_W
            out.append(bar(x0, y, wb, BAR_H, c["baseline"]))
            out.append(f'<text x="{x0 + wb + 7:.1f}" y="{y + 13.5}" font-size="13" fill="{c["muted"]}">'
                       f'{fmt(g["baseline"])} {g["best_baseline"]}</text>')
            y2 = y + BAR_H + BAR_GAP
            out.append(bar(x0, y2, wm, BAR_H, c["method"]))
            out.append(f'<text x="{x0 + wm + 7:.1f}" y="{y2 + 13.5}" font-size="13" font-weight="600" fill="{c["ink"]}">'
                       f'{fmt(g["dsl"])}</text>')
            y = y2 + BAR_H + GROUP_GAP
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main():
    data = json.loads((HERE / "results_data.json").read_text())
    for name, colors in THEMES.items():
        (HERE / name).write_text(draw(data, colors))
        print("wrote", HERE / name)


if __name__ == "__main__":
    main()
