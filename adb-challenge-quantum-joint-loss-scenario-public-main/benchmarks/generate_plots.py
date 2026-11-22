"""Generate publication-grade SVG evaluation plots (zero external dependencies)."""

from __future__ import annotations

import math
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"

STYLE = """
  text { font-family: 'Segoe UI', Arial, sans-serif; fill: #1a202c; }
  .title { font-size: 16px; font-weight: 700; fill: #0b3d91; }
  .subtitle { font-size: 11.5px; fill: #4a5568; }
  .axis { stroke: #4a5568; stroke-width: 1.2; }
  .grid { stroke: #e2e8f0; stroke-width: 1; stroke-dasharray: 3,3; }
  .label { font-size: 11px; fill: #2d3748; }
  .val { font-size: 10.5px; font-weight: 600; fill: #1a202c; }
  .legend { font-size: 11px; fill: #2d3748; }
"""


def _svg_wrap(w: int, h: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" style="background:#ffffff">\n'
        f"<style>{STYLE}</style>\n{body}\n</svg>\n"
    )


def fig1_query_comparison() -> None:
    """Log-scale bar chart: rare exhaustion target & total portfolio calls on adb-base."""
    w, h = 820, 400
    items = [
        ("Plain Monte Carlo (Classical)", 3_161_151, "#a0aec0"),
        ("Untilted IQAE (Pure Quantum)", 270_900, "#718096"),
        ("ADB Importance Sampling (all 5)", 65_536, "#4a5568"),
        ("Our Tilted ML-IQAE (all 5)", 38_900, "#2b6cb0"),
        ("Our Adaptive Hybrid (all 5)", 14_496, "#0b3d91"),
        ("Our Tilted ML-IQAE (Exh. only)", 400, "#0e7c7b"),
    ]
    left, right, top, bottom = 235, 760, 75, 340
    plot_w = right - left
    min_log, max_log = 2.0, 6.7  # 10^2 to ~5e6

    lines = [
        f'<text x="{left}" y="30" class="title">Query Efficiency on adb-base (Log Scale)</text>',
        f'<text x="{left}" y="49" class="subtitle">Model evaluations / oracle calls required to meet evaluation-v1 tolerances</text>',
    ]
    for exp in range(2, 7):
        x = left + (exp - min_log) / (max_log - min_log) * plot_w
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" class="grid"/>')
        lines.append(f'<text x="{x:.1f}" y="{bottom + 20}" text-anchor="middle" class="label">10^{exp}</text>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>')
    lines.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>')

    row_h = (bottom - top) / len(items)
    for i, (label, val, color) in enumerate(items):
        y = top + i * row_h + 9
        bar_h = row_h - 18
        lv = math.log10(val)
        bw = (lv - min_log) / (max_log - min_log) * plot_w
        lines.append(f'<text x="{left - 10}" y="{y + bar_h/2 + 4:.1f}" text-anchor="end" class="label">{label}</text>')
        lines.append(f'<rect x="{left}" y="{y:.1f}" width="{bw:.1f}" height="{bar_h:.1f}" rx="3" fill="{color}"/>')
        lines.append(f'<text x="{left + bw + 7:.1f}" y="{y + bar_h/2 + 4:.1f}" class="val">{val:,}</text>')

    lines.append(f'<text x="{(left + right)/2:.1f}" y="{h - 14}" text-anchor="middle" class="subtitle">Model Evaluations / Oracle Calls (lower is better)</text>')
    (OUT_DIR / "fig1_adb_query_comparison.svg").write_text(_svg_wrap(w, h, "\n".join(lines)), encoding="utf-8")


def fig2_all13_norm_rmse() -> None:
    """Horizontal bar chart of worst normalised RMSE across all 13 benchmark cases."""
    w, h = 820, 500
    cases = [
        ("adb-base (tilted_iqae, 13q)", 0.407, "#0b3d91"),
        ("adb-base (hybrid, 13q)", 0.749, "#0e7c7b"),
        ("adb-independent (tilted_iqae, 13q)", 0.192, "#0b3d91"),
        ("adb-common-shock (tilted_iqae, 13q)", 0.214, "#0b3d91"),
        ("adb-rain-stress (tilted_iqae, 13q)", 0.272, "#0b3d91"),
        ("adb-rain-cover (tilted_iqae, 13q)", 0.280, "#0b3d91"),
        ("kaz-kgz-base (WB 2c, 20q)", 0.175, "#2b6cb0"),
        ("tjk-uzb-base (WB 2c, 20q)", 0.213, "#2b6cb0"),
        ("holdout case-001 (ADB T=1, 13q)", 0.398, "#1a365d"),
        ("holdout case-002 (WB 2c, 20q)", 0.332, "#2b6cb0"),
        ("holdout case-003 (ADB T=3, 13q)", 0.181, "#1a365d"),
        ("holdout case-004 (WB 3c, 20q)", 0.188, "#2b6cb0"),
        ("holdout case-005 (ADB T=10, 13q)", 0.299, "#1a365d"),
        ("holdout case-006 (WB 5c, 20q)", 0.170, "#2b6cb0"),
    ]
    left, right, top, bottom = 245, 760, 72, 445
    plot_w = right - left
    max_x = 1.15

    lines = [
        f'<text x="{left}" y="28" class="title">Evaluation-v1 Accuracy Across All 13 Benchmark Cases (100% Pass)</text>',
        f'<text x="{left}" y="47" class="subtitle">Worst normalised RMSE across all 5 targets per case (values &lt;= 1.0 pass)</text>',
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = left + (tick / max_x) * plot_w
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" class="grid"/>')
        lines.append(f'<text x="{x:.1f}" y="{bottom + 18}" text-anchor="middle" class="label">{tick:.2f}</text>')

    pass_x = left + (1.0 / max_x) * plot_w
    lines.append(f'<line x1="{pass_x:.1f}" y1="{top - 6}" x2="{pass_x:.1f}" y2="{bottom}" stroke="#c53030" stroke-width="2" stroke-dasharray="6,4"/>')
    lines.append(f'<text x="{pass_x - 4:.1f}" y="{top - 10}" text-anchor="end" style="font-size:10.5px;font-weight:700;fill:#c53030">PASS LIMIT (1.00)</text>')

    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>')
    lines.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>')

    row_h = (bottom - top) / len(cases)
    for i, (label, val, color) in enumerate(cases):
        y = top + i * row_h + 4
        bar_h = row_h - 8
        bw = (val / max_x) * plot_w
        lines.append(f'<text x="{left - 10}" y="{y + bar_h/2 + 4:.1f}" text-anchor="end" class="label">{label}</text>')
        lines.append(f'<rect x="{left}" y="{y:.1f}" width="{bw:.1f}" height="{bar_h:.1f}" rx="2" fill="{color}"/>')
        lines.append(f'<text x="{left + bw + 6:.1f}" y="{y + bar_h/2 + 4:.1f}" class="val">{val:.3f}</text>')

    lines.append(f'<text x="{(left + right)/2:.1f}" y="{h - 12}" text-anchor="middle" class="subtitle">Worst Normalised RMSE (lower is better; &lt;= 1.00 passes)</text>')
    (OUT_DIR / "fig2_all13_norm_rmse.svg").write_text(_svg_wrap(w, h, "\n".join(lines)), encoding="utf-8")


def fig3_cx_and_mps_scaling() -> None:
    """Two-panel figure: (A) Factorised CX reduction, (B) Exact chi=10 MPS n=2..50 scaling."""
    w, h = 860, 380
    lines = [
        '<text x="30" y="28" class="title">Circuit Gate Reduction &amp; Multi-Country Tensor-Network Scaling</text>',
        '<text x="30" y="47" class="subtitle">(Left) Monolithic vs. factorised CX gates on adb-base; (Right) Exact chi=10 Causal MPS runtime up to n=50 (10^50 states)</text>',
    ]

    # Left panel: CX comparison
    lx1, lx2, ly1, ly2 = 65, 395, 85, 315
    targets = [("P(L>T)", 4218, 534, "7.9x"), ("E[(L-T)+]", 4218, 568, "7.4x"), ("P(both exh.)", 4218, 568, "7.4x")]
    max_cx = 4800
    for tick in (0, 1000, 2000, 3000, 4000):
        y = ly2 - (tick / max_cx) * (ly2 - ly1)
        lines.append(f'<line x1="{lx1}" y1="{y:.1f}" x2="{lx2}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{lx1 - 8}" y="{y + 4:.1f}" text-anchor="end" class="label">{tick:,}</text>')
    lines.append(f'<line x1="{lx1}" y1="{ly1}" x2="{lx1}" y2="{ly2}" class="axis"/>')
    lines.append(f'<line x1="{lx1}" y1="{ly2}" x2="{lx2}" y2="{ly2}" class="axis"/>')

    gw = (lx2 - lx1) / len(targets)
    for i, (name, mono, fact, ratio) in enumerate(targets):
        cx_center = lx1 + (i + 0.5) * gw
        h_m = (mono / max_cx) * (ly2 - ly1)
        h_f = (fact / max_cx) * (ly2 - ly1)
        lines.append(f'<rect x="{cx_center - 32:.1f}" y="{ly2 - h_m:.1f}" width="28" height="{h_m:.1f}" fill="#a0aec0"/>')
        lines.append(f'<rect x="{cx_center + 4:.1f}" y="{ly2 - h_f:.1f}" width="28" height="{h_f:.1f}" fill="#0e7c7b"/>')
        lines.append(f'<text x="{cx_center - 18:.1f}" y="{ly2 - h_m - 5:.1f}" text-anchor="middle" class="val">{mono:,}</text>')
        lines.append(f'<text x="{cx_center + 18:.1f}" y="{ly2 - h_f - 5:.1f}" text-anchor="middle" class="val">{fact}</text>')
        lines.append(f'<text x="{cx_center:.1f}" y="{ly2 + 18}" text-anchor="middle" class="label">{name}</text>')
        lines.append(f'<text x="{cx_center:.1f}" y="{ly2 + 33}" text-anchor="middle" style="font-size:10px;font-weight:700;fill:#0e7c7b">({ratio} less)</text>')

    lines.append('<rect x="195" y="64" width="12" height="10" fill="#a0aec0"/><text x="211" y="73" class="legend">Monolithic (13q)</text>')
    lines.append('<rect x="305" y="64" width="12" height="10" fill="#0e7c7b"/><text x="321" y="73" class="legend">Factorised</text>')

    # Right panel: MPS scaling (evenly spaced categorical n for zero label overlap)
    rx1, rx2, ry1, ry2 = 490, 830, 85, 315
    mps_pts = [(2, 0.64), (5, 2.91), (10, 16.73), (20, 74.12), (50, 506.53)]
    max_ms = 580.0
    for tick in (0, 100, 200, 300, 400, 500):
        y = ry2 - (tick / max_ms) * (ry2 - ry1)
        lines.append(f'<line x1="{rx1}" y1="{y:.1f}" x2="{rx2}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{rx1 - 8}" y="{y + 4:.1f}" text-anchor="end" class="label">{tick} ms</text>')
    lines.append(f'<line x1="{rx1}" y1="{ry1}" x2="{rx1}" y2="{ry2}" class="axis"/>')
    lines.append(f'<line x1="{rx1}" y1="{ry2}" x2="{rx2}" y2="{ry2}" class="axis"/>')

    step_x = (rx2 - rx1 - 60) / (len(mps_pts) - 1)
    xs = [rx1 + 30 + i * step_x for i in range(len(mps_pts))]
    poly = []
    for px, (n, ms) in zip(xs, mps_pts):
        py = ry2 - (ms / max_ms) * (ry2 - ry1)
        poly.append(f"{px:.1f},{py:.1f}")
    lines.append(f'<polyline fill="none" stroke="#0b3d91" stroke-width="2.5" points="{" ".join(poly)}"/>')
    for idx, (px, (n, ms)) in enumerate(zip(xs, mps_pts)):
        py = ry2 - (ms / max_ms) * (ry2 - ry1)
        lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.8" fill="#0b3d91"/>')
        ly = py - 14 if idx < 4 else py - 14
        lx = px - 14 if idx == 4 else px
        lines.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" class="val">{ms:.1f} ms</text>')
        lines.append(f'<text x="{px:.1f}" y="{ry2 + 18}" text-anchor="middle" class="label">n={n}</text>')
        lines.append(f'<text x="{px:.1f}" y="{ry2 + 32}" text-anchor="middle" style="font-size:9.5px;fill:#4a5568">10^{n}</text>')

    lines.append(f'<text x="{(rx1 + rx2)/2:.1f}" y="{ry2 + 50}" text-anchor="middle" class="subtitle">Countries n &amp; State Space (Exact chi=10 MPS, 0.0 error)</text>')
    (OUT_DIR / "fig3_cx_and_mps_scaling.svg").write_text(_svg_wrap(w, h, "\n".join(lines)), encoding="utf-8")


def fig4_braket_qpu_zne() -> None:
    """Grouped bar chart: Raw vs. Readout+ZNE error across Braket QPU profiles."""
    w, h = 820, 380
    rows = [
        ("IQM Emerald\n(k=0, 450 CX)", 9.76, 7.66),
        ("IonQ Forte\n(k=0, 450 CX)", 5.56, 2.17),
        ("IonQ Forte\n(k=1, 1450 CX)", 27.30, 24.36),
        ("QuEra Libra FT\n(k=0, 450 CX)", 0.05, 0.00),
        ("QuEra Libra FT\n(k=4, 4450 CX)", 1.99, 0.01),
    ]
    left, right, top, bottom = 75, 780, 82, 310
    max_y = 30.0
    lines = [
        '<text x="75" y="28" class="title">Simulated AWS Braket QPU Noise &amp; Readout + ZNE Mitigation</text>',
        '<text x="75" y="47" class="subtitle">Absolute objective probability error (%) on factorised exhaustion circuit (a = 0.61)</text>',
        '<rect x="520" y="60" width="12" height="10" fill="#a0aec0"/><text x="537" y="69" class="legend">Raw Noise Error (%)</text>',
        '<rect x="655" y="60" width="12" height="10" fill="#0b3d91"/><text x="672" y="69" class="legend">Readout + ZNE (%)</text>',
    ]
    for tick in (0, 5, 10, 15, 20, 25, 30):
        y = bottom - (tick / max_y) * (bottom - top)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="label">{tick}%</text>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>')
    lines.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>')

    gw = (right - left) / len(rows)
    for i, (label, raw, zne) in enumerate(rows):
        cx = left + (i + 0.5) * gw
        hr = (raw / max_y) * (bottom - top)
        hz = (zne / max_y) * (bottom - top)
        lines.append(f'<rect x="{cx - 32:.1f}" y="{bottom - hr:.1f}" width="28" height="{hr:.1f}" fill="#a0aec0"/>')
        lines.append(f'<rect x="{cx + 4:.1f}" y="{bottom - hz:.1f}" width="28" height="{hz:.1f}" fill="#0b3d91"/>')
        lines.append(f'<text x="{cx - 18:.1f}" y="{bottom - hr - 5:.1f}" text-anchor="middle" class="val">{raw:.2f}%</text>')
        lines.append(f'<text x="{cx + 18:.1f}" y="{bottom - hz - 5:.1f}" text-anchor="middle" class="val">{zne:.2f}%</text>')
        l1, l2 = label.split("\n")
        lines.append(f'<text x="{cx:.1f}" y="{bottom + 18}" text-anchor="middle" class="label">{l1}</text>')
        lines.append(f'<text x="{cx:.1f}" y="{bottom + 32}" text-anchor="middle" class="subtitle">{l2}</text>')

    (OUT_DIR / "fig4_braket_qpu_zne.svg").write_text(_svg_wrap(w, h, "\n".join(lines)), encoding="utf-8")


def fig5_quantum_circuit_13q() -> None:
    """Publication-grade 13-qubit Causal Bayesian-Network + Tilted Payoff circuit diagram."""
    w, h = 1000, 545
    wires = [
        ("s[0..2]  (3q) Shared EQ", 116),
        ("a[0..2]  (3q) KGZ EQ", 172),
        ("b[0..2]  (3q) TJK EQ", 228),
        ("z        (1q) Shared Rain", 288),
        ("r1       (1q) KGZ Rain", 344),
        ("r2       (1q) TJK Rain", 400),
        ("obj      (1q) Objective", 464),
    ]
    x0, x1 = 195, 955

    lines = [
        '<text x="30" y="28" class="title">13-Qubit Causal Bayesian-Network Loader &amp; Tilted IQAE Circuit (ADB Track)</text>',
        '<text x="30" y="47" class="subtitle">Non-circular state preparation A (T-independent for T = 1, 3, 10 yr) followed by live Grover iterations Q^k and measurement</text>',
        # Stage background bands
        '<rect x="205" y="64" width="120" height="430" rx="6" fill="#f7fafc" stroke="#cbd5e0" stroke-dasharray="4,3"/>',
        '<text x="265" y="83" text-anchor="middle" style="font-size:10.5px;font-weight:700;fill:#4a5568">1. Latent Shocks</text>',
        '<rect x="338" y="64" width="195" height="430" rx="6" fill="#ebf8ff" stroke="#90cdf4" stroke-dasharray="4,3"/>',
        '<text x="435" y="83" text-anchor="middle" style="font-size:10.5px;font-weight:700;fill:#0b3d91">2. Country Conditionals</text>',
        '<rect x="546" y="64" width="160" height="430" rx="6" fill="#e6fffa" stroke="#81e6d9" stroke-dasharray="4,3"/>',
        '<text x="626" y="83" text-anchor="middle" style="font-size:10.5px;font-weight:700;fill:#0e7c7b">3. Tilted Payoff</text>',
        '<rect x="719" y="64" width="232" height="430" rx="6" fill="#faf5ff" stroke="#d6bcfa" stroke-dasharray="4,3"/>',
        '<text x="835" y="83" text-anchor="middle" style="font-size:10.5px;font-weight:700;fill:#553c9a">4. Grover Q^k &amp; Measure</text>',
    ]

    for label, y in wires:
        lines.append(f'<text x="186" y="{y + 4}" text-anchor="end" style="font-family:Consolas,monospace;font-size:11px;fill:#1a202c">{label}</text>')
        lines.append(f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="#2d3748" stroke-width="1.5"/>')
        if "(3q)" in label:
            lines.append(f'<line x1="{x0 + 8}" y1="{y + 5}" x2="{x0 + 16}" y2="{y - 5}" stroke="#2d3748" stroke-width="1.3"/>')
            lines.append(f'<text x="{x0 + 18}" y="{y - 5}" style="font-size:9px;fill:#4a5568">3</text>')

    def gate_box(cx: float, cy: float, bw: float, bh: float, text1: str, text2: str = "", fill: str = "#0b3d91") -> None:
        lines.append(f'<rect x="{cx - bw/2:.1f}" y="{cy - bh/2:.1f}" width="{bw}" height="{bh}" rx="4" fill="{fill}" stroke="#1a202c" stroke-width="1"/>')
        if text2:
            lines.append(f'<text x="{cx:.1f}" y="{cy - 2:.1f}" text-anchor="middle" style="font-size:10.5px;font-weight:700;fill:#ffffff">{text1}</text>')
            lines.append(f'<text x="{cx:.1f}" y="{cy + 11:.1f}" text-anchor="middle" style="font-size:9px;fill:#e2e8f0">{text2}</text>')
        else:
            lines.append(f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" style="font-size:10.5px;font-weight:700;fill:#ffffff">{text1}</text>')

    def ctrl_wire(x: float, y_ctrl: float, y_tgt: float, color: str = "#1a202c") -> None:
        lines.append(f'<line x1="{x:.1f}" y1="{y_ctrl:.1f}" x2="{x:.1f}" y2="{y_tgt:.1f}" stroke="{color}" stroke-width="1.6"/>')
        lines.append(f'<circle cx="{x:.1f}" cy="{y_ctrl:.1f}" r="4.5" fill="{color}"/>')

    gate_box(265, 116, 90, 32, "UCRY q(s)", "Poisson(cT)", "#2b6cb0")
    gate_box(265, 288, 90, 32, "Ry q(z)", "Shared Rain", "#2b6cb0")

    ctrl_wire(390, 116, 172, "#0b3d91")
    gate_box(390, 172, 82, 32, "UCRY", "q(a | s)", "#0b3d91")
    ctrl_wire(480, 116, 228, "#0b3d91")
    gate_box(480, 228, 82, 32, "UCRY", "q(b | s)", "#0b3d91")
    ctrl_wire(390, 288, 344, "#0b3d91")
    gate_box(390, 344, 82, 32, "UCRY", "q(r1 | z)", "#0b3d91")
    ctrl_wire(480, 288, 400, "#0b3d91")
    gate_box(480, 400, 82, 32, "UCRY", "q(r2 | z)", "#0b3d91")

    lines.append('<line x1="626" y1="116" x2="626" y2="464" stroke="#0e7c7b" stroke-width="1.8"/>')
    for _, wy in wires[:-1]:
        lines.append(f'<circle cx="626" cy="{wy}" r="4.5" fill="#0e7c7b"/>')
    gate_box(626, 464, 130, 36, "Ry(2 arcsin sqrt(g))", "g = f * (p/q) / W", "#0e7c7b")

    # Stage 4: wider Q^k box + clean measurement meter
    lines.append('<rect x="736" y="102" width="114" height="380" rx="6" fill="#553c9a" stroke="#1a202c" stroke-width="1.2"/>')
    lines.append('<text x="793" y="274" text-anchor="middle" style="font-size:16px;font-weight:700;fill:#ffffff">Q^k</text>')
    lines.append('<text x="793" y="294" text-anchor="middle" style="font-size:10.5px;fill:#e9d8fd">Grover Operator</text>')
    lines.append('<text x="793" y="314" text-anchor="middle" style="font-size:9.5px;fill:#d6bcfa">(2|psi_A&gt;&lt;psi_A|-I)</text>')
    lines.append('<text x="793" y="328" text-anchor="middle" style="font-size:9.5px;fill:#d6bcfa">x (I - 2 P_good)</text>')

    mx, my = 900, 464
    lines.append(f'<rect x="{mx - 22}" y="{my - 18}" width="44" height="36" rx="4" fill="#ffffff" stroke="#1a202c" stroke-width="1.5"/>')
    lines.append(f'<path d="M {mx - 13} {my + 7} A 13 13 0 0 1 {mx + 13} {my + 7}" fill="none" stroke="#1a202c" stroke-width="1.4"/>')
    lines.append(f'<line x1="{mx}" y1="{my + 7}" x2="{mx + 10}" y2="{my - 8}" stroke="#c53030" stroke-width="1.6"/>')
    lines.append(f'<text x="{mx - 12}" y="{my + 33}" text-anchor="middle" style="font-size:10px;font-weight:700;fill:#1a202c">P(1)=sin^2((2k+1)theta)</text>')

    lines.append('<text x="500" y="528" text-anchor="middle" class="subtitle">Classical prep: 65 local conditional entries (grows as 7 + 29n)  |  Exact agreement with ADB term_distribution() to &lt; 10^-15</text>')
    (OUT_DIR / "fig5_quantum_circuit_13q.svg").write_text(_svg_wrap(w, h, "\n".join(lines)), encoding="utf-8")


def fig6_factorized_vs_monolithic_circuit() -> None:
    """Visual schematic comparing Monolithic 12-control payoff vs Factorised linear-depth ancillas."""
    w, h = 920, 390
    lines = [
        '<text x="30" y="28" class="title">Factorised Linear-Depth Payoff Construction (7.4x - 7.9x CX Gate Reduction)</text>',
        '<text x="30" y="47" class="subtitle">Replacing a monolithic 12-control UCRY (exponential O(2^n)) with per-node likelihood-ratio ancillas + MCX (linear O(n))</text>',
        # Left box: Monolithic
        '<rect x="30" y="70" width="380" height="290" rx="8" fill="#fff5f5" stroke="#feb2b2" stroke-width="1.5"/>',
        '<text x="220" y="94" text-anchor="middle" style="font-size:13px;font-weight:700;fill:#9b2c2c">Monolithic Payoff (13 Qubits)</text>',
        '<text x="220" y="112" text-anchor="middle" style="font-size:10.5px;fill:#742a2a">4,218 CX Gates  |  Scales Exponentially O(2^n)</text>',
        # Monolithic wires
        '<line x1="60" y1="150" x2="380" y2="150" stroke="#2d3748" stroke-width="1.5"/>',
        '<text x="60" y="143" class="label">All 12 state qubits (s, a, b, z, r1, r2)</text>',
        '<line x1="60" y1="265" x2="380" y2="265" stroke="#2d3748" stroke-width="1.5"/>',
        '<text x="60" y="232" class="label">Objective qubit (obj)</text>',
        '<line x1="220" y1="150" x2="220" y2="265" stroke="#c53030" stroke-width="2"/>',
        '<circle cx="220" cy="150" r="6" fill="#c53030"/>',
        '<rect x="125" y="243" width="190" height="44" rx="5" fill="#c53030"/>',
        '<text x="220" y="262" text-anchor="middle" style="font-size:11px;font-weight:700;fill:#ffffff">12-Control UCRYGate</text>',
        '<text x="220" y="278" text-anchor="middle" style="font-size:9.5px;fill:#fed7d7">4,096 CX (brute-force table)</text>',
        '<text x="220" y="330" text-anchor="middle" style="font-size:10px;fill:#742a2a">Each added country multiplies CX by ~16x</text>',
        # Arrow between panels
        '<line x1="422" y1="215" x2="462" y2="215" stroke="#0b3d91" stroke-width="3"/>',
        '<polygon points="462,208 476,215 462,222" fill="#0b3d91"/>',
        # Right box: Factorised
        '<rect x="488" y="70" width="402" height="290" rx="8" fill="#f0fff4" stroke="#9ae6b4" stroke-width="1.5"/>',
        '<text x="689" y="94" text-anchor="middle" style="font-size:13px;font-weight:700;fill:#0e7c7b">Factorised Payoff (17-20 Qubits)</text>',
        '<text x="689" y="112" text-anchor="middle" style="font-size:10.5px;fill:#22543d">534 - 568 CX Gates (7.9x Less)  |  Scales Linearly O(n)</text>',
        # Factorised mini-blocks
        '<rect x="510" y="132" width="165" height="32" rx="4" fill="#0e7c7b"/>',
        '<text x="592" y="152" text-anchor="middle" style="font-size:10px;font-weight:700;fill:#ffffff">anc_f : Payoff f(a,b,r1,r2)</text>',
        '<rect x="510" y="174" width="165" height="32" rx="4" fill="#2b6cb0"/>',
        '<text x="592" y="194" text-anchor="middle" style="font-size:10px;font-weight:700;fill:#ffffff">anc_s : Tilt ratio r_s(s)</text>',
        '<rect x="510" y="216" width="165" height="32" rx="4" fill="#2b6cb0"/>',
        '<text x="592" y="236" text-anchor="middle" style="font-size:10px;font-weight:700;fill:#ffffff">anc_a,b : r_a(a|s), r_b(b|s)</text>',
        '<rect x="510" y="258" width="165" height="32" rx="4" fill="#2b6cb0"/>',
        '<text x="592" y="278" text-anchor="middle" style="font-size:10px;font-weight:700;fill:#ffffff">anc_z,r : r_z(z), r_r(r|z)</text>',
        # Lines to MCX
        '<line x1="675" y1="148" x2="740" y2="215" stroke="#0e7c7b" stroke-width="1.6"/>',
        '<line x1="675" y1="190" x2="740" y2="215" stroke="#2b6cb0" stroke-width="1.6"/>',
        '<line x1="675" y1="232" x2="740" y2="215" stroke="#2b6cb0" stroke-width="1.6"/>',
        '<line x1="675" y1="274" x2="740" y2="215" stroke="#2b6cb0" stroke-width="1.6"/>',
        '<rect x="740" y="185" width="128" height="60" rx="6" fill="#0b3d91"/>',
        '<text x="804" y="210" text-anchor="middle" style="font-size:11.5px;font-weight:700;fill:#ffffff">MCX (AND)</text>',
        '<text x="804" y="226" text-anchor="middle" style="font-size:9.5px;fill:#bee3f8">into obj qubit</text>',
        '<text x="804" y="239" text-anchor="middle" style="font-size:9px;fill:#e2e8f0">P(1) = mu / (scale*W)</text>',
        '<text x="689" y="330" text-anchor="middle" style="font-size:10px;fill:#22543d">Each added country adds only 1 small local rotation (~64 CX)</text>',
    ]
    (OUT_DIR / "fig6_factorized_vs_monolithic_circuit.svg").write_text(_svg_wrap(w, h, "\n".join(lines)), encoding="utf-8")


def fig7_hybrid_architecture_diagram() -> None:
    """Executive end-to-end hybrid system architecture diagram."""
    w, h = 920, 400
    lines = [
        '<text x="30" y="28" class="title">End-to-End Hybrid Quantum-Classical Catastrophe Pricing Architecture</text>',
        '<text x="30" y="47" class="subtitle">Drop-in qjlb / team_solver.py architecture spanning ADB DRB (13q), World Bank SFRARR (20q), and 50-country Exact MPS</text>',
        # Input box
        '<rect x="30" y="155" width="165" height="90" rx="6" fill="#edf2f7" stroke="#0b3d91" stroke-width="1.8"/>',
        '<text x="112" y="184" text-anchor="middle" style="font-size:11.5px;font-weight:700;fill:#0b3d91">Hazard &amp; Bond Input</text>',
        '<text x="112" y="203" text-anchor="middle" class="label">ADB Poisson Rates</text>',
        '<text x="112" y="219" text-anchor="middle" class="label">WB SFRARR 421,968 ELT</text>',
        '<text x="112" y="235" text-anchor="middle" class="label">Contract Terms (f)</text>',
        # Router box
        '<rect x="235" y="155" width="145" height="90" rx="6" fill="#0b3d91"/>',
        '<text x="307" y="188" text-anchor="middle" style="font-size:11.5px;font-weight:700;fill:#ffffff">Adaptive Router</text>',
        '<text x="307" y="206" text-anchor="middle" style="font-size:9.5px;fill:#bee3f8">Dispatches by</text>',
        '<text x="307" y="220" text-anchor="middle" style="font-size:9.5px;fill:#bee3f8">target rarity &amp; n</text>',
        '<line x1="195" y1="200" x2="235" y2="200" stroke="#0b3d91" stroke-width="2"/>',
        # Branch 1: Classical Sobol
        '<rect x="430" y="75" width="240" height="72" rx="6" fill="#f7fafc" stroke="#4a5568" stroke-width="1.5"/>',
        '<text x="550" y="98" text-anchor="middle" style="font-size:11px;font-weight:700;fill:#2d3748">Track 1: Classical Scrambled Sobol</text>',
        '<text x="550" y="116" text-anchor="middle" class="label">Bulk targets: E[L], P(both pay)</text>',
        '<text x="550" y="133" text-anchor="middle" class="subtitle">4,096 QMC samples (~0.01 s)</text>',
        # Branch 2: Quantum Tilted ML-IQAE
        '<rect x="430" y="164" width="240" height="82" rx="6" fill="#e6fffa" stroke="#0e7c7b" stroke-width="1.8"/>',
        '<text x="550" y="186" text-anchor="middle" style="font-size:11px;font-weight:700;fill:#0e7c7b">Track 2: Tilted Live-SV ML-IQAE</text>',
        '<text x="550" y="204" text-anchor="middle" class="label">Rare tails: P(L&gt;T), Stop-Loss, Exh.</text>',
        '<text x="550" y="220" text-anchor="middle" class="subtitle">13q Causal ADB (534 CX) | 20q WB ELT</text>',
        '<text x="550" y="235" text-anchor="middle" class="subtitle">10.1k-13.3k Q-calls | Braket ZNE ready</text>',
        # Branch 3: Exact Causal MPS
        '<rect x="430" y="263" width="240" height="72" rx="6" fill="#ebf8ff" stroke="#2b6cb0" stroke-width="1.5"/>',
        '<text x="550" y="286" text-anchor="middle" style="font-size:11px;font-weight:700;fill:#2b6cb0">Track 3: Exact Causal MPS (chi=10)</text>',
        '<text x="550" y="304" text-anchor="middle" class="label">Multi-country scaling: n = 2 .. 50</text>',
        '<text x="550" y="321" text-anchor="middle" class="subtitle">10^50 states in 0.51 s (0.0 error)</text>',
        # Output box
        '<rect x="720" y="145" width="170" height="110" rx="6" fill="#1a365d"/>',
        '<text x="805" y="174" text-anchor="middle" style="font-size:11.5px;font-weight:700;fill:#ffffff">Verified Output</text>',
        '<text x="805" y="194" text-anchor="middle" style="font-size:9.5px;fill:#bee3f8">5 Target Estimates</text>',
        '<text x="805" y="210" text-anchor="middle" style="font-size:9.5px;fill:#bee3f8">95% Confidence Intervals</text>',
        '<text x="805" y="226" text-anchor="middle" style="font-size:9.5px;fill:#bee3f8">circuit_sha256 + Logs</text>',
        '<text x="805" y="242" text-anchor="middle" style="font-size:9.5px;font-weight:700;fill:#81e6d9">13/13 evaluation-v1 PASS</text>',
        # Connectors
        '<line x1="380" y1="180" x2="430" y2="111" stroke="#4a5568" stroke-width="1.6"/>',
        '<line x1="380" y1="200" x2="430" y2="205" stroke="#0e7c7b" stroke-width="2"/>',
        '<line x1="380" y1="220" x2="430" y2="299" stroke="#2b6cb0" stroke-width="1.6"/>',
        '<line x1="670" y1="111" x2="720" y2="180" stroke="#4a5568" stroke-width="1.6"/>',
        '<line x1="670" y1="205" x2="720" y2="200" stroke="#0e7c7b" stroke-width="2"/>',
        '<line x1="670" y1="299" x2="720" y2="220" stroke="#2b6cb0" stroke-width="1.6"/>',
    ]
    (OUT_DIR / "fig7_hybrid_architecture_diagram.svg").write_text(_svg_wrap(w, h, "\n".join(lines)), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig1_query_comparison()
    fig2_all13_norm_rmse()
    fig3_cx_and_mps_scaling()
    fig4_braket_qpu_zne()
    fig5_quantum_circuit_13q()
    fig6_factorized_vs_monolithic_circuit()
    fig7_hybrid_architecture_diagram()
    print("Generated 7 SVG figures in:", OUT_DIR)


if __name__ == "__main__":
    main()
