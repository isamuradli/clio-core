#!/usr/bin/env python3
"""Three views of a run's selection log, from one reader.

    ./viz_selection.py actions --out DIR --sel 0.001:RUN --sel 0.01:RUN
    ./viz_selection.py bound   --out DIR --run nyx:0.1:RUN --run lammps:0.1:RUN
    ./viz_selection.py chunks  --out DIR --run RUN [--field density]

Every figure here reads `selection.csv` and nothing else, which is why they
share a file: the blob-name parsing, the store-or-csv path resolution and the
palette were previously copied into three scripts, and the copies had drifted
-- one of them could not parse WarpX's `step00010/E_x/chunk_0` at all.

    actions   which action the selector picked, dump by dump, as the data
              evolves. An action is the tuple (library, shuffle, quantize,
              preset), and the plate draws all of it: lane = library,
              colour = error bound, filled = quantize, square = 4-byte shuffle.
              The point is that the selection MOVES -- the same policy on the
              same field picks differently as the physics fills the box.

    bound     requested against applied quantization. A positive error bound
              unmasks the 16 quantize actions and `quantize=1` records that one
              was CHOSEN; it does not record that quantization RAN. Those are
              different events and on a float64 workload they come apart
              completely -- `actual_psnr > 0` is the only witness that it ran.

    chunks    what the model saw, what it said, what happened: the three
              statistics it consumes, then predicted against actual ratio with
              a rug marking the chunks that produced an SGD gradient.
"""
import argparse
import collections
import csv
import json
import os
import re
import sys

# plt00007 (nyx/vpic), step00010 (warpx, no separator), step_140 (lammps).
# The separator is optional: requiring it silently dropped every WarpX row.
FRAME_RE = re.compile(r"^(?:plt|step_?)(\d+)$")
FIELD_RE = re.compile(r"^fab\d+_comp\d+_")

# Reference categorical palette, in fixed order -- validates on every adjacent
# pair in both light and dark.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
GATE = "#8046B5"


# ---------------------------------------------------------------------------
# One reader, three callers.
# ---------------------------------------------------------------------------
def selection_csv(store):
    """A run_config.sh store, or a selection.csv named directly."""
    p = store if store.endswith(".csv") else os.path.join(store, "selection.csv")
    if not os.path.exists(p):
        sys.exit(f"no selection log at {p}")
    return p


def split_blob(blob):
    """-> (field, frame) or (None, None).

    Two shapes, told apart by which half looks like a frame counter rather than
    by which benchmark wrote the file:

        plt00007/fab0000_comp00_density/chunk_0     nyx, vpic  -- frame first
        step00010/E_x/chunk_0                       warpx      -- frame first
        position/step_140/chunk_0                   lammps     -- field first

    The frame is whatever number the counter carries -- a dump index or a
    timestep -- so it is comparable within a run but not across workloads.
    """
    parts = blob.split("/")
    if len(parts) < 2:
        return None, None
    a, b = parts[0], parts[1]
    ma, mb = FRAME_RE.match(a), FRAME_RE.match(b)
    if ma and not mb:
        return FIELD_RE.sub("", b), int(ma.group(1))
    if mb and not ma:
        return FIELD_RE.sub("", a), int(mb.group(1))
    return None, None


def read_rows(path, primary_only=False, want_fields=None):
    """selection.csv -> [(field, frame, row)] in run order.

    `primary_only` matters wherever a chunk is drawn once. Exploration and best
    mode log an `adopted` row per chunk as well -- the alternative that actually
    got stored -- and mixing the two puts two points at one x with no way to
    tell which the model predicted.
    """
    out = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if primary_only and r.get("role") not in (None, "", "primary"):
                continue
            field, frame = split_blob(r["blob"])
            if want_fields and field not in want_fields:
                continue
            out.append((field, frame, r))
    return out


def by_field_frame(rows):
    """[(field, frame, row)] -> {field: {frame: row}}."""
    out = collections.defaultdict(dict)
    for field, frame, r in rows:
        if field is not None and frame is not None:
            out[field][frame] = r
    return out


def fnum(r, key, default=0.0):
    try:
        return float(r[key])
    except (KeyError, TypeError, ValueError):
        return default


def agg():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------
def cmd_actions(a):
    plt = agg()
    from matplotlib.lines import Line2D

    runs = []
    for spec in a.sel:
        if ":" not in spec:
            sys.exit(f"--sel wants EB:DIR, got {spec!r}")
        eb, d = spec.split(":", 1)
        p = selection_csv(d)
        runs.append((float(eb), p, by_field_frame(read_rows(p))))
    runs.sort(key=lambda r: r[0])

    presets = {r["preset"] for _, _, sel in runs for f in sel for r in sel[f].values()}
    if len(presets) > 1:
        print(f"NOTE: preset varies across these runs ({sorted(presets)}); "
              f"the plate does not encode it.")

    known = ["density", "xmom", "ymom", "zmom", "rho_E", "rho_e",
             "position", "velocity", "force"]
    present = [f for f in known if any(f in sel for _, _, sel in runs)]
    extra = sorted({f for _, _, sel in runs for f in sel} - set(present))
    fields = a.field or (present + extra)
    if not fields:
        sys.exit("no recognisable fields in these logs")

    # Lanes shared across subplots so the plates are comparable, ordered by how
    # much work the codec does -- roughly fastest to strongest.
    ORDER = ["nvcomp-bitcomp", "nvcomp-lz4", "nvcomp-snappy", "nvcomp-ans",
             "nvcomp-cascaded", "nvcomp-gdeflate", "nvcomp-zstd"]
    seen = {r["lib_name"] for _, _, sel in runs for f in sel for r in sel[f].values()}
    lanes = [l for l in ORDER if l in seen] + sorted(seen - set(ORDER))
    lane_of = {l: i for i, l in enumerate(lanes)}

    os.makedirs(a.out, exist_ok=True)
    allframes = sorted({d for _, _, sel in runs for f in sel for d in sel[f]})
    # LAMMPS frames are timesteps (0, 20, 40...), nyx frames are dump indices
    # (0, 1, 2...). Plot the real keys so the axis means something in both.
    xlab = ("dump  (the blast expanding)" if max(allframes) < len(allframes) * 2
            else "timestep  (the lattice melting)")

    fig, axes = plt.subplots(
        len(fields), 1, squeeze=False, sharex=True,
        figsize=(11, 1.05 * len(lanes) * len(fields) * 0.55 + 1.6 * len(fields)))
    for ax, field in zip(axes[:, 0], fields):
        for k, (eb, _, sel) in enumerate(runs):
            rows = sel.get(field, {})
            if not rows:
                continue
            dumps = sorted(rows)
            # A small vertical offset per bound: several bounds often land in
            # the same lane on the same dump and stacked markers hide that.
            off = (k - (len(runs) - 1) / 2) * 0.18
            ys = [lane_of[rows[d]["lib_name"]] + off for d in dumps]
            ax.plot(dumps, ys, "-", color=SERIES[k % len(SERIES)], lw=1,
                    alpha=.45, zorder=1)
            for d, y in zip(dumps, ys):
                r = rows[d]
                ax.plot(d, y,
                        marker="s" if r["shuffle"] != "0" else "o",
                        ms=6.5 if r["shuffle"] != "0" else 6,
                        mfc=SERIES[k % len(SERIES)] if r["quantize"] == "1" else "none",
                        mec=SERIES[k % len(SERIES)], mew=1.5, zorder=3)
        ax.set(yticks=range(len(lanes)),
               yticklabels=[l.replace("nvcomp-", "") for l in lanes],
               ylim=(-0.7, len(lanes) - 0.3))
        ax.grid(alpha=0.25, axis="both")
        ax.set_ylabel(field, fontsize=10, rotation=0, ha="right", va="center",
                      labelpad=44)
    axes[-1, 0].set_xlabel(xlab)
    pad = (max(allframes) - min(allframes)) * 0.03 + 0.5
    axes[-1, 0].set_xlim(min(allframes) - pad, max(allframes) + pad)

    handles = [Line2D([], [], color=SERIES[k % len(SERIES)], marker="o", ls="-",
                      label=f"eb = {eb:g}") for k, (eb, _, _) in enumerate(runs)]
    handles += [
        Line2D([], [], color="0.35", marker="o", ls="none", mfc="0.35", label="quantized"),
        Line2D([], [], color="0.35", marker="o", ls="none", mfc="none", label="kept lossless"),
        Line2D([], [], color="0.35", marker="s", ls="none", mfc="none", label="4-byte shuffle"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Action selected per dump - lane = codec, fill = quantize, "
                 "square = shuffle", fontsize=11, y=1.035)
    fig.tight_layout()
    out = os.path.join(a.out, "actions.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  {out}")

    # How much the selection actually moves -- the claim the plate makes.
    for eb, _, sel in runs:
        for field in fields:
            rows = sel.get(field, {})
            if not rows:
                continue
            acts = [(r["lib_name"], r["shuffle"], r["quantize"])
                    for _, r in sorted(rows.items())]
            switches = sum(1 for i in range(1, len(acts)) if acts[i] != acts[i - 1])
            print(f"eb={eb:<6g} {field:<8} {len(set(acts))} distinct action(s), "
                  f"{switches} switch(es) over {len(acts)} dumps")


# ---------------------------------------------------------------------------
# bound
# ---------------------------------------------------------------------------
def cmd_bound(a):
    import numpy as np
    plt = agg()

    runs = []
    for spec in a.run:
        parts = spec.split(":", 2)
        if len(parts) != 3:
            sys.exit(f"--run wants LABEL:EB:STORE, got {spec!r}")
        label, eb, store = parts
        rows = [r for _, _, r in read_rows(selection_csv(store))]
        req = sum(1 for r in rows if r["quantize"] == "1")
        # actual_psnr > 0 is the only record that the quantizer really ran.
        ran = sum(1 for r in rows if fnum(r, "actual_psnr", -1) > 0)
        runs.append({"label": f"{label}\neb={eb}", "n": len(rows), "req": req,
                     "ran": ran,
                     "ratio": float(np.mean([fnum(r, "actual_ratio") for r in rows]))})
        print(f"{label} eb={eb}: {len(rows)} chunks, quantize requested {req}, "
              f"applied {ran}, mean actual ratio {runs[-1]['ratio']:.3f}")

    ncol = 2 + (1 if a.reinterpret else 0)
    fig, ax = plt.subplots(1, ncol, figsize=(5.0 * ncol, 4.2), squeeze=False)
    ax = ax[0]

    y = np.arange(len(runs))
    h = 0.38
    ax[0].barh(y + h / 2, [r["req"] for r in runs], height=h,
               color=SERIES[0], label="quantize chosen")
    ax[0].barh(y - h / 2, [r["ran"] for r in runs], height=h,
               color=SERIES[1], label="quantize applied")
    for i, r in enumerate(runs):
        if r["ran"] == 0 and r["req"] > 0:
            ax[0].text(r["req"] * 0.02, i - h / 2, " none applied", va="center",
                       fontsize=8.5, color=SERIES[1], weight="bold")
        ax[0].text(r["req"], i + h / 2, f" {r['req']}/{r['n']}", va="center", fontsize=8)
    ax[0].set(yticks=y, yticklabels=[r["label"] for r in runs],
              xlabel="chunks", title="chosen is not applied")
    ax[0].legend(fontsize=8, loc="upper center", ncol=2, framealpha=.92)
    ax[0].set_xlim(0, max(max(r["req"] for r in runs), 1) * 1.28)

    ax[1].barh(y, [r["ratio"] for r in runs], color=SERIES[2])
    ax[1].axvline(1.0, ls="--", lw=1, color="0.4")
    ax[1].set(yticks=y, yticklabels=[r["label"] for r in runs], xscale="log",
              xlabel="mean actual ratio (dashed = no compression)",
              title="and what it was worth")
    for i, r in enumerate(runs):
        ax[1].text(r["ratio"], i, f" {r['ratio']:.2f}x", va="center", fontsize=8)

    for k, spec in enumerate(a.reinterpret):
        parts = spec.split(":", 2)
        if len(parts) != 3:
            sys.exit(f"--reinterpret wants LABEL:DTYPE:FILE, got {spec!r}")
        label, dt, path = parts
        buf = np.fromfile(path, dtype=np.dtype(dt))
        as32 = buf.view(np.float32)
        fin = np.isfinite(as32)
        bad = int((~fin).sum())
        # Log y: the float32 case is one narrow spike holding nearly every
        # word, and on a linear axis it flattens the float64 case -- the one
        # that matters -- into the baseline.
        ax[2].hist(np.log10(np.abs(as32[fin & (as32 != 0)])), bins=90,
                   color=SERIES[k % len(SERIES)], alpha=.7,
                   label=f"{label} — {bad} non-finite of {as32.size}")
        ax[2].set(xlabel="log10 |value| when the buffer is read as float32",
                  ylabel="words", yscale="log",
                  title="what the quantizer actually sees")
        print(f"{label}: {path}\n  real range {buf.min():.4g} .. {buf.max():.4g} "
              f"({dt});  as float32 -> {bad}/{as32.size} non-finite, "
              f"finite span {as32[fin].min():.3g} .. {as32[fin].max():.3g}")
    if a.reinterpret:
        ax[2].legend(fontsize=7.5, loc="upper left")

    for b in ax:
        b.grid(alpha=0.3, axis="x")
    fig.suptitle("A positive error bound only matters if the quantizer accepts it",
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "bound_applied.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  {out}")


# ---------------------------------------------------------------------------
# chunks
# ---------------------------------------------------------------------------
def cost_model_of(run_dir):
    """What cost weights the run used, from meta.json. None when unrecorded."""
    p = os.path.join(run_dir, "meta.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as fh:
            return json.load(fh).get("cost_model")
    except (ValueError, OSError):
        return None


def cmd_chunks(a):
    csvp = selection_csv(a.run)
    run_dir = os.path.dirname(csvp)
    want = set(a.field) if a.field else None
    rows = read_rows(csvp, primary_only=True, want_fields=want)
    if not rows:
        sys.exit(f"no primary rows in {csvp}"
                 + (f" for field(s) {sorted(want)}" if want else ""))

    ent = [fnum(r, "entropy") for _, _, r in rows]
    mad = [fnum(r, "mad") for _, _, r in rows]
    dv = [fnum(r, "second_deriv") for _, _, r in rows]
    pr = [fnum(r, "pred_ratio") for _, _, r in rows]
    ar = [fnum(r, "actual_ratio") for _, _, r in rows]
    x = list(range(len(rows)))

    # THE GATE RUG IS ONLY EXACT UNDER RATIO-ONLY COST WEIGHTS. Phase-1 SGD
    # fires on error_pct > mape_threshold, where error_pct is the MAPE of the
    # whole cost -- w_ct*ct + w_dt*dt + bytes/(ratio*bw). With the latency
    # weights zeroed that reduces to |1 - actual/predicted| on the capped
    # ratios, which is what this computes. Under the balanced weights the two
    # constants dominate and this would overstate the gate badly.
    cm = cost_model_of(run_dir)
    draw_gate = a.force_gate or cm == "ratio"
    if not draw_gate:
        print(f"NOTE: cost_model={cm!r} in meta.json, not 'ratio' -- the SGD gate "
              f"is computed from the FULL cost there and cannot be derived from "
              f"selection.csv. Rug omitted; pass --force-gate to draw it anyway.")
    gate = []
    for pv, av in zip(pr, ar):
        pc, ac = min(a.cap, pv), min(a.cap, av)
        gate.append(pc > 0 and ac > 0 and abs(1.0 - ac / pc) > a.mape)

    plt = agg()
    from matplotlib.lines import Line2D

    os.makedirs(a.out, exist_ok=True)
    title_bit = ", ".join(sorted(want)) if want else "all variables"

    # ---- figure 1: the model's inputs -------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.2), sharex=True)
    for ax, vals, label, color, logy in (
            (axes[0], ent, "Shannon entropy  (bits)", SERIES[0], False),
            (axes[1], mad, "mean absolute deviation", SERIES[1], True),
            (axes[2], dv, "second derivative", SERIES[2], True)):
        ax.plot(x, vals, "-", color=color, lw=0.9)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.25)
        if logy:
            pos = [v for v in vals if v > 0]
            if pos:
                ax.set_yscale("log")
                nz = len(vals) - len(pos)
                if nz:
                    # A log axis drops non-positive points silently. Say so
                    # rather than letting a gap read as missing data.
                    ax.text(0.995, 0.06,
                            f"{nz} chunk(s) exactly 0, not drawable on log",
                            transform=ax.transAxes, ha="right", fontsize=7.5,
                            color="0.45")
    axes[-1].set_xlabel("chunk index  (the order the run produced them)")
    axes[-1].set_xlim(0, max(1, len(rows) - 1))
    fig.suptitle(f"What the model sees per chunk - {title_bit}", fontsize=11)
    fig.tight_layout()
    out1 = os.path.join(a.out, "chunk_stats.png")
    fig.savefig(out1, dpi=120, bbox_inches="tight")
    print(f"  {out1}")

    # ---- figure 2: prediction against outcome -----------------------------
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(x, ar, "-", color=SERIES[2], lw=1.0)
    ax.plot(x, pr, "-", color=SERIES[1], lw=1.0)
    finite = [v for v in ar + pr if v > 0]
    if finite and max(finite) / max(min(finite), 1e-9) > 20:
        ax.set_yscale("log")
    if finite and max(finite) > a.cap:
        ax.axhline(a.cap, color="0.45", ls="--", lw=1.2)
        # Left edge with an opaque box: at the right edge this label lands on
        # the densest part of the trace on every workload tried.
        ax.text(len(rows) * 0.008, a.cap, f" {a.cap:g}x model cap ",
                ha="left", va="center", fontsize=8, color="0.25",
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5), zorder=5)
    ax.set_ylabel("compression ratio")
    ax.set_xlabel("chunk index")
    ax.set_xlim(0, max(1, len(rows) - 1))
    ax.grid(alpha=0.25)
    handles = [Line2D([], [], color=SERIES[2], label="actual ratio"),
               Line2D([], [], color=SERIES[1], label="predicted ratio")]
    if draw_gate:
        lo, hi = ax.get_ylim()
        y = lo * 1.06 if ax.get_yscale() == "log" else lo + (hi - lo) * 0.02
        fired = [i for i in x if gate[i]]
        ax.plot(fired, [y] * len(fired), "|", color=GATE, ms=7, mew=0.9)
        handles.append(Line2D([], [], color=GATE, marker="|", ls="none",
                              label=f"SGD gate fired ({len(fired)})"))
        ax.set_ylim(lo, hi)
    ax.legend(handles=handles, fontsize=8.5, frameon=False, ncol=len(handles),
              loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig.suptitle(f"Predicted against delivered - {title_bit}", fontsize=11, y=1.02)
    fig.tight_layout()
    out2 = os.path.join(a.out, "chunk_prediction.png")
    fig.savefig(out2, dpi=120, bbox_inches="tight")
    print(f"  {out2}")

    # ---- the numbers behind the pictures ----------------------------------
    err = [abs(av - pv) / av if av > 0 else 0.0 for pv, av in zip(pr, ar)]
    n = len(rows)
    half = n // 2
    first = sum(err[:half]) / max(1, half)
    second = sum(err[half:]) / max(1, n - half)
    capped = sum(1 for v in pr if v >= a.cap - 1e-3)
    over = sum(1 for v in ar if v > a.cap)
    print(f"chunks {n}  mean ratio MAPE {sum(err)/n:.3f}  "
          f"first half {first:.3f} -> second half {second:.3f} "
          f"({'improved' if second < first else 'no improvement'})")
    print(f"predicted at the {a.cap:g}x cap: {capped} ({100.0*capped/n:.0f}%)   "
          f"ACTUAL above it: {over} ({100.0*over/n:.0f}%)"
          + ("   <- unreachable targets" if over else ""))
    if draw_gate:
        print(f"gate at MAPE > {a.mape:g}: {sum(gate)} of {n} "
              f"({100.0*sum(gate)/n:.1f}%)")
    if not want:
        per = {}
        for (f, _, _), e in zip(rows, err):
            per.setdefault(f, []).append(e)
        if len(per) > 1:
            print("per variable, first half -> second half:")
            for f in sorted(per, key=lambda k: -sum(per[k]) / len(per[k])):
                v = per[f]
                h = len(v) // 2
                if h == 0:
                    continue
                fa, sb = sum(v[:h]) / h, sum(v[h:]) / (len(v) - h)
                if fa > 0:
                    print(f"  {str(f):<12} {fa:7.3f} -> {sb:7.3f}  "
                          f"{100.0*(sb-fa)/fa:+7.1f}%")
                else:
                    print(f"  {str(f):<12} {fa:7.3f} -> {sb:7.3f}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("actions", help="which action was selected, dump by dump")
    q.add_argument("--sel", action="append", required=True, metavar="EB:DIR",
                   help="repeatable; the bound and its run_config.sh store")
    q.add_argument("--out", required=True)
    q.add_argument("--field", action="append")
    q.add_argument("--plt", help="AMReX plotfile dir, to label the x axis in sim time")
    q.set_defaults(fn=cmd_actions)

    q = sub.add_parser("bound", help="requested against applied quantization")
    q.add_argument("--run", action="append", required=True, metavar="LABEL:EB:STORE")
    q.add_argument("--reinterpret", action="append", default=[],
                   metavar="LABEL:DTYPE:FILE",
                   help="repeatable; show a real buffer read as float32")
    q.add_argument("--out", required=True)
    q.set_defaults(fn=cmd_bound)

    q = sub.add_parser("chunks", help="model inputs, prediction, and outcome")
    q.add_argument("--run", required=True, metavar="DIR")
    q.add_argument("--out", required=True)
    q.add_argument("--field", action="append", help="repeatable; restrict to these")
    q.add_argument("--mape", type=float, default=0.30,
                   help="the run's neuropress_mape_threshold (default 0.30)")
    q.add_argument("--cap", type=float, default=100.0,
                   help="the model's ratio cap (default 100)")
    q.add_argument("--force-gate", action="store_true",
                   help="draw the gate rug even when meta.json is not ratio-only")
    q.set_defaults(fn=cmd_chunks)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
