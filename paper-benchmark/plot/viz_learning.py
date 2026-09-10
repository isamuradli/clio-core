#!/usr/bin/env python3
"""Does online learning actually move the model? One run, one curve.

    ./viz_learning.py --out DIR --run nyx:learn:RUN --run nyx:learn-ratio:RUN

`learn` and `learn-ratio` turn NeuroPress's online SGD on with exploration OFF,
so the model trains on the one action it actually picked, chunk after chunk.
This asks whether that changes anything, by recomputing the quantity the
runtime itself gates training on and watching it over the run.

THE GATE IS RECOMPUTED, NOT READ. selection.csv has no `sgd` column, but every
input to the gate is in it, so the arithmetic below is the runtime's
(compressor_runtime.cc:1486-1500) transcribed:

    cost(ct, dt, r) = w_ct*max(1,ct) + w_dt*max(1,dt) + w_io*bytes/(min(r,cap)*bw)
    predicted       = cost(pred_ct_ms,   pred_dt_ms, pred_ratio)
    actual          = cost(actual_ct_ms, pred_dt_ms, actual_ratio)
    error_pct       = |actual - predicted| / actual        SGD fires above 0.30

Note the second argument on both lines: decompression time is NOT measured at
write time, so the runtime scores the PREDICTED dt on both sides and the term
cancels out of the difference. Substituting anything else here would give a
curve that no training decision was ever made from.

TWO THINGS MAKE A FLAT CURVE MEAN NOTHING, and both are reported per run:

  reused    the `reused` column is THREE-valued, not a boolean, and the three
            populations answer the question differently. 0 = a forward pass ran
            for this chunk. 1 = the previous ranking was reused, so the error is
            inherited rather than earned. -1 = the chunk never entered
            NeuroPressRankChunk at all (neuropress_telemetry.cc:221), so no
            gradient could come from it under any weights. Only the 0 rows
            report on weights that the run itself trained, which is why the
            table carries a second error pair restricted to them.
  capped    under ratio-only weights the cost is bytes/(min(ratio,cap)*bw) and
            nothing else, so once the predicted AND actual ratio both clear the
            100x cap the two costs are the same number by arithmetic. error_pct
            is then exactly 0 and SGD cannot fire at any threshold. On highly
            compressible data this is most of the run.

A falling curve is not by itself evidence of learning either -- the data gets
easier or harder on its own. Pass the same workload's `dynamic`/`dynamic-ratio`
run as a `:control` policy and it is drawn dashed against the learning curve;
the gap between them is the part learning is responsible for.
"""
import argparse
import collections
import csv
import gzip
import json
import os
import re
import sys

SERIES = {"learn": "#2a78d6", "learn-ratio": "#eb6834"}
GATE_C = "#8046B5"
CTRL = {"learn": "#8bb6e8", "learn-ratio": "#f2a883"}
SGD_THRESHOLD = 0.30          # compressor_tasks.h: neuropress_mape_threshold_
WEIGHTS = {"balance": (1.0, 1.0, 1.0), "ratio": (0.0, 0.0, 1.0)}


# plt00007/fab0000_comp00_density/chunk_3 (nyx, vpic), step00010/E_x/chunk_0
# (warpx), position/step_0/chunk_0 (lammps). The field is the component that
# is neither a frame nor the chunk index.
FRAME_RE = re.compile(r"^(?:plt|step_?)(\d+)$")
FIELD_RE = re.compile(r"^fab\d+_comp\d+_")


def field_of(blob):
    parts = [p for p in blob.split("/") if p and not p.startswith("chunk_")]
    named = [p for p in parts if not FRAME_RE.match(p)]
    return FIELD_RE.sub("", named[-1]) if named else blob


def agg():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fnum(row, key, default=0.0):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def resolve(spec):
    """A run store, or a selection log named directly -- gzipped or not.

    Two layouts have to work. A live run store is `<dir>/selection.csv` beside
    `<dir>/meta.json`. The kept record is `<wl>/<config>.selection.csv.gz`
    beside `<wl>/<config>.meta.json`, because a directory per run would be
    sixteen directories holding two files each. Returns (csv path, meta path).
    """
    if os.path.isdir(spec):
        for name in ("selection.csv", "selection.csv.gz"):
            c = os.path.join(spec, name)
            if os.path.exists(c):
                return c, os.path.join(spec, "meta.json")
        sys.exit(f"no selection log in {spec}")
    if not os.path.exists(spec):
        sys.exit(f"no such selection log: {spec}")
    # <prefix>.selection.csv[.gz] -> <prefix>.meta.json
    stem = spec[:-3] if spec.endswith(".gz") else spec
    stem = stem[:-len(".selection.csv")] if stem.endswith(".selection.csv") \
        else os.path.splitext(stem)[0]
    cand = stem + ".meta.json"
    return spec, cand if os.path.exists(cand) else \
        os.path.join(os.path.dirname(spec), "meta.json")


def meta_of(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        sys.exit(f"no readable {path} -- needed for the cost weights")


def load(run_dir):
    """selection log + meta.json -> per-chunk records in run order."""
    csvp, metap = resolve(run_dir)
    meta = meta_of(metap)
    cm = meta.get("cost_model")
    if cm not in WEIGHTS:
        sys.exit(f"{run_dir}: cost_model={cm!r}, which has no weights here")
    w_ct, w_dt, w_io = WEIGHTS[cm]
    bw = float(meta.get("bw_bytes_per_ms", 5e6))
    cap = 100.0                                  # upstream RATIO_CAP

    def cost(nbytes, ct, dt, ratio):
        rc = min(cap, ratio)
        io = w_io * nbytes / (rc * bw) if rc > 0 else 1e30
        return w_ct * max(1.0, ct) + w_dt * max(1.0, dt) + io

    opener = gzip.open if csvp.endswith(".gz") else open
    out = []
    with opener(csvp, "rt") as fh:
        for r in csv.DictReader(fh):
            if r.get("role") not in (None, "", "primary"):
                continue            # exploration alternates; none under learn
            nbytes = fnum(r, "chunk_bytes")
            pr, ar = fnum(r, "pred_ratio"), fnum(r, "actual_ratio")
            pdt = fnum(r, "pred_dt_ms")
            pc = cost(nbytes, fnum(r, "pred_ct_ms"), pdt, pr)
            ac = cost(nbytes, fnum(r, "actual_ct_ms"), pdt, ar)
            err = abs(ac - pc) / ac if ac > 0 else 0.0
            out.append({
                "err": err,
                "fired": err > SGD_THRESHOLD,
                "ratio": ar,
                "pred": pr,
                "capped": pr >= cap and ar >= cap,
                "reused": int(fnum(r, "reused", -1.0)),
                "field": field_of(r.get("blob", "")),
                "action": (r.get("lib_name", ""), r.get("quantize", ""),
                           r.get("shuffle", ""), r.get("preset", "")),
            })
    if not out:
        sys.exit(f"{csvp}: no primary rows")
    return meta, out


def rolling(vals, win):
    """Centre-free trailing mean -- x is 'chunks seen so far', so a trailing
    window is what a reader of that axis expects."""
    win = max(1, min(win, len(vals)))
    acc, out = 0.0, []
    q = collections.deque()
    for v in vals:
        q.append(v); acc += v
        if len(q) > win:
            acc -= q.popleft()
        out.append(acc / len(q))
    return out


def halves(vals):
    h = len(vals) // 2
    if h == 0:
        return 0.0, 0.0, 0.0
    a = sum(vals[:h]) / h
    b = sum(vals[h:]) / len(vals[h:])
    delta = (b - a) / a * 100.0 if a > 0 else 0.0
    return a, b, delta


def cmd_perchunk(a):
    """One run, one chunk per x-position, and every SGD gradient marked.

    The trend view smooths; this one does not. It is here to answer a
    different question: not "did the error fall" but "which chunk taught the
    model what, and did the model's own prediction move afterwards".
    """
    plt = agg()
    os.makedirs(a.out, exist_ok=True)
    for spec in a.run:
        parts = spec.split(":")
        if len(parts) < 2:
            sys.exit(f"--run {spec}: want NAME:DIR")
        name, path = parts[0], ":".join(parts[1:])
        meta, rows = load(path)
        if a.field:
            want = set(a.field)
            rows = [r for r in rows if r["field"] in want]
            if not rows:
                sys.exit(f"{path}: no chunks for field(s) {sorted(want)}")
            name = name + "_" + "+".join(sorted(want))
        n = len(rows)
        x = list(range(n))
        fired = [i for i, r in enumerate(rows) if r["fired"]]
        cap = 100.0

        fig, axes = plt.subplots(4, 1, figsize=(12.5, 10.4), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 3, 1.5, 2.4]})

        # ---- 1. what the model said against what happened ------------------
        ax = axes[0]
        ax.plot(x, [r["ratio"] for r in rows], "-", color=SERIES["learn-ratio"],
                lw=0.7, alpha=0.55, label="actual ratio")
        ax.plot(x, [r["pred"] for r in rows], "-", color=SERIES["learn"],
                lw=0.7, label="predicted ratio")
        ax.axhline(cap, color="0.45", ls="--", lw=1.0)
        ax.text(n * 0.004, cap, " 100x model cap ", va="bottom", fontsize=7.5,
                color="0.35", bbox=dict(fc="white", ec="none", pad=1.2))
        pos = [v for r in rows for v in (r["ratio"], r["pred"]) if v > 0]
        if pos and max(pos) / min(pos) > 20:
            ax.set_yscale("log")
        ax.set_ylabel("compression ratio", fontsize=9)
        ax.legend(fontsize=8, ncol=2, framealpha=0.9)
        ax.grid(alpha=0.25)
        # The prediction line is what LEARNING moves. Everything else on this
        # figure is the model's input or its consequence.
        ax.set_title(f"{name}  -  {n} chunks, cost model {meta.get('cost_model')}, "
                     f"{len(fired)} SGD gradients ({100.0*len(fired)/n:.1f}% of chunks)",
                     fontsize=10.5)

        # ---- 2. the gate itself, per chunk ---------------------------------
        ax = axes[1]
        ax.plot(x, [r["err"] for r in rows], "-", color="0.62", lw=0.6,
                label="cost-model error")
        ax.plot(fired, [rows[i]["err"] for i in fired], ".", color=GATE_C,
                ms=3.2, ls="none", label=f"SGD fired (> {SGD_THRESHOLD:g})")
        ax.axhline(SGD_THRESHOLD, color=GATE_C, ls=":", lw=1.2)
        ax.set_ylabel("|actual - predicted| / actual", fontsize=9)
        ax.set_ylim(0, min(2.0, max(0.6, max(r["err"] for r in rows) * 1.05)))
        ax.legend(fontsize=8, ncol=2, framealpha=0.9)
        ax.grid(alpha=0.25)

        # ---- 3. every gradient, and how they accumulate --------------------
        ax = axes[2]
        ax.vlines(fired, 0, 1, color=GATE_C, lw=0.35, alpha=0.75)
        ax.set_yticks([])
        ax.set_ylabel("gradients", fontsize=9)
        ax.set_xlim(0, max(1, n - 1))
        cum, run = [], 0
        for r in rows:
            run += 1 if r["fired"] else 0
            cum.append(run)
        ax2 = ax.twinx()
        ax2.plot(x, cum, "-", color="0.25", lw=1.2)
        ax2.set_ylabel("cumulative", fontsize=8.5, color="0.25")
        ax2.tick_params(labelsize=7.5, colors="0.25")
        # A straight cumulative line means the model is being corrected at a
        # constant rate and is not converging; a knee is where it stopped
        # needing correction.

        # ---- 4. what it actually picked ------------------------------------
        ax = axes[3]
        libs = []
        for r in rows:
            if r["action"][0] not in libs:
                libs.append(r["action"][0])
        libs.sort()
        lane = {lib: i for i, lib in enumerate(libs)}
        for quant, mfc, lbl in ((True, "full", "quantized"),
                                (False, "none", "lossless")):
            xs = [i for i, r in enumerate(rows)
                  if (r["action"][1] == "1") == quant]
            if not xs:
                continue
            ys = [lane[rows[i]["action"][0]] for i in xs]
            ax.plot(xs, ys, ".", ms=2.6, ls="none",
                    color=SERIES["learn"] if quant else SERIES["learn-ratio"],
                    markerfacecolor=None if mfc == "full" else "none",
                    label=lbl)
        ax.set_yticks(range(len(libs)))
        ax.set_yticklabels([l.replace("nvcomp-", "") for l in libs], fontsize=8)
        ax.set_ylim(-0.6, len(libs) - 0.4)
        ax.set_ylabel("chosen codec", fontsize=9)
        ax.set_xlabel("chunk index  (the order the run produced them)", fontsize=9)
        ax.legend(fontsize=8, ncol=2, framealpha=0.9)
        ax.grid(alpha=0.25, axis="y")

        fig.tight_layout()
        out = os.path.join(a.out, f"perchunk_{name}.png")
        fig.savefig(out, dpi=125, bbox_inches="tight")
        plt.close(fig)

        # A gradient only matters if the prediction moved afterwards. WITHIN
        # ONE FIELD, though: consecutive chunks of a run belong to different
        # variables, so a run-order delta measures which variable came next and
        # not what the model learned. Grouping by field first is what makes the
        # two populations comparable.
        seq = collections.defaultdict(list)
        for r in rows:
            seq[r["field"]].append(r)
        after_fire, after_quiet = [], []
        for chain in seq.values():
            for i in range(len(chain) - 1):
                d = abs(chain[i + 1]["pred"] - chain[i]["pred"])
                (after_fire if chain[i]["fired"] else after_quiet).append(d)
        mf = sum(after_fire) / len(after_fire) if after_fire else 0.0
        mq = sum(after_quiet) / len(after_quiet) if after_quiet else 0.0
        print(f"  {out}\n"
              f"      {len(fired)}/{n} gradients over {len(seq)} field(s); "
              f"within a field, mean |d predicted ratio| is {mf:.3f} across a "
              f"boundary that fired vs {mq:.3f} across one that did not "
              f"(n={len(after_fire)} / {len(after_quiet)})")


def cmd_trend(a):
    runs = collections.OrderedDict()
    for spec in a.run:
        parts = spec.split(":")
        if len(parts) < 3:
            sys.exit(f"--run {spec}: want WORKLOAD:POLICY:DIR")
        wl, pol, path = parts[0], parts[1], ":".join(parts[2:])
        control = pol.endswith("-control")
        base = pol[:-len("-control")] if control else pol
        if base not in SERIES:
            sys.exit(f"--run {spec}: policy {base!r} is not learn/learn-ratio")
        meta, rows = load(path)
        runs.setdefault(wl, []).append((base, control, meta, rows))

    os.makedirs(a.out, exist_ok=True)
    plt = agg()
    order = list(runs)
    ncol = 2 if len(order) > 1 else 1
    nrow = (len(order) + ncol - 1) // ncol

    summary = {}
    for figname, key, ylabel, logy in (
            ("learning_error.png", "err",
             f"cost-model error  (SGD fires above {SGD_THRESHOLD:g})", False),
            ("learning_ratio.png", "ratio", "achieved compression ratio", True)):
        fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 3.5 * nrow),
                                 squeeze=False)
        for i, wl in enumerate(order):
            ax = axes[i // ncol][i % ncol]
            for base, control, meta, rows in runs[wl]:
                vals = [r[key] for r in rows]
                y = rolling(vals, a.window)
                ax.plot(range(len(y)), y,
                        color=(CTRL if control else SERIES)[base],
                        ls="--" if control else "-", lw=1.4,
                        label=base + (" (no learning)" if control else ""))
            if key == "err":
                ax.axhline(SGD_THRESHOLD, color="0.45", ls=":", lw=1.1)
            if logy:
                ax.set_yscale("log")
            ax.set_title(wl, fontsize=10)
            ax.grid(alpha=0.25)
            ax.set_xlabel("chunks seen", fontsize=8.5)
            ax.set_ylabel(ylabel, fontsize=8.5)
            ax.legend(fontsize=7.5, framealpha=0.9)
        for j in range(len(order), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.suptitle("Online learning, trailing mean over "
                     f"{a.window} chunks", fontsize=11)
        fig.tight_layout()
        p = os.path.join(a.out, figname)
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  {p}")

    # ---- the table the figures are read against ---------------------------
    lines = ["workload  policy               chunks  fired%  capped%  "
             "infer%  reuse%  nomodel%"
             "   err(1st)  err(2nd)   delta%"
             "   inferred-only: err(1st)  err(2nd)   delta%      n"
             "   ratio(1st)  ratio(2nd)  delta%"]
    for wl in order:
        for base, control, meta, rows in runs[wl]:
            n = len(rows)
            e1, e2, ed = halves([r["err"] for r in rows])
            r1, r2, rd = halves([r["ratio"] for r in rows])
            inf = [r["err"] for r in rows if r["reused"] == 0]
            i1, i2, idl = halves(inf)
            pct = lambda f: 100.0 * sum(1 for r in rows if f(r)) / n
            name = base + ("-control" if control else "")
            lines.append(
                f"{wl:<9} {name:<20} {n:>6}  "
                f"{pct(lambda r: r['fired']):>5.1f}  "
                f"{pct(lambda r: r['capped']):>6.1f}  "
                f"{pct(lambda r: r['reused'] == 0):>5.1f}  "
                f"{pct(lambda r: r['reused'] == 1):>5.1f}  "
                f"{pct(lambda r: r['reused'] < 0):>7.1f}  "
                f"{e1:>9.4f} {e2:>9.4f} {ed:>8.1f}  "
                f"{i1:>21.4f} {i2:>9.4f} {idl:>8.1f} {len(inf):>6}  "
                f"{r1:>11.3f} {r2:>11.3f} {rd:>7.1f}")
            summary[f"{wl}/{name}"] = {
                "chunks": n, "cost_model": meta.get("cost_model"),
                "fired_pct": pct(lambda r: r["fired"]),
                "capped_pct": pct(lambda r: r["capped"]),
                "inferred_pct": pct(lambda r: r["reused"] == 0),
                "reused_pct": pct(lambda r: r["reused"] == 1),
                "never_reached_model_pct": pct(lambda r: r["reused"] < 0),
                "err_first_half": e1, "err_second_half": e2, "err_delta_pct": ed,
                "inferred_only": {"n": len(inf), "err_first_half": i1,
                                  "err_second_half": i2, "err_delta_pct": idl},
                "ratio_first_half": r1, "ratio_second_half": r2,
                "ratio_delta_pct": rd,
                "distinct_actions": len({r["action"] for r in rows}),
            }
    table = "\n".join(lines)
    print("\n" + table)
    with open(os.path.join(a.out, "learning_table.txt"), "w") as fh:
        fh.write(table + "\n")
    with open(os.path.join(a.out, "learning_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    print(f"\n  {a.out}/learning_table.txt\n  {a.out}/learning_summary.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("trend", help="smoothed curves, learning against control")
    t.add_argument("--out", required=True)
    t.add_argument("--run", action="append", required=True,
                   metavar="WORKLOAD:POLICY:DIR",
                   help="POLICY is learn, learn-ratio, or either with "
                        ":control appended for a learning-off baseline")
    t.add_argument("--window", type=int, default=200,
                   help="trailing window, in chunks (default 200)")
    t.set_defaults(fn=cmd_trend)

    c = sub.add_parser("perchunk", help="one run unsmoothed, every gradient marked")
    c.add_argument("--out", required=True)
    c.add_argument("--run", action="append", required=True, metavar="NAME:DIR")
    c.add_argument("--field", action="append",
                   help="restrict to one variable; repeatable. All fields "
                        "interleaved is unreadable past a few hundred chunks")
    c.set_defaults(fn=cmd_perchunk)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
