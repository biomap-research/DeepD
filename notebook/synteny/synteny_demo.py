#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-contained DeepD synteny demo and full-chromosome plotter.

Purpose
-------
This single module downloads demo assets when needed, audits the packaged
DAGchainer result, and renders the formal full-chromosome plot. The plotting
implementation fixes the query track first, usually the complete generated
sequence, and then infers the subject interval from anchors that are
collinear/aligned to that query. It is designed for figures such as:

    complete generated query sequence
        vs.
    only the reference/subject region that has synteny to the query

It also reads query and subject GFF3 files and highlights features whose GFF3
third column is exactly tRNA or rRNA (case-insensitive).

Coordinate note
---------------
GFF3 coordinates are converted to 0-based half-open intervals internally
(start = GFF_start - 1, end = GFF_end), matching common BED-like plotting.
DAGchainer coordinates are used as stored in the .pos.aligncoords file.
A one-base visual offset is negligible at kb-scale, but --axis-one-based can be
used for left tick display if desired.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import unquote

import matplotlib
matplotlib.use("Agg")
# Keep text as editable text objects in vector outputs such as PDF/SVG.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch, Polygon, Rectangle


# Reserved Hugging Face dataset; synteny data/ can be re-uploaded later.
DEFAULT_HF_DATASET_REPO = "biomap-research/DeepD"
DEFAULT_HF_REVISION = "main"
DEFAULT_SYNTENY_REMOTE_PREFIX = "synteny"


# ----------------------------- basic helpers -----------------------------


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr, flush=True)


def open_text(path: str | Path, mode: str = "rt"):
    p = str(path)
    if p == "-":
        return sys.stdin if "r" in mode else sys.stdout
    if p.endswith(".gz"):
        return gzip.open(p, mode)
    return open(p, mode, encoding=None if "b" in mode else "utf-8")


def natural_key(s: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(s))]


def parse_int(x: str) -> int:
    return int(float(x.replace(",", "")))


def fmt_bp(x: int) -> str:
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.2f} Mb"
    if abs(x) >= 1_000:
        return f"{x / 1_000:.1f} kb"
    return f"{x} bp"


def parse_attrs(attr_text: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for item in attr_text.strip().split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
            attrs[key.strip()] = unquote(value.strip().strip('"'))
        else:
            m = re.match(r'([^\s]+)\s+"?([^";]+)"?$', item)
            if m:
                attrs[m.group(1)] = unquote(m.group(2).strip())
    return attrs


def first_attr(attrs: Dict[str, str], keys: Sequence[str]) -> Optional[str]:
    for k in keys:
        v = attrs.get(k)
        if v:
            return v
    return None


def shorten_id(gid: str, max_len: int = 26) -> str:
    if len(gid) <= max_len:
        return gid
    left = max(7, (max_len - 2) // 2)
    right = max(7, max_len - 2 - left)
    return gid[:left] + ".." + gid[-right:]


# ----------------------------- data classes ------------------------------


@dataclass(frozen=True)
class Feature:
    seqid: str
    start: int
    end: int
    fid: str
    strand: str
    ftype: str
    source: str = ""

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2.0

    @property
    def is_special(self) -> bool:
        return self.ftype.lower() in {"trna", "rrna"}

    @property
    def special_label(self) -> str:
        low = self.ftype.lower()
        if low == "trna":
            return "tRNA"
        if low == "rrna":
            return "rRNA"
        return self.ftype


def context_feature_class(ftype: str, attrs: Dict[str, str]) -> str:
    text = f"{ftype} {' '.join(attrs.values())}".lower()
    if "telomer" in text:
        return "telomere"
    if "centromer" in text:
        return "centromere"
    return ""


@dataclass(frozen=True)
class Anchor:
    block: str
    qseq: str
    qgene: str
    qstart: int
    qend: int
    sseq: str
    sgene: str
    sstart: int
    send: int
    evalue: str = ""
    dag_score: str = ""
    block_order: int = 0

    @property
    def qlo(self) -> int:
        return min(self.qstart, self.qend)

    @property
    def qhi(self) -> int:
        return max(self.qstart, self.qend)

    @property
    def slo(self) -> int:
        return min(self.sstart, self.send)

    @property
    def shi(self) -> int:
        return max(self.sstart, self.send)

    @property
    def qmid(self) -> float:
        return (self.qlo + self.qhi) / 2.0

    @property
    def smid(self) -> float:
        return (self.slo + self.shi) / 2.0


# ----------------------------- readers -----------------------------------


def read_gff_features(path: str | Path) -> Tuple[List[Feature], Dict[str, List[Feature]], Dict[str, int]]:
    """Read gene/tRNA/rRNA-like top-level features from GFF3.

    Exact GFF3 third-column tRNA/rRNA features are kept and highlighted.
    Gene-level rows are kept for the background gene track. If an exact-overlap
    gene row and tRNA/rRNA row both exist, the special row is preferred to avoid
    drawing duplicate boxes.
    """
    feats: List[Feature] = []
    seq_lengths: Dict[str, int] = {}
    with open_text(path, "rt") as fh:
        in_fasta = False
        for line_no, line in enumerate(fh, start=1):
            raw = line.rstrip("\n")
            if raw.startswith("##sequence-region"):
                # GFF3 sequence-region line: ##sequence-region seqid start end
                parts_sr = raw.split()
                if len(parts_sr) >= 4:
                    try:
                        seqid_sr = parts_sr[1]
                        end_sr = parse_int(parts_sr[3])
                        seq_lengths[seqid_sr] = max(seq_lengths.get(seqid_sr, 0), end_sr)
                    except Exception:
                        pass
                continue
            if raw.startswith("##FASTA"):
                in_fasta = True
                continue
            if in_fasta or not raw.strip() or raw.startswith("#"):
                continue
            parts = raw.split("\t")
            if len(parts) < 9:
                continue
            seqid, source, ftype, start_s, end_s, _score, strand, _phase, attrs_s = parts[:9]
            ftype_low = ftype.lower()
            keep = (ftype_low == "gene" or ftype_low.endswith("gene") or ftype_low in {"trna", "rrna"})
            if not keep:
                continue
            try:
                # GFF3 is 1-based inclusive; convert to 0-based half-open.
                start = max(0, parse_int(start_s) - 1)
                end = parse_int(end_s)
            except Exception:
                eprint(f"[WARN] skip GFF line with bad coordinates: {path}:{line_no}")
                continue
            if end < start:
                start, end = end, start
            attrs = parse_attrs(attrs_s)
            fid = first_attr(attrs, ["ID", "Name", "gene", "locus_tag", "transcript_id", "product"])
            if not fid:
                fid = f"{ftype}:{seqid}:{start + 1}-{end}"
            if strand not in {"+", "-", "."}:
                strand = "."
            feats.append(Feature(seqid, start, end, fid, strand, ftype, source))

    # Deduplicate exact same intervals, preferring tRNA/rRNA over generic gene.
    by_key: Dict[Tuple[str, int, int, str], Feature] = {}
    for f in feats:
        key = (f.seqid, f.start, f.end, f.strand)
        old = by_key.get(key)
        if old is None:
            by_key[key] = f
        elif (not old.is_special) and f.is_special:
            by_key[key] = f
    feats = sorted(by_key.values(), key=lambda f: (natural_key(f.seqid), f.start, f.end, natural_key(f.fid)))
    by_seq: Dict[str, List[Feature]] = defaultdict(list)
    for f in feats:
        by_seq[f.seqid].append(f)
    return feats, dict(by_seq), seq_lengths


def read_context_features(path: str | Path) -> Tuple[List[Feature], Dict[str, List[Feature]]]:
    """Read structural markers from a published GFF without using its genes."""
    features: List[Feature] = []
    with open_text(path, "rt") as fh:
        in_fasta = False
        for line_no, line in enumerate(fh, start=1):
            raw = line.rstrip("\n")
            if raw.startswith("##FASTA"):
                in_fasta = True
                continue
            if in_fasta or not raw.strip() or raw.startswith("#"):
                continue
            parts = raw.split("\t")
            if len(parts) < 9:
                continue
            seqid, source, ftype, start_s, end_s, _score, strand, _phase, attrs_s = parts[:9]
            attrs = parse_attrs(attrs_s)
            marker_class = context_feature_class(ftype, attrs)
            if not marker_class:
                continue
            try:
                start = max(0, parse_int(start_s) - 1)
                end = parse_int(end_s)
            except Exception:
                eprint(f"[WARN] skip context GFF line with bad coordinates: {path}:{line_no}")
                continue
            fid = first_attr(attrs, ["ID", "Name", "gene", "locus_tag", "product"]) or f"{marker_class}:{seqid}:{start + 1}-{end}"
            features.append(Feature(seqid, start, end, fid, strand if strand in {"+", "-", "."} else ".", marker_class, source))
    deduplicated = {
        (f.seqid, f.start, f.end, f.ftype): f
        for f in features
    }
    result = sorted(deduplicated.values(), key=lambda f: (natural_key(f.seqid), f.start, f.end, f.ftype))
    by_seq: Dict[str, List[Feature]] = defaultdict(list)
    for feature in result:
        by_seq[feature.seqid].append(feature)
    return result, dict(by_seq)


def resolve_context_seqid(subject_seqid: str, context_by_seq: Dict[str, List[Feature]]) -> str:
    """Resolve common published/self seqid naming differences for one chromosome."""
    if subject_seqid in context_by_seq:
        return subject_seqid
    subject_low = subject_seqid.lower()
    contained = [
        seqid for seqid in context_by_seq
        if seqid.lower() in subject_low or subject_low in seqid.lower()
    ]
    if len(contained) == 1:
        return contained[0]
    if len(context_by_seq) == 1:
        return next(iter(context_by_seq))
    return ""


def read_fasta_lengths(path: str | Path) -> Dict[str, int]:
    """Read FASTA sequence lengths using the first whitespace-delimited token as seqid."""
    lengths: Dict[str, int] = {}
    cur_id: Optional[str] = None
    cur_len = 0
    with open_text(path, "rt") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    lengths[cur_id] = max(lengths.get(cur_id, 0), cur_len)
                cur_id = line[1:].strip().split()[0]
                cur_len = 0
            else:
                cur_len += len(line.strip())
        if cur_id is not None:
            lengths[cur_id] = max(lengths.get(cur_id, 0), cur_len)
    return lengths


def merge_lengths(primary: Dict[str, int], secondary: Dict[str, int]) -> Dict[str, int]:
    merged = dict(primary)
    for k, v in secondary.items():
        merged[k] = max(merged.get(k, 0), v)
    return merged


def read_aligncoords(path: str | Path) -> List[Anchor]:
    anchors: List[Anchor] = []
    current_block = "block000000"
    block_counter = 0
    row_in_block = 0
    header_re = re.compile(r"alignment\s+(\S+)\s+vs\.?\s+(\S+)\s+Alignment\s+#(\d+)", re.IGNORECASE)
    aln_no_re = re.compile(r"Alignment\s+#(\d+)", re.IGNORECASE)
    with open_text(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("##"):
                block_counter += 1
                row_in_block = 0
                m = header_re.search(line)
                if m:
                    qseq_h, sseq_h, aln_no = m.group(1), m.group(2), m.group(3)
                    safe_q = re.sub(r"[^A-Za-z0-9_.-]+", "_", qseq_h)
                    safe_s = re.sub(r"[^A-Za-z0-9_.-]+", "_", sseq_h)
                    current_block = f"block{block_counter:06d}_{safe_q}_vs_{safe_s}_A{aln_no}"
                else:
                    m2 = aln_no_re.search(line)
                    aln_no = m2.group(1) if m2 else str(block_counter)
                    current_block = f"block{block_counter:06d}_A{aln_no}"
                continue
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                row_in_block += 1
                anchors.append(Anchor(
                    block=current_block,
                    qseq=parts[0], qgene=parts[1], qstart=parse_int(parts[2]), qend=parse_int(parts[3]),
                    sseq=parts[4], sgene=parts[5], sstart=parse_int(parts[6]), send=parse_int(parts[7]),
                    evalue=parts[8] if len(parts) > 8 else "",
                    dag_score=parts[9] if len(parts) > 9 else "",
                    block_order=row_in_block,
                ))
            except Exception:
                continue
    return anchors


def swap_anchor(a: Anchor) -> Anchor:
    return Anchor(
        block=a.block,
        qseq=a.sseq, qgene=a.sgene, qstart=a.sstart, qend=a.send,
        sseq=a.qseq, sgene=a.qgene, sstart=a.qstart, send=a.qend,
        evalue=a.evalue, dag_score=a.dag_score, block_order=a.block_order,
    )


def orient_anchors(
    anchors: List[Anchor],
    query_seqids_from_gff: set[str],
    subject_seqids_from_gff: set[str],
    query_seqid: Optional[str] = None,
    subject_seqid: Optional[str] = None,
    force_swap: bool = False,
    auto_orient: bool = True,
) -> List[Anchor]:
    if not anchors:
        return anchors
    if force_swap:
        eprint("[WARN] --swap-aligncoords was set; swapping q/s columns for all anchors")
        return [swap_anchor(a) for a in anchors]
    if not auto_orient:
        return anchors

    cur = 0
    rev = 0
    for a in anchors:
        if query_seqid:
            cur += 3 if a.qseq == query_seqid else 0
            rev += 3 if a.sseq == query_seqid else 0
        else:
            cur += 1 if a.qseq in query_seqids_from_gff else 0
            rev += 1 if a.sseq in query_seqids_from_gff else 0
        if subject_seqid:
            cur += 3 if a.sseq == subject_seqid else 0
            rev += 3 if a.qseq == subject_seqid else 0
        else:
            cur += 1 if a.sseq in subject_seqids_from_gff else 0
            rev += 1 if a.qseq in subject_seqids_from_gff else 0

    if rev > cur:
        eprint(f"[WARN] aligncoords orientation appears reversed (current_score={cur}, reversed_score={rev}); swapping q/s columns")
        return [swap_anchor(a) for a in anchors]
    eprint(f"[INFO] aligncoords orientation kept (current_score={cur}, reversed_score={rev})")
    return anchors


# ----------------------------- interval helpers ---------------------------


def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) < min(a1, b1)


def features_overlapping(features: Sequence[Feature], start: int, end: int) -> List[Feature]:
    return [f for f in features if overlaps(f.start, f.end, start, end)]


def infer_seq_len(
    seqid: str,
    by_seq: Dict[str, List[Feature]],
    seq_lengths: Dict[str, int],
    anchors: Sequence[Anchor],
    side: str,
) -> int:
    vals: List[int] = []
    if seqid in seq_lengths:
        vals.append(seq_lengths[seqid])
    if seqid in by_seq and by_seq[seqid]:
        vals.extend(f.end for f in by_seq[seqid])
    if side == "query":
        vals.extend(a.qhi for a in anchors if a.qseq == seqid)
    else:
        vals.extend(a.shi for a in anchors if a.sseq == seqid)
    if not vals:
        raise SystemExit(f"[ERROR] cannot infer sequence length for {side} seqid={seqid}; provide --{side}-len or check GFF/aligncoords")
    return max(vals)


def choose_most_common_seqid(anchors: Sequence[Anchor], side: str, restrict_other: Optional[Tuple[str, str]] = None) -> str:
    c: Counter[str] = Counter()
    for a in anchors:
        if restrict_other:
            other_side, other_seqid = restrict_other
            if other_side == "query" and a.qseq != other_seqid:
                continue
            if other_side == "subject" and a.sseq != other_seqid:
                continue
        c[a.qseq if side == "query" else a.sseq] += 1
    if not c:
        raise SystemExit(f"[ERROR] cannot choose {side} seqid from anchors")
    return c.most_common(1)[0][0]


def parse_region(region: str) -> Tuple[str, Optional[int], Optional[int]]:
    """Parse seqid:whole or seqid:start-end. Coordinates are used as given."""
    if ":" not in region:
        return region, None, None
    seqid, rest = region.split(":", 1)
    if rest.lower() in {"whole", "all", "full"}:
        return seqid, None, None
    if "-" not in rest:
        raise ValueError(f"bad region: {region}; expected seqid:start-end or seqid:whole")
    s0, s1 = rest.replace(",", "").split("-", 1)
    return seqid, parse_int(s0), parse_int(s1)


def cluster_anchors(anchors: Sequence[Anchor], by: str = "query", gap: int = 30_000) -> List[List[Anchor]]:
    if not anchors:
        return []
    if by == "subject":
        arr = sorted(anchors, key=lambda a: (a.slo, a.shi, a.qlo))
        get0 = lambda a: a.slo
        get1 = lambda a: a.shi
    else:
        arr = sorted(anchors, key=lambda a: (a.qlo, a.qhi, a.slo))
        get0 = lambda a: a.qlo
        get1 = lambda a: a.qhi
    clusters: List[List[Anchor]] = []
    cur: List[Anchor] = [arr[0]]
    cur_end = get1(arr[0])
    for a in arr[1:]:
        if get0(a) - cur_end <= gap:
            cur.append(a)
            cur_end = max(cur_end, get1(a))
        else:
            clusters.append(cur)
            cur = [a]
            cur_end = get1(a)
    clusters.append(cur)
    return clusters


def infer_orientation(anchors: Sequence[Anchor]) -> str:
    if len(anchors) < 2:
        return "."
    xs = [a.qmid for a in anchors]
    ys = [a.smid for a in anchors]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return "+" if cov >= 0 else "-"


# ----------------------------- plotting -----------------------------------


def axis_x_end(start: int, end: int, plot_span: Optional[int] = None) -> float:
    span = max(1, end - start)
    denom = max(1, plot_span or span)
    return min(1.0, max(0.0, span / denom))


def map_x(pos: float, start: int, end: int, plot_span: Optional[int] = None) -> float:
    span = max(1, end - start)
    denom = max(1, plot_span or span)
    return (pos - start) / denom


def clipped_map(pos: float, start: int, end: int, plot_span: Optional[int] = None) -> float:
    right = axis_x_end(start, end, plot_span)
    return min(right, max(0.0, map_x(pos, start, end, plot_span)))


def choose_tick_label_interval(region_len: int, tick_interval: int, requested: int) -> int:
    if requested and requested > 0:
        return max(tick_interval, requested)
    # Keep labels readable while preserving 50-kb tick marks.
    target_labels = 8
    raw = max(tick_interval, math.ceil(max(1, region_len) / target_labels))
    return int(math.ceil(raw / tick_interval) * tick_interval)


def iter_axis_ticks(start: int, end: int, tick_interval: int) -> List[int]:
    if tick_interval <= 0:
        tick_interval = 50_000
    ticks: List[int] = []
    first = int(math.ceil(start / tick_interval) * tick_interval)
    if start not in ticks:
        ticks.append(start)
    t = first
    while t <= end:
        if t >= start and t not in ticks:
            ticks.append(t)
        t += tick_interval
    if end not in ticks:
        ticks.append(end)
    return sorted(ticks)


def draw_axis(
    ax,
    y: float,
    start: int,
    end: int,
    label: str,
    tick_side: str = "above",
    one_based_left: bool = False,
    plot_span: Optional[int] = None,
    tick_interval: int = 50_000,
    tick_label_interval: int = 0,
) -> None:
    right = axis_x_end(start, end, plot_span)
    ax.plot([0, right], [y, y], color="#222222", lw=0.8, zorder=2)
    tick_y = y + 0.042 if tick_side == "above" else y - 0.042
    label_y = y + 0.090 if tick_side == "above" else y - 0.090
    region_len = max(1, end - start)
    label_every = choose_tick_label_interval(region_len, tick_interval, tick_label_interval)
    last_label_x: Optional[float] = None
    for t in iter_axis_ticks(start, end, tick_interval):
        x = clipped_map(t, start, end, plot_span)
        is_boundary = (t == start or t == end)
        is_labeled = is_boundary or (t % label_every == 0)
        tick_len = 0.020 if is_labeled else 0.012
        tick_lw = 0.55 if is_labeled else 0.35
        ax.plot([x, x], [y - tick_len, y + tick_len], color="#222222", lw=tick_lw, zorder=2)
        if is_labeled:
            # Avoid near-duplicate labels at the right boundary. The tick line is still drawn.
            if last_label_x is None or abs(x - last_label_x) >= 0.045 or is_boundary:
                show_t = t + 1 if one_based_left and t == start else t
                ax.text(x, tick_y, fmt_bp(show_t), ha="center", va="center", fontsize=6.6)
                last_label_x = x
    ax.text(0.0, label_y, label, ha="left", va="center", fontsize=8.7)


def projected_query_centromeres(subject_centromeres: Sequence[Feature], anchors: Sequence[Anchor], query_seqid: str) -> List[Feature]:
    """Project published reference centromere midpoints through local synteny anchors."""
    if not subject_centromeres or not anchors:
        return []
    points = sorted((anchor.smid, anchor.qmid) for anchor in anchors)
    projected: List[Feature] = []
    for marker in subject_centromeres:
        target = marker.mid
        if len(points) == 1:
            query_mid = points[0][1] + (target - points[0][0])
        else:
            right_index = next((index for index, point in enumerate(points) if point[0] >= target), len(points) - 1)
            left_index = max(0, right_index - 1)
            if right_index == left_index:
                right_index = min(len(points) - 1, left_index + 1)
            s_left, q_left = points[left_index]
            s_right, q_right = points[right_index]
            if s_right == s_left:
                query_mid = (q_left + q_right) / 2.0
            else:
                query_mid = q_left + (target - s_left) * (q_right - q_left) / (s_right - s_left)
        half_width = max(1.0, (marker.end - marker.start) / 2.0)
        projected.append(Feature(
            query_seqid,
            max(0, int(round(query_mid - half_width))),
            max(1, int(round(query_mid + half_width))),
            f"projected_{marker.fid}",
            ".",
            "centromere",
            "projected_from_published_reference_by_synteny",
        ))
    return projected


def draw_structural_markers(
    ax,
    *,
    region: Tuple[int, int],
    track_y: float,
    axis_y: float,
    centromeres: Sequence[Feature],
    telomeres: Sequence[Feature],
    infer_end_telomeres: bool,
    plot_span: int,
) -> None:
    start, end = region
    right = axis_x_end(start, end, plot_span)
    # Published telomere intervals are context features, never evaluation features.
    marker_width = 0.006
    marker_height = 0.026
    telomere_positions: List[Tuple[float, float]] = []
    for marker in telomeres:
        if overlaps(marker.start, marker.end, start, end):
            x1 = clipped_map(marker.start, start, end, plot_span)
            x2 = clipped_map(marker.end, start, end, plot_span)
            telomere_positions.append(((x1 + x2) / 2.0, max(marker_width, abs(x2 - x1))))
    if not telomere_positions and infer_end_telomeres:
        telomere_positions = [(0.0, marker_width), (right, marker_width)]
    for x, width in telomere_positions:
        ax.add_patch(Rectangle(
            (x - width / 2.0, track_y - marker_height / 2.0),
            width,
            marker_height,
            facecolor="#E41A1C",
            edgecolor="#B00000",
            linewidth=0.45,
            zorder=7,
            clip_on=False,
        ))
    for marker in centromeres:
        if not overlaps(marker.start, marker.end, start, end):
            continue
        x = clipped_map(marker.mid, start, end, plot_span)
        direction = -1 if axis_y > track_y else 1
        ax.scatter([x], [axis_y + direction * 0.012], marker="v" if direction < 0 else "^",
                   s=48, color="#E69F00", edgecolor="#A66A00", linewidth=0.45, zorder=8)


def draw_feature(
    ax,
    f: Feature,
    start: int,
    end: int,
    y: float,
    height: float,
    normal_color: str,
    normal_edge: str,
    trna_color: str,
    rrna_color: str,
    triangle_threshold: float = 0.010,
    plot_span: Optional[int] = None,
) -> None:
    x1 = clipped_map(f.start, start, end, plot_span)
    x2 = clipped_map(f.end, start, end, plot_span)
    if x2 < x1:
        x1, x2 = x2, x1
    if x2 <= 0 or x1 >= 1:
        return
    width = max(0.0, x2 - x1)
    y0 = y - height / 2.0
    y1 = y + height / 2.0
    low = f.ftype.lower()
    if low == "trna":
        face, edge = trna_color, "#8a5a00"
    elif low == "rrna":
        face, edge = rrna_color, "#6f2f74"
    else:
        face, edge = normal_color, normal_edge
    strand = f.strand if f.strand in {"+", "-"} else "."
    if width < triangle_threshold and strand in {"+", "-"}:
        c = (x1 + x2) / 2.0
        w = max(width, triangle_threshold * 0.55)
        x1t = max(0.0, c - w / 2.0)
        x2t = min(1.0, c + w / 2.0)
        verts = [(x1t, y0), (x1t, y1), (x2t, y)] if strand == "+" else [(x2t, y0), (x2t, y1), (x1t, y)]
    elif strand == "+":
        head = min(width * 0.42, 0.018)
        verts = [(x1, y0), (x2 - head, y0), (x2, y), (x2 - head, y1), (x1, y1)] if width > head else [(x1, y0), (x1, y1), (x2, y)]
    elif strand == "-":
        head = min(width * 0.42, 0.018)
        verts = [(x2, y0), (x1 + head, y0), (x1, y), (x1 + head, y1), (x2, y1)] if width > head else [(x2, y0), (x2, y1), (x1, y)]
    else:
        verts = [(x1, y0), (x2, y0), (x2, y1), (x1, y1)]
    lw = 0.55 if f.is_special else 0.35
    ax.add_patch(Polygon(verts, closed=True, facecolor=face, edgecolor=edge, linewidth=lw, zorder=4 if f.is_special else 3))


def draw_ribbon(
    ax,
    top_a0: int,
    top_a1: int,
    top_region: Tuple[int, int],
    bottom_a0: int,
    bottom_a1: int,
    bottom_region: Tuple[int, int],
    y_top: float,
    y_bottom: float,
    alpha: float,
    color: str = "#9b9b9b",
    plot_span: Optional[int] = None,
) -> None:
    x1 = clipped_map(min(top_a0, top_a1), top_region[0], top_region[1], plot_span)
    x2 = clipped_map(max(top_a0, top_a1), top_region[0], top_region[1], plot_span)
    xb1 = clipped_map(min(bottom_a0, bottom_a1), bottom_region[0], bottom_region[1], plot_span)
    xb2 = clipped_map(max(bottom_a0, bottom_a1), bottom_region[0], bottom_region[1], plot_span)
    min_w = 0.0025
    if x2 - x1 < min_w:
        c = (x1 + x2) / 2.0
        x1, x2 = max(0.0, c - min_w / 2.0), min(1.0, c + min_w / 2.0)
    if xb2 - xb1 < min_w:
        c = (xb1 + xb2) / 2.0
        xb1, xb2 = max(0.0, c - min_w / 2.0), min(1.0, c + min_w / 2.0)
    mid_y = (y_top + y_bottom) / 2.0
    verts = [
        (x1, y_top),
        (x1, mid_y), (xb1, mid_y), (xb1, y_bottom),
        (xb2, y_bottom),
        (xb2, mid_y), (x2, mid_y), (x2, y_top),
        (x1, y_top),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha, zorder=1))


def label_features(ax, features: Sequence[Feature], start: int, end: int, y: float, above: bool, mode: str, anchor_ids: set[str], plot_span: Optional[int] = None) -> None:
    if mode == "none":
        return
    if mode == "all":
        show = list(features)
    elif mode == "anchors":
        show = [f for f in features if f.fid in anchor_ids]
    elif mode == "special":
        show = [f for f in features if f.is_special]
    else:  # auto
        show = [f for f in features if f.is_special]
        if len(anchor_ids) <= 30:
            seen = {id(f) for f in show}
            show.extend([f for f in features if f.fid in anchor_ids and id(f) not in seen])
    if not show:
        return
    dy = 0.060 if above else -0.060
    va = "bottom" if above else "top"
    rot = 28 if above else -28
    for f in show:
        x = clipped_map(f.mid, start, end, plot_span)
        # Do not print tRNA/rRNA feature names; keep only the type if labels are explicitly enabled.
        label = f.special_label if f.is_special else shorten_id(f.fid, 22)
        ax.text(x, y + dy, label, ha="center", va=va, fontsize=5.4, rotation=rot, zorder=5)


def plot_pair(
    out_pdf: Path,
    title: str,
    query_name: str,
    subject_name: str,
    query_seqid: str,
    subject_seqid: str,
    query_region: Tuple[int, int],
    subject_region: Tuple[int, int],
    query_features: Sequence[Feature],
    subject_features: Sequence[Feature],
    anchors: Sequence[Anchor],
    subject_context_features: Sequence[Feature],
    project_centromere: bool,
    query_on_top: bool,
    label_mode: str,
    one_based_axis: bool,
    fig_width: float,
    fig_height: float,
    ribbon_alpha: float,
    tick_interval: int,
    tick_label_interval: int,
) -> None:
    q0, q1 = query_region
    s0, s1 = subject_region
    q_feats = features_overlapping(query_features, q0, q1)
    s_feats = features_overlapping(subject_features, s0, s1)
    orient = infer_orientation(anchors)
    # One shared bp-to-x scale for both tracks. The longer chromosome spans the full plotting width;
    # the shorter one occupies a proportionally shorter axis.
    plot_span = max(1, q1 - q0, s1 - s0)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(-0.035, 1.035)
    ax.set_ylim(0, 1)
    ax.axis("off")

    if query_on_top:
        yq, ys = 0.68, 0.32
        q_tick, s_tick = "above", "below"
        q_axis_y, s_axis_y = 0.77, 0.23
    else:
        ys, yq = 0.68, 0.32
        s_tick, q_tick = "above", "below"
        s_axis_y, q_axis_y = 0.77, 0.23
    gene_h = 0.055

    # ribbons first
    for a in anchors:
        if query_on_top:
            draw_ribbon(ax, a.qlo, a.qhi, (q0, q1), a.slo, a.shi, (s0, s1), yq - gene_h / 2.1, ys + gene_h / 2.1, alpha=ribbon_alpha, plot_span=plot_span)
        else:
            draw_ribbon(ax, a.slo, a.shi, (s0, s1), a.qlo, a.qhi, (q0, q1), ys - gene_h / 2.1, yq + gene_h / 2.1, alpha=ribbon_alpha, plot_span=plot_span)

    q_special = sum(1 for f in q_feats if f.is_special)
    s_special = sum(1 for f in s_feats if f.is_special)
    q_label = f"Query {query_name} | generated annotation | {query_seqid}:{q0:,}-{q1:,} ({fmt_bp(q1 - q0)}, {len(q_feats)} features, {q_special} tRNA/rRNA, {len(anchors)} anchors, orient {orient})"
    s_label = f"Reference {subject_name} | self annotation; published structural markers | {subject_seqid}:{s0:,}-{s1:,} ({fmt_bp(s1 - s0)}, {len(s_feats)} features, {s_special} tRNA/rRNA)"

    draw_axis(ax, q_axis_y, q0, q1, q_label, tick_side=q_tick, one_based_left=one_based_axis, plot_span=plot_span, tick_interval=tick_interval, tick_label_interval=tick_label_interval)
    draw_axis(ax, s_axis_y, s0, s1, s_label, tick_side=s_tick, one_based_left=one_based_axis, plot_span=plot_span, tick_interval=tick_interval, tick_label_interval=tick_label_interval)

    for f in q_feats:
        draw_feature(ax, f, q0, q1, yq, gene_h, normal_color="#86bdd8", normal_edge="#1f4e6b", trna_color="#F2B6D0", rrna_color="#CC79A7", plot_span=plot_span)
    for f in s_feats:
        draw_feature(ax, f, s0, s1, ys, gene_h, normal_color="#2e86ab", normal_edge="#16384a", trna_color="#F2B6D0", rrna_color="#CC79A7", plot_span=plot_span)

    subject_centromeres = [feature for feature in subject_context_features if feature.ftype == "centromere"]
    subject_telomeres = [feature for feature in subject_context_features if feature.ftype == "telomere"]
    query_centromeres = projected_query_centromeres(subject_centromeres, anchors, query_seqid) if project_centromere else []
    draw_structural_markers(ax, region=(q0, q1), track_y=yq, axis_y=q_axis_y,
                            centromeres=query_centromeres, telomeres=[], infer_end_telomeres=True,
                            plot_span=plot_span)
    draw_structural_markers(ax, region=(s0, s1), track_y=ys, axis_y=s_axis_y,
                            centromeres=subject_centromeres, telomeres=subject_telomeres,
                            infer_end_telomeres=True, plot_span=plot_span)

    q_anchor_ids = {a.qgene for a in anchors}
    s_anchor_ids = {a.sgene for a in anchors}
    label_features(ax, q_feats, q0, q1, yq, above=(query_on_top), mode=label_mode, anchor_ids=q_anchor_ids, plot_span=plot_span)
    label_features(ax, s_feats, s0, s1, ys, above=(not query_on_top), mode=label_mode, anchor_ids=s_anchor_ids, plot_span=plot_span)

    # legend
    ax.text(0.5, 0.995, title, ha="center", va="top", fontsize=11, fontweight="bold")
    legend_handles = [
        Patch(facecolor="#2E86AB", edgecolor="#16384A", label="coding gene"),
        Patch(facecolor="#F2B6D0", edgecolor="#8A5A78", label="tRNA"),
        Line2D([0], [0], marker="v", color="none", markerfacecolor="#E69F00", markeredgecolor="#A66A00", markersize=7, label="centromere"),
        Patch(facecolor="#E41A1C", edgecolor="#B00000", label="telomere/end"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.965),
              ncol=4, frameon=False, fontsize=7.5, handlelength=1.3, columnspacing=1.4)
    fig.tight_layout(pad=0.4)
    with PdfPages(out_pdf) as pdf:
        pdf.savefig(fig)
    plt.close(fig)


# ----------------------------- CLI ----------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Full-chromosome synteny plot with one shared coordinate scale for query and subject."
    )
    p.add_argument("--aligncoords", required=True, help="DAGchainer .pos.aligncoords file")
    p.add_argument("--query-gff", required=True, help="Query GFF3/GFF3.gz path")
    p.add_argument("--subject-gff", required=True, help="Subject GFF3/GFF3.gz path")
    p.add_argument("--subject-context-gff", default="", help="Published subject GFF used only for centromere/telomere context markers")
    p.add_argument("--project-centromere", action="store_true", help="Project published subject centromere positions to the query through local synteny anchors")
    p.add_argument("--query-fasta", help="Optional query FASTA/FASTA.gz path for exact chromosome/contig length inference")
    p.add_argument("--subject-fasta", help="Optional subject FASTA/FASTA.gz path for exact chromosome/contig length inference")
    p.add_argument("--query-name", default="query", help="Display name for query genome/sample")
    p.add_argument("--subject-name", default="subject", help="Display name for subject genome/sample")
    p.add_argument("--query-seqid", help="Query sequence ID to display. Default: seqid with most anchors after orientation.")
    p.add_argument("--subject-seqid", help="Subject sequence ID to display. Default: seqid with most anchors for selected query.")
    p.add_argument("--query-region", help="Optional query region, seqid:start-end or seqid:whole. Default: whole selected query seqid.")
    p.add_argument("--subject-region", help="Optional subject region, seqid:start-end or seqid:whole. Default: whole selected subject seqid.")
    p.add_argument("--query-len", type=int, help="Query sequence length if it cannot be inferred from GFF/anchors")
    p.add_argument("--subject-len", type=int, help="Subject sequence length if it cannot be inferred from GFF/anchors")
    p.add_argument("--subject-pad", type=int, default=5_000, help="Padding around anchor-inferred subject span, only used with --subject-region-from-anchors. Default: 5000")
    p.add_argument("--subject-region-from-anchors", action="store_true", help="Legacy behavior: draw only the subject span inferred from anchors + --subject-pad instead of the full subject chromosome.")
    p.add_argument("--query-pad", type=int, default=0, help="Padding around query region when --query-region is not whole. Default: 0")
    p.add_argument("--swap-aligncoords", action="store_true", help="Force swap q/s columns in aligncoords")
    p.add_argument("--no-auto-orient", action="store_true", help="Disable automatic q/s orientation detection using GFF seqids")
    p.add_argument("--keep-largest-cluster", action="store_true", help="Keep only the largest anchor cluster before inferring subject span")
    p.add_argument("--cluster-by", choices=["query", "subject"], default="query", help="Coordinate axis used for anchor clustering. Default: query")
    p.add_argument("--cluster-gap", type=int, default=30_000, help="Max gap between neighboring anchors in one cluster. Default: 30000")
    p.add_argument("--min-cluster-anchors", type=int, default=2, help="Ignore clusters with fewer anchors when choosing largest cluster. Default: 2")
    p.add_argument("--min-anchors", type=int, default=1, help="Minimum anchors required after filtering. Default: 1")
    p.add_argument("--label-features", choices=["none", "special", "anchors", "auto", "all"], default="none", help="Feature label mode. Default: none. tRNA/rRNA remain colored but their names are not printed.")
    p.add_argument("--query-on-top", action="store_true", help="Draw query track on top instead of bottom")
    p.add_argument("--axis-one-based", action="store_true", help="Add 1 to each left tick label")
    p.add_argument("--tick-interval", type=int, default=50_000, help="Tick-mark interval in bp. Default: 50000")
    p.add_argument("--tick-label-interval", type=int, default=0, help="Tick-label interval in bp. Default: auto; set 50000 to label every 50-kb tick.")
    p.add_argument("--fig-width", type=float, default=13.5, help="Figure width in inches. Default: 13.5")
    p.add_argument("--fig-height", type=float, default=5.6, help="Figure height in inches. Default: 5.6")
    p.add_argument("--ribbon-alpha", type=float, default=0.26, help="Ribbon transparency. Default: 0.26")
    p.add_argument("--out-prefix", required=True, help="Output prefix; writes <prefix>.pdf and <prefix>.summary.tsv")
    return p


def plot_main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    q_feats_all, q_by_seq, q_seq_lengths = read_gff_features(args.query_gff)
    s_feats_all, s_by_seq, s_seq_lengths = read_gff_features(args.subject_gff)
    subject_context_all: List[Feature] = []
    subject_context_by_seq: Dict[str, List[Feature]] = {}
    if args.subject_context_gff:
        subject_context_all, subject_context_by_seq = read_context_features(args.subject_context_gff)
        eprint(f"[INFO] loaded published context markers: {len(subject_context_all)} from {args.subject_context_gff}")
    eprint(f"[INFO] loaded query GFF features: {len(q_feats_all)} from {args.query_gff}")
    eprint(f"[INFO] loaded subject GFF features: {len(s_feats_all)} from {args.subject_gff}")
    if args.query_fasta:
        q_fasta_lengths = read_fasta_lengths(args.query_fasta)
        q_seq_lengths = merge_lengths(q_seq_lengths, q_fasta_lengths)
        eprint(f"[INFO] loaded query FASTA lengths: {len(q_fasta_lengths)} from {args.query_fasta}")
    if args.subject_fasta:
        s_fasta_lengths = read_fasta_lengths(args.subject_fasta)
        s_seq_lengths = merge_lengths(s_seq_lengths, s_fasta_lengths)
        eprint(f"[INFO] loaded subject FASTA lengths: {len(s_fasta_lengths)} from {args.subject_fasta}")

    anchors = read_aligncoords(args.aligncoords)
    eprint(f"[INFO] loaded anchors: {len(anchors)} from {args.aligncoords}")
    if not anchors:
        raise SystemExit("[ERROR] no anchors read from aligncoords")

    # Query/subject seqids can be specified in --query-region/--subject-region too.
    query_seqid = args.query_seqid
    query_region_req = None
    if args.query_region:
        qseq_r, q0_r, q1_r = parse_region(args.query_region)
        query_seqid = qseq_r
        query_region_req = (q0_r, q1_r)

    subject_seqid = args.subject_seqid
    subject_region_req = None
    if args.subject_region:
        sseq_r, s0_r, s1_r = parse_region(args.subject_region)
        subject_seqid = sseq_r
        subject_region_req = (s0_r, s1_r)

    anchors = orient_anchors(
        anchors,
        query_seqids_from_gff=set(q_by_seq),
        subject_seqids_from_gff=set(s_by_seq),
        query_seqid=query_seqid,
        subject_seqid=subject_seqid,
        force_swap=args.swap_aligncoords,
        auto_orient=(not args.no_auto_orient),
    )

    if not query_seqid:
        query_seqid = choose_most_common_seqid(anchors, "query")
    if not subject_seqid:
        subject_seqid = choose_most_common_seqid(anchors, "subject", restrict_other=("query", query_seqid))

    q_len = args.query_len or infer_seq_len(query_seqid, q_by_seq, q_seq_lengths, anchors, "query")
    s_len = args.subject_len or infer_seq_len(subject_seqid, s_by_seq, s_seq_lengths, anchors, "subject")

    if query_region_req is None or query_region_req[0] is None or query_region_req[1] is None:
        q0, q1 = 0, q_len
    else:
        q0 = max(0, min(query_region_req[0] - args.query_pad, q_len))
        q1 = max(q0 + 1, min(query_region_req[1] + args.query_pad, q_len))

    pair_anchors = [a for a in anchors if a.qseq == query_seqid and a.sseq == subject_seqid and overlaps(a.qlo, a.qhi, q0, q1)]
    eprint(f"[INFO] anchors on selected query/subject before clustering: {len(pair_anchors)}")
    if args.keep_largest_cluster:
        clusters = cluster_anchors(pair_anchors, by=args.cluster_by, gap=args.cluster_gap)
        clusters = [c for c in clusters if len(c) >= args.min_cluster_anchors] or clusters
        if clusters:
            clusters.sort(key=lambda c: (len(c), sum(a.qhi - a.qlo for a in c), sum(a.shi - a.slo for a in c)), reverse=True)
            pair_anchors = clusters[0]
            eprint(f"[INFO] kept largest {args.cluster_by}-cluster: anchors={len(pair_anchors)}; total_clusters={len(clusters)}")
    if len(pair_anchors) < args.min_anchors:
        raise SystemExit(f"[ERROR] too few anchors after filtering: {len(pair_anchors)} < {args.min_anchors}")

    if args.subject_region_from_anchors:
        if subject_region_req is None or subject_region_req[0] is None or subject_region_req[1] is None:
            s0 = max(0, min(a.slo for a in pair_anchors) - args.subject_pad)
            s1 = min(s_len, max(a.shi for a in pair_anchors) + args.subject_pad)
        else:
            s0 = max(0, min(subject_region_req[0], s_len))
            s1 = max(s0 + 1, min(subject_region_req[1], s_len))
    else:
        # New default: draw the complete selected subject chromosome/contig, not only the syntenic span.
        if subject_region_req is None or subject_region_req[0] is None or subject_region_req[1] is None:
            s0, s1 = 0, s_len
        else:
            s0 = max(0, min(subject_region_req[0], s_len))
            s1 = max(s0 + 1, min(subject_region_req[1], s_len))

    # Keep only anchors visible on both axes.
    visible_anchors = [a for a in pair_anchors if overlaps(a.qlo, a.qhi, q0, q1) and overlaps(a.slo, a.shi, s0, s1)]
    if len(visible_anchors) < args.min_anchors:
        raise SystemExit(f"[ERROR] too few visible anchors: {len(visible_anchors)} < {args.min_anchors}")

    q_feats = q_by_seq.get(query_seqid, [])
    s_feats = s_by_seq.get(subject_seqid, [])
    subject_context_seqid = resolve_context_seqid(subject_seqid, subject_context_by_seq)
    subject_context_features = subject_context_by_seq.get(subject_context_seqid, [])
    if args.subject_context_gff and subject_context_seqid and subject_context_seqid != subject_seqid:
        eprint(f"[INFO] mapped published context seqid {subject_context_seqid} to subject seqid {subject_seqid}")
    elif args.subject_context_gff and not subject_context_seqid:
        eprint(f"[WARN] no published context seqid could be matched to subject seqid {subject_seqid}")
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_pdf = out_prefix.with_suffix(".pdf")
    title = f"{args.query_name} vs {args.subject_name} - full-chromosome synteny"

    plot_pair(
        out_pdf=out_pdf,
        title=title,
        query_name=args.query_name,
        subject_name=args.subject_name,
        query_seqid=query_seqid,
        subject_seqid=subject_seqid,
        query_region=(q0, q1),
        subject_region=(s0, s1),
        query_features=q_feats,
        subject_features=s_feats,
        anchors=visible_anchors,
        subject_context_features=subject_context_features,
        project_centromere=args.project_centromere,
        query_on_top=args.query_on_top,
        label_mode=args.label_features,
        one_based_axis=args.axis_one_based,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
        ribbon_alpha=args.ribbon_alpha,
        tick_interval=args.tick_interval,
        tick_label_interval=args.tick_label_interval,
    )

    summary_path = out_prefix.with_suffix(".summary.tsv")
    q_visible_feats = features_overlapping(q_feats, q0, q1)
    s_visible_feats = features_overlapping(s_feats, s0, s1)
    s_visible_context = features_overlapping(subject_context_features, s0, s1)
    projected_centromeres = projected_query_centromeres(
        [feature for feature in s_visible_context if feature.ftype == "centromere"],
        visible_anchors,
        query_seqid,
    ) if args.project_centromere else []
    with open_text(summary_path, "wt") as oh:
        oh.write("key\tvalue\n")
        rows = {
            "aligncoords": str(args.aligncoords),
            "query_name": args.query_name,
            "subject_name": args.subject_name,
            "query_seqid": query_seqid,
            "query_start": q0,
            "query_end": q1,
            "query_length": q1 - q0,
            "subject_seqid": subject_seqid,
            "subject_start": s0,
            "subject_end": s1,
            "subject_length": s1 - s0,
            "shared_plot_span": max(q1 - q0, s1 - s0),
            "tick_interval": args.tick_interval,
            "tick_label_interval": args.tick_label_interval if args.tick_label_interval else "auto",
            "anchor_count": len(visible_anchors),
            "orientation": infer_orientation(visible_anchors),
            "query_feature_count": len(q_visible_feats),
            "query_tRNA_rRNA_count": sum(1 for f in q_visible_feats if f.is_special),
            "subject_feature_count": len(s_visible_feats),
            "subject_tRNA_rRNA_count": sum(1 for f in s_visible_feats if f.is_special),
            "evaluation_reference_annotation": str(args.subject_gff),
            "evaluation_reference_annotation_role": "self annotation",
            "published_context_annotation": str(args.subject_context_gff),
            "published_context_seqid": subject_context_seqid,
            "published_context_usage": "centromere/telomere markers only; excluded from quantitative evaluation",
            "subject_published_centromere_count": sum(1 for f in s_visible_context if f.ftype == "centromere"),
            "subject_published_telomere_count": sum(1 for f in s_visible_context if f.ftype == "telomere"),
            "query_projected_centromere_count": len(projected_centromeres),
            "subject_telomere_marker_count": sum(1 for f in s_visible_context if f.ftype == "telomere") or 2,
            "query_telomere_end_marker_count": 2,
            "pdf": str(out_pdf),
        }
        for k, v in rows.items():
            oh.write(f"{k}\t{v}\n")

    eprint(f"[OK] wrote {out_pdf}")
    eprint(f"[OK] wrote {summary_path}")
    return 0


# ----------------------------- demo orchestration ------------------------

SYNTENY_REQUIRED_ASSETS = (
    "data/resolved_manifest.tsv",
    "data/AIYeast00.fa",
    "data/AIYeast00.combined.qc_pass.gff3",
    "data/S288C_chrIII.fa",
    "data/S288C_chrIII.combined.qc_pass.gff3",
    "data/S288C_chrIII.structural_context.gff3",
    "data/60_protein_synteny_work/02.dag/AIYeast00_S288C_chrIII.pos.aligncoords",
    "data/60_protein_synteny_work/run_parameters.json",
)


def _missing_assets(root: Path) -> list[str]:
    return [relative for relative in SYNTENY_REQUIRED_ASSETS if not (root / relative).is_file()]


def _copy_absent_tree(source: Path, dest: Path) -> None:
    """Copy files from source into dest without replacing existing files."""
    if not source.is_dir():
        return
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        target = dest / path.relative_to(source)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def ensure_demo_assets(
    root: str | Path,
    *,
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    remote_prefix: Optional[str] = None,
) -> dict[str, object]:
    """Download the synteny data subtree from a Hugging Face dataset if absent.

    Defaults to the ``biomap-research/DeepD`` dataset (override with arguments
    or ``DEEPD_DATA_REPO_ID`` / ``DEEPD_DATA_REVISION`` / ``DEEPD_SYNTENY_DATA_PREFIX``).
    The remote prefix should contain a ``data/`` directory matching this module.
    Existing local assets are never overwritten.
    """
    root = Path(root).resolve()
    missing = _missing_assets(root)
    if not missing:
        return {"downloaded": False, "root": str(root), "asset_count": len(SYNTENY_REQUIRED_ASSETS)}

    repo_id = repo_id or os.environ.get("DEEPD_DATA_REPO_ID") or DEFAULT_HF_DATASET_REPO
    revision = revision or os.environ.get("DEEPD_DATA_REVISION") or DEFAULT_HF_REVISION
    remote_prefix = remote_prefix or os.environ.get(
        "DEEPD_SYNTENY_DATA_PREFIX", DEFAULT_SYNTENY_REMOTE_PREFIX
    )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download missing demo assets. "
            "Install the dependencies from requirements.txt."
        ) from exc

    prefix = remote_prefix.strip("/")
    pattern = f"{prefix}/data/**" if prefix else "data/**"
    try:
        snapshot = Path(snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=[pattern],
            token=os.environ.get("HF_TOKEN"),
        ))
    except Exception as exc:
        raise FileNotFoundError(
            f"Synteny demo assets are missing locally and could not be downloaded "
            f"from Hugging Face dataset {repo_id}. Upload synteny/data "
            f"to that dataset, or place the files under {root}. "
            f"Missing: {', '.join(missing)}"
        ) from exc
    source_root = snapshot / prefix if prefix else snapshot
    source_data = source_root / "data"
    if not source_data.is_dir():
        raise FileNotFoundError(
            f"Downloaded snapshot does not contain the expected data directory: {source_data}"
        )
    _copy_absent_tree(source_data, root / "data")

    missing = _missing_assets(root)
    if missing:
        raise FileNotFoundError(
            "The Hugging Face snapshot is incomplete. Missing files after download: "
            + ", ".join(missing)
        )
    return {
        "downloaded": True,
        "root": str(root),
        "repo_id": repo_id,
        "revision": revision,
        "remote_prefix": prefix,
        "asset_count": len(SYNTENY_REQUIRED_ASSETS),
    }


def _read_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: value or "" for key, value in row.items()} for row in csv.DictReader(handle, delimiter="\t")]


def _asset_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def audit_synteny_inputs(root: str | Path) -> dict[str, object]:
    root = Path(root).resolve()
    parameters_path = root / "data/60_protein_synteny_work/run_parameters.json"
    aligncoords = root / "data/60_protein_synteny_work/02.dag/AIYeast00_S288C_chrIII.pos.aligncoords"
    parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
    lines = aligncoords.read_text(encoding="utf-8").splitlines()
    headers = [line for line in lines if line.startswith("## alignment")]
    anchors = [line for line in lines if line and not line.startswith("#")]
    return {
        "blastp_evalue": parameters["evalue"],
        "blastp_max_target_seqs": parameters["max_target_seqs"],
        "dag_d": parameters["dag_d"],
        "dag_g": parameters["dag_g"],
        "dag_a": parameters["dag_a"],
        "alignment_count": len(headers),
        "anchor_count": len(anchors),
    }


def run_synteny_demo(
    root: str | Path,
    *,
    download_if_missing: bool = True,
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    remote_prefix: Optional[str] = None,
) -> dict[str, object]:
    """Ensure assets, render every generated sample, and return a compact report."""
    root = Path(root).resolve()
    if download_if_missing:
        ensure_demo_assets(root, repo_id=repo_id, revision=revision, remote_prefix=remote_prefix)
    missing = _missing_assets(root)
    if missing:
        raise FileNotFoundError("Missing synteny demo assets: " + ", ".join(missing))

    manifest_path = root / "data/resolved_manifest.tsv"
    rows = _read_manifest_rows(manifest_path)
    references = [row for row in rows if row.get("source") == "reference"]
    generated_rows = [row for row in rows if row.get("source") == "generated"]
    if len(references) != 1:
        raise ValueError(f"Expected exactly one reference row; found {len(references)}")
    reference = references[0]

    results_root = root / "results/formal"
    results_root.mkdir(parents=True, exist_ok=True)
    run_parameters = {
        "tick_interval": 50_000,
        "tick_label_interval": 0,
        "fig_width": 13.5,
        "fig_height": 5.6,
        "ribbon_alpha": 0.26,
        "label_features": "none",
        "project_centromere": True,
    }
    (results_root / "run_parameters.json").write_text(
        json.dumps(run_parameters, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    status_lines = ["sample\tstatus\taligncoords\tpdf\n"]
    pdf_paths: list[str] = []

    for generated in generated_rows:
        sample = generated["sample"]
        reference_sample = reference["sample"]
        aligncoords = root / "data/60_protein_synteny_work/02.dag" / f"{sample}_{reference_sample}.pos.aligncoords"
        prefix = results_root / sample / f"{sample}_vs_{reference_sample}_fullchrom_synteny"
        prefix.parent.mkdir(parents=True, exist_ok=True)
        args = [
            "--aligncoords", str(aligncoords),
            "--query-gff", str(_asset_path(root, generated["combined_gff"])),
            "--subject-gff", str(_asset_path(root, reference["combined_gff"])),
            "--query-fasta", str(_asset_path(root, generated["genome_fasta"])),
            "--subject-fasta", str(_asset_path(root, reference["genome_fasta"])),
            "--query-name", sample,
            "--subject-name", reference_sample,
            "--tick-interval", "50000",
            "--tick-label-interval", "0",
            "--fig-width", "13.5",
            "--fig-height", "5.6",
            "--ribbon-alpha", "0.26",
            "--label-features", "none",
            "--out-prefix", str(prefix),
            "--project-centromere",
        ]
        context_gff = reference.get("context_annotation_gff", "")
        if context_gff:
            args.extend(["--subject-context-gff", str(_asset_path(root, context_gff))])
        plot_main(args)
        pdf = prefix.with_suffix(".pdf")
        pdf_paths.append(str(pdf.relative_to(root)))
        status_lines.append(
            f"{sample}\tok\t{aligncoords.relative_to(root)}\t{pdf.relative_to(root)}\n"
        )

    (results_root / "synteny_plot_status.tsv").write_text("".join(status_lines), encoding="utf-8")
    report = audit_synteny_inputs(root)
    report.update({"status": "passed", "samples": len(generated_rows), "pdfs": pdf_paths})
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the self-contained DeepD synteny demo.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--repo-id",
        default=None,
        help=f"Hugging Face dataset repository ID (default: {DEFAULT_HF_DATASET_REPO})",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--remote-prefix", default=None)
    args = parser.parse_args(argv)
    report = run_synteny_demo(
        args.root,
        repo_id=args.repo_id,
        revision=args.revision,
        remote_prefix=args.remote_prefix,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
