from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import mathtext
from matplotlib.font_manager import FontProperties, findfont
from scipy import stats

from analysis_all_common import clean_axis, significance_symbol


COLORS = {
    "R": (0.70, 0.55, 0.95),
    "T": (0.00, 0.784, 0.784),
    "RT": (0.00, 0.784, 0.784),
    "TR": (0.70, 0.55, 0.95),
}


# ``UnicodeFonts`` was public in older Matplotlib releases, then moved to the
# private ``_mathtext`` implementation module.  Keep the Arial MathText
# renderer usable with both layouts.
try:
    _UnicodeFonts = mathtext.UnicodeFonts
except AttributeError:
    from matplotlib import _mathtext

    _UnicodeFonts = _mathtext.UnicodeFonts


_SYMBOL_GREEK_CODEPOINTS = {
    # Greek capitals in the legacy Microsoft Symbol character map.
    0x0391: 0xF041,  # Alpha
    0x0392: 0xF042,  # Beta
    0x0393: 0xF047,  # Gamma
    0x0394: 0xF044,  # Delta
    0x0395: 0xF045,  # Epsilon
    0x0396: 0xF05A,  # Zeta
    0x0397: 0xF048,  # Eta
    0x0398: 0xF051,  # Theta
    0x0399: 0xF049,  # Iota
    0x039A: 0xF04B,  # Kappa
    0x039B: 0xF04C,  # Lambda
    0x039C: 0xF04D,  # Mu
    0x039D: 0xF04E,  # Nu
    0x039E: 0xF058,  # Xi
    0x039F: 0xF04F,  # Omicron
    0x03A0: 0xF050,  # Pi
    0x03A1: 0xF052,  # Rho
    0x03A3: 0xF053,  # Sigma
    0x03A4: 0xF054,  # Tau
    0x03A5: 0xF055,  # Upsilon
    0x03A6: 0xF046,  # Phi
    0x03A7: 0xF043,  # Chi
    0x03A8: 0xF059,  # Psi
    0x03A9: 0xF057,  # Omega
    # Greek lowercase and the variants used by Matplotlib MathText.
    0x03B1: 0xF061,  # alpha
    0x03B2: 0xF062,  # beta
    0x03B3: 0xF067,  # gamma
    0x03B4: 0xF064,  # delta
    0x03B5: 0xF065,  # epsilon
    0x03B6: 0xF07A,  # zeta
    0x03B7: 0xF068,  # eta
    0x03B8: 0xF071,  # theta
    0x03B9: 0xF069,  # iota
    0x03BA: 0xF06B,  # kappa
    0x03BB: 0xF06C,  # lambda
    0x03BC: 0xF06D,  # mu
    0x03BD: 0xF06E,  # nu
    0x03BE: 0xF078,  # xi
    0x03BF: 0xF06F,  # omicron
    0x03C0: 0xF070,  # pi
    0x03C1: 0xF072,  # rho
    0x03C2: 0xF056,  # final sigma
    0x03C3: 0xF073,  # sigma
    0x03C4: 0xF074,  # tau
    0x03C5: 0xF075,  # upsilon
    0x03C6: 0xF066,  # phi
    0x03C7: 0xF063,  # chi
    0x03C8: 0xF079,  # psi
    0x03C9: 0xF077,  # omega
    0x03D1: 0xF04A,  # vartheta
    0x03D5: 0xF06A,  # phi symbol (MathText \phi)
    0x03D6: 0xF076,  # varpi
    0x03F1: 0xF072,  # varrho
    0x03F5: 0xF065,  # lunate epsilon
}


class _ArialMathFonts(_UnicodeFonts):
    """Use Arial for text-like math and Symbol for Greek math glyphs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fontmap["symbol"] = findfont(FontProperties(family="Symbol"))
        # Windows' Symbol font exposes its glyphs through the MS Symbol
        # charmap (``symb``), using private-use code points U+F020-U+F0FE.
        self._get_font("symbol").select_charmap(0x73796D62)

    def _get_glyph(self, fontname, font_class, sym, *args, **kwargs):
        try:
            unicode_index = ord(sym) if len(sym) == 1 else None
        except TypeError:
            unicode_index = None
        if unicode_index is None:
            try:
                from matplotlib._mathtext import get_unicode_index

                unicode_index = get_unicode_index(sym)
            except (ImportError, ValueError):
                unicode_index = None

        symbol_index = _SYMBOL_GREEK_CODEPOINTS.get(unicode_index)
        if symbol_index is not None:
            font = self._get_font("symbol")
            glyph_index = font.get_char_index(symbol_index)
            if glyph_index != 0:
                if args or "fontsize" in kwargs:
                    fontsize = args[0] if args else kwargs["fontsize"]
                    return (
                        font,
                        symbol_index,
                        font.get_glyph_name(glyph_index),
                        fontsize,
                        False,
                    )
                return font, symbol_index, False

        # Arial lacks U+22EF (MIDLINE HORIZONTAL ELLIPSIS), which makes
        # ``\cdots`` fall back to STIXGeneral in editable SVG output.  Arial's
        # U+2026 ellipsis is the closest single-glyph substitute.
        if sym == r"\cdots":
            font = self._get_font("rm")
            unicode_index = 0x2026
            glyph_index = font.get_char_index(unicode_index)
            if glyph_index != 0:
                if args or "fontsize" in kwargs:
                    fontsize = args[0] if args else kwargs["fontsize"]
                    return (
                        font,
                        unicode_index,
                        font.get_glyph_name(glyph_index),
                        fontsize,
                        False,
                    )
                return font, unicode_index, False
        return super()._get_glyph(fontname, font_class, sym, *args, **kwargs)


# Keep Matplotlib's public "custom" fontset while routing Greek glyphs through
# Symbol and Latin/text-like mathematical glyphs through Arial.
mathtext.MathTextParser._font_type_mapping["custom"] = _ArialMathFonts


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.sf": "Arial",
            "mathtext.tt": "Arial",
            "mathtext.cal": "Arial:italic",
            "axes.titlesize": 6,
            "axes.labelsize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "text.color": "black",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def normalize(value: np.ndarray | float, center: float, half_range: float):
    return (np.asarray(value, dtype=float) - center) / half_range


def build_matched_rows(condition_rows):
    probe = {}
    normal = {}
    for row in condition_rows:
        key = (str(row["subject"]), float(row["stimx"]), float(row["stimt"]))
        trialtype = str(row["trialtype"]).lower()
        if "probe" in trialtype:
            probe[key] = row
        elif "normal" in trialtype:
            normal[key] = row

    matched = []
    for key in sorted(set(probe) & set(normal)):
        p = probe[key]
        n = normal[key]
        matched.append(
            {
                "subject": key[0],
                "stim_r": key[1],
                "stim_t": key[2],
                "perceived_r": float(p["respx"]),
                "perceived_t": float(p["respt"]),
                "produced_r": float(n["respx"]),
                "produced_t": float(n["respt"]),
                "probe_n": int(p["n"]),
                "normal_n": int(n["n"]),
            }
        )
    return matched


def fit_subject_models(matched_rows, normalization):
    subjects = sorted({row["subject"] for row in matched_rows})
    coefficients = []
    subject_arrays = {}
    for subject in subjects:
        rows = [row for row in matched_rows if row["subject"] == subject]
        r_in = normalize(
            [row["perceived_r"] for row in rows], normalization["R"]["center"], normalization["R"]["half_range"]
        )
        t_in = normalize(
            [row["perceived_t"] for row in rows], normalization["T"]["center"], normalization["T"]["half_range"]
        )
        r_out = normalize(
            [row["produced_r"] for row in rows], normalization["R"]["center"], normalization["R"]["half_range"]
        )
        t_out = normalize(
            [row["produced_t"] for row in rows], normalization["T"]["center"], normalization["T"]["half_range"]
        )
        valid = np.isfinite(r_in) & np.isfinite(t_in) & np.isfinite(r_out) & np.isfinite(t_out)
        design = np.column_stack([np.ones(int(valid.sum())), r_in[valid], t_in[valid]])
        beta_r, *_ = np.linalg.lstsq(design, r_out[valid], rcond=None)
        beta_t, *_ = np.linalg.lstsq(design, t_out[valid], rcond=None)
        coefficients.append(
            {
                "subject": subject,
                "n_conditions": int(valid.sum()),
                "intercept_R": float(beta_r[0]),
                "beta_RR": float(beta_r[1]),
                "beta_RT": float(beta_r[2]),
                "intercept_T": float(beta_t[0]),
                "beta_TR": float(beta_t[1]),
                "beta_TT": float(beta_t[2]),
                "condition_number": float(np.linalg.cond(design)),
            }
        )
        subject_arrays[subject] = {
            "R_in": r_in[valid],
            "T_in": t_in[valid],
            "R_out": r_out[valid],
            "T_out": t_out[valid],
        }
    return coefficients, subject_arrays


def fit_probe_models(condition_rows, normalization):
    probe_rows = [row for row in condition_rows if "probe" in str(row["trialtype"]).lower()]
    subjects = sorted({str(row["subject"]) for row in probe_rows})
    coefficients = []
    subject_arrays = {}
    for subject in subjects:
        rows = [row for row in probe_rows if str(row["subject"]) == subject]
        r_in = normalize(
            [row["stimx"] for row in rows], normalization["R"]["center"], normalization["R"]["half_range"]
        )
        t_in = normalize(
            [row["stimt"] for row in rows], normalization["T"]["center"], normalization["T"]["half_range"]
        )
        r_out = normalize(
            [row["respx"] for row in rows], normalization["R"]["center"], normalization["R"]["half_range"]
        )
        t_out = normalize(
            [row["respt"] for row in rows], normalization["T"]["center"], normalization["T"]["half_range"]
        )
        valid = np.isfinite(r_in) & np.isfinite(t_in) & np.isfinite(r_out) & np.isfinite(t_out)
        design = np.column_stack([np.ones(int(valid.sum())), r_in[valid], t_in[valid]])
        beta_r, *_ = np.linalg.lstsq(design, r_out[valid], rcond=None)
        beta_t, *_ = np.linalg.lstsq(design, t_out[valid], rcond=None)
        coefficients.append(
            {
                "subject": subject,
                "n_conditions": int(valid.sum()),
                "intercept_R": float(beta_r[0]),
                "beta_RR": float(beta_r[1]),
                "beta_RT": float(beta_r[2]),
                "intercept_T": float(beta_t[0]),
                "beta_TR": float(beta_t[1]),
                "beta_TT": float(beta_t[2]),
                "condition_number": float(np.linalg.cond(design)),
            }
        )
        subject_arrays[subject] = {
            "R_in": r_in[valid],
            "T_in": t_in[valid],
            "R_out": r_out[valid],
            "T_out": t_out[valid],
        }
    return coefficients, subject_arrays, probe_rows


def component_plus_residual_within_subject(subject_arrays, x_key: str, y_key: str, covariate_key: str):
    all_x = []
    all_y = []
    for arrays in subject_arrays.values():
        x = arrays[x_key]
        y = arrays[y_key]
        covariate = arrays[covariate_key]
        design = np.column_stack([np.ones(covariate.size), x, covariate])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        adjusted_y = y - beta[0] - beta[2] * covariate
        all_x.append(x)
        all_y.append(adjusted_y)
    return np.concatenate(all_x), np.concatenate(all_y)


def plot_partial(ax, x, y, title, xlabel, ylabel, color, identity=False):
    ax.scatter(x, y, s=10, alpha=0.20, color=color, edgecolor="none")
    slope, intercept = np.polyfit(x, y, 1)
    xx = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    if identity:
        ax.plot(xx, xx, color="0.45", lw=0.8, ls="--")
    ax.plot(xx, slope * xx + intercept, color=color, lw=1.5)
    ax.axhline(0, color="0.65", lw=0.7, ls="--")
    ax.axvline(0, color="0.65", lw=0.7, ls="--")
    r = float(np.corrcoef(x, y)[0, 1])
    ax.set_title(f"{title}\nPooled descriptive $r$ = {r:.2f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    clean_axis(ax)
    return {
        "n_condition_means": int(x.size),
        "r_descriptive": r,
        "slope_descriptive": float(slope),
        "identity_line": bool(identity),
        "display_role": "Pooled condition-level descriptive visualization only.",
        "inference": "No inferential p value is displayed in this relationship panel; group inference uses subject-specific coefficients in the corresponding coefficient panel.",
    }


def bootstrap_median_ci(values, seed=20260624, n_boot=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.RandomState(seed)
    samples = rng.choice(values, size=(n_boot, values.size), replace=True)
    medians = np.median(samples, axis=1)
    return [float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))]


def _wilcoxon_signed_rank(*args):
    """Return the exact two-sided Wilcoxon statistic across SciPy versions."""
    if len(args) == 1:
        difference = np.asarray(args[0], dtype=float)
    elif len(args) == 2:
        difference = np.asarray(args[0], dtype=float) - np.asarray(args[1], dtype=float)
    else:
        raise TypeError("Wilcoxon signed-rank test expects one difference array or two paired arrays.")

    if not np.all(np.isfinite(difference)):
        raise ValueError("Wilcoxon signed-rank differences must be finite.")

    # Match zero_method="wilcox": zero differences do not enter the rank sum or
    # the sign-randomization distribution.
    difference = difference[difference != 0]
    if difference.size == 0:
        return SimpleNamespace(statistic=0.0, pvalue=1.0)

    ranks = stats.rankdata(np.abs(difference), method="average")
    positive_rank_sum = float(np.sum(ranks[difference > 0]))
    negative_rank_sum = float(np.sum(ranks[difference < 0]))
    statistic = min(positive_rank_sum, negative_rank_sum)

    # Average ranks are integers or half-integers. Doubling makes the exact
    # subset-sum distribution integer-valued and therefore also supports ties.
    doubled_ranks = np.rint(2.0 * ranks).astype(int)
    total_rank = int(np.sum(doubled_ranks))
    counts = [0] * (total_rank + 1)
    counts[0] = 1
    reachable_total = 0
    for rank in doubled_ranks:
        for rank_sum in range(reachable_total, -1, -1):
            if counts[rank_sum]:
                counts[rank_sum + rank] += counts[rank_sum]
        reachable_total += int(rank)

    observed = int(round(2.0 * statistic))
    favourable = sum(
        count
        for rank_sum, count in enumerate(counts)
        if min(rank_sum, total_rank - rank_sum) <= observed
    )
    pvalue = min(1.0, favourable / float(2 ** difference.size))
    return SimpleNamespace(statistic=statistic, pvalue=pvalue)


def wilcoxon_test(x, y=None, reference=0.0):
    x = np.asarray(x, dtype=float)
    if y is None:
        difference = x - float(reference)
        result = _wilcoxon_signed_rank(difference)
    else:
        y = np.asarray(y, dtype=float)
        result = _wilcoxon_signed_rank(x, y)
        difference = x - y
    return {
        "test": "two-sided exact Wilcoxon signed-rank",
        "n": int(difference.size),
        "statistic": float(result.statistic),
        "p_raw": float(result.pvalue),
        "reference": None if y is not None else float(reference),
        "median_difference": float(np.median(difference)),
        "direction": "positive" if np.median(difference) > 0 else "negative" if np.median(difference) < 0 else "zero",
    }


def p_text(p):
    if p < 0.001:
        exponent = int(np.floor(np.log10(p)))
        mantissa = p / (10.0 ** exponent)
        return rf"$p$ = {mantissa:.2f} $\times$ 10$^{{{exponent}}}$"
    return f"$p$ = {p:.3f}"


def p_stars(p):
    return significance_symbol(p)


def q_text(q):
    if q < 0.001:
        exponent = int(np.floor(np.log10(q)))
        mantissa = q / (10.0 ** exponent)
        return rf"$q$ = {mantissa:.2f} $\times$ 10$^{{{exponent}}}$"
    return f"$q$ = {q:.3f}"


def adjusted_p_text(p):
    return significance_symbol(p)


def adjusted_p_value_text(p):
    return significance_symbol(p)


def bh_fdr(pvalues):
    """Benjamini-Hochberg FDR adjusted p-values (q-values) for one test family."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(p))
    if finite.size == 0:
        return out
    order = finite[np.argsort(p[finite])]
    ranked = p[order]
    k = ranked.size
    q = ranked * k / np.arange(1, k + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]  # enforce monotonicity
    q = np.clip(q, 0.0, 1.0)
    out[order] = q
    return out


def group_summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "bootstrap_95_ci_median": bootstrap_median_ci(values),
        "positive": int(np.sum(values > 0)),
        "negative": int(np.sum(values < 0)),
        "zero": int(np.sum(values == 0)),
    }


def plot_coefficient_comparison(
    ax,
    first,
    second,
    labels,
    colors,
    title,
    references=(0.0, 0.0),
    compact=False,
    compact_group_names=("R", "T"),
    compact_inline_group=False,
    outside_adjusted_p_only=False,
):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    data = [first, second]
    box = ax.boxplot(
        data,
        positions=[0, 1],
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "lw": 1.2},
        whiskerprops={"color": "black", "lw": 0.8},
        capprops={"color": "black", "lw": 0.8},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)

    jitter = np.linspace(-0.07, 0.07, first.size)
    for i in range(first.size):
        ax.plot([jitter[i], 1 + jitter[i]], [first[i], second[i]], color="0.70", lw=0.5, alpha=0.55, zorder=1)
    ax.scatter(jitter, first, s=13, color=colors[0], edgecolor="white", linewidth=0.3, zorder=2)
    ax.scatter(1 + jitter, second, s=13, color=colors[1], edgecolor="white", linewidth=0.3, zorder=2)
    visual_reference = references[0] if references[0] == references[1] else 0.0
    ax.axhline(visual_reference, color="0.35", lw=0.8, ls="--")

    tests = [
        wilcoxon_test(first, reference=references[0]),
        wilcoxon_test(second, reference=references[1]),
        wilcoxon_test(first, second),
    ]
    q_values = bh_fdr([t["p_raw"] for t in tests])
    for t, q in zip(tests, q_values):
        t["p_bh"] = float(q)

    span = max(float(np.ptp(np.concatenate(data))), 0.5)
    ymax = float(np.max(np.concatenate(data)))
    ymin = float(np.min(np.concatenate(data)))
    if outside_adjusted_p_only:
        data_span = max(ymax - ymin, 1e-6)
        padding = 0.08 * data_span
        lower = min(ymin, visual_reference) - padding
        upper = max(ymax, visual_reference) + padding
        axis_transform = ax.get_xaxis_transform()
        ax.plot(
            [0, 0, 1, 1],
            [1.01, 1.04, 1.04, 1.01],
            color="black",
            lw=0.8,
            transform=axis_transform,
            clip_on=False,
        )
        ax.text(
            0.5,
            1.055,
            adjusted_p_text(tests[2]["p_bh"]),
            ha="center",
            va="bottom",
            fontsize=10,
            transform=axis_transform,
            clip_on=False,
        )
        # Keep narrow publication panels readable: the dashed reference line
        # identifies the null value, while each adjusted significance symbol
        # sits above its corresponding box.  Stacking formula, null value, and
        # symbol into every x tick made the direct-export version illegible.
        for position, test in zip((0, 1), tests[:2]):
            ax.text(
                position,
                0.985,
                adjusted_p_text(test["p_bh"]),
                ha="center",
                va="top",
                transform=axis_transform,
                clip_on=False,
            )
        ax.set_ylim(lower, upper)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_title(title, pad=24)
        ax.set_ylabel("Normalized gain")
        clean_axis(ax)
        return {
            labels[0]: group_summary(first),
            labels[1]: group_summary(second),
            "tests": {
                f"{labels[0]}_vs_{references[0]:g}": tests[0],
                f"{labels[1]}_vs_{references[1]:g}": tests[1],
                f"{labels[0]}_vs_{labels[1]}": tests[2],
                "multiple_comparison": "Benjamini-Hochberg FDR across the 3 tests in this panel; raw p (p_raw) and adjusted p (p_bh) are stored, while the figure displays significance symbols derived from p_bh.",
            },
        }

    top = ymax + 0.11 * span
    ax.plot([0, 0, 1, 1], [top, top + 0.04 * span, top + 0.04 * span, top], color="black", lw=0.8)
    ax.text(0.5, top + 0.05 * span,
            p_stars(tests[2]['p_bh']),
            ha="center", va="bottom", fontsize=10)
    if compact:
        first_name, second_name = compact_group_names
        if compact_inline_group:
            compact_text = (
                f"{first_name} vs {references[0]:g}: {p_stars(tests[0]['p_bh'])}\n"
                f"{second_name} vs {references[1]:g}: {p_stars(tests[1]['p_bh'])}"
            )
            lower_scale = 0.72
        else:
            compact_text = (
                f"{first_name} vs {references[0]:g}:\n{p_stars(tests[0]['p_bh'])}\n"
                f"{second_name} vs {references[1]:g}:\n{p_stars(tests[1]['p_bh'])}"
            )
            lower_scale = 0.82
        ax.text(
            0.5,
            ymin - 0.11 * span,
            compact_text,
            ha="center",
            va="top",
            fontsize=10,
        )
        ax.set_ylim(ymin - lower_scale * span, top + 0.30 * span)
    else:
        ax.text(0, ymin - 0.11 * span,
                f"vs {references[0]:g}:\n{p_stars(tests[0]['p_bh'])}",
                ha="center", va="top", fontsize=10)
        ax.text(1, ymin - 0.11 * span,
                f"vs {references[1]:g}:\n{p_stars(tests[1]['p_bh'])}",
                ha="center", va="top", fontsize=10)
        ax.set_ylim(ymin - 0.50 * span, top + 0.30 * span)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel("Normalized gain")
    clean_axis(ax)
    return {
        labels[0]: group_summary(first),
        labels[1]: group_summary(second),
        "tests": {
            f"{labels[0]}_vs_{references[0]:g}": tests[0],
            f"{labels[1]}_vs_{references[1]:g}": tests[1],
            f"{labels[0]}_vs_{labels[1]}": tests[2],
            "multiple_comparison": "Benjamini-Hochberg FDR across the 3 tests in this panel; raw p (p_raw) and BH-adjusted p (p_bh) are stored, while the figure displays significance symbols derived from p_bh.",
        },
    }


def label_panel(ax, label):
    panel_label = str(label).lower()
    return ax.text(
        -0.20,
        1.16,
        panel_label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
        gid=f"panel_label_{panel_label}",
    )
