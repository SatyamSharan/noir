
# schwartz_radar_plus_breakdown.py
# Streamlit app: Radar view + Per-Value Breakdown (roomier breakdown chart)
#
# HOW TO RUN:
#   pip install streamlit pandas matplotlib openpyxl numpy
#   streamlit run schwartz_radar_plot.py

import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Schwartz Values", layout="wide")

# ---------- CSS: center fixed-width plot container ----------
st.markdown(
    '''
    <style>
    .fixed-plot-wrap { width: var(--plot-width, 760px); margin: 0 auto; }
    .small-note { color: #666; font-size: 0.85rem; }
    </style>
    ''',
    unsafe_allow_html=True
)

# ---------- Helpers ----------

def normalize_name(n, post_suffix="1"):
    s = str(n).strip()
    if post_suffix and s.endswith(post_suffix):
        return s[: -len(post_suffix)], True
    return s, False

def detect_name_col(df):
    for cand in ["Name", "Student", "Student Name", "ID"]:
        if cand in df.columns:
            return cand
    return df.columns[0]

def numeric_param_cols(df, name_col):
    cols = [c for c in df.columns if c != name_col]
    num_cols = []
    for c in cols:
        try:
            pd.to_numeric(df[c], errors="raise")
            num_cols.append(c)
        except Exception:
            pass
    return num_cols

def polygon_centroid_xy(xs, ys):
    if xs[0] != xs[-1] or ys[0] != ys[-1]:
        xs = list(xs) + [xs[0]]
        ys = list(ys) + [ys[0]]
    A = 0.0
    Cx = 0.0
    Cy = 0.0
    for i in range(len(xs)-1):
        step = xs[i]*ys[i+1] - xs[i+1]*ys[i]
        A += step
        Cx += (xs[i] + xs[i+1]) * step
        Cy += (ys[i] + ys[i+1]) * step
    A *= 0.5
    if abs(A) < 1e-9:
        Cx = np.mean(xs[:-1]); Cy = np.mean(ys[:-1])
    else:
        Cx = Cx / (6.0 * A); Cy = Cy / (6.0 * A)
    return Cx, Cy

def radar_to_xy(r_vals):
    n = len(r_vals)
    thetas = np.linspace(0, 2*np.pi, n, endpoint=False)
    xs = r_vals * np.cos(thetas)
    ys = r_vals * np.sin(thetas)
    return xs, ys, thetas

def xy_to_polar(x, y):
    r = math.hypot(x, y); theta = math.atan2(y, x)
    if theta < 0: theta += 2*math.pi
    return r, theta

def plot_radar(ax, categories, pre_vals=None, post_vals=None, show_centroids=True,
               show_arrow=False, title=None, show_polygons=True, show_labels=True,
               pre_color=None, post_color=None, rlimit=None):
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    angles_closed = np.concatenate((angles, [angles[0]]))

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    if show_labels: ax.set_thetagrids(angles * 180/np.pi, categories)
    else: ax.set_thetagrids([])
    ax.grid(True, linestyle="--", alpha=0.3)

    if rlimit is None:
        rmax = 1
        for v in [pre_vals, post_vals]:
            if v is not None and len(v) > 0 and np.isfinite(v).any():
                rmax = max(rmax, float(np.nanmax(v)))
        if rmax <= 0: rmax = 1
        ax.set_ylim(0, rmax * 1.05)
    else:
        ax.set_ylim(0, rlimit)

    pre_centroid_polar = None; post_centroid_polar = None

    if show_polygons and pre_vals is not None:
        pre_closed = np.concatenate((pre_vals, [pre_vals[0]]))
        ax.plot(angles_closed, pre_closed, linewidth=2, label="Pre", color=pre_color)
        ax.fill(angles_closed, pre_closed, alpha=0.12, color=pre_color)
    if show_polygons and post_vals is not None:
        post_closed = np.concatenate((post_vals, [post_vals[0]]))
        ax.plot(angles_closed, post_closed, linewidth=2, linestyle="--", label="Post", color=post_color)
        ax.fill(angles_closed, post_closed, alpha=0.12, color=post_color)

    if pre_vals is not None:
        xs, ys, _ = radar_to_xy(pre_vals)
        xs_closed = np.concatenate((xs, [xs[0]])); ys_closed = np.concatenate((ys, [ys[0]]))
        cx, cy = polygon_centroid_xy(xs_closed, ys_closed)
        r_c, th_c = xy_to_polar(cx, cy); pre_centroid_polar = (th_c, r_c)
        if show_centroids: ax.scatter([th_c], [r_c], s=80, marker="o", zorder=5, label="Pre centroid", color=pre_color)

    if post_vals is not None:
        xs, ys, _ = radar_to_xy(post_vals)
        xs_closed = np.concatenate((xs, [xs[0]])); ys_closed = np.concatenate((ys, [ys[0]]))
        cx, cy = polygon_centroid_xy(xs_closed, ys_closed)
        r_c, th_c = xy_to_polar(cx, cy); post_centroid_polar = (th_c, r_c)
        if show_centroids: ax.scatter([th_c], [r_c], s=110, marker="^", zorder=6, label="Post centroid", color=post_color)

    if show_arrow and pre_centroid_polar and post_centroid_polar:
        ax.annotate("", xy=(post_centroid_polar[0], post_centroid_polar[1]),
                    xytext=(pre_centroid_polar[0], pre_centroid_polar[1]),
                    arrowprops=dict(arrowstyle="->", linewidth=2), zorder=7)

    if title: ax.set_title(title, pad=18, fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.15))

def prepare_pairs(df, post_suffix="1"):
    name_col = detect_name_col(df)
    params = numeric_param_cols(df, name_col)
    if len(params) < 3:
        st.error("Could not detect numeric parameter columns. Please check your file."); st.stop()
    pairs = {}
    for _, row in df.iterrows():
        raw_name = row[name_col]; base, is_post = normalize_name(raw_name, post_suffix=post_suffix)
        entry = pairs.setdefault(base, {"pre": None, "post": None})
        values = row[params].astype(float).values
        if is_post: entry["post"] = values
        else: entry["pre"] = values
    return pairs, params, name_col

def build_records(pairs, categories, use_pre=True, use_post=True):
    recs = []
    for nm, d in pairs.items():
        if use_pre and d.get("pre") is not None:
            rec = {"Name": nm, "Kind": "Pre"}
            rec.update({categories[i]: float(d["pre"][i]) for i in range(len(categories))})
            recs.append(rec)
        if use_post and d.get("post") is not None:
            rec = {"Name": nm, "Kind": "Post"}
            rec.update({categories[i]: float(d["post"][i]) for i in range(len(categories))})
            recs.append(rec)
    cols = ["Name", "Kind"] + list(categories)
    return pd.DataFrame(recs, columns=cols) if recs else pd.DataFrame(columns=cols)

# ---------- Sidebar ----------

st.title("Schwartz Values")

with st.sidebar:
    st.header("1) Upload your Excel")
    uploaded = st.file_uploader("Excel file (.xlsx)", type=["xlsx"])
    sheet_name = st.text_input("Sheet name or index (blank = first sheet)", value="")
    post_suffix = st.text_input("Post suffix", value="1", help="Suffix marking post rows (e.g., '1' so 'Riya1')")

    st.header("2) Radar options")
    radar_mode = st.radio("Radar view", ["Single", "Compare multiple"], horizontal=True)
    show_polygons = st.checkbox("Show radar polygons (shapes)", value=True)
    show_labels = st.checkbox("Show category labels", value=True, help="Turn off for clean centroids-only view")
    show_centroids = st.checkbox("Show centroids", value=True)
    show_arrow = st.checkbox("Draw arrow from pre→post centroids", value=False)
    fixed_rmax = st.checkbox("Fix radar scale to 0–7", value=True)

    st.header("3) Colours")
    pre_color = st.color_picker("Pre colour", "#1f77b4")
    post_color = st.color_picker("Post colour", "#d62728")

    st.header("4) Fixed size")
    width_px = st.slider("Radar chart width (px)", min_value=420, max_value=2000, value=760, step=20)
    dpi = st.slider("DPI", min_value=80, max_value=220, value=130, step=10)
    cols_per_row = st.slider("Radar columns per row", min_value=1, max_value=3, value=2)

    st.header("5) Per-Value Breakdown — layout")
    auto_widen = st.checkbox("Auto widen chart by number of students", value=True, help="Makes the chart wider when many students are shown")
    px_per_student = st.slider("Pixels per student (when auto-widen)", min_value=16, max_value=80, value=36, step=2)
    min_plot_width_px = st.slider("Minimum chart width (px)", min_value=600, max_value=2400, value=1000, step=20)
    bar_width = st.slider("Bar width (0.2–0.6)", min_value=0.2, max_value=0.6, value=0.36, step=0.02)
    label_rotation = st.slider("X label rotation", min_value=0, max_value=90, value=35, step=5)

    st.header("6) Per-Value Breakdown — data")
    sort_by = st.selectbox("Sort students by", ["Name (A→Z)", "Pre value", "Post value", "Δ (Post−Pre)"], index=0)
    show_mean_lines = st.checkbox("Show cohort mean lines", value=True)
    anonymize = st.checkbox("Anonymize names (ID1, ID2, …)", value=False)
    fix_bar_ylim = st.checkbox("Fix bar chart Y to 0–7", value=True)

if uploaded is None:
    st.info("⬆️ Upload an Excel (.xlsx) file to begin."); st.stop()

# ---------- Load data ----------
try:
    sn = None if sheet_name.strip() == "" else sheet_name.strip()
    df_raw = pd.read_excel(uploaded, sheet_name=sn, engine="openpyxl")
except Exception as e:
    st.error(f"Failed to read Excel: {e}"); st.stop()

pairs, categories, name_col = prepare_pairs(df_raw, post_suffix=post_suffix)
names = sorted(pairs.keys())
if not names:
    st.warning("No names detected. Check the name column and the post suffix pattern."); st.stop()

# ---------- Tabs: Radar + Breakdown ----------
tab_radar, tab_break = st.tabs(["📈 Radar view", "🪜 Per-Value Breakdown"])

# -------- Radar Tab --------
def compute_centroids_row(name, vals, kind):
    xs, ys, _ = radar_to_xy(vals)
    xs_closed = np.concatenate((xs, [xs[0]])); ys_closed = np.concatenate((ys, [ys[0]]))
    cx, cy = polygon_centroid_xy(xs_closed, ys_closed)
    r_c, th_c = xy_to_polar(cx, cy)
    return {"Name": name, "Kind": kind, "Centroid_X": cx, "Centroid_Y": cy, "Centroid_R": r_c, "Angle_deg": np.degrees(th_c)}

def render_radar(name, d, categories):
    size_in = width_px / dpi
    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(size_in, size_in), dpi=dpi)
    plot_radar(ax, categories, d.get("pre"), d.get("post"), show_centroids=show_centroids, show_arrow=show_arrow,
               title=name, show_polygons=show_polygons, show_labels=show_labels,
               pre_color=pre_color, post_color=post_color, rlimit=7 if fixed_rmax else None)
    st.markdown(f"<div class='fixed-plot-wrap' style='--plot-width:{width_px}px'>", unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=False, clear_figure=True); st.markdown("</div>", unsafe_allow_html=True)

with tab_radar:
    if radar_mode == "Single":
        name = st.selectbox("Choose a student", names, index=0, key="radar_single_select")
        d = pairs[name]; render_radar(name, d, categories)
        rows = []
        if d.get("pre") is not None: rows.append(compute_centroids_row(name, d["pre"], "Pre"))
        if d.get("post") is not None: rows.append(compute_centroids_row(name, d["post"], "Post"))
        if rows:
            df_cent = pd.DataFrame(rows)[["Name","Kind","Centroid_R","Angle_deg","Centroid_X","Centroid_Y"]]
            st.subheader("Centroid values"); st.dataframe(df_cent, use_container_width=True)
            csv = df_cent.to_csv(index=False).encode("utf-8")
            st.download_button("Download centroids (CSV)", data=csv, file_name=f"{name}_centroids.csv", mime="text/csv")
    else:
        chosen = st.multiselect("Choose students to compare", names, default=names[:min(4, len(names))], key="radar_multi_select")
        if chosen:
            n = len(chosen); rows = int(np.ceil(n / cols_per_row)); idx = 0
            for r in range(rows):
                cols = st.columns(cols_per_row, gap="large")
                for c in range(cols_per_row):
                    if idx >= n: break
                    with cols[c]:
                        nm = chosen[idx]; render_radar(nm, pairs[nm], categories); idx += 1
            cent_rows = []
            for nm in chosen:
                d = pairs[nm]
                if d.get("pre") is not None: cent_rows.append(compute_centroids_row(nm, d["pre"], "Pre"))
                if d.get("post") is not None: cent_rows.append(compute_centroids_row(nm, d["post"], "Post"))
            if cent_rows:
                df_cent_all = pd.DataFrame(cent_rows)[["Name","Kind","Centroid_R","Angle_deg","Centroid_X","Centroid_Y"]]
                st.subheader("Centroid values (selected students)"); st.dataframe(df_cent_all, use_container_width=True)
                csv = df_cent_all.to_csv(index=False).encode("utf-8")
                st.download_button("Download centroids (CSV)", data=csv, file_name="selected_students_centroids.csv", mime="text/csv")

# -------- Per-Value Breakdown Tab --------
with tab_break:
    st.write("Pick one parameter to see Pre vs Post for **all students**.")
    param = st.selectbox("Value (parameter)", categories, index=0)

    # Build long-form DataFrame
    records = []
    idx_param = list(categories).index(param)
    for nm, d in pairs.items():
        pre_v = d.get("pre")[idx_param] if d.get("pre") is not None else np.nan
        post_v = d.get("post")[idx_param] if d.get("post") is not None else np.nan
        delta = post_v - pre_v if np.isfinite(pre_v) and np.isfinite(post_v) else np.nan
        records.append({"Name": nm, "Pre": pre_v, "Post": post_v, "Delta": delta})
    plot_df = pd.DataFrame(records)

    # Sorting
    if sort_by == "Name (A→Z)": plot_df = plot_df.sort_values("Name")
    elif sort_by == "Pre value": plot_df = plot_df.sort_values("Pre", na_position="last")
    elif sort_by == "Post value": plot_df = plot_df.sort_values("Post", na_position="last")
    else: plot_df = plot_df.sort_values("Delta", na_position="last")

    display_names = [f"ID{i+1}" for i in range(len(plot_df))] if anonymize else plot_df["Name"].tolist()

    # --- Relaxed layout controls ---
    n_students = len(plot_df)
    # Compute dynamic width in pixels
    base_plot_px = int(px_per_student * n_students) + 240 if auto_widen else 0
    plot_width_px = max(min_plot_width_px if auto_widen else 0, base_plot_px if auto_widen else min_plot_width_px)
    size_in = plot_width_px / dpi

    # Draw bars
    x = np.arange(n_students)
    fig, ax = plt.subplots(figsize=(size_in, max(4.5, size_in*0.45)), dpi=dpi)
    ax.bar(x - bar_width/2, plot_df["Pre"].values, bar_width, label="Pre", color="#9ecae1", edgecolor="none")
    ax.bar(x + bar_width/2, plot_df["Post"].values, bar_width, label="Post", color="#3182bd", edgecolor="none")

    if show_mean_lines:
        pre_mean = np.nanmean(plot_df["Pre"].values); post_mean = np.nanmean(plot_df["Post"].values)
        ax.axhline(pre_mean, linestyle="--", linewidth=1, alpha=0.7, label=f"Pre mean: {pre_mean:.2f}")
        ax.axhline(post_mean, linestyle=":", linewidth=1.2, alpha=0.8, label=f"Post mean: {post_mean:.2f}")

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=label_rotation, ha="right")
    ax.set_ylabel(param); ax.set_title(f"Per-Value Breakdown • {param} (all students)")

    if fix_bar_ylim: ax.set_ylim(0, 7)
    fig.subplots_adjust(bottom=0.28)  # more space for long names
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", ncols=2)

    st.markdown(f"<div class='fixed-plot-wrap' style='--plot-width:{plot_width_px}px'>", unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=False, clear_figure=True); st.markdown("</div>", unsafe_allow_html=True)
    st.caption(f"Showing {n_students} students • Chart width ≈ {plot_width_px}px  |  Bar width = {bar_width:.2f}")

    # Data table + download
    show_tbl = st.checkbox("Show data table", value=False)
    if show_tbl:
        show_df = plot_df.copy(); show_df.insert(0, "DisplayName", display_names)
        st.dataframe(show_df, use_container_width=True)

    csv = plot_df.to_csv(index=False).encode("utf-8")
    st.download_button(f"Download {param} values (CSV)",
                       data=csv, file_name=f"{param}_pre_post_all_students.csv", mime="text/csv")
