#!/usr/bin/env python
"""
build_sidewalkphys.py -- sidewalk-roughness layer from the founder's own
phone-IMU walking sessions (SensorLogger app, iPhone 16 Pro).

METHOD: WPRI-inspired walk-vibration index. This is honestly labelled as such:
it is NOT an ASTM-certified roughness measure (no calibrated wheel, no rated
profiler -- a phone held in hand while walking).

Pipeline, per session directory in RAW_DIR:
  1. Load Accelerometer.csv + Location.csv (+ Gravity.csv, Barometer.csv).
     iOS SensorLogger "Accelerometer" = gravity-removed user acceleration in
     the DEVICE frame; "Gravity" = gravity vector in the same frame. Units are
     verified at runtime from |gravity|: median ~9.81 -> m/s^2 (if ~1.0 the
     data would be in g and is rescaled; not the case for these sessions).
  2. Vertical axis: project user acceleration onto the gravity unit vector,
     a_vert = dot(a, g) / |g|  (signed, along gravity; sign irrelevant to RMS).
     Fallback if Gravity.csv were absent (never triggers on this dataset):
     if one raw axis has |mean| ~ 9.8 use it (mean-removed), else device z.
  3. Band-pass a_vert 0.5-8 Hz (4th-order Butterworth, zero-phase sosfiltfilt)
     to strip gait DC / slow tilt below and sensor noise above.
  4. 2-second non-overlapping windows; per window: RMS of band-passed vertical
     accel converted to g (RMS_ms2 / 9.80665), window midpoint time matched to
     the nearest Location fix (<= 8 s away; two sessions log GPS only every
     ~12 s, and 8 s of walking is ~11 m -- negligible vs a ~350 m res-9 hex).
     Walking speed per fix: iOS Doppler `speed` where valid (>= 0); where
     invalid (-1; 58-86% of fixes in the sparse-GPS sessions) a fallback
     position-derived speed is used -- haversine distance over a >= 8 s
     centered span of neighbouring fixes (span capped at 60 s). Because GPS
     jitter can fake 0.5-2 m/s while STANDING (measured on this data: during
     Doppler-confirmed standing the position-derived speed medians 0.49 m/s),
     an independent walking gate from Pedometer.csv (CMPedometer cumulative
     steps, ~2.6 s cadence) is also applied: step rate over a centered 8 s
     span must be >= 1.0 steps/s. Cadence is surface-independent (step timing,
     not vibration amplitude), so this gate does not bias the roughness
     metric. Windows are DISCARDED when: partial (<80% of expected samples),
     no GPS fix within tolerance, horizontalAccuracy > 50 m, no usable speed,
     speed < 0.5 m/s (standing) or > 2.5 m/s (not walking), step rate
     < 1.0 steps/s (standing despite apparent GPS speed), or barometric
     vertical rate |dz/dt| > 0.5 m/s (elevator/stairs -- observed ~50 m
     relativeAltitude excursions where sessions start/end inside a building;
     that vibration is not sidewalk, and it would poison the baro slope hint).
  5. Speed normalisation: roughness RMS scales with walking speed, so both are
     reported: raw g-RMS and speed-normalised g-RMS / speed (units g/(m/s)).
  6. Aggregate per H3 res-9 hex across sessions (medians are robust to the
     small window counts): swp_rough_g_rms, swp_rough_norm, swp_speed_mps,
     swp_windows, swp_walks, swp_baro_range.

swp_baro_range: Barometer relativeAltitude (m) is session-relative, so ranges
are computed per (session, hex) first -- robust range (p95 - p05) over kept
windows, requiring >= 3 baro-matched windows -- then the median across
sessions is taken. A slope hint, not a measured grade: despite the |dz/dt|
elevator filter, hexes containing a session's start/end building can retain
indoor-altitude leakage (level walking on an upper floor passes every gate
when GPS is stale); ranges > 10 m are flagged at build time and should be
treated as suspect rather than slope.

OUTPUT: data/sidewalkphys_nyc_h3.parquet, keyed h3_index (H3 res 9, string).

CAVEATS (also see final report): phone-in-hand, not a standardised wheel;
6 sessions on 5 streets; GPS in Manhattan canyons is 10-30 m accurate, so
res-9 hexes (~350 m across) absorb most position error but hex-edge windows
can bleed into neighbours.
"""

import glob
import os
import sys

import duckdb
import h3
import numpy as np
import pandas as pd
from scipy import signal, stats

RAW_DIR = os.path.expanduser(
    "~/Downloads/Final Semester/geofm-global/data/raw/sensorlogger"
)
OUT_PATH = os.path.expanduser("~/Downloads/curbai/data/sidewalkphys_nyc_h3.parquet")
CROSSCHECK_PATH = os.path.expanduser(
    "~/Downloads/Final Semester/geofm-global/data/processed/features/res9_curbside.parquet"
)

G0 = 9.80665          # m/s^2 per g
H3_RES = 9
WIN_S = 2.0           # window length, seconds
BAND_HZ = (0.5, 8.0)  # band-pass edges, Hz
SPEED_MIN = 0.5       # m/s -- below: standing
SPEED_MAX = 2.5       # m/s -- above: not walking
LOC_TOL_S = 8.0       # max age of nearest GPS fix (2 sessions log every ~12 s)
POS_SPEED_SPAN_S = 8.0   # min centered span for position-derived speed
POS_SPEED_MAX_SPAN_S = 60.0  # beyond this, fallback speed is fiction -> NaN
STEP_RATE_MIN = 1.0   # steps/s over STEP_SPAN_S -- walking is ~1.5-2.2
STEP_SPAN_S = 8.0     # centered span for step-rate estimate
VERT_RATE_MAX = 0.5   # m/s baro dz/dt -- above: elevator/stairs, not sidewalk
BARO_RANGE_MIN_WIN = 3   # min baro-matched windows for a session-hex range
BARO_RANGE_FLAG_M = 10.0  # in-hex range above this is not plausible slope
BARO_TOL_S = 5.0      # max age of nearest barometer sample
HACC_MAX_M = 50.0     # discard windows with worse horizontal accuracy
MIN_FILL = 0.8        # window must have >= 80% of expected samples
MIN_FS_HZ = 20.0      # need Nyquist comfortably above 8 Hz band edge


def read_sensor(session_dir, name, cols):
    """Read one SensorLogger CSV; return sorted df with epoch-seconds col 't'."""
    path = os.path.join(session_dir, f"{name}.csv")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    df = pd.read_csv(path)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"    ! {name}.csv missing columns {missing}; ignoring file")
        return None
    df = df[cols].dropna()
    if df.empty:
        return None
    df = df.copy()
    df["t"] = df["time"].astype(np.float64) * 1e-9  # ns epoch -> s epoch
    return df.sort_values("t").drop_duplicates("t").reset_index(drop=True)


def haversine_m(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((la2 - la1) / 2) ** 2
         + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2)
    return 6371000.0 * 2 * np.arcsin(np.sqrt(a))


def effective_speed(loc):
    """Per-fix walking speed: iOS Doppler `speed` where valid (>= 0), else a
    position-derived fallback = haversine distance over a centered span of
    >= POS_SPEED_SPAN_S (<= POS_SPEED_MAX_SPAN_S) of neighbouring fixes.
    Returns (speed_eff, n_fallback_used)."""
    t = loc["t"].to_numpy()
    lat = loc["latitude"].to_numpy()
    lon = loc["longitude"].to_numpy()
    half = POS_SPEED_SPAN_S / 2.0
    j = np.searchsorted(t, t - half, side="right") - 1  # last fix <= t-half
    k = np.searchsorted(t, t + half, side="left")       # first fix >= t+half
    j = np.clip(j, 0, len(t) - 1)
    k = np.clip(k, 0, len(t) - 1)
    span = t[k] - t[j]
    with np.errstate(divide="ignore", invalid="ignore"):
        v_pos = haversine_m(lat[j], lon[j], lat[k], lon[k]) / span
    v_pos = np.where(
        (span >= POS_SPEED_SPAN_S * 0.75) & (span <= POS_SPEED_MAX_SPAN_S),
        v_pos, np.nan,
    )
    doppler = loc["speed"].to_numpy()
    use_fallback = doppler < 0
    speed_eff = np.where(use_fallback, v_pos, doppler)
    return speed_eff, int((use_fallback & ~np.isnan(v_pos)).sum())


def vertical_accel(acc, grav):
    """Return (t, a_vert_ms2, note). Projects device-frame user accel onto
    gravity when Gravity.csv exists; otherwise falls back per docstring."""
    if grav is not None:
        g = grav.rename(columns={"x": "gx", "y": "gy", "z": "gz"})
        m = pd.merge_asof(
            acc, g[["t", "gx", "gy", "gz"]], on="t",
            direction="nearest", tolerance=0.05,
        ).dropna(subset=["gx", "gy", "gz"])
        gmag = np.sqrt(m.gx**2 + m.gy**2 + m.gz**2).to_numpy()
        gmed = float(np.median(gmag))
        scale = G0 if gmed < 3.0 else 1.0  # logged in g units? -> m/s^2
        if not 8.0 < gmed * scale < 11.5:
            raise ValueError(f"unexpected gravity magnitude {gmed:.3f}")
        a_vert = (m.x * m.gx + m.y * m.gy + m.z * m.gz).to_numpy() / gmag * scale
        amag = np.sqrt(m.x**2 + m.y**2 + m.z**2).to_numpy() * scale
        note = (f"gravity-projection (|g| median {gmed * scale:.2f} m/s^2, "
                f"mean |userAccel| {amag.mean():.2f} m/s^2)")
        return m["t"].to_numpy(), a_vert, note
    # Fallback -- not exercised on this dataset (all sessions have Gravity.csv)
    means = acc[["x", "y", "z"]].mean()
    ax = means.abs().idxmax()
    if abs(means[ax]) > 7.0:  # raw accel incl. gravity: use that axis, de-mean
        a_vert = (acc[ax] - means[ax]).to_numpy()
        note = f"no Gravity.csv; axis '{ax}' has |mean|~9.8 -> vertical, de-meaned"
    else:  # gravity-removed but frame unknown: device z, documented approximation
        a_vert = acc["z"].to_numpy()
        note = "no Gravity.csv and no ~9.8 axis; using device z (approximation)"
    return acc["t"].to_numpy(), a_vert, note


def process_session(session_dir):
    """Return (kept_windows_df, stats_dict) or (None, reason)."""
    name = os.path.basename(session_dir.rstrip("/"))
    acc = read_sensor(session_dir, "Accelerometer", ["time", "x", "y", "z"])
    grav = read_sensor(session_dir, "Gravity", ["time", "x", "y", "z"])
    loc = read_sensor(
        session_dir, "Location",
        ["time", "latitude", "longitude", "speed", "horizontalAccuracy"],
    )
    baro = read_sensor(session_dir, "Barometer", ["time", "relativeAltitude"])
    ped = read_sensor(session_dir, "Pedometer", ["time", "steps"])

    if acc is None or len(acc) < 1000:
        return None, f"{name}: no usable Accelerometer.csv"
    if loc is None or loc.empty:
        return None, f"{name}: no usable Location.csv"

    fs = 1.0 / float(np.median(np.diff(acc["t"].to_numpy())))
    if fs < MIN_FS_HZ:
        return None, f"{name}: sample rate {fs:.1f} Hz too low for 8 Hz band"

    t, a_vert, vert_note = vertical_accel(acc, grav)

    sos = signal.butter(4, BAND_HZ, btype="bandpass", fs=fs, output="sos")
    v_bp = signal.sosfiltfilt(sos, a_vert)

    # 2-second windows
    t0 = t[0]
    widx = np.floor((t - t0) / WIN_S).astype(np.int64)
    per_win = pd.DataFrame({"w": widx, "v2": v_bp**2}).groupby("w")["v2"].agg(
        ["mean", "count"]
    )
    need = MIN_FILL * fs * WIN_S
    n_partial = int((per_win["count"] < need).sum())
    per_win = per_win[per_win["count"] >= need]
    win = pd.DataFrame({
        "t": t0 + (per_win.index.to_numpy(dtype=np.float64) + 0.5) * WIN_S,
        "rough_g_rms": np.sqrt(per_win["mean"].to_numpy()) / G0,
    }).sort_values("t")

    # attach nearest GPS fix
    loc = loc[
        loc["latitude"].between(-90, 90)
        & loc["longitude"].between(-180, 180)
        & ((loc["latitude"] != 0) | (loc["longitude"] != 0))
    ].sort_values("t").reset_index(drop=True)
    loc["speed_eff"], n_fallback = effective_speed(loc)
    win = pd.merge_asof(
        win,
        loc[["t", "latitude", "longitude", "speed_eff", "horizontalAccuracy"]],
        on="t", direction="nearest", tolerance=LOC_TOL_S,
    )

    # step-rate walking gate from cumulative pedometer steps
    if ped is not None and len(ped) >= 2:
        tm = win["t"].to_numpy()
        s_hi = np.interp(tm + STEP_SPAN_S / 2, ped["t"], ped["steps"])
        s_lo = np.interp(tm - STEP_SPAN_S / 2, ped["t"], ped["steps"])
        win["step_rate"] = (s_hi - s_lo) / STEP_SPAN_S
    else:
        win["step_rate"] = np.nan  # no pedometer -> gate passes, noted

    # attach nearest barometer sample + vertical rate (elevator/stairs gate)
    if baro is not None and len(baro) >= 5:
        alt = baro["relativeAltitude"].rolling(5, center=True, min_periods=1)\
            .median().to_numpy()
        baro = baro.assign(
            relativeAltitude=alt,
            baro_dzdt=np.gradient(alt, baro["t"].to_numpy()),
        )
        win = pd.merge_asof(
            win, baro[["t", "relativeAltitude", "baro_dzdt"]],
            on="t", direction="nearest", tolerance=BARO_TOL_S,
        )
    else:
        win["relativeAltitude"] = np.nan
        win["baro_dzdt"] = np.nan

    no_fix = win["latitude"].isna()
    bad_acc = ~no_fix & (win["horizontalAccuracy"] > HACC_MAX_M)
    sp = win["speed_eff"]
    ok_so_far = ~no_fix & ~bad_acc
    no_speed = ok_so_far & sp.isna()
    slow = ok_so_far & (sp < SPEED_MIN)
    fast = ok_so_far & (sp > SPEED_MAX)
    ok_so_far = ok_so_far & ~(no_speed | slow | fast)
    no_steps = ok_so_far & (win["step_rate"] < STEP_RATE_MIN)
    ok_so_far = ok_so_far & ~no_steps
    vert = ok_so_far & (win["baro_dzdt"].abs() > VERT_RATE_MAX)
    kept = win[
        ~(no_fix | bad_acc | no_speed | slow | fast | no_steps | vert)
    ].copy()
    kept = kept.rename(columns={"speed_eff": "speed"})

    kept["h3_index"] = [
        h3.geo_to_h3(la, lo, H3_RES)
        for la, lo in zip(kept["latitude"], kept["longitude"])
    ]
    kept["rough_norm"] = kept["rough_g_rms"] / kept["speed"]
    kept["session"] = name

    st = {
        "session": name,
        "fs_hz": fs,
        "vert_note": vert_note,
        "n_accel_rows": len(acc),
        "windows_formed": int(len(win)),
        "n_partial": n_partial,
        "n_no_fix": int(no_fix.sum()),
        "n_bad_hacc": int(bad_acc.sum()),
        "n_no_speed": int(no_speed.sum()),
        "n_slow": int(slow.sum()),
        "n_fast": int(fast.sum()),
        "n_no_steps": int(no_steps.sum()),
        "n_vert": int(vert.sum()),
        "n_speed_fallback_fixes": n_fallback,
        "has_pedometer": bool(ped is not None and len(ped) >= 2),
        "n_kept": int(len(kept)),
        "med_rough_g": float(kept["rough_g_rms"].median()) if len(kept) else np.nan,
        "med_norm": float(kept["rough_norm"].median()) if len(kept) else np.nan,
        "med_speed": float(kept["speed"].median()) if len(kept) else np.nan,
        "n_hexes": int(kept["h3_index"].nunique()),
    }
    cols = ["session", "h3_index", "t", "rough_g_rms", "rough_norm",
            "speed", "relativeAltitude"]
    return kept[cols], st


def main():
    session_dirs = sorted(
        d for d in glob.glob(os.path.join(RAW_DIR, "*")) if os.path.isdir(d)
    )
    print(f"sessions found: {len(session_dirs)} in {RAW_DIR}")

    parts, session_stats = [], []
    for d in session_dirs:
        res, st = process_session(d)
        if res is None:
            print(f"  SKIP {st}")
            continue
        parts.append(res)
        session_stats.append(st)
        print(
            f"  {st['session']}\n"
            f"    fs={st['fs_hz']:.1f} Hz, accel rows={st['n_accel_rows']}, "
            f"vertical={st['vert_note']}\n"
            f"    windows: formed={st['windows_formed']} kept={st['n_kept']} "
            f"(partial={st['n_partial']}, no_fix={st['n_no_fix']}, "
            f"hacc>{HACC_MAX_M:.0f}m={st['n_bad_hacc']}, "
            f"no_speed={st['n_no_speed']}, slow={st['n_slow']}, "
            f"fast={st['n_fast']}, no_steps={st['n_no_steps']}, "
            f"vert={st['n_vert']}; "
            f"pos-derived speed on {st['n_speed_fallback_fixes']} fixes, "
            f"pedometer={'yes' if st['has_pedometer'] else 'NO'})\n"
            f"    median g-RMS={st['med_rough_g']:.4f} g, "
            f"median norm={st['med_norm']:.4f} g/(m/s), "
            f"median speed={st['med_speed']:.2f} m/s, hexes={st['n_hexes']}"
        )

    if not parts:
        sys.exit("no sessions produced windows; aborting")

    pool = pd.concat(parts, ignore_index=True)
    total_formed = sum(s["windows_formed"] for s in session_stats)
    total_partial = sum(s["n_partial"] for s in session_stats)
    total_kept = len(pool)
    print(
        f"\nTOTALS: sessions parsed={len(session_stats)}/{len(session_dirs)}, "
        f"windows formed={total_formed} (+{total_partial} partial dropped), "
        f"kept={total_kept}, discarded={total_formed - total_kept}"
    )

    # ----- aggregate per hex -----
    agg = (
        pool.groupby("h3_index")
        .agg(
            swp_rough_g_rms=("rough_g_rms", "median"),
            swp_rough_norm=("rough_norm", "median"),
            swp_speed_mps=("speed", "median"),
            swp_windows=("rough_g_rms", "size"),
            swp_walks=("session", "nunique"),
        )
        .reset_index()
    )
    agg["swp_windows"] = agg["swp_windows"].astype(np.int32)
    agg["swp_walks"] = agg["swp_walks"].astype(np.int32)

    # barometer range: per (session, hex) first (relativeAltitude is
    # session-relative), robust p95-p05, then median across sessions
    b = (
        pool.dropna(subset=["relativeAltitude"])
        .groupby(["session", "h3_index"])["relativeAltitude"]
        .agg(
            p95=lambda s: s.quantile(0.95),
            p05=lambda s: s.quantile(0.05),
            count="count",
        )
    )
    b = b[b["count"] >= BARO_RANGE_MIN_WIN]
    if len(b):
        rng = (b["p95"] - b["p05"]).rename("rng").reset_index()
        baro_hex = (
            rng.groupby("h3_index")["rng"].median().rename("swp_baro_range")
            .reset_index()
        )
        agg = agg.merge(baro_hex, on="h3_index", how="left")
    else:
        agg["swp_baro_range"] = np.nan
    suspect = agg[agg["swp_baro_range"] > BARO_RANGE_FLAG_M]
    for _, row in suspect.iterrows():
        print(f"  ! swp_baro_range {row['swp_baro_range']:.1f} m at "
              f"{row['h3_index']} exceeds {BARO_RANGE_FLAG_M:.0f} m -- "
              f"likely indoor-altitude leakage (session start/end building), "
              f"not sidewalk slope")

    agg = agg.sort_values("h3_index").reset_index(drop=True)
    print(f"hexes (H3 res {H3_RES}): {len(agg)}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    con = duckdb.connect()
    con.register("agg_df", agg)
    con.execute(f"COPY (SELECT * FROM agg_df) TO '{OUT_PATH}' (FORMAT PARQUET)")

    # ----- verify readback -----
    chk = con.execute(f"SELECT * FROM '{OUT_PATH}'").df()
    assert len(chk) == len(agg) and chk["h3_index"].is_unique, "readback mismatch"
    print(f"\nwrote {OUT_PATH}  rows={len(chk)}")
    print("schema:")
    print(con.execute(f"DESCRIBE SELECT * FROM '{OUT_PATH}'").df()
          [["column_name", "column_type"]].to_string(index=False))
    print("\nsample rows:")
    print(chk.head(5).to_string(index=False))
    print("\nlayer stats:")
    print(chk[["swp_rough_g_rms", "swp_rough_norm", "swp_speed_mps",
               "swp_windows", "swp_baro_range"]].describe().to_string())

    # ----- cross-check against existing aggregate -----
    # NOTE (verified during development): mot_motion_accel_mag_median in
    # res9_curbside is ZERO-FILLED across the full 10,686-hex universe -- only
    # hexes with mot_n_walks > 0 carry a real observation (4 hexes, built from
    # a 1-2 walk subset of this same raw data). A naive join therefore mostly
    # compares roughness against 0.0 = "no data". Both views are printed.
    if os.path.exists(CROSSCHECK_PATH):
        cc = con.execute(
            f"""
            SELECT a.h3_index, a.swp_rough_g_rms, a.swp_rough_norm,
                   c.mot_motion_accel_mag_median, c.mot_n_walks
            FROM '{OUT_PATH}' a
            JOIN '{CROSSCHECK_PATH}' c USING (h3_index)
            WHERE c.mot_motion_accel_mag_median IS NOT NULL
            """
        ).df()
        print(f"\ncross-check vs res9_curbside.mot_motion_accel_mag_median: "
              f"{len(cc)}/{len(chk)} hexes overlap")
        if len(cc) >= 3:
            r, p = stats.pearsonr(cc["swp_rough_g_rms"],
                                  cc["mot_motion_accel_mag_median"])
            print(f"  naive (incl. zero-filled no-data hexes): "
                  f"r={r:.3f} (p={p:.2e}, n={len(cc)}) -- NOT meaningful, "
                  f"{int((cc['mot_n_walks'] == 0).sum())}/{len(cc)} mot values "
                  f"are zero-fill")
        real = cc[cc["mot_n_walks"] > 0]
        print(f"  hexes with a real mot_ observation (mot_n_walks>0): "
              f"{len(real)}")
        if len(real) >= 3:
            r, p = stats.pearsonr(real["swp_rough_g_rms"],
                                  real["mot_motion_accel_mag_median"])
            print(f"  restricted Pearson r (swp_rough_g_rms vs mot_...) = "
                  f"{r:.3f} (p={p:.2e}, n={len(real)})")
        elif len(real) > 0:
            print("  too few for a correlation; pairs "
                  "(h3, swp_rough_g_rms, mot_accel_mag_median):")
            for _, row in real.iterrows():
                print(f"    {row['h3_index']}  {row['swp_rough_g_rms']:.4f} g"
                      f"   {row['mot_motion_accel_mag_median']:.4f}")
    else:
        print(f"\ncross-check parquet not found at {CROSSCHECK_PATH}")

    print("\nper-session median g-RMS (kept windows):")
    for s in session_stats:
        print(f"  {s['session']:<35} {s['med_rough_g']:.4f} g   "
              f"(norm {s['med_norm']:.4f}, speed {s['med_speed']:.2f} m/s, "
              f"windows {s['n_kept']})")


if __name__ == "__main__":
    main()
