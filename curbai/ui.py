"""Shared Janus house style + helpers for CurbIndex and the Hex Atlas."""

from __future__ import annotations

import streamlit as st

BRAND_CSS = """<style>
:root{--paper:#F6F4EF;--ink:#1A1815;--cobalt:#1E3A8A;--muted:#8C887E;--line:#E0DDD4;--fg2:#4E4B45;}
[data-testid="stHeader"],#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stStatusWidget"],[data-testid="stDecoration"]{visibility:hidden;height:0;display:none;}
.stApp,[data-testid="stAppViewContainer"],[data-testid="stSidebar"]{background:var(--paper);}
html,body,[data-testid="stAppViewContainer"] *{color:var(--ink);}
.block-container{padding-top:2.4rem;max-width:1180px;}
h1,h2,h3,h4{font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif!important;letter-spacing:-.01em;color:var(--ink)!important;}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] *{font-family:ui-monospace,"SF Mono",Menlo,monospace!important;color:var(--muted)!important;letter-spacing:.02em;}
[data-baseweb="tab-list"]{gap:2px;border-bottom:1px solid var(--line);}
button[data-baseweb="tab"]{font-family:ui-monospace,"SF Mono",Menlo,monospace!important;letter-spacing:.06em;text-transform:uppercase;font-size:.7rem!important;}
[data-baseweb="tab-highlight"],[data-baseweb="tab-border"]{background:var(--cobalt)!important;}
.stButton>button,button[kind="secondary"]{border:1px solid var(--line)!important;border-radius:4px!important;color:var(--ink)!important;background:#fff!important;font-family:ui-monospace,Menlo,monospace!important;font-size:.72rem!important;letter-spacing:.04em;}
[data-testid="stMetricValue"]{font-family:"Iowan Old Style",Palatino,Georgia,serif!important;color:var(--cobalt)!important;}
a{color:var(--cobalt)!important;}
.janus-eyebrow{font-family:ui-monospace,"SF Mono",Menlo,monospace;font-size:.7rem;letter-spacing:.22em;text-transform:uppercase;color:var(--cobalt);margin:0 0 6px;}
.janus-title{font-family:"Iowan Old Style",Palatino,Georgia,serif;font-size:2.5rem;font-weight:600;letter-spacing:-.015em;color:var(--ink);margin:0 0 8px;line-height:1.03;}
.janus-dek{font-family:"Iowan Old Style",Palatino,Georgia,serif;font-size:1.04rem;color:var(--fg2);margin:0;max-width:66ch;line-height:1.5;}
.janus-dek .sig{color:var(--cobalt);font-style:italic;}
.janus-rule{height:1px;background:var(--line);margin:16px 0 4px;}
/* raw data card */
.jx-card{border:1px solid var(--line);border-radius:6px;background:#fff;padding:18px 20px;}
.jx-cid{font-family:ui-monospace,Menlo,monospace;font-size:.62rem;letter-spacing:.1em;text-transform:uppercase;color:var(--cobalt);}
.jx-big{font-family:"Iowan Old Style",Palatino,Georgia,serif;font-size:2.6rem;font-weight:600;line-height:1;margin:4px 0 2px;color:var(--ink);}
.jx-big small{font-size:.9rem;color:var(--muted);font-family:ui-monospace,Menlo,monospace;letter-spacing:.04em;}
.jx-row{display:flex;justify-content:space-between;gap:12px;font-size:.9rem;padding:5px 0;border-top:1px solid var(--line);}
.jx-row .k{color:var(--fg2);} .jx-row .v{font-family:ui-monospace,Menlo,monospace;font-size:.8rem;text-align:right;}
.jx-row .v b{color:var(--cobalt);font-weight:600;}
.jx-lab{font-family:ui-monospace,Menlo,monospace;font-size:.58rem;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin:14px 0 6px;}
.jx-hist{display:flex;align-items:flex-end;gap:2px;height:46px;}
.jx-hist .b{flex:1;background:var(--cobalt);border-radius:1px 1px 0 0;min-height:1px;opacity:.85;}
.jx-hticks{display:flex;justify-content:space-between;font-family:ui-monospace,Menlo,monospace;font-size:.55rem;color:var(--faint);margin-top:3px;}
.jx-fac{font-size:.86rem;padding:3px 0;display:flex;justify-content:space-between;}
.jx-fac .c{font-family:ui-monospace,Menlo,monospace;font-size:.72rem;color:var(--muted);}
</style>"""


def inject() -> None:
    st.markdown(BRAND_CSS, unsafe_allow_html=True)


def header(eyebrow: str, title: str, dek: str) -> None:
    st.markdown(
        f'<p class="janus-eyebrow">{eyebrow}</p>'
        f'<div class="janus-title">{title}</div>'
        f'<p class="janus-dek">{dek}</p>'
        f'<div class="janus-rule"></div>',
        unsafe_allow_html=True,
    )


def count_color(t: float) -> list[int]:
    """Paper→cobalt ramp for a normalised raw count (t in 0..1). Not a score — a
    visual encoding of the raw number. Alpha rises with the count."""
    t = max(0.0, min(1.0, float(t)))
    stops = [(0.0, (223, 221, 212)), (0.35, (170, 182, 205)),
             (0.70, (70, 105, 180)), (1.0, (30, 58, 138))]
    alpha = int(70 + t * 165)
    for (t0, c0), (t1, c1) in zip(stops[:-1], stops[1:]):
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0 + 1e-9)
            return [int(c0[i] + f * (c1[i] - c0[i])) for i in range(3)] + [alpha]
    return [30, 58, 138, 235]
