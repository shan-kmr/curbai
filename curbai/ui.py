"""Shared Janus house style + helpers — Graphite & Signal edition.

Premium white, near-black ink, gray structure, one signal-orange accent
used sparingly (links, selection, key numbers — never large fills).
Grotesk everything; mono for data. (2026-07-10 retheme: the old
paper/cobalt/serif look was retired as reading AI-designed.)
"""

from __future__ import annotations

import streamlit as st

BRAND_CSS = """<style>
:root{--ground:#FCFCFB;--ink:#16181A;--fg2:#4A4E54;--muted:#7A7F85;--faint:#B9BCC1;
      --line:#E9EAEC;--line2:#F2F3F4;--accent:#D9480F;}
[data-testid="stHeader"],#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stStatusWidget"],[data-testid="stDecoration"]{visibility:hidden;height:0;display:none;}
.stApp,[data-testid="stAppViewContainer"],[data-testid="stSidebar"]{background:var(--ground);}
html,body,[data-testid="stAppViewContainer"] *{color:var(--ink);}
.block-container{padding-top:2.2rem;max-width:1180px;}
h1,h2,h3,h4{font-family:-apple-system,"Helvetica Neue",Helvetica,Arial,sans-serif!important;
  letter-spacing:-.015em;color:var(--ink)!important;font-weight:650!important;}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] *{font-family:ui-monospace,"SF Mono",Menlo,monospace!important;color:var(--muted)!important;letter-spacing:.02em;}
[data-baseweb="tab-list"]{gap:2px;border-bottom:1px solid var(--line);}
button[data-baseweb="tab"]{font-family:ui-monospace,"SF Mono",Menlo,monospace!important;letter-spacing:.06em;text-transform:uppercase;font-size:.7rem!important;}
[data-baseweb="tab-highlight"],[data-baseweb="tab-border"]{background:var(--ink)!important;}
.stButton>button,button[kind="secondary"]{border:1px solid var(--line)!important;border-radius:6px!important;color:var(--ink)!important;background:#fff!important;font-family:ui-monospace,Menlo,monospace!important;font-size:.72rem!important;letter-spacing:.04em;}
[data-testid="stMetricValue"]{font-family:-apple-system,"Helvetica Neue",Arial,sans-serif!important;color:var(--ink)!important;font-weight:650!important;}
a{color:var(--ink)!important;text-decoration:none;border-bottom:1px solid var(--line);}
.janus-eyebrow{font-family:ui-monospace,"SF Mono",Menlo,monospace;font-size:.68rem;letter-spacing:.22em;text-transform:uppercase;color:var(--muted);margin:0 0 8px;}
.janus-title{font-family:-apple-system,"Helvetica Neue",Helvetica,Arial,sans-serif;font-size:2.3rem;font-weight:700;letter-spacing:-.02em;color:var(--ink);margin:0 0 8px;line-height:1.04;}
.janus-dek{font-family:-apple-system,"Helvetica Neue",Arial,sans-serif;font-size:1rem;color:var(--fg2);margin:0;max-width:66ch;line-height:1.55;}
.janus-dek .sig{color:var(--ink);font-weight:600;}
.janus-rule{height:1px;background:var(--line);margin:16px 0 4px;}
/* raw data card */
.jx-card{border:1px solid var(--line);border-radius:8px;background:#fff;padding:18px 20px;}
.jx-cid{font-family:ui-monospace,Menlo,monospace;font-size:.62rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);}
.jx-big{font-family:-apple-system,"Helvetica Neue",Arial,sans-serif;font-size:2.4rem;font-weight:700;line-height:1;margin:4px 0 2px;color:var(--ink);letter-spacing:-.02em;}
.jx-big small{font-size:.85rem;color:var(--muted);font-family:ui-monospace,Menlo,monospace;letter-spacing:.04em;font-weight:400;}
.jx-row{display:flex;justify-content:space-between;gap:12px;font-size:.9rem;padding:5px 0;border-top:1px solid var(--line2);}
.jx-row .k{color:var(--fg2);} .jx-row .v{font-family:ui-monospace,Menlo,monospace;font-size:.8rem;text-align:right;}
.jx-row .v b{color:var(--ink);font-weight:650;}
.jx-lab{font-family:ui-monospace,Menlo,monospace;font-size:.58rem;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin:14px 0 6px;}
.jx-hist{display:flex;align-items:flex-end;gap:2px;height:46px;}
.jx-hist .b{flex:1;background:var(--ink);border-radius:1px 1px 0 0;min-height:1px;opacity:.78;}
.jx-hticks{display:flex;justify-content:space-between;font-family:ui-monospace,Menlo,monospace;font-size:.55rem;color:var(--faint);margin-top:3px;}
.jx-fac{font-size:.86rem;padding:3px 0;display:flex;justify-content:space-between;}
.jx-fac .c{font-family:ui-monospace,Menlo,monospace;font-size:.72rem;color:var(--muted);}
.jx-txt{font-family:-apple-system,"Helvetica Neue",Arial,sans-serif;font-size:.86rem;color:var(--fg2);line-height:1.5;margin-top:4px;}
.jx-sec{margin-top:15px;}
/* card entrance — kept from v1 */
@keyframes jxIn{from{opacity:0;transform:translateY(9px)}to{opacity:1;transform:none}}
@keyframes jxPop{from{opacity:0;transform:scale(.94)}to{opacity:1;transform:scale(1)}}
@keyframes jxBar{from{transform:scaleY(0)}to{transform:scaleY(1)}}
.jx-card{animation:jxIn .32s ease-out both;}
.jx-big{transform-origin:left bottom;animation:jxPop .38s cubic-bezier(.2,.9,.3,1.15) both .08s;}
.jx-card .jx-lab{animation:jxIn .3s ease-out both;}
.jx-card .jx-lab:nth-of-type(1){animation-delay:.04s}.jx-card .jx-lab:nth-of-type(2){animation-delay:.08s}
.jx-card .jx-lab:nth-of-type(3){animation-delay:.12s}.jx-card .jx-lab:nth-of-type(4){animation-delay:.16s}
.jx-card .jx-lab:nth-of-type(5){animation-delay:.20s}.jx-card .jx-lab:nth-of-type(6){animation-delay:.24s}
.jx-card .jx-lab:nth-of-type(7){animation-delay:.28s}.jx-card .jx-lab:nth-of-type(8){animation-delay:.32s}
.jx-card .jx-lab:nth-of-type(9){animation-delay:.36s}.jx-card .jx-lab:nth-of-type(10){animation-delay:.40s}
.jx-card .jx-lab:nth-of-type(n+11){animation-delay:.44s}
.jx-hist .b{transform-origin:bottom;animation:jxBar .5s ease-out both .18s;}
@media (prefers-reduced-motion: reduce){.jx-card,.jx-card *{animation:none!important}}
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
    """White→graphite density ramp for a normalised raw count (t in 0..1).
    Data as ink density — the accent never fills."""
    t = max(0.0, min(1.0, float(t)))
    stops = [(0.0, (233, 234, 236)), (0.5, (158, 162, 168)), (1.0, (42, 45, 49))]
    alpha = int(60 + t * 160)
    for (t0, c0), (t1, c1) in zip(stops[:-1], stops[1:]):
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0 + 1e-9)
            return [int(c0[i] + f * (c1[i] - c0[i])) for i in range(3)] + [alpha]
    return [42, 45, 49, 220]
