"""Visual building blocks for the EngineTwin cockpit."""

import base64
from html import escape
from pathlib import Path


COCKPIT_CSS = """
<style>
600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root {
  --et-bg:#050504; --et-panel:#0b0a07; --et-panel-2:#121006; --et-line:#3b3115;
  --et-cyan:#d8aa32; --et-green:#e6bc55; --et-amber:#f4c95d; --et-red:#d78b80;
  --et-text:#fff9e8; --et-muted:#9d916f;
}
.stApp { background:radial-gradient(circle at 76% -8%,#221b08 0,#090806 27%,#030303 64%); color:var(--et-text); }
[data-testid="stAppViewContainer"] > .main { background:transparent; }
.block-container { max-width: 1560px; padding: 2.2rem 2.6rem 4rem; }
[data-testid="stHeader"] { background:rgba(3,3,3,.78); backdrop-filter:blur(12px); }
[data-testid="stSidebar"] { background:#080704; border-right:1px solid var(--et-line); }
h1,h2,h3 { font-family:'Arial',sans-serif !important; letter-spacing:.03em; }
p, label, [data-testid="stMetricLabel"] { font-family:'Arial',sans-serif; }
[data-baseweb="tab-list"] { gap:.25rem; border-bottom:1px solid var(--et-line); }
[data-baseweb="tab"] { height:3rem; padding:0 .9rem; color:#7595a7; font-family:'Arial',sans-serif; font-size:.78rem; text-transform:uppercase; }
[aria-selected="true"][data-baseweb="tab"] { color:var(--et-cyan)!important; background:#151106; border-radius:7px 7px 0 0; }
[data-baseweb="tab-highlight"] { background-color:var(--et-cyan)!important; }
[data-testid="stPlotlyChart"] { border:1px solid #243645; border-radius:10px; overflow:hidden; background:rgba(8,25,37,.72); }
[data-testid="stAlert"] { border-radius:8px; border:1px solid #5b4920; background:rgba(30,24,8,.88); }
.et-kicker { color:var(--et-cyan); font:600 .72rem 'Arial',sans-serif; letter-spacing:.24em; text-transform:uppercase; }
.et-hero { display:flex; justify-content:space-between; align-items:end; padding-bottom:1.15rem; border-bottom:1px solid var(--et-line); margin-bottom:.65rem; }
.et-hero h1 { margin:.2rem 0 0; font:800 2.4rem/1 'Arial',sans-serif; letter-spacing:.02em; color:var(--et-text); }
.et-hero p { margin:.45rem 0 0; color:var(--et-muted); font:.72rem 'Arial',sans-serif; }
.et-live { display:flex; align-items:center; gap:.65rem; color:var(--et-cyan); border:1px solid #7b6120; border-radius:5px; padding:.65rem .9rem; font:600 .68rem 'Arial',sans-serif; letter-spacing:.13em; }
.et-dot { width:8px; height:8px; border-radius:50%; background:var(--et-cyan); box-shadow:0 0 14px var(--et-cyan); animation:etpulse 1.3s infinite; }
@keyframes etpulse { 50% { opacity:.35; transform:scale(.72); } }
.et-toolbar { display:flex; gap:.7rem; align-items:center; margin:.55rem 0 1rem; padding:.65rem .8rem; background:#0c0a05; border:1px solid var(--et-line); border-radius:7px; }
.et-chip { color:#9db5c1; border-right:1px solid var(--et-line); padding-right:1rem; font:500 .68rem 'Arial',sans-serif; }
.et-chip:last-child { border:0; } .et-chip strong { color:#f3fbfd; font-size:.76rem; }
.et-section { color:#a9c1cc; font:600 .66rem 'Arial',sans-serif; letter-spacing:.12em; text-transform:uppercase; margin:.35rem 0 .55rem; }
.et-demo { display:flex; gap:.65rem; align-items:center; margin:0 0 .75rem; padding:.55rem .72rem; color:#d7b85f; background:rgba(77,61,15,.26); border:1px solid #66551f; border-radius:6px; font:500 .6rem 'Arial',sans-serif; }
.et-demo b { flex:0 0 auto; color:#ffd869; letter-spacing:.1em; }
.et-card { position:relative; min-height:94px; padding:.85rem .95rem .72rem; border:1px solid var(--et-line); border-radius:8px; background:linear-gradient(145deg,rgba(22,18,7,.96),rgba(7,7,5,.98)); overflow:hidden; }
.et-card:after { content:""; position:absolute; left:0; right:0; bottom:0; height:3px; background:var(--accent,var(--et-cyan)); box-shadow:0 0 15px var(--accent,var(--et-cyan)); }
.et-label { color:#9b8b64; font:600 .63rem 'Arial',sans-serif; letter-spacing:.12em; text-transform:uppercase; }
.et-value { margin-top:.25rem; color:#f3fbfd; font:800 1.9rem/1 'Arial',sans-serif; }
.et-unit { color:var(--accent,var(--et-cyan)); font:600 .72rem 'Arial',sans-serif; }
.et-sub { margin-top:.4rem; color:#817554; font:500 .6rem 'Arial',sans-serif; }
.et-engine-card { border:1px solid var(--et-line); border-radius:10px; background:linear-gradient(180deg,#121006,#070705); padding:.75rem; min-height:388px; }
.et-engine-head { display:flex; justify-content:space-between; color:#aa9869; font:600 .63rem 'Arial',sans-serif; letter-spacing:.12em; text-transform:uppercase; }
.et-engine-state { color:var(--state,var(--et-green)); }
.et-engine-svg { width:100%; height:275px; display:block; filter:drop-shadow(0 14px 30px rgba(0,0,0,.45)); }
.et-piston { transform-box:fill-box; transform-origin:center; animation:etpiston .52s ease-in-out infinite alternate; }
.et-p2 { animation-delay:-.26s; } .et-p3 { animation-delay:-.39s; } .et-p4 { animation-delay:-.13s; }
@keyframes etpiston { from { transform:translateY(-11px); } to { transform:translateY(13px); } }
.et-crank { transform-box:fill-box; transform-origin:center; animation:etspin 1s linear infinite; }
@keyframes etspin { to { transform:rotate(360deg); } }
.et-hot { animation:ethot .8s ease-in-out infinite alternate; }
@keyframes ethot { to { opacity:.28; } }
.et-engine-foot { display:grid; grid-template-columns:1fr 1fr; gap:.5rem; }
.et-mini { background:#090805; border:1px solid #3b3115; border-radius:6px; padding:.55rem .65rem; }
.et-mini span { display:block; color:#8f805d; font:500 .56rem 'Arial',sans-serif; text-transform:uppercase; }
.et-mini strong { color:#e8f7fb; font:700 .83rem 'Arial',sans-serif; }
.et-health { display:flex; align-items:center; gap:1rem; border:1px solid var(--et-line); border-radius:9px; background:#0b0904; padding:.8rem 1rem; margin-top:.75rem; }
.et-ring { --health:100; width:88px; height:88px; border-radius:50%; display:grid; place-items:center; background:conic-gradient(var(--et-green) calc(var(--health)*1%),#15313d 0); position:relative; }
.et-ring:after { content:""; position:absolute; inset:9px; border-radius:50%; background:#080704; }
.et-ring strong { z-index:1; color:white; font:800 1.5rem 'Arial',sans-serif; }
.et-health-copy b { display:block; color:var(--et-green); font:600 .67rem 'Arial',sans-serif; letter-spacing:.11em; }
.et-health-copy span { color:#708d9c; font:500 .58rem 'Arial',sans-serif; }
.et-sensor-grid { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:.55rem; margin:.3rem 0 .8rem; }
.et-sensor { min-height:78px; border:1px solid var(--et-line); border-left:3px solid var(--sensor,var(--et-green)); border-radius:7px; padding:.65rem .72rem; background:#0b0905; }
.et-sensor small { display:block; color:#94845e; font:600 .55rem 'Arial',sans-serif; text-transform:uppercase; letter-spacing:.08em; }
.et-sensor strong { display:block; margin:.3rem 0 .2rem; color:#edf9fc; font:700 1.25rem 'Arial',sans-serif; }
.et-sensor em { color:var(--sensor,var(--et-green)); font:600 .53rem 'Arial',sans-serif; font-style:normal; }
.et-log { border:1px solid var(--et-line); border-radius:8px; background:#0a0905; padding:.75rem; }
.et-log-row { display:grid; grid-template-columns:74px 1fr auto; gap:.6rem; align-items:center; padding:.58rem .25rem; border-bottom:1px solid #102d3c; font:500 .6rem 'Arial',sans-serif; }
.et-log-row:last-child { border:0; } .et-log-row time { color:#607f90; } .et-log-row b { color:#b8cdd6; } .et-log-row span { color:var(--et-green); }
.et-drive { position:relative; height:360px; margin:.75rem 0 1rem; overflow:hidden; border:1px solid #4a3b16; border-radius:10px; background:radial-gradient(circle at 55% 36%,#2a210c 0,#100d06 30%,#050505 70%); }
.et-drive:after { content:""; position:absolute; left:0; right:0; top:64%; height:1px; background:#c79b31; box-shadow:0 0 20px #c79b31; }
.et-drive-meta { position:absolute; z-index:4; left:1.1rem; top:1rem; color:#d8aa32; font:600 .62rem 'Arial',sans-serif; letter-spacing:.12em; }
.et-drive-meta b { display:block; margin-top:.25rem; font:800 1.5rem 'Arial',sans-serif; }
.et-drive-status { position:absolute; z-index:4; right:1rem; top:1rem; padding:.45rem .65rem; border:1px solid #8e6e24; color:#d8aa32; font:600 .58rem 'Arial',sans-serif; letter-spacing:.1em; }
.et-marquee { position:absolute; top:22px; left:0; display:flex; width:max-content; color:rgba(216,170,50,.09); font:800 13rem/1 'Arial',sans-serif; letter-spacing:-.04em; white-space:nowrap; animation:etmarquee 18s linear infinite; }
.et-marquee span { padding-right:1.2rem; }
@keyframes etmarquee { to { transform:translateX(-50%); } }
.et-powertrain { position:absolute; z-index:3; left:13%; bottom:92px; width:420px; max-width:72%; animation:ettravel 8s ease-in-out infinite alternate; filter:drop-shadow(0 15px 12px rgba(0,0,0,.24)); }
@keyframes ettravel { from { transform:translateX(-7%); } to { transform:translateX(78%); } }
.et-wheel { transform-box:fill-box; transform-origin:center; animation:etspin 1.15s linear infinite; }
.et-reticle { transform-box:fill-box; transform-origin:center; animation:etspin 7s linear infinite; }
.et-scan { animation:etscan 2s ease-in-out infinite alternate; }
@keyframes etscan { from { transform:translateX(-70px); opacity:.15; } to { transform:translateX(80px); opacity:.7; } }
.et-drive-bottom { position:absolute; z-index:4; left:1.1rem; right:1.1rem; bottom:1rem; display:flex; justify-content:space-between; align-items:end; color:#a9976d; font:500 .58rem 'Arial',sans-serif; }
.et-drive-bottom strong { display:block; color:#f2c65b; font:700 .72rem 'Arial',sans-serif; letter-spacing:.08em; }
.et-scroll-cue { display:flex; align-items:center; gap:.6rem; color:#d8aa32; letter-spacing:.12em; }
.et-scroll-cue i { display:block; width:22px; height:34px; border:1px solid #a77f24; border-radius:12px; position:relative; }
.et-scroll-cue i:after { content:""; position:absolute; left:9px; top:6px; width:2px; height:7px; background:#d8aa32; animation:etscroll 1.4s infinite; }
@keyframes etscroll { to { transform:translateY(11px); opacity:0; } }
@media (max-width:900px) { .et-hero { align-items:start; flex-direction:column; gap:1rem; } .et-sensor-grid { grid-template-columns:repeat(2,1fr); } .block-container { padding:1.2rem; } }
@media (max-width:900px) { .et-drive { height:310px; } .et-marquee { font-size:8rem; top:55px; } .et-powertrain { width:310px; max-width:78%; bottom:88px; animation:ettravel-mobile 6s ease-in-out infinite alternate; } @keyframes ettravel-mobile { from { transform:translateX(-10%); } to { transform:translateX(28%); } } }

/* Showroom-style first view. It deliberately has its own editorial rhythm. */
.stApp:has(.et-showroom) { background:#070706; }
.stApp:has(.et-showroom) [data-testid="stHeader"],
.stApp:has(.et-showroom) [data-testid="stSidebar"],
.stApp:has(.et-showroom) [data-testid="stToolbar"] { display:none!important; }
.stApp:has(.et-showroom) .block-container { max-width:none; padding:0; }
.et-showroom { position:relative; min-height:100vh; overflow:hidden; isolation:isolate; color:#f4efe2; background:
  radial-gradient(circle at 66% 44%,rgba(177,133,45,.12),transparent 28%),
  linear-gradient(112deg,#080807 0%,#0d0c09 54%,#050505 100%); }
.et-showroom:before { content:""; position:absolute; inset:0; z-index:-1; opacity:.34; pointer-events:none; background-image:
  linear-gradient(rgba(255,255,255,.024) 1px,transparent 1px),
  linear-gradient(90deg,rgba(255,255,255,.024) 1px,transparent 1px); background-size:72px 72px; mask-image:linear-gradient(to bottom,black,transparent 82%); }
.et-showroom__top { position:absolute; top:0; left:0; right:0; height:86px; display:flex; align-items:center; justify-content:space-between; padding:0 4.2vw; border-bottom:1px solid rgba(198,157,65,.19); animation:et-reveal-down .7s both; }
.et-wordmark { display:flex; align-items:center; gap:13px; font:600 11px/1 'Arial',sans-serif; letter-spacing:.18em; }
.et-wordmark__mark { position:relative; width:26px; height:26px; border:1px solid #caa348; transform:rotate(45deg); }
.et-wordmark__mark:after { content:""; position:absolute; inset:6px; border:1px solid #caa348; }
.et-system { display:flex; align-items:center; gap:10px; color:#b8aa87; font:500 9px 'Arial',sans-serif; letter-spacing:.18em; }
.et-system i { width:6px; height:6px; border-radius:50%; background:#d7ae4b; box-shadow:0 0 13px #d7ae4b; animation:etpulse 1.5s infinite; }
.et-showroom__rail { position:absolute; left:4.2vw; top:145px; z-index:5; width:180px; animation:et-reveal-right .75s .25s both; }
.et-model-no { color:#d2aa4c; font:600 9px 'Arial',sans-serif; letter-spacing:.18em; }
.et-showroom__rail h1 { margin:12px 0 55px; max-width:180px; color:#f7f2e7; font:700 28px/1.02 'Arial',sans-serif; letter-spacing:.015em; }
.et-showroom__nav { display:grid; gap:20px; color:#766f61; font:500 10px 'Arial',sans-serif; }
.et-showroom__nav span { position:relative; padding-left:31px; }
.et-showroom__nav span:first-child { color:#f4efe2; }
.et-showroom__nav span:first-child:before { content:""; position:absolute; left:0; top:6px; width:18px; height:1px; background:#d0a646; }
.et-stage { position:absolute; inset:95px 4vw 155px 19vw; display:grid; place-items:center; }
.et-sequence { position:absolute; z-index:3; right:4.2vw; top:102px; color:rgba(244,239,226,.9); font:300 42px/1 'Arial',sans-serif; letter-spacing:-.02em; animation:et-reveal-down .7s .35s both; }
.et-sequence small { display:block; margin:7px 2px 0 0; color:#837a68; text-align:right; font:500 8px 'Arial',sans-serif; letter-spacing:.16em; }
.et-gear-label { position:absolute; z-index:3; top:112px; left:50%; color:#9b927f; font:500 8px 'Arial',sans-serif; letter-spacing:.16em; animation:et-reveal-down .7s .48s both; }
.et-register { position:absolute; z-index:4; color:#b9aa86; font:300 18px/1 'Arial',sans-serif; opacity:.7; }
.et-register--tl { top:100px; left:20px; } .et-register--tr { top:100px; right:20px; }
.et-register--bl { bottom:18px; left:20px; } .et-register--br { bottom:18px; right:20px; }
.et-stage__ghost { position:absolute; top:1%; left:3%; color:rgba(255,255,255,.025); font:800 clamp(11rem,27vw,31rem)/.75 'Arial',sans-serif; letter-spacing:-.065em; user-select:none; animation:et-ghost-in 1s .15s both; }
.et-stage__floor { position:absolute; width:min(72vw,1000px); height:180px; bottom:2%; border:1px solid rgba(214,174,79,.16); border-radius:50%; transform:perspective(700px) rotateX(67deg); box-shadow:0 25px 75px rgba(0,0,0,.85),inset 0 0 55px rgba(205,163,68,.035); animation:et-floor-in 1s .5s both; }
.et-engine-product { position:relative; z-index:2; width:min(68vw,920px); max-height:68vh; object-fit:contain; transform-origin:52% 72%; filter:drop-shadow(0 38px 38px rgba(0,0,0,.52)); animation:et-engine-in 1.35s cubic-bezier(.16,.8,.26,1) .25s both, et-engine-float 6s ease-in-out 1.6s infinite; }
.et-hotspot { position:absolute; z-index:4; display:flex; align-items:center; gap:8px; color:#b7ad97; font:500 8px 'Arial',sans-serif; letter-spacing:.08em; text-transform:uppercase; animation:et-hotspot-in .6s var(--delay) both; }
.et-hotspot:before { content:""; width:8px; height:8px; border:1px solid #d7ad49; border-radius:50%; background:#0b0a07; box-shadow:0 0 0 5px rgba(215,173,73,.08),0 0 15px rgba(215,173,73,.3); }
.et-hotspot:after { content:""; position:absolute; left:4px; width:1px; height:34px; bottom:10px; background:linear-gradient(#d7ad49,transparent); }
.et-hotspot--one { top:26%; left:48%; --delay:1.25s; }
.et-hotspot--two { top:46%; right:13%; --delay:1.4s; }
.et-hotspot--three { bottom:20%; left:31%; --delay:1.55s; }
.et-showroom__copy { position:absolute; left:4.2vw; bottom:48px; width:255px; color:#aaa18e; font:400 10px/1.7 'Arial',sans-serif; animation:et-reveal-up .7s .8s both; }
.et-showroom__copy b { display:block; margin-bottom:9px; color:#d6ad4d; font-size:9px; letter-spacing:.14em; }
.et-showroom__specs { position:absolute; left:34vw; right:4.2vw; bottom:48px; display:grid; grid-template-columns:repeat(4,1fr); gap:2vw; animation:et-reveal-up .7s .95s both; }
.et-spec { padding-top:12px; border-top:1px solid rgba(201,161,68,.2); }
.et-spec small { display:block; margin-bottom:7px; color:#9d927d; font:500 8px 'Arial',sans-serif; letter-spacing:.1em; text-transform:uppercase; }
.et-spec strong { color:#eee8dc; font:600 15px 'Arial',sans-serif; letter-spacing:.04em; }
.et-enter { position:absolute; right:4.2vw; bottom:112px; z-index:10; display:grid; place-items:center; width:186px; height:43px; border:1px solid #c9a144; color:#15120a!important; background:#d4ac4d; text-decoration:none!important; font:600 9px 'Arial',sans-serif; letter-spacing:.13em; animation:et-reveal-up .7s 1.05s both; transition:transform .2s ease,background .2s ease; cursor:pointer; }
.et-enter:hover { border-color:#f3cf78; color:#15120a!important; background:#f0c96d; transform:translateY(-2px); }
.stApp:has(.et-showroom) div[data-testid="stButton"] { position:fixed; right:4.2vw; bottom:112px; z-index:1000; width:186px; height:43px; opacity:.002; }
.stApp:has(.et-showroom) div[data-testid="stButton"] button { width:186px; height:43px; }
@keyframes et-engine-in { from { opacity:0; transform:translateX(13%) scale(.9) rotate(1.5deg); filter:blur(7px) drop-shadow(0 38px 38px rgba(0,0,0,.3)); } to { opacity:1; transform:translateX(0) scale(1) rotate(0); filter:blur(0) drop-shadow(0 38px 38px rgba(0,0,0,.52)); } }
@keyframes et-engine-float { 0%,100% { transform:translateY(0); } 50% { transform:translateY(-7px); } }
@keyframes et-ghost-in { from { opacity:0; transform:translateX(5%); } to { opacity:1; transform:none; } }
@keyframes et-floor-in { from { opacity:0; transform:perspective(700px) rotateX(67deg) scale(.7); } to { opacity:1; transform:perspective(700px) rotateX(67deg) scale(1); } }
@keyframes et-hotspot-in { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:none; } }
@keyframes et-reveal-up { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:none; } }
@keyframes et-reveal-down { from { opacity:0; transform:translateY(-10px); } to { opacity:1; transform:none; } }
@keyframes et-reveal-right { from { opacity:0; transform:translateX(-12px); } to { opacity:1; transform:none; } }
@media (max-width:900px) {
  .et-showroom { min-height:940px; }
  .et-showroom__top { padding:0 22px; }
  .et-showroom__rail { left:22px; top:120px; }
  .et-showroom__rail h1 { margin-bottom:24px; }
  .et-showroom__nav { display:none; }
  .et-sequence { right:22px; top:116px; font-size:30px; } .et-gear-label { display:none; }
  .et-stage { inset:220px 8px 320px; }
  .et-stage__ghost { font-size:48vw; top:16%; left:2%; }
  .et-engine-product { width:96vw; }
  .et-hotspot--one { top:34%; left:48%; } .et-hotspot--two { top:51%; right:3%; } .et-hotspot--three { bottom:23%; left:18%; }
  .et-showroom__copy { left:22px; right:22px; bottom:188px; width:auto; }
  .et-showroom__specs { left:22px; right:22px; bottom:92px; grid-template-columns:repeat(2,1fr); row-gap:14px; }
  .et-enter { right:22px; bottom:25px; }
  .stApp:has(.et-showroom) div[data-testid="stButton"] { right:22px; bottom:25px; }
}
@media (prefers-reduced-motion:reduce) { .et-showroom * { animation:none!important; } }

/* Readable, restrained shared presentation. */
:root { --et-text:#f5f4ef; --et-muted:#b4b3aa; --et-line:#34342e; }
html, body, .stApp, .stApp input, .stApp button, .stApp select, .stApp textarea {
  font-family:Helvetica,Arial,sans-serif!important;
}
h1,h2,h3,h4,p,label,[class^="et-"],[class^="et-"] * {
  font-family:Helvetica,Arial,sans-serif!important;
}
.stApp { background:#10110f; }
[data-testid="stSidebar"] { background:#171815; border-right:1px solid #34342e; }
.block-container { max-width:1480px; padding-top:2rem; }
h1,h2,h3 { letter-spacing:-.025em; font-weight:600!important; }
p, label { line-height:1.55; }
.et-hero { margin-bottom:1.4rem; padding-bottom:1.3rem; }
.et-hero h1 { font-size:2rem; }
.et-hero p { font-size:.95rem; line-height:1.5; }
.et-kicker { font-size:.75rem; letter-spacing:.08em; }
.et-live { font-size:.72rem; letter-spacing:.04em; border-color:#57503b; }
.et-dot { animation:none; box-shadow:none; }
.et-demo { font-size:.82rem; line-height:1.5; border-color:#504b36; }
.et-card { background:#191a17; border-color:#34342e; padding:1rem; }
.et-card:after { height:2px; box-shadow:none; }
.et-label,.et-section,.et-sub,.et-chip,.et-engine-head,.et-mini span,.et-health-copy span,.et-sensor small,.et-sensor em,.et-log-row {
  font-size:.8rem; line-height:1.45; letter-spacing:.02em; text-transform:none; color:#b4b3aa;
}
.et-value { font-size:1.9rem; font-weight:600; }
.et-unit { font-size:.9rem; }
.et-sensor strong { font-size:1.45rem; }
.et-engine-foot { gap:.75rem; }
[data-testid="stPlotlyChart"],.et-live-plot { border:1px solid #34342e; border-radius:10px; background:#151614; overflow:hidden; }
.et-showroom__rail h1 { width:max-content; max-width:90vw; font-size:clamp(28px,3.1vw,52px); line-height:1.15; letter-spacing:-.04em; }
.et-showroom__copy { font-size:13px; width:285px; }
.et-showroom__copy b,.et-spec small,.et-model-no,.et-wordmark,.et-system,.et-enter { font-size:11px; letter-spacing:.06em; }
.et-spec strong { font-size:17px; }
.et-showroom__specs { left:37vw; }
.et-hotspot { font-size:11px; }
@media(max-width:900px) { .et-showroom__specs { left:22px; } .et-showroom__copy { width:auto; } }

</style>
"""


def showroom_landing(image_path: str | Path) -> str:
    """A quiet product-configurator opening scene for the engine twin."""
    image = Path(image_path).read_bytes()
    encoded = base64.b64encode(image).decode("ascii")
    return f"""
    <main class="et-showroom" aria-label="EngineTwin introduction">
      <header class="et-showroom__top">
        <div class="et-wordmark"><i class="et-wordmark__mark"></i><span>POWER RANGER / ENGINE TWIN</span></div>
        <div class="et-system"><i></i><span>DIGITAL TWIN ONLINE</span></div>
      </header>
      <aside class="et-showroom__rail">
        <div class="et-model-no">ET-04 / INLINE FOUR</div>
        <h1>COMBUSTION<br>ENGINE TWIN</h1>
      </aside>
      <div class="et-gear-label">GEARS&nbsp;&nbsp;(04)</div>
      <div class="et-sequence">(00-08)<small>SIMULATION WINDOW / SEC</small></div>
      <i class="et-register et-register--tl">+</i><i class="et-register et-register--tr">+</i>
      <i class="et-register et-register--bl">+</i><i class="et-register et-register--br">+</i>
      <section class="et-stage">
        <div class="et-stage__ghost">TWIN</div>
        <div class="et-stage__floor"></div>
        <img class="et-engine-product" src="data:image/png;base64,{encoded}" alt="Studio rendering of an inline-four combustion engine">
        <div class="et-hotspot et-hotspot--one">Ignition array</div>
        <div class="et-hotspot et-hotspot--two">Air path</div>
        <div class="et-hotspot et-hotspot--three">Vibration node</div>
      </section>
      <div class="et-showroom__copy"><b>REAL-TIME ENGINE INTELLIGENCE</b>A synchronized view of combustion sound, vibration and ECU telemetry, translated into explainable fault evidence.</div>
      <section class="et-showroom__specs" aria-label="Engine specifications">
        <div class="et-spec"><small>Configuration</small><strong>INLINE-4</strong></div>
        <div class="et-spec"><small>Sensor streams</small><strong>10 CHANNELS</strong></div>
        <div class="et-spec"><small>Analysis</small><strong>FFT + STFT</strong></div>
        <div class="et-spec"><small>Guard state</small><strong>READY</strong></div>
      </section>
      <div class="et-enter" aria-hidden="true">ENTER LIVE TWIN&nbsp; →</div>
    </main>
    """


def hero() -> str:
    return """
    <div class="et-hero">
      <div><div class="et-kicker">EngineTwin / unified simulation</div>
      <h1>EngineTwin</h1>
      <p>Understand your engine. Explore its signals, condition and performance.</p></div>
      <div class="et-live"><i class="et-dot"></i> ENGINE WORKSPACE</div>
    </div>
    """


def toolbar(mode: str, position: float, fault: str, snr: int, duration: float = 8.0) -> str:
    state = "NOMINAL" if fault == "healthy" else fault.replace("_", " ").upper()
    return f"""
    <div class="et-toolbar">
      <div class="et-chip">MODE<br><strong>{escape(mode.upper())}</strong></div>
      <div class="et-chip">TIMELINE<br><strong>{position:04.1f} s / {duration:04.1f} s</strong></div>
      <div class="et-chip">AI GUARD<br><strong>{escape(state)}</strong></div>
      <div class="et-chip">SIGNAL QUALITY<br><strong>{snr} dB SNR</strong></div>
    </div>
    """


def metric_card(label: str, value: str, unit: str, subtitle: str, accent="#d8b453") -> str:
    return f"""
    <div class="et-card" style="--accent:{accent}">
      <div class="et-label">{escape(label)}</div>
      <div class="et-value">{escape(value)} <span class="et-unit">{escape(unit)}</span></div>
      <div class="et-sub">{escape(subtitle)}</div>
    </div>
    """


def scroll_engine_simulation(label: str, rpm: float, load: float) -> str:
    """Editorial engine-only motion scene inspired by a horizontal scroll sequence."""
    fault = label != "healthy"
    state = "NOMINAL" if not fault else label.replace("_", " ").upper()
    state_colour = "#d8aa32" if not fault else "#d94b45"
    return f"""
    <section class="et-drive" aria-label="Scrolling engine simulation">
      <div class="et-drive-meta">LIVE POWERTRAIN SCAN<b>{rpm:,.0f} RPM</b></div>
      <div class="et-drive-status" style="color:{state_colour}">● {escape(state)} · {load:.1f}% LOAD</div>
      <div class="et-marquee" aria-hidden="true"><span>ENGINE TWIN / COMBUSTION / </span><span>ENGINE TWIN / COMBUSTION / </span></div>
      <svg class="et-powertrain" viewBox="0 0 560 245" role="img" aria-label="Animated inline four cylinder engine moving through a diagnostic scan">
        <defs>
          <linearGradient id="goldCase" x1="0" x2="1"><stop stop-color="#16130b"/><stop offset=".5" stop-color="#493918"/><stop offset="1" stop-color="#0b0a07"/></linearGradient>
          <filter id="goldGlow"><feGaussianBlur stdDeviation="4" result="g"/><feMerge><feMergeNode in="g"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
        </defs>
        <g class="et-reticle" opacity=".52" stroke="#b88d29" fill="none">
          <circle cx="282" cy="119" r="103" stroke-dasharray="8 12"/><circle cx="282" cy="119" r="91" opacity=".45"/>
          <path d="M282 4v28M282 206v28M167 119h28M369 119h28"/>
        </g>
        <g class="et-scan" stroke="#e9be52" opacity=".45"><path d="M282 18v202"/><path d="M274 18v202" opacity=".2"/><path d="M290 18v202" opacity=".2"/></g>
        <g>
          <path d="M133 71h262l39 43-18 92H139l-34-34 5-75z" fill="url(#goldCase)" stroke="#e0b54d" stroke-width="3"/>
          <path d="M153 48h218l31 23H129z" fill="#17130a" stroke="#b88d29" stroke-width="3"/>
          <path d="M166 88h190v91H166z" fill="#080806" stroke="#7f6427"/>
          <g fill="#c99c32" opacity=".9">
            <rect class="et-piston" x="177" y="98" width="34" height="37" rx="5"/>
            <rect class="et-piston et-p2" x="221" y="98" width="34" height="37" rx="5"/>
            <rect class="et-piston et-p3" x="265" y="98" width="34" height="37" rx="5"/>
            <rect class="et-piston et-p4" x="309" y="98" width="34" height="37" rx="5"/>
          </g>
          <g stroke="#8f7438" stroke-width="4"><path d="M194 136v38"/><path d="M238 136v38"/><path d="M282 136v38"/><path d="M326 136v38"/></g>
          <g class="et-crank"><circle cx="260" cy="180" r="22" fill="#090806" stroke="#efc45d" stroke-width="5" stroke-dasharray="10 6"/><circle cx="260" cy="180" r="6" fill="#efc45d"/></g>
          <circle class="et-wheel" cx="111" cy="151" r="37" fill="#090806" stroke="#d3a842" stroke-width="8"/><circle cx="111" cy="151" r="10" fill="#d3a842"/>
          <circle class="et-wheel" cx="429" cy="151" r="37" fill="#090806" stroke="#d3a842" stroke-width="8"/><circle cx="429" cy="151" r="10" fill="#d3a842"/>
          <path d="M72 151H24M466 151h70" stroke="#9a782d" stroke-width="9" stroke-linecap="round"/>
          <circle cx="{194 if fault else 326}" cy="104" r="12" fill="{state_colour}" opacity="{'.9' if fault else '.32'}" filter="url(#goldGlow)"/>
        </g>
      </svg>
      <div class="et-drive-bottom"><div><strong>ENGINE-ONLY DIGITAL TWIN</strong>Four cylinders · crank · combustion signature · diagnostic targeting</div>
        <div class="et-scroll-cue"><i></i><span>SCROLL TO INSPECT<br>THE LIVE ENGINE</span></div></div>
    </section>
    """


def engine_panel(label: str, severity: float, rpm: float, load: float) -> str:
    nominal = label == "healthy"
    colour = "#d8aa32" if nominal else "#d78b80"
    state = "NOMINAL" if nominal else label.replace("_", " ").upper()
    hot_x = {"misfire": 103, "knock": 174, "valve_wear": 242}.get(label.split("+")[0], 174)
    health = max(18, round(100 - severity * (34 if not nominal else 0)))
    return f"""
    <div class="et-engine-card" style="--state:{colour}">
      <div class="et-engine-head"><span>Interactive engine twin</span><span class="et-engine-state">● {state}</span></div>
      <svg class="et-engine-svg" viewBox="0 0 360 280" role="img" aria-label="Animated four cylinder engine">
        <defs>
          <linearGradient id="block" x1="0" x2="1"><stop stop-color="#3a2e13"/><stop offset="1" stop-color="#0b0a06"/></linearGradient>
          <filter id="glow"><feGaussianBlur stdDeviation="5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
        </defs>
        <path d="M65 70 L292 70 320 112 302 219 74 219 43 185 47 105Z" fill="url(#block)" stroke="#a47f29" stroke-width="2"/>
        <path d="M78 55 H280 L300 72 H62Z" fill="#211a09" stroke="#ba9131"/>
        <g stroke="#766027" fill="#090806">
          <rect x="82" y="79" width="48" height="105" rx="12"/><rect x="133" y="79" width="48" height="105" rx="12"/>
          <rect x="184" y="79" width="48" height="105" rx="12"/><rect x="235" y="79" width="48" height="105" rx="12"/>
        </g>
        <g fill="#d7aa3d" opacity=".82">
          <rect class="et-piston" x="89" y="99" width="34" height="36" rx="5"/>
          <rect class="et-piston et-p2" x="140" y="99" width="34" height="36" rx="5"/>
          <rect class="et-piston et-p3" x="191" y="99" width="34" height="36" rx="5"/>
          <rect class="et-piston et-p4" x="242" y="99" width="34" height="36" rx="5"/>
        </g>
        <g stroke="#95783a" stroke-width="4"><path d="M106 137 106 190"/><path d="M157 137 157 190"/><path d="M208 137 208 190"/><path d="M259 137 259 190"/></g>
        <g class="et-crank"><circle cx="182" cy="193" r="25" fill="none" stroke="#f6b93b" stroke-width="5" stroke-dasharray="12 8"/><circle cx="182" cy="193" r="7" fill="#f6b93b"/></g>
        <path d="M44 117 C18 117 17 170 42 170" fill="none" stroke="#b88d29" stroke-width="7"/><path d="M305 103 C343 105 344 155 306 158" fill="none" stroke="#e0b64e" stroke-width="7"/>
        <circle class="et-hot" cx="{hot_x}" cy="101" r="17" fill="{colour}" opacity="{'.08' if nominal else '.82'}" filter="url(#glow)"/>
        <g fill="#aa8b44"><circle cx="70" cy="202" r="5"/><circle cx="292" cy="202" r="5"/><circle cx="67" cy="88" r="5"/><circle cx="291" cy="88" r="5"/></g>
        <text x="180" y="250" text-anchor="middle" fill="#9d8b61" font-family="Arial" font-size="9">INLINE-4 · FIRING ORDER 1-3-4-2</text>
      </svg>
      <div class="et-engine-foot">
        <div class="et-mini"><span>Crankshaft</span><strong>{rpm:,.0f} RPM</strong></div>
        <div class="et-mini"><span>Calculated load</span><strong>{load:.1f}%</strong></div>
      </div>
      <div class="et-health"><div class="et-ring" style="--health:{health}"><strong>{health}%</strong></div>
        <div class="et-health-copy"><b>CALCULATED HEALTH</b><span>{state}<br>Digital-twin confidence layer</span></div></div>
    </div>
    """


def sensor_grid(readings) -> str:
    cards = []
    for label, value, unit, state, colour in readings:
        cards.append(
            f'<div class="et-sensor" style="--sensor:{colour}"><small>{escape(label)}</small>'
            f'<strong>{escape(value)} <span style="font-size:.65em;color:#b99a4c">{escape(unit)}</span></strong><em>{escape(state)}</em></div>'
        )
    return '<div class="et-section">Live sensor array · synchronized telemetry channels</div><div class="et-sensor-grid">' + "".join(cards) + "</div>"


def event_log(label: str, confidence: float, position: float) -> str:
    fault = label != "healthy"
    return f"""
    <div class="et-section">Event log</div><div class="et-log">
      <div class="et-log-row"><time>00:00.0</time><b>10 streams synchronized</b><span>ONLINE</span></div>
      <div class="et-log-row"><time>00:03.0</time><b>{'Fault signature entered simulation' if fault else 'Healthy baseline locked'}</b><span>{'ALERT' if fault else 'NOMINAL'}</span></div>
      <div class="et-log-row"><time>{position:05.1f}</time><b>EngineGuard inference complete</b><span>{confidence:.0%} CONF.</span></div>
    </div>
    """
