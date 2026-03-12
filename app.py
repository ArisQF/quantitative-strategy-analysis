"""
╔══════════════════════════════════════════════════════╗
║         QUANT RISK DASHBOARD — WIN1! BMFBOVESPA      ║
║   Hybrid Volatility Stop + LRI · Métricas & Monte    ║
╚══════════════════════════════════════════════════════╝

Como rodar:
    streamlit run dashboard.py

Como passar seus dados reais:
    Salve o array de trades no seu script principal:
        import numpy as np
        trades = res['Trade'][res['Trade'] != 0].values
        np.save("trades_real.npy", trades)
    Depois carregue aqui (ver instrução na sidebar).
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quant Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── TEMA ESCURO CUSTOMIZADO ───────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace !important; }

    .main { background-color: #0a0e1a; }
    .stApp { background-color: #0a0e1a; color: #e2e8f0; }
    section[data-testid="stSidebar"] { background-color: #0f1628 !important; border-right: 1px solid #1e2d50; }
    .stTabs [data-baseweb="tab-list"] { background-color: #0f1628; border-bottom: 1px solid #1e2d50; gap: 0; }
    .stTabs [data-baseweb="tab"] { background-color: transparent; color: #64748b; font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 1.5px; text-transform: uppercase; padding: 14px 20px; }
    .stTabs [aria-selected="true"] { color: #00d4ff !important; border-bottom: 2px solid #00d4ff; }
    .stTabs [data-baseweb="tab-panel"] { background-color: #0a0e1a; padding-top: 24px; }
    div[data-testid="metric-container"] { background: #141c35; border: 1px solid #1e2d50; border-radius: 12px; padding: 14px 18px; }
    div[data-testid="metric-container"] > label { color: #64748b !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 1.5px; font-family: 'JetBrains Mono', monospace !important; }
    div[data-testid="metric-container"] > div { font-family: 'JetBrains Mono', monospace !important; }
    h1, h2, h3 { color: #e2e8f0 !important; font-family: 'JetBrains Mono', monospace !important; }
    .stSlider > div > div > div { background-color: #00d4ff !important; }
    .block-container { padding: 1.5rem 2rem 2rem; }
    hr { border-color: #1e2d50; }
    .stFileUploader { background: #141c35; border: 1px dashed #1e2d50; border-radius: 10px; }
    .stAlert { background: #141c35 !important; border: 1px solid #1e2d50 !important; }
</style>
""", unsafe_allow_html=True)


# ── ENGINES ───────────────────────────────────────────────────────────────────

@st.cache_data
def calc_metrics(trades: np.ndarray) -> dict:
    n = len(trades)
    if n == 0:
        return {}

    wins   = trades[trades > 0]
    losses = trades[trades < 0]
    zeros  = trades[trades == 0]

    win_rate = len(wins) / max(len(wins) + len(losses), 1)
    avg_win  = wins.mean()  if len(wins)   > 0 else 0.0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0
    payoff   = avg_win / avg_loss if avg_loss > 0 else np.inf
    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

    # Equity curve
    equity_curve = np.cumsum(trades)
    peak = np.maximum.accumulate(equity_curve)
    dd   = np.where(peak > 0, (peak - equity_curve) / peak, 0.0)
    max_dd = dd.max()

    # Sharpe / Sortino
    mu  = trades.mean()
    std = trades.std()
    neg = trades[trades < 0]
    down_std = np.sqrt((neg**2).sum() / n) if len(neg) > 0 else 1e-9
    sharpe  = (mu / std)      * np.sqrt(n) if std > 0 else 0.0
    sortino = (mu / down_std) * np.sqrt(n)

    # Calmar
    total_ret = equity_curve[-1]
    calmar = total_ret / (max_dd * abs(total_ret) + 1e-9) if max_dd > 0 else np.inf

    # Profit factor
    gp = wins.sum()
    gl = abs(losses.sum())
    pf = gp / gl if gl > 0 else np.inf

    # Consecutivas
    max_cw = max_cl = cw = cl = 0
    for t in trades:
        if t > 0:  cw += 1; cl = 0; max_cw = max(max_cw, cw)
        elif t < 0: cl += 1; cw = 0; max_cl = max(max_cl, cl)
        else: cw = cl = 0

    # VaR / CVaR / Skew / Kurt
    sorted_t = np.sort(trades)
    idx5 = max(int(n * 0.05), 1)
    var95  = sorted_t[idx5]
    cvar95 = sorted_t[:idx5].mean() if idx5 > 0 else var95
    skew = ((trades - mu)**3).mean() / (std**3 + 1e-9)
    kurt = ((trades - mu)**4).mean() / (std**4 + 1e-9) - 3

    # Rolling Sharpe
    w = min(50, n // 4)
    roll_sharpe = []
    for i in range(w, n + 1):
        sl = trades[i-w:i]
        s, m2 = sl.std(), sl.mean()
        roll_sharpe.append(m2 / s * np.sqrt(w) if s > 0 else 0.0)

    # Rolling win rate (window=30)
    ww = min(30, n // 4)
    roll_wr = [(trades[i-ww:i] > 0).mean() for i in range(ww, n + 1)]

    return dict(
        n=n, wins=len(wins), losses=len(losses), zeros=len(zeros),
        win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss,
        payoff=payoff, expectancy=expectancy,
        total_ret=total_ret, max_dd=max_dd,
        sharpe=sharpe, sortino=sortino, calmar=calmar, pf=pf,
        max_cw=max_cw, max_cl=max_cl,
        equity_curve=equity_curve, dd=dd, peak=peak,
        mu=mu, std=std,
        gp=gp, gl=gl,
        var95=var95, cvar95=cvar95, skew=skew, kurt=kurt,
        roll_sharpe=roll_sharpe, roll_wr=roll_wr,
        sorted_t=sorted_t,
    )


@st.cache_data
def run_monte_carlo(win_rate, avg_win, avg_loss, n_trades, n_sims, capital):
    rng = np.random.default_rng(42)
    outcomes = rng.random((n_sims, n_trades))
    deltas   = np.where(outcomes < win_rate, avg_win, -avg_loss)
    paths    = np.hstack([
        np.full((n_sims, 1), capital),
        capital + np.cumsum(deltas, axis=1)
    ])
    final   = paths[:, -1]
    peaks   = np.maximum.accumulate(paths, axis=1)
    max_dds = ((peaks - paths) / np.maximum(peaks, 1e-9)).max(axis=1)
    return paths, final, max_dds


def percentile_fan(paths, step=1):
    n_pts = paths.shape[1]
    idx   = np.arange(0, n_pts, max(1, step))
    ps    = np.percentile(paths[:, idx], [5, 25, 50, 75, 95], axis=0)
    return idx, ps   # shape (5, len(idx))


# ── PALETA ────────────────────────────────────────────────────────────────────
C = dict(bg="#0a0e1a", surface="#0f1628", card="#141c35",
         border="#1e2d50", accent="#00d4ff", accent2="#7c3aed",
         green="#00e676", red="#ff1744", yellow="#ffd600",
         text="#e2e8f0", muted="#64748b")

PLOT_LAYOUT = dict(
    paper_bgcolor=C["card"], plot_bgcolor=C["surface"],
    font=dict(color=C["text"], family="JetBrains Mono"),
    xaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"]),
    yaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"]),
    margin=dict(l=50, r=20, t=30, b=40),
)


# ── HELPERS UI ────────────────────────────────────────────────────────────────

def section(title: str):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin:28px 0 12px">
      <div style="width:4px;height:20px;background:linear-gradient({C['accent']},{C['accent2']});border-radius:2px"></div>
      <span style="color:{C['text']};font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:2px">{title}</span>
      <div style="flex:1;height:1px;background:{C['border']}"></div>
    </div>""", unsafe_allow_html=True)


def health_bar(label, value, bad, ok, good, fmt_fn):
    score = max(0.0, min(1.0, (value - bad) / max(good - bad, 1e-9)))
    color = C["green"] if score > 0.66 else C["yellow"] if score > 0.33 else C["red"]
    pct   = int(score * 100)
    st.markdown(f"""
    <div style="background:{C['card']};border:1px solid {C['border']};border-radius:10px;padding:12px 16px;margin-bottom:8px">
      <div style="display:flex;justify-content:space-between;margin-bottom:6px">
        <span style="color:{C['muted']};font-size:11px;text-transform:uppercase;letter-spacing:1px">{label}</span>
        <span style="color:{color};font-size:13px;font-weight:700">{fmt_fn(value)}</span>
      </div>
      <div style="height:6px;background:{C['border']};border-radius:3px;overflow:hidden">
        <div style="height:100%;width:{pct}%;background:{color};border-radius:3px"></div>
      </div>
      <div style="display:flex;justify-content:space-between;margin-top:3px">
        <span style="color:{C['red']};font-size:9px">RUIM</span>
        <span style="color:{C['yellow']};font-size:9px">OK</span>
        <span style="color:{C['green']};font-size:9px">BOM</span>
      </div>
    </div>""", unsafe_allow_html=True)


def fmts(v, d=2):
    return f"{v:.{d}f}" if np.isfinite(v) else "∞"


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h3 style='color:{C['accent']};letter-spacing:2px'>⚡ CONFIGURAÇÃO</h3>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown(f"<span style='color:{C['muted']};font-size:11px;text-transform:uppercase;letter-spacing:1px'>Fonte de Dados</span>", unsafe_allow_html=True)
    data_source = st.radio("", ["Dados Sintéticos (demo)", "Upload .npy / .csv"], label_visibility="collapsed")

    trades_arr: np.ndarray | None = None

    if data_source == "Upload .npy / .csv":
        uploaded = st.file_uploader("Arquivo com trades (vetor 1-D)", type=["npy", "csv"])
        if uploaded:
            try:
                if uploaded.name.endswith(".npy"):
                    trades_arr = np.load(uploaded).flatten()
                else:
                    trades_arr = pd.read_csv(uploaded, header=None).values.flatten().astype(float)
                st.success(f"✓ {len(trades_arr)} trades carregados")
            except Exception as e:
                st.error(f"Erro ao carregar: {e}")
        else:
            st.info("Salve no Python:\n```python\nnp.save('trades.npy',\n  res['Trade'][res['Trade']!=0].values)\n```")
    else:
        st.markdown("---")
        st.markdown(f"<span style='color:{C['muted']};font-size:11px;text-transform:uppercase;letter-spacing:1px'>Parâmetros Sintéticos</span>", unsafe_allow_html=True)
        syn_wr  = st.slider("Win Rate (%)", 35, 70, 52) / 100
        syn_po  = st.slider("Payoff", 0.5, 4.0, 1.8, 0.1)
        syn_n   = st.slider("Nº de Trades", 100, 2000, 800, 50)
        np.random.seed(0)
        avg_w = 100.0
        avg_l = avg_w / syn_po
        raw = np.random.random(syn_n)
        trades_arr = np.where(raw < syn_wr, avg_w * (0.5 + np.random.random(syn_n)),
                              -avg_l * (0.5 + np.random.random(syn_n)))

    st.markdown("---")
    st.markdown(f"<span style='color:{C['muted']};font-size:11px;text-transform:uppercase;letter-spacing:1px'>Monte Carlo</span>", unsafe_allow_html=True)
    mc_sims   = st.slider("Simulações",    100, 3000, 500, 100)
    mc_trades = st.slider("Trades futuros", 50, 1000, 200, 50)
    mc_cap    = st.number_input("Capital Inicial (R$)", 1000, 500000, 10000, 1000)

    st.markdown("---")
    st.markdown(f"<span style='color:{C['muted']};font-size:10px'>WIN1! · BMFBOVESPA · 1-MIN<br>Hybrid Vol Stop + LRI · Δ_bin</span>", unsafe_allow_html=True)


# ── GUARD ─────────────────────────────────────────────────────────────────────
if trades_arr is None or len(trades_arr) == 0:
    st.warning("Nenhum dado carregado.")
    st.stop()

m = calc_metrics(trades_arr)
active_trades = trades_arr[trades_arr != 0]

# ── HEADER ────────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown(f"""
    <h1 style='color:{C["accent"]};letter-spacing:3px;margin-bottom:4px;font-size:22px'>
    📈 QUANT RISK DASHBOARD
    </h1>
    <span style='color:{C["muted"]};font-size:11px'>
    WIN1! · BMFBOVESPA · Hybrid Volatility Stop · LRI-Delta Filter
    </span>""", unsafe_allow_html=True)
with col_h2:
    status_color = C["green"] if m["total_ret"] > 0 else C["red"]
    st.markdown(f"""
    <div style='text-align:right;margin-top:10px'>
      <span style='color:{status_color};font-size:11px'>● BACKTEST LOADED</span><br>
      <span style='color:{C["muted"]};font-size:11px'>{m["n"]} trades · {m["wins"]}W / {m["losses"]}L</span>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["OVERVIEW", "EQUITY & DRAWDOWN", "MONTE CARLO", "DISTRIBUIÇÃO"])


# ═══════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════
with tab1:
    section("Resumo de Performance")
    c1, c2, c3, c4 = st.columns(4)
    ret_color = "normal" if m["total_ret"] >= 0 else "inverse"
    c1.metric("Total P&L", f"R$ {m['total_ret']:,.0f}")
    c2.metric("Win Rate",  f"{m['win_rate']*100:.1f}%", f"{m['wins']}W / {m['losses']}L")
    c3.metric("Payoff",    fmts(m['payoff'], 2) + "x",  f"Avg W R${m['avg_win']:.1f} | L R${m['avg_loss']:.1f}")
    c4.metric("Expectância/Trade", f"R$ {m['expectancy']:.2f}")

    section("Métricas de Risco")
    r1, r2, r3, r4, r5, r6 = st.columns(6)
    r1.metric("Sharpe",        fmts(m['sharpe'], 2))
    r2.metric("Sortino",       fmts(m['sortino'], 2))
    r3.metric("Calmar",        fmts(m['calmar'], 2))
    r4.metric("Profit Factor", fmts(m['pf'], 2))
    r5.metric("Max Drawdown",  f"{m['max_dd']*100:.1f}%")
    r6.metric("Std Dev/trade", f"R$ {m['std']:.1f}")

    section("Trades")
    t1, t2, t3, t4, t5, t6 = st.columns(6)
    t1.metric("Total Trades",     m['n'])
    t2.metric("Avg Ganho",        f"R$ {m['avg_win']:.1f}")
    t3.metric("Avg Perda",        f"R$ {m['avg_loss']:.1f}")
    t4.metric("Gross Profit",     f"R$ {m['gp']:,.0f}")
    t5.metric("Gross Loss",       f"R$ {m['gl']:,.0f}")
    t6.metric("Consec. Perda Max", m['max_cl'])

    section("Avaliação do Modelo")
    hc1, hc2 = st.columns(2)
    with hc1:
        health_bar("Win Rate",      m['win_rate'], 0.40, 0.50, 0.62, lambda v: f"{v*100:.1f}%")
        health_bar("Payoff",        m['payoff'] if np.isfinite(m['payoff']) else 5.0,
                                     0.8, 1.2, 2.5, lambda v: f"{v:.2f}x")
        health_bar("Profit Factor", m['pf'] if np.isfinite(m['pf']) else 5.0,
                                     0.9, 1.3, 2.5, lambda v: f"{v:.2f}")
    with hc2:
        health_bar("Sharpe Ratio",  m['sharpe'],              0.0,  0.5,  2.0,  lambda v: f"{v:.2f}")
        health_bar("Max Drawdown",  1 - m['max_dd'],           0.70, 0.85, 0.95, lambda v: f"{m['max_dd']*100:.1f}%")
        health_bar("Expectância",   m['expectancy'],          -10,   0,   30,   lambda v: f"R$ {v:.1f}")


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — EQUITY & DRAWDOWN
# ═══════════════════════════════════════════════════════════════════
with tab2:
    section("Curva de Equity Acumulada")
    eq_idx = np.arange(len(m["equity_curve"]))

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=eq_idx, y=m["equity_curve"],
        mode="lines", name="Equity",
        line=dict(color=C["accent"], width=1.5),
        fill="tozeroy",
        fillcolor=f"rgba(0,212,255,0.08)",
    ))
    fig_eq.add_hline(y=0, line_dash="dash", line_color=C["muted"], line_width=1)
    fig_eq.update_layout(**PLOT_LAYOUT, height=300,
                         yaxis_title="R$", xaxis_title="Trade #")
    st.plotly_chart(fig_eq, use_container_width=True)

    col_dd, col_rs = st.columns(2)

    with col_dd:
        section("Drawdown (%)")
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=eq_idx, y=-m["dd"] * 100,
            mode="lines", name="Drawdown",
            line=dict(color=C["red"], width=1.5),
            fill="tozeroy",
            fillcolor=f"rgba(255,23,68,0.12)",
        ))
        fig_dd.update_layout(**PLOT_LAYOUT, height=250,
                             yaxis_title="%", xaxis_title="Trade #")
        st.plotly_chart(fig_dd, use_container_width=True)

    with col_rs:
        section("Rolling Sharpe (50 trades)")
        rs_idx = np.arange(len(m["roll_sharpe"]))
        fig_rs = go.Figure()
        fig_rs.add_hline(y=0, line_dash="dash", line_color=C["muted"], line_width=1)
        fig_rs.add_hline(y=1, line_dash="dot",  line_color=C["green"], line_width=1, opacity=0.5)
        fig_rs.add_trace(go.Scatter(
            x=rs_idx, y=m["roll_sharpe"],
            mode="lines", name="Sharpe",
            line=dict(color=C["accent2"], width=1.5),
        ))
        fig_rs.update_layout(**PLOT_LAYOUT, height=250,
                             yaxis_title="Sharpe", xaxis_title="Trade #")
        st.plotly_chart(fig_rs, use_container_width=True)

    section("Rolling Win Rate (30 trades)")
    wr_idx = np.arange(len(m["roll_wr"]))
    fig_wr = go.Figure()
    fig_wr.add_hrect(y0=0.5, y1=1.0, fillcolor=f"rgba(0,230,118,0.05)", line_width=0)
    fig_wr.add_hline(y=0.5, line_dash="dash", line_color=C["yellow"], line_width=1)
    fig_wr.add_trace(go.Scatter(
        x=wr_idx, y=np.array(m["roll_wr"]) * 100,
        mode="lines", name="Win Rate %",
        line=dict(color=C["green"], width=1.5),
        fill="tozeroy", fillcolor="rgba(0,230,118,0.07)",
    ))
    fig_wr.update_layout(**PLOT_LAYOUT, height=220,
                         yaxis_title="%", xaxis_title="Trade #")
    st.plotly_chart(fig_wr, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — MONTE CARLO
# ═══════════════════════════════════════════════════════════════════
with tab3:
    avg_w = m["avg_win"]
    avg_l = m["avg_loss"]
    wr    = m["win_rate"]

    with st.spinner("Rodando simulações..."):
        paths, final_eq, max_dds = run_monte_carlo(wr, avg_w, avg_l, mc_trades, mc_sims, mc_cap)

    prob_profit = (final_eq > mc_cap).mean() * 100
    prob_ruin   = (final_eq < mc_cap * 0.5).mean() * 100
    med_final   = np.median(final_eq)
    p5_final    = np.percentile(final_eq, 5)
    p95_final   = np.percentile(final_eq, 95)
    avg_max_dd  = max_dds.mean() * 100

    section("KPIs da Simulação")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Prob. Lucro",        f"{prob_profit:.1f}%")
    k2.metric("Prob. Ruin (>50%DD)", f"{prob_ruin:.1f}%")
    k3.metric("Mediana Final",      f"R$ {med_final:,.0f}")
    k4.metric("P5 Final",           f"R$ {p5_final:,.0f}")
    k5.metric("Avg Max Drawdown",   f"{avg_max_dd:.1f}%")

    section("Fan Chart — Percentis de Equity Futura")
    step = max(1, mc_trades // 100)
    idx, ps = percentile_fan(paths, step)

    fig_fan = go.Figure()
    # Faixas
    fig_fan.add_trace(go.Scatter(x=np.concatenate([idx, idx[::-1]]),
        y=np.concatenate([ps[4], ps[0][::-1]]),
        fill="toself", fillcolor="rgba(0,212,255,0.04)",
        line=dict(color="rgba(0,0,0,0)"), name="P5–P95", showlegend=True))
    fig_fan.add_trace(go.Scatter(x=np.concatenate([idx, idx[::-1]]),
        y=np.concatenate([ps[3], ps[1][::-1]]),
        fill="toself", fillcolor="rgba(0,212,255,0.10)",
        line=dict(color="rgba(0,0,0,0)"), name="P25–P75", showlegend=True))
    # Linhas de borda
    fig_fan.add_trace(go.Scatter(x=idx, y=ps[4], mode="lines",
        line=dict(color=C["green"], width=1, dash="dot"), name="P95"))
    fig_fan.add_trace(go.Scatter(x=idx, y=ps[0], mode="lines",
        line=dict(color=C["red"],   width=1, dash="dot"), name="P5"))
    # Mediana
    fig_fan.add_trace(go.Scatter(x=idx, y=ps[2], mode="lines",
        line=dict(color=C["accent"], width=2.5), name="Mediana"))
    # Capital inicial
    fig_fan.add_hline(y=mc_cap, line_dash="dash",
                      line_color=C["yellow"], line_width=1.5,
                      annotation_text="Capital Inicial",
                      annotation_font_color=C["yellow"])
    fig_fan.update_layout(**PLOT_LAYOUT, height=340,
                          xaxis_title="Trades", yaxis_title="R$",
                          legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)))
    st.plotly_chart(fig_fan, use_container_width=True)

    col_fd, col_dd_mc = st.columns(2)

    with col_fd:
        section("Distribuição do Capital Final")
        colors_fd = [C["red"] if v < mc_cap else C["green"] for v in final_eq]
        fig_fd = go.Figure()
        fig_fd.add_trace(go.Histogram(
            x=final_eq, nbinsx=40,
            marker=dict(color=colors_fd, line=dict(width=0)),
            name="Capital Final",
        ))
        fig_fd.add_vline(x=mc_cap, line_dash="dash",
                         line_color=C["yellow"], line_width=2,
                         annotation_text="Capital Inicial",
                         annotation_font_color=C["yellow"])
        fig_fd.update_layout(**PLOT_LAYOUT, height=260,
                             xaxis_title="R$", yaxis_title="Freq.")
        st.plotly_chart(fig_fd, use_container_width=True)

    with col_dd_mc:
        section("Distribuição do Max Drawdown (simulações)")
        fig_mdd = go.Figure()
        fig_mdd.add_trace(go.Histogram(
            x=max_dds * 100, nbinsx=40,
            marker=dict(color=C["red"], opacity=0.75),
            name="Max DD %",
        ))
        fig_mdd.add_vline(x=m["max_dd"] * 100, line_dash="dash",
                          line_color=C["accent"], line_width=2,
                          annotation_text="Backtest DD",
                          annotation_font_color=C["accent"])
        fig_mdd.update_layout(**PLOT_LAYOUT, height=260,
                              xaxis_title="%", yaxis_title="Freq.")
        st.plotly_chart(fig_mdd, use_container_width=True)

    section("Tabela de Cenários")
    scenarios = pd.DataFrame({
        "Cenário":       ["Pessimista (P5)", "Conservador (P25)", "Base (Mediana)", "Otimista (P75)", "Excelente (P95)"],
        "Capital Final": [f"R$ {np.percentile(final_eq, p):,.0f}" for p in [5, 25, 50, 75, 95]],
        "Retorno (%)":   [f"{(np.percentile(final_eq, p)/mc_cap - 1)*100:.1f}%" for p in [5, 25, 50, 75, 95]],
        "Prob. de Ocorrer": ["5%", "25%", "50%", "25%", "5%"],
    })
    st.dataframe(scenarios, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 4 — DISTRIBUIÇÃO
# ═══════════════════════════════════════════════════════════════════
with tab4:
    col_h, col_sc = st.columns(2)

    with col_h:
        section("Histograma de Retornos")
        colors_h = [C["red"] if v < 0 else C["green"] for v in active_trades]
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=active_trades, nbinsx=40,
            marker=dict(color=colors_h, line=dict(width=0)),
        ))
        fig_hist.add_vline(x=0, line_dash="dash", line_color=C["muted"], line_width=1)
        fig_hist.add_vline(x=m["mu"], line_dash="dot", line_color=C["accent"],
                           annotation_text=f"Média R${m['mu']:.1f}",
                           annotation_font_color=C["accent"])
        fig_hist.update_layout(**PLOT_LAYOUT, height=280,
                               xaxis_title="R$", yaxis_title="Freq.")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_sc:
        section("Scatter — Trade # vs Retorno")
        sc_idx = np.arange(len(active_trades))
        sc_colors = [C["green"] if v > 0 else C["red"] for v in active_trades]
        fig_sc = go.Figure()
        fig_sc.add_hline(y=0, line_dash="dash", line_color=C["muted"], line_width=1)
        fig_sc.add_trace(go.Scatter(
            x=sc_idx, y=active_trades,
            mode="markers",
            marker=dict(color=sc_colors, size=3, opacity=0.6),
            name="Trade",
        ))
        fig_sc.update_layout(**PLOT_LAYOUT, height=280,
                             xaxis_title="Trade #", yaxis_title="R$")
        st.plotly_chart(fig_sc, use_container_width=True)

    section("Estatísticas Avançadas")
    st_cols = st.columns(4)
    stats = [
        ("VaR 95%",   f"R$ {m['var95']:.1f}",  "Pior 5% dos trades"),
        ("CVaR 95%",  f"R$ {m['cvar95']:.1f}", "Média dos piores 5%"),
        ("Skewness",  f"{m['skew']:.3f}",       "Cauda à direita ✓" if m['skew'] > 0 else "Cauda à esquerda ⚠"),
        ("Kurtosis",  f"{m['kurt']:.3f}",       "Leptocúrtica" if m['kurt'] > 0 else "Platicúrtica"),
        ("Mediana",   f"R$ {np.median(active_trades):.1f}", "P50 dos trades"),
        ("P10",       f"R$ {np.percentile(active_trades, 10):.1f}", "Pior decil"),
        ("P90",       f"R$ {np.percentile(active_trades, 90):.1f}", "Melhor decil"),
        ("Std Dev",   f"R$ {m['std']:.1f}",     "Desvio por trade"),
    ]
    for i, (lbl, val, sub) in enumerate(stats):
        st_cols[i % 4].metric(lbl, val, sub)

    section("Q-Q Plot — Normalidade dos Retornos")
    from scipy import stats as sp_stats
    (osm, osr), (slope, intercept, r) = sp_stats.probplot(active_trades, dist="norm")
    qq_line = slope * np.array(osm) + intercept

    fig_qq = go.Figure()
    fig_qq.add_trace(go.Scatter(x=osm, y=osr, mode="markers",
        marker=dict(color=C["accent"], size=3, opacity=0.6), name="Dados"))
    fig_qq.add_trace(go.Scatter(x=osm, y=qq_line, mode="lines",
        line=dict(color=C["red"], width=2, dash="dash"), name="Normal teórica"))
    fig_qq.update_layout(**PLOT_LAYOUT, height=280,
                         xaxis_title="Quantis Teóricos", yaxis_title="Quantis Observados",
                         annotations=[dict(x=0.05, y=0.95, xref="paper", yref="paper",
                                          text=f"R² = {r**2:.4f}",
                                          font=dict(color=C["accent"], size=12),
                                          showarrow=False)])
    st.plotly_chart(fig_qq, use_container_width=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style='text-align:center;color:{C["muted"]};font-size:10px;letter-spacing:1px;padding:8px'>
⚠ BACKTEST NÃO GARANTE RESULTADOS FUTUROS · WIN1! 1-MIN · HYBRID VOL STOP + LRI-DELTA FILTER
</div>""", unsafe_allow_html=True)