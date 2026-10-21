"""MXN vs G10 FX Options Risk Workbench - Streamlit corporate hedging-desk prototype."""
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from risk_engine import (
    COLUMNS, DEFAULT_MXN_RATE, DEFAULT_USDMXN, INSTRUMENTS, NUMERIC, OPTION_TYPES, PAIRS,
    ROUNDING_MODES, SPECS, TEXT_COLS, base_ccy, clean_ledger, ensure_columns, exposure_units,
    forward_rate, gk_greeks, gk_price, gk_price_vec, implied_volatility, ledger_issues,
    ledger_risk, min_variance_beta, normalize_pair, pair_summary, size_hedge, spec_for,
    strike_from_delta, stress_grid, to_foreign, zero_cost_collar_strike)

st.set_page_config(page_title='MXN FX Options Risk Workbench', layout='wide')
st.title('MXN vs G10 FX Options Risk Workbench')
st.caption('Pre-trade option analytics • hedge sizing (forward / option / zero-cost collar) • trade ledger • '
           'portfolio Greeks • scenario risk • all risk reported in MXN • Created by Gerardo Ferreira')


PURPOSES = ['Hedge', 'Hedge / optionality', 'Risk reduction', 'Trading / opportunistic']
PAYABLE = 'Payable: need to buy foreign ccy (importer / FX debt service)'
RECEIVABLE = 'Receivable: will sell foreign ccy (exporter / FX income)'
label_of = lambda p: SPECS[p]['label']


def dp_of(pair):
    return spec_for(pair)['dp']


def step_of(pair):
    return float(spec_for(pair)['pip'] * 100)


def fmt_of(pair):
    return f'%.{dp_of(pair)}f'


def default_premium(pair, days=90):
    s = spec_for(pair)
    return round(gk_price(s['spot'], s['spot'], days / 365, DEFAULT_MXN_RATE, s['foreign_rate'], s['vol'], 'Call'),
                 s['dp'] + 1)


def sample_ledger():
    today = str(date.today())
    us, eu = SPECS['USDMXN'], SPECS['EURMXN']
    prem = round(gk_price(us['spot'], 19.00, 90 / 365, DEFAULT_MXN_RATE, us['foreign_rate'], us['vol'], 'Call'), 4)
    fwd = round(float(forward_rate(eu['spot'], 180 / 365, DEFAULT_MXN_RATE, eu['foreign_rate'])), 4)
    base = dict(trade_date=today, mxn_rate=DEFAULT_MXN_RATE, notes='Example only')
    return pd.DataFrame([
        {**base, 'trade_id': 'DEMO-001', 'pair': 'USDMXN', 'maturity': 'Jan-2027', 'instrument_type': 'Call',
         'signed_notional': 1_000_000.0, 'trade_price': prem, 'spot': us['spot'], 'strike': 19.00,
         'days_to_expiry': 90.0, 'foreign_rate': us['foreign_rate'], 'implied_volatility': us['vol'],
         'market_option_premium': prem, 'purpose': 'Hedge / optionality'},
        {**base, 'trade_id': 'DEMO-002', 'pair': 'EURMXN', 'maturity': 'Apr-2027', 'instrument_type': 'Forward',
         'signed_notional': -500_000.0, 'trade_price': round(fwd + 0.15, 4), 'spot': eu['spot'], 'strike': 0.0,
         'days_to_expiry': 180.0, 'foreign_rate': eu['foreign_rate'], 'implied_volatility': 0.0,
         'market_option_premium': 0.0, 'purpose': 'Hedge'},
    ], columns=COLUMNS)


def typed(df):
    out = df.copy()
    for c in NUMERIC:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    for c in TEXT_COLS:
        out[c] = out[c].fillna('').astype(str)
    return out[COLUMNS]


def next_trade_id(ledger):
    stamp = date.today().strftime('%Y%m%d')
    existing = set(ledger['trade_id'].astype(str))
    n = len(ledger) + 1
    while f'TR-{stamp}-{n:03d}' in existing:
        n += 1
    return f'TR-{stamp}-{n:03d}'


def add_trade(row):
    ledger = st.session_state.ledger
    full = {c: row.get(c, '' if c in TEXT_COLS else 0.0) for c in COLUMNS}
    if not full['trade_id']:
        full['trade_id'] = next_trade_id(ledger)
    if not full['trade_date']:
        full['trade_date'] = str(date.today())
    st.session_state.ledger = typed(pd.concat([ledger, pd.DataFrame([full])], ignore_index=True))
    return full['trade_id']


if 'ledger' not in st.session_state:
    st.session_state.ledger = sample_ledger()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header('Pre-trade option ticket')
    pair = st.selectbox('Currency pair (G10 vs MXN)', PAIRS, index=0, format_func=label_of)
    s = spec_for(pair)
    base, dp = s['base'], s['dp']
    spot = st.number_input(f'Spot (MXN per 1 {base})', min_value=1e-6, value=float(s['spot']),
                           step=step_of(pair), format=fmt_of(pair), key=f'sb_spot_{pair}')
    opt_type = st.selectbox('Option type', ['Call', 'Put'],
                            help=f'Call = right to BUY {base} / sell MXN (protects a {base} payable). '
                                 f'Put = right to SELL {base} / buy MXN (protects a {base} receivable).')
    days = st.number_input('Days to expiry', min_value=1, value=90, step=1)
    vol_pct = st.number_input('Volatility (%)', min_value=0.01, value=float(s['vol'] * 100), step=0.25,
                              key=f'sb_vol_{pair}')
    rd_pct = st.number_input('MXN rate (%, cont. comp.)', value=DEFAULT_MXN_RATE * 100, step=0.25, key='sb_rd')
    rf_pct = st.number_input(f'{base} rate (%, cont. comp.)', value=float(s['foreign_rate'] * 100), step=0.25,
                             key=f'sb_rf_{pair}')
    vol, rd, rf, T = vol_pct / 100, rd_pct / 100, rf_pct / 100, days / 365
    strike_mode = st.radio('Strike input', ['Absolute strike', 'Delta (spot, premium excl.)'], horizontal=True)
    if strike_mode.startswith('Absolute'):
        k = st.number_input('Strike (MXN per 1 FX)', min_value=1e-6, value=float(s['spot']), step=step_of(pair),
                            format=fmt_of(pair), key=f'sb_k_{pair}')
    else:
        d_pct = st.number_input('Absolute delta (%)', min_value=1.0, max_value=99.0, value=25.0, step=1.0)
        try:
            k = strike_from_delta(spot, T, rd, rf, vol, d_pct / 100, opt_type)
        except ValueError as e:
            st.error(str(e))
            k = spot
        st.caption(f'{d_pct:.0f}Δ {opt_type.lower()} strike = **{k:,.{dp}f}**')
    market_prem = st.number_input('Observed market premium (MXN per 1 FX; 0 = none)', min_value=0.0,
                                  value=default_premium(pair), step=step_of(pair) / 10,
                                  format=f'%.{dp + 1}f', key=f'sb_prem_{pair}')
    qty = st.number_input(f'Proposed signed notional ({base}; + buy / − sell)', value=float(s['exposure']),
                          step=float(s['lot']), format='%.0f', key=f'sb_qty_{pair}')
    st.divider()
    usdmxn = st.number_input('USDMXN reference spot (for USD-equivalent reporting)', min_value=1e-6,
                             value=float(DEFAULT_USDMXN), step=0.05, format='%.4f', key='sb_usdmxn',
                             help='Placeholder: update with the live rate. Used only to show USD equivalents.')

kpi_box = st.container()

# ------------------------------------------------- manual trade entry panel
with st.expander('➕ Add a trade to the portfolio', expanded=True):
    c = st.columns([1.1, 1.6, 1.1, 1.2, 1.5, 1.2])
    t_date = c[0].date_input('Trade date', value=date.today(), key='add_date')
    t_pair = c[1].selectbox('Pair', PAIRS, key='add_pair', format_func=label_of)
    t_inst = c[2].selectbox('Instrument', list(INSTRUMENTS), key='add_inst')
    t_side = c[3].selectbox('Side', ['Buy (long)', 'Sell (short)'], key='add_side')
    t_ts = spec_for(t_pair)
    t_base, t_dp = t_ts['base'], t_ts['dp']
    t_notional = c[4].number_input(f'Notional ({t_base})', min_value=1.0, value=float(t_ts['exposure']),
                                   step=float(t_ts['lot']), format='%.0f', key=f'add_notional_{t_pair}')
    t_mat = c[5].text_input('Maturity label', placeholder='Jan-2027', key='add_mat')
    t_opt, t_der = t_inst in OPTION_TYPES, t_inst != 'Spot'
    d = st.columns(4)
    entry_label = {'Spot': 'Entry spot / budget rate', 'Forward': 'Contracted forward rate'}.get(
        t_inst, 'Entry premium (MXN per 1 FX)')
    t_entry = d[0].number_input(entry_label, min_value=1e-6,
                                value=float(default_premium(t_pair)) if t_opt else float(t_ts['spot']),
                                step=step_of(t_pair) / (10 if t_opt else 1), format=f'%.{t_dp + (1 if t_opt else 0)}f',
                                key=f'add_entry_{t_pair}_{t_inst}',
                                help=f'Quote: MXN per 1 {t_base}. Notional is in {t_base}.')
    t_mkt = d[1].number_input('Current spot', min_value=1e-6, value=float(t_ts['spot']), step=step_of(t_pair),
                              format=fmt_of(t_pair), key=f'add_mkt_{t_pair}')
    d[2].metric('Notional (MXN)', f'{t_notional * t_mkt:,.0f}')
    t_purpose = d[3].selectbox('Purpose', PURPOSES, key='add_purpose')
    t_strike, t_days, t_vol, t_rd, t_rf, t_mop = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if t_der:
        o1 = st.columns(3)
        t_days = float(o1[0].number_input('Days to maturity / expiry', min_value=1, value=90, step=1, key='add_days'))
        t_rd = o1[1].number_input('MXN rate (%)', value=DEFAULT_MXN_RATE * 100, step=0.25, key='add_rd') / 100
        t_rf = o1[2].number_input(f'{t_base} rate (%)', value=float(t_ts['foreign_rate'] * 100), step=0.25,
                                  key=f'add_rf_{t_pair}') / 100
    if t_opt:
        o2 = st.columns(3)
        t_strike = o2[0].number_input('Strike', min_value=1e-6, value=float(t_ts['spot']), step=step_of(t_pair),
                                      format=fmt_of(t_pair), key=f'add_k_{t_pair}')
        t_vol = o2[1].number_input('Implied vol (%)', min_value=0.01, value=float(t_ts['vol'] * 100), step=0.25,
                                   key=f'add_vol_{t_pair}') / 100
        t_mop = o2[2].number_input('Current market premium (0 = use entry)', min_value=0.0,
                                   value=float(default_premium(t_pair)), step=step_of(t_pair) / 10,
                                   format=f'%.{t_dp + 1}f', key=f'add_mop_{t_pair}_{t_inst}')
        try:
            mdl = gk_greeks(t_mkt, t_strike, t_days / 365, t_rd, t_rf, t_vol, t_inst)
            msg = f'Model premium {mdl["Theoretical premium"]:,.{t_dp + 1}f} • spot delta {mdl["Delta (spot)"]:.3f}'
            if t_mop > 0:
                msg += (' • implied vol at current premium '
                        f'{implied_volatility(t_mop, t_mkt, t_strike, t_days / 365, t_rd, t_rf, t_inst):.2%}')
            st.caption(msg)
        except Exception as e:  # noqa: BLE001
            st.caption(f'Model check unavailable: {e}')
    elif t_inst == 'Forward':
        mf = float(forward_rate(t_mkt, t_days / 365, t_rd, t_rf))
        st.caption(f'Market outright forward for these inputs: {mf:,.{t_dp}f} '
                   f'({(mf - t_mkt) / t_ts["pip"]:+,.0f} pips over spot).')
    t_notes = st.text_input('Notes', key='add_notes')
    if st.button('Add trade to ledger', type='primary', key='add_btn'):
        sign = 1.0 if t_side.startswith('Buy') else -1.0
        tid = add_trade({
            'trade_date': str(t_date), 'pair': t_pair, 'maturity': t_mat, 'instrument_type': t_inst,
            'signed_notional': sign * float(t_notional), 'trade_price': t_entry, 'spot': t_mkt,
            'strike': t_strike if t_opt else 0.0, 'days_to_expiry': t_days if t_der else 0.0,
            'mxn_rate': t_rd if t_der else 0.0, 'foreign_rate': t_rf if t_der else 0.0,
            'implied_volatility': t_vol if t_opt else 0.0,
            'market_option_premium': (t_mop if t_mop > 0 else t_entry) if t_opt else 0.0,
            'purpose': t_purpose, 'notes': t_notes})
        st.success(f'Added {tid}: {t_side.split()[0]} {t_notional:,.0f} {t_base} × {label_of(t_pair)} {t_inst}.')

pre, hedge_tab, ledger_tab, risk_tab = st.tabs(
    ['1 · Pre-trade analytics', '2 · Hedge sizing', '3 · Trade ledger & CSV', '4 · Portfolio risk & stress'])

# ----------------------------------------------------------- 1 pre-trade
with pre:
    st.subheader(f'{s["label"]}: {opt_type} option pre-trade check')
    st.caption(s['liquidity'])
    try:
        g = gk_greeks(spot, k, T, rd, rf, vol, opt_type)
        model = g['Theoretical premium']
        diff = market_prem - model
        F = float(forward_rate(spot, T, rd, rf))
        prem_used = market_prem if market_prem > 0 else model
        c = st.columns(4)
        c[0].metric('Model premium (MXN per 1 FX)', f'{model:,.{dp + 1}f}',
                    delta=f'{g["Premium (% foreign notional)"]:.3f}% of notional', delta_color='off')
        c[1].metric('Market − model', f'{diff:,.{dp + 1}f}',
                    delta=f'{(diff / model * 100 if model else 0):.2f}% vs model')
        c[2].metric(f'Delta-adjusted exposure ({base})', f'{qty * g["Delta (spot)"]:,.0f}')
        c[3].metric('Position vega / 1 vol pt (MXN)', f'{qty * g["Vega (per 1 vol pt)"]:,.0f}')
        c2 = st.columns(4)
        c2[0].metric('Outright forward', f'{F:,.{dp}f}', delta=f'{(F - spot) / s["pip"]:+,.0f} pips fwd points',
                     delta_color='off')
        be = k + prem_used if opt_type == 'Call' else k - prem_used
        c2[1].metric('Break-even at expiry (ex-financing)', f'{be:,.{dp}f}')
        c2[2].metric('Strike vs forward', f'{(k / F - 1):+.2%}')
        c2[3].metric('Premium paid' if qty >= 0 else 'Premium received', f'{abs(qty) * prem_used:,.0f} MXN',
                     delta=f'≈ USD {abs(qty) * prem_used / usdmxn:,.0f}', delta_color='off')
        st.dataframe(pd.DataFrame({'Metric': list(g.keys()), 'Value': list(g.values())}), hide_index=True)
        if market_prem > 0:
            try:
                iv = implied_volatility(market_prem, spot, k, T, rd, rf, opt_type)
                st.metric('Implied volatility from observed premium', f'{iv:.2%}',
                          delta=f'{(iv - vol) * 100:+.2f} vol points vs assumption')
                if iv > vol + .03:
                    st.warning('Market premium implies volatility materially above your assumption. Check event risk '
                               '(Banxico, Fed, US trade headlines), skew, expiry and quote quality before trading.')
                elif iv < vol - .03:
                    st.info('Market premium implies volatility materially below your assumption. Confirm the '
                            'bid/offer, liquidity and model inputs.')
                else:
                    st.success('Observed premium is broadly consistent with the entered volatility under '
                               'Garman-Kohlhagen. This is not, by itself, a trade recommendation.')
            except Exception as e:  # noqa: BLE001
                st.warning(str(e))
        grid = np.linspace(spot * 0.8, spot * 1.2, 81)
        intrinsic = np.maximum(grid - k, 0.0) if opt_type == 'Call' else np.maximum(k - grid, 0.0)
        st.line_chart(pd.DataFrame({'Model premium today': gk_price_vec(grid, k, T, rd, rf, vol, opt_type == 'Call'),
                                    'Intrinsic at expiry': intrinsic},
                                   index=pd.Index(grid.round(dp + 1), name=f'Spot (MXN per {base})')))
        st.caption('Premium comparison is meaningful only when market premium, spot, strike, expiry and rates refer to '
                   'the same deal terms. MXN trades with a pronounced skew (typically richer USD calls / MXN puts), '
                   'so a flat-vol comparison away from ATM is biased: use the vol quoted for that strike.')
        st.subheader('Proposed option position')
        st.write(f'Premium value = notional × premium = **{qty * prem_used:,.0f} MXN** '
                 f'(≈ USD {qty * prem_used / usdmxn:,.0f}); notional in MXN at spot = **{abs(qty) * spot:,.0f}**.')
        if st.button('Add this option to the portfolio ledger', key='pre_add'):
            if qty == 0:
                st.warning('Proposed notional is zero: nothing to add.')
            else:
                tid = add_trade({'pair': pair, 'instrument_type': opt_type, 'signed_notional': qty,
                                 'trade_price': prem_used, 'spot': spot, 'strike': k,
                                 'days_to_expiry': float(days), 'mxn_rate': rd, 'foreign_rate': rf,
                                 'implied_volatility': vol, 'market_option_premium': prem_used,
                                 'purpose': 'Hedge / optionality', 'notes': 'Added from pre-trade ticket'})
                st.success(f'Added {tid} at premium {prem_used:,.{dp + 1}f}.')
    except Exception as e:  # noqa: BLE001
        st.error(str(e))

# ------------------------------------------------------- 2 hedge sizing
with hedge_tab:
    st.subheader('Hedge sizing: FX exposure → forward / option / collar notional')
    st.caption('Convert a foreign-currency or MXN exposure into a hedge notional (rounded to dealing lots), with '
               'forward carry, a protective option, or a zero-cost collar.')
    h1, h2, h3 = st.columns(3)
    h_pair = h1.selectbox('Currency pair', PAIRS, key='h_pair', format_func=label_of)
    hs = spec_for(h_pair)
    hb, hdp = hs['base'], hs['dp']
    h_long = h1.radio('Underlying exposure', [PAYABLE, RECEIVABLE], key='h_dir').startswith('Payable')
    units = exposure_units(h_pair)
    h_qty = h2.number_input('Exposure amount', min_value=0.0, step=0.5, key=f'h_qty_{h_pair}',
                            value=float(hs['exposure'] / 1e6))
    h_unit = h2.selectbox('Measure', units, index=2, key=f'h_unit_{h_pair}')
    h_ratio = h3.slider('Hedge ratio (%)', 0, 150, 100, 5, key='h_ratio')
    h_beta = h3.number_input('Basis / proxy-hedge beta', min_value=0.01, value=1.0, step=0.05, key='h_beta',
                             help='Hedge notional per unit of exposure. 1.00 = hedge in the same pair. For a proxy '
                                  'hedge use the value-based minimum-variance beta (calculator below).')
    r4 = st.columns(4)
    h_spot = r4[0].number_input(f'Spot (MXN per 1 {hb})', min_value=1e-6, value=float(hs['spot']),
                                step=step_of(h_pair), format=fmt_of(h_pair), key=f'h_spot_{h_pair}')
    h_lot = r4[1].number_input(f'Rounding lot ({hb})', min_value=1.0, value=float(hs['lot']), step=float(hs['lot']),
                               format='%.0f', key=f'h_lot_{h_pair}', help='Dealing-size increment of the hedge.')
    h_days = r4[2].number_input('Days to maturity of exposure', min_value=1, value=90, step=1, key='h_days')
    h_round = r4[3].selectbox('Rounding', list(ROUNDING_MODES), key='h_round')
    r5 = st.columns(4)
    h_rd = r5[0].number_input('MXN rate (%)', value=DEFAULT_MXN_RATE * 100, step=0.25, key='h_rd') / 100
    h_rf = r5[1].number_input(f'{hb} rate (%)', value=float(hs['foreign_rate'] * 100), step=0.25,
                              key=f'h_rf_{h_pair}') / 100
    h_inst = r5[2].selectbox('Hedge instrument', ['Forward', 'Protective option (long)', 'Zero-cost collar'],
                             key='h_inst')
    h_opt, h_collar = h_inst.startswith('Protective'), h_inst.startswith('Zero-cost')
    h_otype = 'Call' if h_long else 'Put'
    h_basis = r5[3].selectbox('Option sizing basis', ['Notional (1:1: locks worst-case rate)', 'Delta-equivalent'],
                              key='h_basis', disabled=not h_opt)
    h_delta, h_prem, h_k, h_vol = None, None, h_spot, hs['vol']
    ks, sold_prem, sold_vol, sold_g, hg = None, None, None, None, None
    default_k = round(h_spot * (1.03 if h_long else 0.97), hdp)
    if h_opt or h_collar:
        oc = st.columns(3 if h_collar else 2)
        h_k = oc[0].number_input(f'{"Cap (bought call)" if h_long else "Floor (bought put)"} strike' if h_collar
                                 else 'Strike', min_value=1e-6, value=float(default_k), step=step_of(h_pair),
                                 format=fmt_of(h_pair), key=f'h_k_{h_pair}_{h_otype}')
        h_vol = oc[1].number_input('Volatility (%)', min_value=0.01, value=float(hs['vol'] * 100), step=0.25,
                                   key=f'h_vol_{h_pair}') / 100
        skew = 0.0
        if h_collar:
            skew = oc[2].number_input('Skew adj. on sold leg (vol pts)', value=0.0, step=0.25, key='h_skew',
                                      help='Added to the vol used to price the sold option. E.g. +1.0 if the sold '
                                           '(OTM) option trades richer than the bought one.') / 100
        try:
            hg = gk_greeks(h_spot, h_k, h_days / 365, h_rd, h_rf, h_vol, h_otype)
            h_delta, h_prem = hg['Delta (spot)'], hg['Theoretical premium']
            if h_collar:
                sold_vol = h_vol + skew
                ks = zero_cost_collar_strike(h_spot, h_k, h_days / 365, h_rd, h_rf, h_vol, h_otype, sold_vol)
                sold_type = 'Put' if h_long else 'Call'
                sold_g = gk_greeks(h_spot, ks, h_days / 365, h_rd, h_rf, sold_vol, sold_type)
                sold_prem = sold_g['Theoretical premium']
        except ValueError as e:
            st.error(str(e))
    st.caption(f'{hs["liquidity"]} • rounding lot {h_lot:,.0f} {hb} • quoted MXN per 1 {hb}. '
               'Verify tenor, lot sizes and dealer quoting conventions.')
    try:
        use_delta = h_opt and h_basis.startswith('Delta') and h_delta is not None
        res = size_hedge(h_pair, h_qty, h_unit, h_ratio / 100, h_beta, h_round, h_lot, h_spot, h_days, h_rd, h_rf,
                         h_delta if use_delta else None)
        opt_notional = res['option_notional_fx'] if use_delta else res['hedge_notional_fx']
        F_h = res['forward_rate']
        direction = 'BUY' if h_long else 'SELL'
        if h_opt and h_delta is not None:
            headline = (f'**Buy {hb} {opt_notional:,.0f} {h_otype.upper()}** (strike {h_k:,.{hdp}f}, spot delta '
                        f'{h_delta:+.3f}, {h_days} days) to cover {res["exposure_fx"] * h_ratio / 100:,.0f} {hb} of exposure'
                        + (f' (delta-equivalent: {res["lots_exact"]:,.2f} lots of forward).' if use_delta else '.'))
        elif h_collar and ks is not None:
            bought_t, sold_t = ('call', 'put') if h_long else ('put', 'call')
            headline = (f'**Zero-cost collar on {hb} {res["hedge_notional_fx"]:,.0f}**: buy {bought_t} @ {h_k:,.{hdp}f}, '
                        f'sell {sold_t} @ {ks:,.{hdp}f}, {h_days} days, net premium ≈ 0.')
        else:
            headline = (f'**{direction} {hb} {res["hedge_notional_fx"]:,.0f} forward** at {F_h:,.{hdp}f} '
                        f'({res["lots"]:,} × {h_lot:,.0f} lots; {res["lots_exact"]:,.2f} exact before rounding).')
        st.success(headline)
        m = st.columns(4)
        m[0].metric(f'Exposure ({hb})', f'{res["exposure_fx"]:,.0f}')
        m[1].metric(f'Hedge volume required ({hb})', f'{res["target_fx"]:,.0f}')
        m[2].metric(f'Hedge notional ({hb})', f'{res["hedge_notional_fx"]:,.0f}',
                    delta=f'{res["lots"]:,} lots / {res["lots_exact"]:,.2f} exact', delta_color='off')
        m[3].metric(f'Unhedged residual ({hb})', f'{res["residual_fx"]:,.0f}',
                    delta=f'{res["hedged_pct"]:.1%} hedged', delta_color='off')
        v = st.columns(4)
        v[0].metric('Hedge notional (MXN)', f'{res["hedge_notional_mxn"]:,.0f}')
        v[1].metric('Hedge P&L per 1% spot move (MXN)', f'{res["pnl_per_1pct_mxn"]:,.0f}')
        v[2].metric('Outright forward', f'{F_h:,.{hdp}f}',
                    delta=f'{res["forward_points"] / hs["pip"]:+,.0f} pips • {res["annualized_carry"]:.2%} p.a.',
                    delta_color='off')
        carry_cost = res['carry_mxn'] if h_long else -res['carry_mxn']
        if h_opt and h_prem is not None:
            outlay = opt_notional * h_prem
            v[3].metric('Premium outlay (MXN)', f'{outlay:,.0f}',
                        delta=f'{h_prem / h_spot:.3%} of spot notional', delta_color='off')
        elif h_collar and sold_prem is not None:
            v[3].metric('Net premium (MXN)', f'{res["hedge_notional_fx"] * (h_prem - sold_prem):,.2f}',
                        delta=f'range {min(h_k, ks):,.{hdp}f} – {max(h_k, ks):,.{hdp}f}', delta_color='off')
        else:
            v[3].metric('Forward carry cost / (benefit) vs spot (MXN)', f'{carry_cost:,.0f}',
                        delta='cost' if carry_cost > 0 else 'benefit', delta_color='off')
        if h_collar and ks is not None:
            st.info(('Payable: your effective buying rate is capped at the bought call strike '
                     f'**{h_k:,.{hdp}f}** and you give up improvement below the sold put strike **{ks:,.{hdp}f}**.')
                    if h_long else
                    ('Receivable: your effective selling rate is floored at the bought put strike '
                     f'**{h_k:,.{hdp}f}** and you give up upside above the sold call strike **{ks:,.{hdp}f}**.'))
            leg_n = res['hedge_notional_fx']
            st.dataframe(pd.DataFrame([
                {'Leg': 'Buy ' + h_otype, 'Strike': h_k, 'Vol %': h_vol * 100, 'Premium (MXN/1 FX)': h_prem,
                 'Spot delta': h_delta, 'Premium (MXN)': leg_n * h_prem},
                {'Leg': 'Sell ' + ('Put' if h_long else 'Call'), 'Strike': ks, 'Vol %': sold_vol * 100,
                 'Premium (MXN/1 FX)': sold_prem, 'Spot delta': sold_g['Delta (spot)'],
                 'Premium (MXN)': -leg_n * sold_prem}]), hide_index=True)
        elif h_opt and hg is not None and h_prem is not None:
            be = h_k + h_prem if h_long else h_k - h_prem
            st.info(f'Worst-case all-in rate at expiry (strike ± premium, ex-financing): **{be:,.{hdp}f}**. '
                    'With 1:1 notional sizing the option caps/floors the whole exposure; delta-equivalent sizing '
                    'only matches today\'s delta and leaves a gap if spot moves through the strike.')
        e1, e2 = st.columns(2)
        with e1:
            st.markdown('**Exposure in other measures**')
            fx_amt = res['exposure_fx']
            other = {hb: fx_amt, f'{hb} thousands': fx_amt / 1e3, f'{hb} millions': fx_amt / 1e6,
                     'MXN at spot': fx_amt * h_spot, 'MXN at forward': fx_amt * F_h}
            st.dataframe(pd.DataFrame({'Measure': list(other), 'Quantity': list(other.values())}), hide_index=True)
        with e2:
            st.markdown('**Hedge-ratio ladder**')
            rows = []
            for pct in sorted({25, 50, 75, 100, 125, h_ratio}):
                r_ = size_hedge(h_pair, h_qty, h_unit, pct / 100, h_beta, h_round, h_lot, h_spot, h_days, h_rd, h_rf)
                rows.append({'Hedge ratio %': pct, 'Lots': r_['lots'], 'Exact': round(r_['lots_exact'], 2),
                             f'Notional ({hb})': round(r_['hedge_notional_fx']),
                             f'Residual ({hb})': round(r_['residual_fx'])})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        with st.expander('Forward curve & carry by tenor'):
            tenors = np.array([30, 60, 90, 180, 270, 365, 540, 730])
            Fs = forward_rate(h_spot, tenors / 365, h_rd, h_rf)
            sign = 1.0 if h_long else -1.0
            st.dataframe(pd.DataFrame({
                'Days': tenors, 'Forward': Fs.round(hdp + 1),
                'Fwd points (pips)': ((Fs - h_spot) / hs['pip']).round(0),
                'Carry p.a. (%)': (np.log(Fs / h_spot) / (tenors / 365) * 100).round(3),
                'Carry cost / (benefit) for this hedge (MXN)': (sign * res['hedge_notional_fx'] * (Fs - h_spot)).round(0)}),
                hide_index=True)
            st.caption('Positive cost = you buy the foreign currency forward above spot (MXN rates > foreign rates '
                       'make a payable hedge expensive and a receivable hedge accretive). Options do not carry this '
                       'forward-point cost but you pay premium up front.')
        with st.expander('Proxy-hedge beta calculator (minimum variance)'):
            pb = st.columns(3)
            vol_e = pb[0].number_input('Exposure pair vol (%)', min_value=0.01, value=14.0, step=0.5, key='pb_ve')
            vol_h = pb[1].number_input('Hedge pair vol (%)', min_value=0.01, value=12.0, step=0.5, key='pb_vh')
            rho = pb[2].number_input('Correlation of MXN returns', min_value=-1.0, max_value=1.0, value=0.9,
                                     step=0.05, key='pb_rho')
            st.write(f'Suggested value-based beta = ρ·σ_exposure/σ_hedge = **{min_variance_beta(vol_e, vol_h, rho):.3f}**. '
                     'Estimate vols and correlation from your own history; enter the result in the beta field above.')
        st.markdown('**Book this hedge**')
        b = st.columns([1.2, 1.2, 1])
        b_month = b[0].text_input('Maturity label', placeholder='Jan-2027', key='h_month')
        if h_collar:
            b[1].metric('Entry premiums', 'model prices (both legs)')
        else:
            b_entry = b[1].number_input('Entry premium (MXN per 1 FX)' if h_opt else 'Contracted forward rate',
                                        min_value=1e-6,
                                        value=float(h_prem) if (h_opt and h_prem) else float(F_h),
                                        step=step_of(h_pair) / (10 if h_opt else 1),
                                        format=f'%.{hdp + (1 if h_opt else 0)}f', key=f'h_entry_{h_pair}_{h_inst}')
        if b[2].button('Add hedge to ledger', type='primary', key='h_add'):
            n_book = opt_notional if (h_opt or h_collar) else res['hedge_notional_fx']
            if n_book == 0:
                st.warning('Hedge notional is zero: nothing to add.')
            else:
                row = {'pair': h_pair, 'maturity': b_month, 'spot': h_spot, 'days_to_expiry': float(h_days),
                       'mxn_rate': h_rd, 'foreign_rate': h_rf, 'purpose': 'Hedge',
                       'notes': f'Hedge sizing: {h_qty:,.2f} {h_unit} at {h_ratio}%'}
                if h_collar and ks is not None:
                    long_leg = {**row, 'instrument_type': h_otype, 'signed_notional': float(n_book), 'strike': h_k,
                                'implied_volatility': h_vol, 'trade_price': h_prem, 'market_option_premium': h_prem,
                                'notes': row['notes'] + ' | collar bought leg'}
                    short_leg = {**row, 'instrument_type': 'Put' if h_long else 'Call', 'signed_notional': -float(n_book),
                                 'strike': ks, 'implied_volatility': sold_vol, 'trade_price': sold_prem,
                                 'market_option_premium': sold_prem, 'notes': row['notes'] + ' | collar sold leg'}
                    ids = [add_trade(long_leg), add_trade(short_leg)]
                    st.success(f'Added collar legs {ids[0]} and {ids[1]} to the ledger.')
                elif h_opt and h_delta is not None:
                    row.update(instrument_type=h_otype, signed_notional=float(n_book), strike=h_k,
                               implied_volatility=h_vol, trade_price=b_entry, market_option_premium=b_entry)
                    st.success(f'Added {add_trade(row)} to the ledger.')
                else:
                    row.update(instrument_type='Forward', signed_notional=float(n_book) * (1 if h_long else -1),
                               trade_price=b_entry)
                    st.success(f'Added {add_trade(row)} to the ledger.')
    except ValueError as e:
        st.error(str(e))

# ------------------------------------------------------------ 3 ledger
with ledger_tab:
    st.subheader('Trade-by-trade portfolio ledger')
    st.write('One row per trade; `signed_notional` is in foreign-currency units (+ long foreign ccy / − short). '
             'Spot rows: `trade_price` = entry spot or budget rate. Forward rows: `trade_price` = contracted forward '
             'rate. Option rows: `trade_price` = premium paid/received per 1 unit of foreign ccy (MXN) and '
             '`market_option_premium` = current observed premium. All quotes are MXN per 1 foreign unit; rates and '
             'vols are decimals (0.07 = 7%).')
    up = st.file_uploader('Import existing portfolio CSV (optional)', type=['csv'], key='ledger_upload')
    if up is not None:
        imp_mode = st.radio('Import mode', ['Replace ledger', 'Append to ledger'], horizontal=True)
        if st.button('Import file'):
            try:
                imp = typed(ensure_columns(pd.read_csv(up)))
                imp['instrument_type'] = imp['instrument_type'].str.strip().str.title()
                imp['pair'] = imp['pair'].map(normalize_pair)
                base_df = st.session_state.ledger if imp_mode.startswith('Append') else sample_ledger().iloc[0:0]
                st.session_state.ledger = typed(pd.concat([base_df, imp], ignore_index=True))
                st.success(f'Imported {len(imp)} rows.')
            except Exception as e:  # noqa: BLE001
                st.error(f'CSV import error: {e}')
    extra = sorted(set(st.session_state.ledger['pair'].astype(str)) - set(PAIRS) - {''})
    cfg = {'pair': st.column_config.SelectboxColumn('pair', options=PAIRS + extra),
           'instrument_type': st.column_config.SelectboxColumn('instrument_type', options=list(INSTRUMENTS))}
    edited = st.data_editor(st.session_state.ledger, num_rows='dynamic', column_config=cfg)
    st.session_state.ledger = edited
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.download_button('Export full portfolio CSV', edited.to_csv(index=False).encode('utf-8'),
                           'fx_trade_ledger.csv', 'text/csv', type='primary')
    with b2:
        st.download_button('Download blank CSV template', pd.DataFrame(columns=COLUMNS).to_csv(index=False).encode('utf-8'),
                           'fx_ledger_template.csv', 'text/csv')
    with b3:
        if st.button('Validate ledger'):
            try:
                found = ledger_issues(clean_ledger(edited))
                if found:
                    st.error('\n\n'.join(found))
                else:
                    st.success(f'Basic validation passed for {len(clean_ledger(edited))} rows. This checks fields, '
                               'not economic suitability, ISDA terms, credit lines or hedge-accounting eligibility.')
            except ValueError as e:
                st.error(str(e))
    with b4:
        if st.button('Clear all rows'):
            st.session_state.ledger = sample_ledger().iloc[0:0]
            st.rerun()

# --------------------------------------------- shared ledger computations
try:
    clean = clean_ledger(st.session_state.ledger)
    issues = ledger_issues(clean)
    risk = ledger_risk(clean) if not issues else pd.DataFrame()
except ValueError as e:
    clean, issues, risk = pd.DataFrame(), [str(e)], pd.DataFrame()

with kpi_box:
    if issues:
        st.warning(f'{len(issues)} ledger issue(s) block portfolio metrics: see the Ledger tab.')
    elif clean.empty:
        st.info('Portfolio is empty: add a trade below.')
    else:
        tot = {'pnl': risk.current_pnl_mxn.sum(), 'delta': risk.delta_mxn.sum(), 'gross': risk.gross_notional_mxn.sum(),
               'vega': risk.vega_per_vol_pt_mxn.sum(), 'theta': risk.theta_daily_mxn.sum()}
        usd = lambda x: f'≈ USD {x / usdmxn:,.0f}'
        k5 = st.columns(5)
        k5[0].metric('Portfolio P&L (MXN)', f'{tot["pnl"]:,.0f}', delta=usd(tot['pnl']), delta_color='off')
        k5[1].metric('Net delta notional (MXN)', f'{tot["delta"]:,.0f}', delta=usd(tot['delta']), delta_color='off')
        k5[2].metric('Gross notional (MXN)', f'{tot["gross"]:,.0f}', delta=usd(tot['gross']), delta_color='off')
        k5[3].metric('Net vega / vol pt (MXN)', f'{tot["vega"]:,.0f}', delta=usd(tot['vega']), delta_color='off')
        k5[4].metric('Net theta / day (MXN)', f'{tot["theta"]:,.0f}', delta=usd(tot['theta']), delta_color='off')


@st.cache_data(show_spinner=False)
def run_stress(df, price_shocks, vol_shocks, g10_shock):
    return stress_grid(df, price_shocks, vol_shocks, g10_shock)


# ------------------------------------------------------- 4 risk & stress
with risk_tab:
    st.subheader('Portfolio Greeks, P&L and scenario stress')
    st.caption('P&L and Greeks are in MXN (every pair is quoted vs MXN). Options are revalued with Garman-Kohlhagen and '
               'marked to the observed premium; forwards are discounted at the MXN rate; spot rows are linear. '
               'Net delta is shown in foreign-currency units per pair because deltas in different currencies cannot be '
               'summed.')
    if issues:
        st.error('Fix these ledger issues first:\n\n' + '\n'.join(f'- {i}' for i in issues[:25]))
    elif clean.empty:
        st.info('Nothing to analyse yet.')
    else:
        st.markdown('**Exposure by pair** (net delta in foreign-currency units; positive = long foreign ccy / short MXN)')
        st.dataframe(pair_summary(risk), hide_index=True)
        st.markdown('**Trade-level risk** (MXN)')
        st.dataframe(risk, hide_index=True)
        st.download_button('Export portfolio risk CSV', risk.to_csv(index=False).encode('utf-8'),
                           'fx_portfolio_risk.csv', 'text/csv')
        st.subheader('Full-revaluation stress scenarios')
        st.write('Spot shock = % move in every XXXMXN pair (**+ = MXN depreciates** against the G10 block, the '
                 'risk-off direction). The optional G10-vs-USD move is layered on top for non-USD pairs, so EURMXN '
                 'moves by (1+MXN shock)×(1+G10 move)−1 while USDMXN moves only by the MXN shock. Vol shocks are '
                 'parallel; rates are held fixed. First-pass grid: not VaR, no skew dynamics, no basis.')
        px = st.multiselect('Spot shock (%, + = MXN weaker)', [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25],
                            default=[-15, -10, -5, 0, 5, 10, 15])
        vx = st.multiselect('Volatility shock (absolute vol points)', [-15, -10, -5, 0, 5, 10, 15, 25],
                            default=[-10, 0, 10, 15])
        g10 = st.slider('Additional G10 ex-USD vs USD move (%, applied to non-USD pairs)', -15.0, 15.0, 0.0, 0.5)
        if not px or not vx:
            st.info('Select at least one spot shock and one volatility shock.')
        else:
            sd = run_stress(clean, tuple(px), tuple(vx), g10)
            worst = sd.loc[sd.change_vs_current_pnl.idxmin()]
            st.metric('Worst scenario in selected grid (MXN)', f'{worst.change_vs_current_pnl:,.0f}',
                      delta=f"{worst['spot_shock_%']:+.0f}% spot / {worst['volatility_shock_pts']:+.0f} vol pts "
                            f"(≈ USD {worst.change_vs_current_pnl / usdmxn:,.0f})", delta_color='off')
            st.markdown('**Change vs current P&L (MXN): spot shock (rows) × vol shock (columns)**')
            st.dataframe(sd.pivot(index='spot_shock_%', columns='volatility_shock_pts', values='change_vs_current_pnl'))
            st.dataframe(sd.sort_values('change_vs_current_pnl'), hide_index=True)
            st.download_button('Export stress scenarios CSV', sd.to_csv(index=False).encode('utf-8'),
                               'fx_stress_scenarios.csv', 'text/csv')

st.divider()
st.caption('Suggested next enhancements: dealer vol-surface input (ATM / 25Δ RR / 25Δ BF) with strike-dependent vol, '
           'premium-adjusted delta conventions, cash-flow ladder against budget rates, historical VaR / Expected '
           'Shortfall with a MXN-G10 covariance matrix, hedge-accounting effectiveness testing, counterparty '
           'credit-line / CVA and margin usage, barrier and seagull structures, and a live market-data feed.')
