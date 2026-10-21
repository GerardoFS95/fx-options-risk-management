"""Garman-Kohlhagen pricing, vectorised portfolio risk and hedge-sizing helpers
for a corporate MXN-vs-G10 FX options book.

Conventions
-----------
* Every pair is quoted XXXMXN = MXN per 1 unit of the G10 (foreign) currency.
  Foreign currency = base (e.g. USD in USDMXN), domestic currency = MXN, so all
  premiums, strikes and P&L are in MXN and the corporate's functional currency
  needs no further conversion. (JPY is quoted as MXN per 1 JPY.)
* Pricing is Garman-Kohlhagen (Black-Scholes-Merton with a continuous foreign
  dividend yield r_f). Rates are continuously-compounded decimals, T = days/365.
* Delta is *spot delta, premium excluded* (e^{-r_f T} N(d1) for a call), expressed
  per 1 unit of foreign currency.
* ``signed_notional`` is in foreign-currency units: positive = long foreign
  currency (buy FX / long call / long put-on-MXN), negative = short.

Every pricing helper is numpy-vectorised, so a whole ledger, or a whole scenario
grid, is revalued in a single call.
"""
import math

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri

# --------------------------------------------------------------------------
# Ledger schema
# --------------------------------------------------------------------------
INSTRUMENTS = ('Spot', 'Forward', 'Call', 'Put')
OPTION_TYPES = ('Call', 'Put')
COLUMNS = ['trade_id', 'trade_date', 'pair', 'maturity', 'instrument_type',
           'signed_notional', 'trade_price', 'spot', 'strike', 'days_to_expiry',
           'mxn_rate', 'foreign_rate', 'implied_volatility',
           'market_option_premium', 'purpose', 'notes']
REQUIRED = ['pair', 'instrument_type', 'signed_notional', 'trade_price', 'spot',
            'strike', 'days_to_expiry', 'mxn_rate', 'foreign_rate',
            'implied_volatility', 'market_option_premium']
NUMERIC = REQUIRED[2:]
TEXT_COLS = ['trade_id', 'trade_date', 'pair', 'maturity', 'instrument_type',
             'purpose', 'notes']
OPTION_ONLY = ['strike', 'implied_volatility', 'market_option_premium']
TENOR_FIELDS = ['days_to_expiry', 'mxn_rate', 'foreign_rate']

# --------------------------------------------------------------------------
# Pair specifications: ILLUSTRATIVE placeholders, overwrite with live quotes.
# spot = MXN per 1 unit of base; lot = default rounding lot (foreign units);
# vol / foreign_rate = placeholder ATM vol and money-market rate (decimals).
# --------------------------------------------------------------------------
SPECS = {
    'USDMXN': {'base': 'USD', 'label': 'USD/MXN', 'spot': 18.40, 'vol': 0.120, 'foreign_rate': 0.0400,
               'dp': 4, 'pip': 1e-4, 'lot': 100_000.0, 'exposure': 1_000_000.0,
               'liquidity': 'Core: deepest OTC and listed (CME 6M) market'},
    'EURMXN': {'base': 'EUR', 'label': 'EUR/MXN', 'spot': 21.40, 'vol': 0.125, 'foreign_rate': 0.0200,
               'dp': 4, 'pip': 1e-4, 'lot': 100_000.0, 'exposure': 1_000_000.0,
               'liquidity': 'Liquid cross'},
    'GBPMXN': {'base': 'GBP', 'label': 'GBP/MXN', 'spot': 24.60, 'vol': 0.135, 'foreign_rate': 0.0375,
               'dp': 4, 'pip': 1e-4, 'lot': 100_000.0, 'exposure': 1_000_000.0,
               'liquidity': 'Liquid cross'},
    'JPYMXN': {'base': 'JPY', 'label': 'JPY/MXN (MXN per 1 JPY)', 'spot': 0.1230, 'vol': 0.160, 'foreign_rate': 0.0075,
               'dp': 5, 'pip': 1e-5, 'lot': 10_000_000.0, 'exposure': 100_000_000.0,
               'liquidity': 'Cross: quoted by dealers as MXN/JPY; wider spreads'},
    'CHFMXN': {'base': 'CHF', 'label': 'CHF/MXN', 'spot': 23.00, 'vol': 0.140, 'foreign_rate': 0.0025,
               'dp': 4, 'pip': 1e-4, 'lot': 100_000.0, 'exposure': 1_000_000.0,
               'liquidity': 'Thin cross: expect wide bid/offer'},
    'CADMXN': {'base': 'CAD', 'label': 'CAD/MXN', 'spot': 13.30, 'vol': 0.120, 'foreign_rate': 0.0250,
               'dp': 4, 'pip': 1e-4, 'lot': 100_000.0, 'exposure': 1_000_000.0,
               'liquidity': 'Liquid cross (USMCA flows)'},
    'AUDMXN': {'base': 'AUD', 'label': 'AUD/MXN', 'spot': 12.10, 'vol': 0.135, 'foreign_rate': 0.0360,
               'dp': 4, 'pip': 1e-4, 'lot': 100_000.0, 'exposure': 1_000_000.0,
               'liquidity': 'Thin cross: expect wide bid/offer'},
    'NZDMXN': {'base': 'NZD', 'label': 'NZD/MXN', 'spot': 10.60, 'vol': 0.145, 'foreign_rate': 0.0300,
               'dp': 4, 'pip': 1e-4, 'lot': 100_000.0, 'exposure': 1_000_000.0,
               'liquidity': 'Very thin cross: consider a synthetic via USDMXN x NZDUSD'},
    'NOKMXN': {'base': 'NOK', 'label': 'NOK/MXN', 'spot': 1.82, 'vol': 0.150, 'foreign_rate': 0.0400,
               'dp': 4, 'pip': 1e-4, 'lot': 1_000_000.0, 'exposure': 10_000_000.0,
               'liquidity': 'Very thin cross: consider a synthetic via USDMXN x USDNOK'},
    'SEKMXN': {'base': 'SEK', 'label': 'SEK/MXN', 'spot': 1.95, 'vol': 0.150, 'foreign_rate': 0.0200,
               'dp': 4, 'pip': 1e-4, 'lot': 1_000_000.0, 'exposure': 10_000_000.0,
               'liquidity': 'Very thin cross: consider a synthetic via USDMXN x USDSEK'},
    'Other': {'base': 'FCY', 'label': 'Other (user-defined, vs MXN)', 'spot': 1.0, 'vol': 0.150,
              'foreign_rate': 0.0300, 'dp': 4, 'pip': 1e-4, 'lot': 100_000.0, 'exposure': 1_000_000.0,
              'liquidity': 'User defined'},
}
PAIRS = list(SPECS)
DEFAULT_MXN_RATE = 0.07     # placeholder MXN money-market rate
DEFAULT_USDMXN = SPECS['USDMXN']['spot']
ROUNDING_MODES = ('Nearest', 'Round down (under-hedge)', 'Round up (over-hedge)')


def normalize_pair(pair):
    """'usd/mxn' -> 'USDMXN'; unknown names are passed through upper-cased."""
    p = str(pair).strip().replace('/', '').replace(' ', '')
    for k in SPECS:
        if p.upper() == k.upper():
            return k
    return p.upper()


def spec_for(pair):
    return SPECS.get(normalize_pair(pair), SPECS['Other'])


def base_ccy(pair):
    p = normalize_pair(pair)
    return SPECS[p]['base'] if p in SPECS else (p[:3] if len(p) >= 6 else 'FCY')


def is_usd_pair(pair):
    return normalize_pair(pair).startswith('USD')


# --------------------------------------------------------------------------
# Garman-Kohlhagen core (vectorised)
# --------------------------------------------------------------------------
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _is_call(option_type):
    return str(option_type).strip().lower().startswith('c')


def _prep(S, K, T, rd, rf, sigma, is_call):
    S, K, T, rd, rf, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, rd, rf, sigma))
    if not all(np.all(np.isfinite(a)) for a in (S, K, T, rd, rf, sigma)):
        raise ValueError('Spot, strike, expiry, rates and volatility must be finite.')
    if np.any(S <= 0) or np.any(K <= 0) or np.any(T <= 0) or np.any(sigma <= 0):
        raise ValueError('Spot, strike, time to expiry and volatility must be greater than zero.')
    return S, K, T, rd, rf, sigma, np.asarray(is_call, dtype=bool)


def _d1_d2(S, K, T, rd, rf, sigma):
    sqT = np.sqrt(T)
    d1 = (np.log(S / K) + (rd - rf + 0.5 * sigma * sigma) * T) / (sigma * sqT)
    return d1, d1 - sigma * sqT, sqT


def forward_rate(S, T, rd, rf):
    """Outright forward (covered interest parity), MXN per 1 foreign unit."""
    return np.asarray(S, float) * np.exp((np.asarray(rd, float) - np.asarray(rf, float)) * np.asarray(T, float))


def gk_price_vec(S, K, T, rd, rf, sigma, is_call):
    S, K, T, rd, rf, sigma, is_call = _prep(S, K, T, rd, rf, sigma, is_call)
    d1, d2, _ = _d1_d2(S, K, T, rd, rf, sigma)
    ed, ef = np.exp(-rd * T), np.exp(-rf * T)
    call = S * ef * ndtr(d1) - K * ed * ndtr(d2)
    put = K * ed * ndtr(-d2) - S * ef * ndtr(-d1)
    return np.where(is_call, call, put)


def gk_greeks_vec(S, K, T, rd, rf, sigma, is_call):
    """Raw Greeks per 1 unit of foreign notional (values in MXN):
    delta = spot delta (premium excluded), delta_fwd = N(d1)-style forward delta,
    gamma per 1 MXN of spot, vega per 1.00 vol, theta per year (dV/dt),
    rho_d / rho_f per 1.00 of MXN / foreign rate."""
    S, K, T, rd, rf, sigma, is_call = _prep(S, K, T, rd, rf, sigma, is_call)
    d1, d2, sqT = _d1_d2(S, K, T, rd, rf, sigma)
    ed, ef = np.exp(-rd * T), np.exp(-rf * T)
    pdf = np.exp(-0.5 * d1 * d1) / _SQRT_2PI
    nd1, nd2, nmd1, nmd2 = ndtr(d1), ndtr(d2), ndtr(-d1), ndtr(-d2)
    price = np.where(is_call, S * ef * nd1 - K * ed * nd2, K * ed * nmd2 - S * ef * nmd1)
    common = -S * ef * pdf * sigma / (2.0 * sqT)
    theta = np.where(is_call,
                     common - rd * K * ed * nd2 + rf * S * ef * nd1,
                     common + rd * K * ed * nmd2 - rf * S * ef * nmd1)
    return {'price': price,
            'delta': np.where(is_call, ef * nd1, -ef * nmd1),
            'delta_fwd': np.where(is_call, nd1, -nmd1),
            'gamma': ef * pdf / (S * sigma * sqT),
            'vega': S * ef * pdf * sqT,
            'theta': theta,
            'rho_d': np.where(is_call, K * T * ed * nd2, -K * T * ed * nmd2),
            'rho_f': np.where(is_call, -S * T * ef * nd1, S * T * ef * nmd1)}


# --------------------------------------------------------------------------
# Scalar wrappers
# --------------------------------------------------------------------------
def gk_price(S, K, T, rd, rf, sigma, option_type='Call'):
    return float(gk_price_vec(S, K, T, rd, rf, sigma, _is_call(option_type)))


def gk_greeks(S, K, T, rd, rf, sigma, option_type='Call'):
    g = {k: float(v) for k, v in gk_greeks_vec(S, K, T, rd, rf, sigma, _is_call(option_type)).items()}
    return {'Theoretical premium': g['price'],
            'Premium (% foreign notional)': g['price'] / float(S) * 100.0,
            'Delta (spot)': g['delta'], 'Delta (forward)': g['delta_fwd'],
            'Gamma (per 1 MXN)': g['gamma'],
            'Gamma (delta chg per 1% spot)': g['gamma'] * float(S) / 100.0,
            'Vega (per 1 vol pt)': g['vega'] / 100.0, 'Vega (per 1.00 vol)': g['vega'],
            'Theta (daily)': g['theta'] / 365.0,
            'Rho MXN (per 1bp)': g['rho_d'] * 1e-4, 'Rho foreign (per 1bp)': g['rho_f'] * 1e-4}


def implied_volatility(premium, S, K, T, rd, rf, option_type='Call'):
    if premium <= 0:
        raise ValueError('Market premium must be positive.')
    call = _is_call(option_type)
    f = lambda v: float(gk_price_vec(S, K, T, rd, rf, v, call)) - premium
    lo, hi = 1e-5, 5.0
    if f(lo) * f(hi) > 0:
        raise ValueError('Cannot solve implied volatility: premium may be outside model bounds or inputs may be inconsistent.')
    return brentq(f, lo, hi, maxiter=500)


def strike_from_delta(S, T, rd, rf, sigma, delta, option_type='Call'):
    """Strike giving a target *absolute* spot delta (premium excluded), e.g. 0.25."""
    if not 0 < delta < 1:
        raise ValueError('Delta must be between 0 and 1 (absolute value).')
    x = delta * math.exp(rf * T)
    if x >= 1:
        raise ValueError('Target delta is not attainable for these rates and tenor.')
    d1 = ndtri(x) if _is_call(option_type) else -ndtri(x)
    return float(S * math.exp(-d1 * sigma * math.sqrt(T) + (rd - rf + 0.5 * sigma * sigma) * T))


def zero_cost_collar_strike(S, K_bought, T, rd, rf, sigma, bought_type='Call', sold_vol=None):
    """Strike of the sold option (opposite type) that makes the collar zero-premium.
    Payable (need to buy FX): buy call K_bought, sell put -> returns the put strike.
    Receivable (will sell FX): buy put K_bought, sell call -> returns the call strike.
    ``sold_vol`` lets you price the sold leg off a different vol (skew)."""
    sold_type = 'Put' if _is_call(bought_type) else 'Call'
    sv = sigma if sold_vol is None else sold_vol
    target = gk_price(S, K_bought, T, rd, rf, sigma, bought_type)
    f = lambda k: gk_price(S, k, T, rd, rf, sv, sold_type) - target
    lo, hi = S * 0.05, S * 20.0
    if f(lo) * f(hi) > 0:
        raise ValueError('No zero-cost strike exists for these inputs.')
    return brentq(f, lo, hi, xtol=1e-12, maxiter=500)


def _mark_pnl(typ, q, S, entry, T, rd, rf, K=None, vol=None, premium=None):
    if typ == 'Spot':
        return q * (S - entry)
    if typ == 'Forward':
        return q * (S * math.exp(-rf * T) - entry * math.exp(-rd * T))
    prem = premium if premium is not None else gk_price(S, K, T, rd, rf, vol, typ)
    return q * (prem - entry)


def signed_position_value(row, spot=None, vol=None):
    """Single-row P&L in MXN (kept for API compatibility and tests; the ledger
    uses ledger_risk). With no overrides the observed premium marks options."""
    typ = str(row['instrument_type']).title()
    if typ not in INSTRUMENTS:
        raise ValueError('instrument_type must be Spot, Forward, Call or Put')
    q, entry = float(row['signed_notional']), float(row['trade_price'])
    S = float(row['spot'] if spot is None else spot)
    T = max(float(row['days_to_expiry']) / 365.0, 1e-6)
    rd, rf = float(row['mxn_rate']), float(row['foreign_rate'])
    if typ in OPTION_TYPES:
        premium = float(row['market_option_premium']) if (spot is None and vol is None) else None
        v = max(1e-4, float(row['implied_volatility'] if vol is None else vol))
        return _mark_pnl(typ, q, S, entry, T, rd, rf, float(row['strike']), v, premium)
    return _mark_pnl(typ, q, S, entry, T, rd, rf)


# --------------------------------------------------------------------------
# Ledger cleaning / validation
# --------------------------------------------------------------------------
def ensure_columns(df):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(missing))
    out = df.copy()
    for c in TEXT_COLS:
        if c not in out.columns:
            out[c] = ''
    return out


def clean_ledger(df):
    """Drop blank rows, normalise names, coerce numbers."""
    out = ensure_columns(df)
    blank = out[['pair', 'instrument_type', 'signed_notional', 'trade_price']].isna().all(axis=1)
    out = out.loc[~blank].copy()
    out['instrument_type'] = out['instrument_type'].astype(str).str.strip().str.title()
    out['pair'] = out['pair'].astype(str).map(normalize_pair)
    out['trade_id'] = out['trade_id'].fillna('').astype(str)
    for c in NUMERIC:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    opt = out['instrument_type'].isin(OPTION_TYPES)
    spot_row = out['instrument_type'] == 'Spot'
    out.loc[~opt, OPTION_ONLY] = out.loc[~opt, OPTION_ONLY].fillna(0.0)
    out.loc[spot_row, TENOR_FIELDS] = out.loc[spot_row, TENOR_FIELDS].fillna(0.0)
    return out.reset_index(drop=True)


def ledger_issues(df):
    """Row-level problems in a cleaned ledger (empty list = OK)."""
    issues = []
    ids = df['trade_id'].astype(str)

    def add(mask, msg):
        for i in df.index[np.asarray(mask, dtype=bool)]:
            issues.append(f'Row {i + 1} ({ids[i] or "no id"}): {msg}')

    add(~df['instrument_type'].isin(INSTRUMENTS), 'invalid instrument_type (use Spot, Forward, Call or Put)')
    for c in NUMERIC:
        add(~np.isfinite(df[c].to_numpy(float)), f'{c} is missing or not a number')
    opt = df['instrument_type'].isin(OPTION_TYPES)
    fwd = df['instrument_type'] == 'Forward'
    add(df['spot'] <= 0, 'spot must be positive')
    add(~opt & (df['trade_price'] <= 0), 'trade_price must be positive (entry spot or contracted forward rate)')
    add((opt | fwd) & (df['days_to_expiry'] <= 0), 'days_to_expiry must be positive for forwards and options')
    add(opt & (df['strike'] <= 0), 'option strike must be positive')
    add(opt & (df['implied_volatility'] <= 0), 'option implied_volatility must be positive')
    add(opt & (df['market_option_premium'] < 0), 'market_option_premium cannot be negative')
    add((opt | fwd) & ((df['mxn_rate'].abs() > 1) | (df['foreign_rate'].abs() > 1)),
        'rates must be decimals (0.07 = 7%), not percentages')
    add(opt & (df['implied_volatility'] > 5), 'implied_volatility must be a decimal (0.12 = 12%)')
    return issues


# --------------------------------------------------------------------------
# Portfolio risk (vectorised); all P&L / Greeks in MXN
# --------------------------------------------------------------------------
def _arrays(df):
    t = df['instrument_type'].to_numpy()
    f = lambda c: df[c].to_numpy(dtype=float)
    T = np.maximum(f('days_to_expiry') / 365.0, 1e-6)
    rd, rf = f('mxn_rate'), f('foreign_rate')
    return {'q': f('signed_notional'), 'S': f('spot'), 'entry': f('trade_price'),
            'K': f('strike'), 'T': T, 'rd': rd, 'rf': rf, 'v': f('implied_volatility'),
            'mop': f('market_option_premium'),
            'ed': np.exp(-rd * T), 'ef': np.exp(-rf * T),
            'is_spot': t == 'Spot', 'is_fwd': t == 'Forward',
            'is_opt': np.isin(t, OPTION_TYPES), 'is_call': t == 'Call',
            'is_usd': np.array([is_usd_pair(p) for p in df['pair']], dtype=bool)}


def _current_marks(a):
    """Per-unit mark minus entry for every line, in MXN."""
    fwd_val = a['S'] * a['ef'] - a['entry'] * a['ed']
    spot_val = a['S'] - a['entry']
    opt_val = a['mop'] - a['entry']
    return np.where(a['is_opt'], opt_val, np.where(a['is_fwd'], fwd_val, spot_val))


def ledger_risk(df):
    a = _arrays(df)
    n = len(df)
    q, S = a['q'], a['S']
    pnl = q * _current_marks(a)

    delta_fx = q.copy()                       # Spot: 1:1
    delta_fx[a['is_fwd']] = (q * a['ef'])[a['is_fwd']]
    gamma1 = np.zeros(n); vega = np.zeros(n); theta = np.zeros(n)
    rho_d = np.zeros(n); rho_f = np.zeros(n)
    fwd_px = np.full(n, np.nan); model = np.full(n, np.nan)

    d = a['is_fwd']
    if d.any():
        theta[d] = q[d] * (a['rf'][d] * S[d] * a['ef'][d] - a['rd'][d] * a['entry'][d] * a['ed'][d]) / 365.0
        rho_d[d] = q[d] * a['entry'][d] * a['T'][d] * a['ed'][d] * 1e-4
        rho_f[d] = -q[d] * S[d] * a['T'][d] * a['ef'][d] * 1e-4
    o = a['is_opt']
    if o.any():
        g = gk_greeks_vec(S[o], a['K'][o], a['T'][o], a['rd'][o], a['rf'][o], a['v'][o], a['is_call'][o])
        model[o] = g['price']
        delta_fx[o] = q[o] * g['delta']
        gamma1[o] = q[o] * g['gamma'] * S[o] / 100.0
        vega[o] = q[o] * g['vega'] / 100.0
        theta[o] = q[o] * g['theta'] / 365.0
        rho_d[o] = q[o] * g['rho_d'] * 1e-4
        rho_f[o] = q[o] * g['rho_f'] * 1e-4
    der = a['is_fwd'] | o
    fwd_px[der] = forward_rate(S[der], a['T'][der], a['rd'][der], a['rf'][der])
    mop = np.where(o, a['mop'], np.nan)

    return pd.DataFrame({
        'trade_id': df['trade_id'].to_numpy(), 'pair': df['pair'].to_numpy(),
        'instrument_type': df['instrument_type'].to_numpy(), 'signed_notional': q,
        'current_pnl_mxn': pnl, 'gross_notional_mxn': np.abs(q * S),
        'market_forward': fwd_px, 'model_premium': model, 'market_option_premium': mop,
        'premium_model_gap': mop - model,
        'delta_fx': delta_fx, 'delta_mxn': delta_fx * S,
        'gamma_per_1pct_fx': gamma1, 'vega_per_vol_pt_mxn': vega,
        'theta_daily_mxn': theta, 'rho_mxn_per_bp': rho_d, 'rho_foreign_per_bp': rho_f})


def pair_summary(risk):
    if risk.empty:
        return pd.DataFrame()
    return (risk.groupby('pair', sort=False)
            .agg(trades=('trade_id', 'count'),
                 net_delta_fx=('delta_fx', 'sum'),
                 pnl_mxn=('current_pnl_mxn', 'sum'),
                 delta_mxn=('delta_mxn', 'sum'),
                 gross_notional_mxn=('gross_notional_mxn', 'sum'),
                 gamma_per_1pct_fx=('gamma_per_1pct_fx', 'sum'),
                 vega_mxn=('vega_per_vol_pt_mxn', 'sum'),
                 theta_mxn=('theta_daily_mxn', 'sum'),
                 rho_mxn_per_bp=('rho_mxn_per_bp', 'sum'))
            .reset_index())


def stress_grid(df, price_shocks, vol_shocks, g10_shock_pct=0.0):
    """Full revaluation of every line for every (spot shock %, vol shock pts) pair
    in one broadcast (scenarios x lines). Results in MXN.

    Spot shock P: every XXXMXN spot moves by P% (+ = MXN depreciates against the
    whole G10 block). g10_shock_pct G: an additional move of the non-USD G10
    currencies against USD (+ = G10 ex-USD strengthens vs USD), applied to every
    non-USD pair, so EURMXN moves by (1+P)(1+G)-1 while USDMXN moves by P.
    Vol shocks are parallel absolute shifts. Rates and carry are held fixed."""
    ps = np.asarray(sorted(price_shocks), dtype=float) / 100.0
    vs = np.asarray(sorted(vol_shocks), dtype=float) / 100.0
    g = float(g10_shock_pct) / 100.0
    if np.any(ps <= -1.0) or g <= -1.0:
        raise ValueError('Spot shocks must be greater than -100%.')
    a = _arrays(df)
    P = np.repeat(ps, len(vs))
    V = np.tile(vs, len(ps))
    q = a['q'][None, :]
    cross = np.where(a['is_usd'], 1.0, 1.0 + g)[None, :]
    newS = a['S'][None, :] * (1.0 + P[:, None]) * cross

    spot_pnl = q * (newS - a['entry'][None, :])
    fwd_pnl = q * (newS * a['ef'][None, :] - (a['entry'] * a['ed'])[None, :])
    pnl = np.where(a['is_fwd'][None, :], fwd_pnl, spot_pnl)
    base = float((a['q'] * _current_marks(a)).sum())

    idx = a['is_opt']
    if idx.any():
        row = lambda x: x[idx][None, :]
        newvol = np.maximum(1e-4, row(a['v']) + V[:, None])
        newprem = gk_price_vec(newS[:, idx], row(a['K']), row(a['T']), row(a['rd']), row(a['rf']),
                               newvol, row(a['is_call']))
        base_model = gk_price_vec(a['S'][idx], a['K'][idx], a['T'][idx], a['rd'][idx], a['rf'][idx],
                                  a['v'][idx], a['is_call'][idx])
        # Anchor to the observed premium; model changes drive the incremental stress.
        scen_mark = row(a['mop']) + newprem - base_model[None, :]
        pnl[:, idx] = q[:, idx] * (scen_mark - row(a['entry']))
    total = pnl.sum(axis=1)
    return pd.DataFrame({'spot_shock_%': P * 100.0, 'volatility_shock_pts': V * 100.0,
                         'stressed_portfolio_pnl': total, 'change_vs_current_pnl': total - base})


# --------------------------------------------------------------------------
# Hedge sizing
# --------------------------------------------------------------------------
def exposure_units(pair):
    b = base_ccy(pair)
    return [b, f'{b} thousands', f'{b} millions', 'MXN', 'MXN millions']


def to_foreign(pair, quantity, unit, spot=None):
    """Convert an exposure to units of the foreign (base) currency."""
    b = base_ccy(pair)
    if unit == b:
        return quantity
    if unit == f'{b} thousands':
        return quantity * 1e3
    if unit == f'{b} millions':
        return quantity * 1e6
    if unit in ('MXN', 'MXN millions'):
        if not spot or spot <= 0:
            raise ValueError('A positive spot is required to convert an MXN exposure.')
        return quantity * (1e6 if unit == 'MXN millions' else 1.0) / spot
    raise ValueError(f'Unknown measure: {unit}')


def round_lots(x, mode='Nearest'):
    if mode.startswith('Round down'):
        return int(math.floor(x + 1e-9))
    if mode.startswith('Round up'):
        return int(math.ceil(x - 1e-9))
    return int(math.floor(x + 0.5))


def min_variance_beta(vol_exposure, vol_hedge, correlation):
    """Value-based minimum-variance hedge ratio: rho * sigma_exposure / sigma_hedge."""
    if vol_exposure <= 0 or vol_hedge <= 0:
        raise ValueError('Volatilities must be positive.')
    if not -1.0 <= correlation <= 1.0:
        raise ValueError('Correlation must be between -1 and 1.')
    return correlation * vol_exposure / vol_hedge


def size_hedge(pair, quantity, unit, hedge_ratio=1.0, beta=1.0, rounding='Nearest',
               lot=None, spot=None, days=None, mxn_rate=None, foreign_rate=None,
               option_delta=None):
    """Notional / lots needed to hedge an FX exposure.

    hedge_ratio: fraction of exposure to hedge (1.0 = 100%).
    beta: basis / proxy-hedge ratio (hedge notional per unit of exposure).
    lot: rounding lot in foreign-currency units.
    spot (+ days, rates): enables MXN notionals, forward rate and carry.
    option_delta: if given, also size a delta-equivalent option position
        (a 1:1 notional option hedge is simply the forward-equivalent notional).
    """
    spec = spec_for(pair)
    size = float(lot) if lot else float(spec['lot'])
    if quantity < 0 or hedge_ratio < 0:
        raise ValueError('Exposure and hedge ratio cannot be negative.')
    if size <= 0 or beta <= 0:
        raise ValueError('Lot size and beta must be positive.')
    exposure = to_foreign(pair, quantity, unit, spot)
    target = exposure * hedge_ratio * beta
    exact = target / size
    lots = round_lots(exact, rounding)
    notional = lots * size
    hedged_equiv = notional / beta
    out = {'foreign_ccy': spec['base'] if normalize_pair(pair) in SPECS else base_ccy(pair),
           'lot': size, 'exposure_fx': exposure, 'target_fx': target,
           'lots_exact': exact, 'lots': lots, 'hedge_notional_fx': notional,
           'residual_fx': exposure - hedged_equiv,
           'hedged_pct': hedged_equiv / exposure if exposure > 0 else 0.0}
    if spot:
        out.update(exposure_mxn=exposure * spot, per_lot_notional_mxn=size * spot,
                   hedge_notional_mxn=notional * spot, pnl_per_1pct_mxn=notional * spot * 0.01)
        if days and days > 0 and mxn_rate is not None and foreign_rate is not None:
            T = days / 365.0
            F = float(forward_rate(spot, T, mxn_rate, foreign_rate))
            out.update(forward_rate=F, forward_points=F - spot,
                       annualized_carry=math.log(F / spot) / T,
                       carry_mxn=notional * (F - spot))
    if option_delta:
        d = abs(option_delta)
        out['option_notional_exact'] = target / d
        out['option_lots_exact'] = exact / d
        out['option_lots'] = round_lots(exact / d, rounding)
        out['option_notional_fx'] = out['option_lots'] * size
    return out
