import math
import unittest

import numpy as np
import pandas as pd

from risk_engine import (PAIRS, SPECS, clean_ledger, forward_rate, gk_greeks, gk_price,
                         implied_volatility, ledger_issues, ledger_risk, min_variance_beta,
                         normalize_pair, signed_position_value, size_hedge, strike_from_delta,
                         stress_grid, to_foreign, zero_cost_collar_strike)

S, K, T, RD, RF, V = 18.40, 19.00, 0.5, 0.07, 0.04, 0.12


def make_ledger():
    base = dict(trade_date='2026-10-21', notes='', maturity='Apr-2027', purpose='Hedge')
    rows = [
        {**base, 'trade_id': 'A', 'pair': 'USDMXN', 'instrument_type': 'Call', 'signed_notional': 1_000_000.0,
         'trade_price': 0.30, 'spot': 18.40, 'strike': 19.00, 'days_to_expiry': 180.0, 'mxn_rate': 0.07,
         'foreign_rate': 0.04, 'implied_volatility': 0.12, 'market_option_premium': 0.34},
        {**base, 'trade_id': 'B', 'pair': 'EURMXN', 'instrument_type': 'Put', 'signed_notional': -500_000.0,
         'trade_price': 0.40, 'spot': 21.40, 'strike': 20.50, 'days_to_expiry': 90.0, 'mxn_rate': 0.07,
         'foreign_rate': 0.02, 'implied_volatility': 0.125, 'market_option_premium': 0.35},
        {**base, 'trade_id': 'C', 'pair': 'EURMXN', 'instrument_type': 'Forward', 'signed_notional': -300_000.0,
         'trade_price': 21.90, 'spot': 21.40, 'strike': 0.0, 'days_to_expiry': 120.0, 'mxn_rate': 0.07,
         'foreign_rate': 0.02, 'implied_volatility': 0.0, 'market_option_premium': 0.0},
        {**base, 'trade_id': 'D', 'pair': 'JPYMXN', 'instrument_type': 'Spot', 'signed_notional': 50_000_000.0,
         'trade_price': 0.1200, 'spot': 0.1230, 'strike': 0.0, 'days_to_expiry': 0.0, 'mxn_rate': 0.0,
         'foreign_rate': 0.0, 'implied_volatility': 0.0, 'market_option_premium': 0.0},
        {**base, 'trade_id': 'E', 'pair': 'GBPMXN', 'instrument_type': 'Call', 'signed_notional': 200_000.0,
         'trade_price': 0.9, 'spot': 24.60, 'strike': 25.5, 'days_to_expiry': 60.0, 'mxn_rate': 0.07,
         'foreign_rate': 0.0375, 'implied_volatility': 0.135, 'market_option_premium': 0.8},
    ]
    return clean_ledger(pd.DataFrame(rows))


def fd(fn, h):
    return (fn(h) - fn(-h)) / (2 * h)


class GarmanKohlhagenTests(unittest.TestCase):
    def test_put_call_parity(self):
        c = gk_price(S, K, T, RD, RF, V, 'Call'); p = gk_price(S, K, T, RD, RF, V, 'Put')
        self.assertAlmostEqual(c - p, S * math.exp(-RF * T) - K * math.exp(-RD * T), places=10)

    def test_equals_black76_on_forward(self):
        from scipy.stats import norm
        F = float(forward_rate(S, T, RD, RF))
        d1 = (math.log(F / K) + 0.5 * V * V * T) / (V * math.sqrt(T)); d2 = d1 - V * math.sqrt(T)
        b76 = math.exp(-RD * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
        self.assertAlmostEqual(gk_price(S, K, T, RD, RF, V, 'Call'), b76, places=10)

    def test_forward_and_carry_sign(self):
        F = float(forward_rate(S, 1.0, RD, RF))
        self.assertAlmostEqual(F, S * math.exp(0.03), places=10)
        self.assertGreater(F, S)  # MXN rates above USD -> forward points positive in USDMXN

    def test_iv_roundtrip(self):
        for typ in ('Call', 'Put'):
            prem = gk_price(S, K, T, RD, RF, 0.137, typ)
            self.assertAlmostEqual(implied_volatility(prem, S, K, T, RD, RF, typ), 0.137, places=7)

    def test_delta_gamma_vega_theta_rho_match_finite_differences(self):
        for typ in ('Call', 'Put'):
            g = gk_greeks(S, K, T, RD, RF, V, typ)
            h = 1e-3
            self.assertAlmostEqual(g['Delta (spot)'], fd(lambda x: gk_price(S + x, K, T, RD, RF, V, typ), h), places=6)
            gam = (gk_price(S + h, K, T, RD, RF, V, typ) - 2 * gk_price(S, K, T, RD, RF, V, typ)
                   + gk_price(S - h, K, T, RD, RF, V, typ)) / h ** 2
            self.assertAlmostEqual(g['Gamma (per 1 MXN)'], gam, places=4)
            self.assertAlmostEqual(g['Vega (per 1.00 vol)'], fd(lambda x: gk_price(S, K, T, RD, RF, V + x, typ), 1e-5), places=4)
            self.assertAlmostEqual(g['Theta (daily)'] * 365, -fd(lambda x: gk_price(S, K, T + x, RD, RF, V, typ), 1e-5), places=4)
            self.assertAlmostEqual(g['Rho MXN (per 1bp)'] * 1e4, fd(lambda x: gk_price(S, K, T, RD + x, RF, V, typ), 1e-6), places=4)
            self.assertAlmostEqual(g['Rho foreign (per 1bp)'] * 1e4, fd(lambda x: gk_price(S, K, T, RD, RF + x, V, typ), 1e-6), places=4)

    def test_forward_delta_is_spot_delta_grossed_up(self):
        g = gk_greeks(S, K, T, RD, RF, V, 'Call')
        self.assertAlmostEqual(g['Delta (spot)'], math.exp(-RF * T) * g['Delta (forward)'], places=12)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            gk_price(S, K, 0, RD, RF, V)
        with self.assertRaises(ValueError):
            gk_price(S, K, T, RD, RF, -0.1)

    def test_negative_foreign_rate_allowed(self):  # JPY / CHF history
        self.assertGreater(gk_price(S, K, T, RD, -0.005, V), 0)

    def test_strike_from_delta_roundtrip(self):
        for typ, d in (('Call', 0.25), ('Put', 0.25), ('Call', 0.10), ('Put', 0.40)):
            k = strike_from_delta(S, T, RD, RF, V, d, typ)
            self.assertAlmostEqual(abs(gk_greeks(S, k, T, RD, RF, V, typ)['Delta (spot)']), d, places=9)
        self.assertGreater(strike_from_delta(S, T, RD, RF, V, 0.25, 'Call'), S)
        self.assertLess(strike_from_delta(S, T, RD, RF, V, 0.25, 'Put'), S)

    def test_zero_cost_collar(self):
        kc = 19.20  # payable: buy call, sell put
        kp = zero_cost_collar_strike(S, kc, T, RD, RF, V, 'Call')
        self.assertLess(kp, S)
        self.assertAlmostEqual(gk_price(S, kc, T, RD, RF, V, 'Call'), gk_price(S, kp, T, RD, RF, V, 'Put'), places=8)
        kp2 = 17.50  # receivable: buy put, sell call
        kc2 = zero_cost_collar_strike(S, kp2, T, RD, RF, V, 'Put')
        self.assertAlmostEqual(gk_price(S, kp2, T, RD, RF, V, 'Put'), gk_price(S, kc2, T, RD, RF, V, 'Call'), places=8)
        # skew: richer vol on the sold put forces a higher (less favourable) sold strike to move closer? cost-neutral still holds
        kp3 = zero_cost_collar_strike(S, kc, T, RD, RF, V, 'Call', sold_vol=V + 0.02)
        self.assertAlmostEqual(gk_price(S, kc, T, RD, RF, V, 'Call'), gk_price(S, kp3, T, RD, RF, V + 0.02, 'Put'), places=8)
        self.assertLess(kp3, kp)  # richer put vol -> put struck further OTM for the same premium


class PairTests(unittest.TestCase):
    def test_g10_coverage(self):
        bases = {SPECS[p]['base'] for p in PAIRS if p != 'Other'}
        self.assertEqual(bases, {'USD', 'EUR', 'JPY', 'GBP', 'CHF', 'CAD', 'AUD', 'NZD', 'NOK', 'SEK'})
        self.assertTrue(all(p.endswith('MXN') for p in PAIRS if p != 'Other'))

    def test_normalize(self):
        self.assertEqual(normalize_pair('usd/mxn'), 'USDMXN')
        self.assertEqual(normalize_pair(' EUR MXN '), 'EURMXN')


class PortfolioTests(unittest.TestCase):
    def test_zero_shock_matches_current_pnl(self):
        sd = stress_grid(make_ledger(), [0], [0], 0.0)
        self.assertAlmostEqual(sd.change_vs_current_pnl.iloc[0], 0.0, places=8)

    def test_vector_stress_matches_scalar_revaluation(self):
        led = make_ledger()
        P, VS, G = 10.0, 5.0, -4.0
        sd = stress_grid(led, [-10, 10], [-5, 5], G)
        row = sd[(sd['spot_shock_%'] == P) & (sd['volatility_shock_pts'] == VS)].iloc[0]
        total, base = 0.0, 0.0
        for _, r in led.iterrows():
            cross = 1.0 if r['pair'].startswith('USD') else 1 + G / 100
            newS = r.spot * (1 + P / 100) * cross
            Tt = max(r.days_to_expiry / 365, 1e-6)
            if r.instrument_type == 'Spot':
                pnl = r.signed_notional * (newS - r.trade_price)
            elif r.instrument_type == 'Forward':
                pnl = r.signed_notional * (newS * math.exp(-r.foreign_rate * Tt) - r.trade_price * math.exp(-r.mxn_rate * Tt))
            else:
                b = gk_price(r.spot, r.strike, Tt, r.mxn_rate, r.foreign_rate, r.implied_volatility, r.instrument_type)
                n = gk_price(newS, r.strike, Tt, r.mxn_rate, r.foreign_rate, r.implied_volatility + VS / 100, r.instrument_type)
                pnl = r.signed_notional * (r.market_option_premium + n - b - r.trade_price)
            total += pnl
            base += signed_position_value(r)
        self.assertAlmostEqual(row.stressed_portfolio_pnl, total, places=5)
        self.assertAlmostEqual(row.change_vs_current_pnl, total - base, places=5)

    def test_g10_shock_leaves_usdmxn_untouched(self):
        led = make_ledger()
        led = led[led.pair == 'USDMXN'].reset_index(drop=True)
        a = stress_grid(led, [5], [0], 0.0).stressed_portfolio_pnl.iloc[0]
        b = stress_grid(led, [5], [0], 7.0).stressed_portfolio_pnl.iloc[0]
        self.assertAlmostEqual(a, b, places=10)

    def test_ledger_risk_matches_row_pnl(self):
        led = make_ledger()
        risk = ledger_risk(led)
        expected = [signed_position_value(r) for _, r in led.iterrows()]
        np.testing.assert_allclose(risk.current_pnl_mxn.to_numpy(), expected)

    def test_ledger_greeks_match_finite_differences(self):
        led = make_ledger()
        risk = ledger_risk(led)
        h = 1e-4
        for i, r in led.iterrows():
            if r.instrument_type == 'Spot':
                continue
            fdv = lambda x: signed_position_value(r, spot=r.spot + x, vol=r.implied_volatility) if r.instrument_type != 'Forward' \
                else signed_position_value(r, spot=r.spot + x)
            # delta in foreign units = dV/dS
            self.assertAlmostEqual(risk.delta_fx[i], fd(fdv, h * r.spot), delta=abs(risk.delta_fx[i]) * 1e-4 + 1e-3)
        fwd = led[led.instrument_type == 'Forward'].iloc[0]
        i = led.index[led.instrument_type == 'Forward'][0]
        v0 = lambda T_: fwd.signed_notional * (fwd.spot * math.exp(-fwd.foreign_rate * T_) - fwd.trade_price * math.exp(-fwd.mxn_rate * T_))
        T0 = fwd.days_to_expiry / 365
        self.assertAlmostEqual(risk.theta_daily_mxn[i] * 365, -fd(lambda x: v0(T0 + x), 1e-6), delta=1e-2)
        self.assertAlmostEqual(risk.rho_mxn_per_bp[i] * 1e4,
                               fd(lambda x: fwd.signed_notional * (fwd.spot * math.exp(-fwd.foreign_rate * T0)
                                                                   - fwd.trade_price * math.exp(-(fwd.mxn_rate + x) * T0)), 1e-6), delta=1e-2)

    def test_spot_row_delta_is_notional(self):
        risk = ledger_risk(make_ledger())
        r = risk[risk.trade_id == 'D'].iloc[0]
        self.assertEqual(r.delta_fx, 50_000_000.0)
        self.assertAlmostEqual(r.delta_mxn, 50_000_000.0 * 0.1230)

    def test_blank_rows_dropped_and_issues_flagged(self):
        raw = make_ledger().copy()
        raw.loc[len(raw)] = [None] * len(raw.columns)
        self.assertEqual(len(clean_ledger(raw)), 5)
        bad = make_ledger().copy(); bad.loc[0, 'strike'] = 0.0
        self.assertTrue(any('strike' in i for i in ledger_issues(bad)))
        bad = make_ledger().copy(); bad.loc[0, 'mxn_rate'] = 7.0
        self.assertTrue(any('decimals' in i for i in ledger_issues(bad)))
        bad = make_ledger().copy(); bad.loc[2, 'days_to_expiry'] = 0.0
        self.assertTrue(any('days_to_expiry' in i for i in ledger_issues(bad)))
        self.assertEqual(ledger_issues(make_ledger()), [])


class HedgeTests(unittest.TestCase):
    def test_units(self):
        self.assertAlmostEqual(to_foreign('USDMXN', 2, 'USD millions'), 2e6)
        self.assertAlmostEqual(to_foreign('EURMXN', 3_000, 'EUR thousands'), 3e6)
        self.assertAlmostEqual(to_foreign('USDMXN', 20_000_000, 'MXN', spot=20.0), 1_000_000.0)
        self.assertAlmostEqual(to_foreign('USDMXN', 20, 'MXN millions', spot=20.0), 1_000_000.0)
        with self.assertRaises(ValueError):
            to_foreign('USDMXN', 1, 'MXN')
        with self.assertRaises(ValueError):
            to_foreign('USDMXN', 1, 'EUR')

    def test_basic_lots(self):
        r = size_hedge('USDMXN', 1_000_000, 'USD', lot=100_000)
        self.assertAlmostEqual(r['lots_exact'], 10.0); self.assertEqual(r['lots'], 10)
        self.assertAlmostEqual(r['hedge_notional_fx'], 1_000_000.0)

    def test_ratio_beta_rounding(self):
        r = size_hedge('USDMXN', 1_000_000, 'USD', hedge_ratio=0.75, rounding='Round down (under-hedge)', lot=100_000)
        self.assertEqual(r['lots'], 7)
        r = size_hedge('USDMXN', 1_020_000, 'USD', rounding='Round up (over-hedge)', lot=100_000)
        self.assertEqual(r['lots'], 11)
        r = size_hedge('USDMXN', 1_000_000, 'USD', beta=0.8, lot=100_000)
        self.assertAlmostEqual(r['lots_exact'], 8.0); self.assertAlmostEqual(r['hedged_pct'], 1.0)

    def test_notional_carry_and_option_sizing(self):
        r = size_hedge('USDMXN', 1_000_000, 'USD', lot=100_000, spot=18.40, days=365, mxn_rate=0.07,
                       foreign_rate=0.04, option_delta=-0.5)
        self.assertAlmostEqual(r['hedge_notional_mxn'], 18_400_000.0)
        self.assertAlmostEqual(r['pnl_per_1pct_mxn'], 184_000.0)
        F = 18.40 * math.exp(0.03)
        self.assertAlmostEqual(r['forward_rate'], F); self.assertAlmostEqual(r['carry_mxn'], 1_000_000 * (F - 18.40), places=4)
        self.assertAlmostEqual(r['annualized_carry'], 0.03)
        self.assertEqual(r['option_lots'], 20); self.assertAlmostEqual(r['option_notional_fx'], 2_000_000.0)

    def test_mxn_exposure_conversion(self):
        r = size_hedge('EURMXN', 214, 'MXN millions', lot=100_000, spot=21.40)
        self.assertAlmostEqual(r['exposure_fx'], 10_000_000.0, places=6)  # 214mm MXN / 21.4 = 10mm EUR
        self.assertEqual(r['lots'], 100)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            size_hedge('USDMXN', -1, 'USD')
        with self.assertRaises(ValueError):
            size_hedge('USDMXN', 1, 'USD', beta=0)

    def test_min_variance_beta(self):
        self.assertAlmostEqual(min_variance_beta(0.14, 0.12, 0.9), 1.05)
        with self.assertRaises(ValueError):
            min_variance_beta(0.1, 0.1, 1.5)


if __name__ == '__main__':
    unittest.main()
