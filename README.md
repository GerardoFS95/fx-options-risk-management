# MXN vs G10 FX Options Risk Workbench

Streamlit hedging-desk prototype for a **Mexican corporate** managing FX exposure against the G10 block: spot, outright forwards and European vanilla options (plus zero-cost collars) on USD, EUR, GBP, JPY, CHF, CAD, AUD, NZD, NOK and SEK, all versus MXN.

Pricing uses **Garman-Kohlhagen** (Black-Scholes-Merton with a foreign interest rate). It is identical to Black-76 on the outright forward, discounted at the MXN rate, so it is consistent with the commodities version.

## Conventions (read these first)
- Every pair is quoted **XXXMXN = MXN per 1 unit of the G10 currency** (JPY is MXN per 1 JPY, e.g. 0.1230). MXN is always the domestic currency, so premium, strike, P&L and every Greek are in **MXN**. No FX conversion is needed to aggregate; USD equivalents are shown for reference using the USDMXN reference spot in the sidebar.
- `signed_notional` is in **foreign-currency units**: positive = long foreign currency (buy FX / long call / long put), negative = short.
- **Call** = right to buy the G10 currency / sell MXN (protects a payable). **Put** = right to sell the G10 currency / buy MXN (protects a receivable).
- Rates are continuously-compounded decimals, `T = days / 365`. Convert dealer money-market quotes (e.g. TIIE is simple ACT/360) before entering them.
- Delta is **spot delta, premium excluded** (`e^(-rf·T)·N(d1)` for a call), in foreign-currency units. Forward delta is shown in the pre-trade table.
- Stress sign convention: **+ spot shock = MXN depreciates** (the risk-off direction).

## What carried over from the commodities app
| Commodities app | FX app |
|---|---|
| Black-76 on a futures price | Garman-Kohlhagen on spot with two rates (MXN, foreign) |
| Futures / Physical / Call / Put | Spot / Forward / Call / Put |
| Contracts × multiplier | Notional in foreign currency (OTC), rounded to a dealing lot |
| Product (corn, soy, palm …) | Pair (USDMXN, EURMXN … 10 G10 pairs + user-defined) |
| Physical long / short | Payable (need to buy FX) / Receivable (will sell FX) |
| Palm oil MYR → USD translation | Everything natively in MXN; USD equivalents for display |
| Protective option sized on delta | 1:1 notional (locks the worst-case rate) **or** delta-equivalent |
| — | Forward points / carry, strike-from-delta, zero-cost collar solver, proxy-hedge beta calculator |

## Features
- **Add a trade (main page):** manual entry for Spot, Forward, Call and Put (pair, side, notional, maturity, entry price/premium, option terms, both rates) appended straight to the ledger.
- **1 · Pre-trade analytics:** premium (MXN per unit and % of notional), Greeks (spot/forward delta, gamma, vega, theta, MXN and foreign rho), implied vol from an observed premium, market-vs-model gap, outright forward and forward points, break-even, premium outlay, premium-vs-spot chart with expiry intrinsic. Strike can be entered as an absolute level or as a **target delta** (e.g. 25Δ).
- **2 · Hedge sizing:** converts an exposure in foreign currency (units / thousands / millions) or **MXN** (e.g. an MXN budget exposed to USD) into a hedge notional with hedge ratio, proxy-hedge beta, rounding lots, notional, P&L per 1% move, a hedge-ratio ladder and a **forward curve / carry table**. Hedge instruments:
  - **Forward:** shows the forward rate, points in pips, annualised carry and the carry cost/(benefit) vs spot (with MXN rates above foreign rates, a payable hedge costs carry and a receivable hedge earns it).
  - **Protective option:** long call (payable) or long put (receivable), notional-matched or delta-equivalent, with premium outlay and all-in worst-case rate.
  - **Zero-cost collar:** you pick the protective strike; the engine solves the financing strike with a root-finder, with an optional skew adjustment (vol pts) on the sold leg. Books both legs.
  - The hedge can be booked into the ledger in one click.
- **3 · Trade ledger & CSV:** editable ledger, CSV import (replace or append), export, blank template, validation (including a guard against rates or vols entered as percentages).
- **4 · Portfolio risk & stress:** P&L, delta (MXN and foreign units), gamma per 1% move, vega, theta, MXN and foreign rho per bp, notional by trade and by pair; a full-revaluation spot × vol stress grid with a **MXN shock layered with an optional G10-vs-USD shock** (so EURMXN moves by `(1+MXN shock)(1+G10 move)−1` while USDMXN moves only by the MXN shock), pivot view and CSV export.

## Efficiency
Pricing, Greeks and the stress grid are fully numpy-vectorised (`scipy.special.ndtr`): the whole ledger and the whole scenario grid are revalued in one broadcast. The ledger lives in `st.session_state` and stress results are cached.

## Run locally
```bash
conda create -n mxn-fx-risk-workbench python=3.11 -y
conda activate mxn-fx-risk-workbench
cd "C:\PATH\TO\mxn_fx_risk_workbench"
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
Run the tests with `python -m unittest test_models.py`.

## Deploy from GitHub (Streamlit Community Cloud)
1. Push `app.py`, `risk_engine.py`, `requirements.txt` (and the rest) to a GitHub repository.
2. Go to share.streamlit.io, choose **New app**, select the repo and branch, and set the main file to `app.py`.
3. Deploy. Every push to the branch redeploys the app.

The ledger is held in the browser session only. Export the CSV before closing the page, or re-import it next time.

## Pair defaults (illustrative placeholders)
| Pair | Foreign ccy | Placeholder spot | Placeholder vol | Rounding lot | Liquidity note |
|---|---|---|---|---|---|
| USDMXN | USD | 18.40 | 12.0% | 100,000 | Core |
| EURMXN | EUR | 21.40 | 12.5% | 100,000 | Liquid cross |
| GBPMXN | GBP | 24.60 | 13.5% | 100,000 | Liquid cross |
| JPYMXN | JPY | 0.1230 | 16.0% | 10,000,000 | Cross, wider spreads |
| CHFMXN | CHF | 23.00 | 14.0% | 100,000 | Thin cross |
| CADMXN | CAD | 13.30 | 12.0% | 100,000 | Liquid cross |
| AUDMXN | AUD | 12.10 | 13.5% | 100,000 | Thin cross |
| NZDMXN | NZD | 10.60 | 14.5% | 100,000 | Very thin: consider synthetic via USDMXN × NZDUSD |
| NOKMXN | NOK | 1.82 | 15.0% | 1,000,000 | Very thin: consider synthetic |
| SEKMXN | SEK | 1.95 | 15.0% | 1,000,000 | Very thin: consider synthetic |

MXN rate placeholder 7.0%; foreign rate placeholders are in `risk_engine.SPECS`. **All spots, rates and vols are illustrative and were not taken from a live feed. Overwrite them with dealer quotes before using any output.**

## Ledger convention
- `signed_notional`: positive long foreign currency, negative short, in foreign-currency units.
- **Spot:** `trade_price` is the entry spot / budget rate. P&L = notional × (spot − entry). Use it to carry an existing unhedged exposure against your budget rate.
- **Forward:** `trade_price` is the contracted forward rate; `days_to_expiry`, `mxn_rate`, `foreign_rate` required. MTM = notional × (S·e^(−rf·T) − K·e^(−rd·T)), i.e. discounted at the MXN rate.
- **Options:** `trade_price` is the entry premium per 1 unit of foreign currency (MXN); `market_option_premium` is the current observed premium. **Current P&L uses the observed market premium** (Garman-Kohlhagen is used for Greeks and for stress revaluation, anchored to that observed premium).
- `mxn_rate`, `foreign_rate` and `implied_volatility` are decimals (e.g. 0.07, 0.04, 0.12).

## Important limitations
This is a prototype and not a production trading system.
- **Flat vol per trade; no smile or skew.** MXN options trade with a material skew, so the model premium and Greeks away from ATM will differ from dealer prices; use the vol quoted for the strike. The collar solver has a skew-adjustment input for this reason, but a real surface (ATM / 25Δ RR / 25Δ BF) is a next step.
- Spot delta ex-premium only; no premium-adjusted delta, no delta-cutoff or ATM-forward conventions.
- No bid/offer, settlement-date logic (T+2 spot, holidays, delivery vs NDF), day-count conventions beyond ACT/365 continuous compounding, or American/barrier/digital features.
- Stress shocks are parallel and deterministic; they are not VaR/ES, ignore skew dynamics and cross-rate correlation beyond the explicit G10-vs-USD input, and hold rates fixed.
- No credit lines/CVA, margin/collateral, hedge-accounting (IFRS 9 / NIF C-10) effectiveness testing, accounting vs economic hedge classification, or tax treatment.
- The proxy-hedge beta is a user input (a minimum-variance calculator is provided, but you must supply vols and correlation).
- Validate calculations against a trusted pricing source (dealer quote or Bloomberg OVML) before using in a live decision process.
