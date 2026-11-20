# Forecast Desk

A Streamlit web app that forecasts Indian (NSE) stock prices. Search any NSE-listed stock, then see either the next 1-30 trading days (machine learning with a backtest) or the next 1-5 years (a range of outcomes).

> **Educational tool. Not financial advice.** No model can reliably predict stock prices. Use the forecasts as scenarios, not signals.

## Features

- **Three lists to pick from:** NSE stocks (about 2,000), Recently listed IPOs, and Upcoming IPOs. The list is downloaded from NSE and cached for a day.
- **Search by name or symbol.** You can also type any NSE symbol directly, for example `ZOMATO`.
- **Recently listed IPOs:** every NSE stock listed in the last year, newest first, forecastable from 5 trading days of prices.
- **Upcoming IPOs:** open and upcoming issues from NSE's calendar (or entered by hand) with weak, moderate and strong listing-price scenarios and a 20-day range. These are scenarios you control, not predictions, because an unlisted stock has no price history.
- **Short-term forecast (1-30 trading days)** using Ridge, Random Forest, Gradient Boosting, or an Ensemble of all three. Shows an 80% range around the forecast.
- **Honest backtest.** The model trains on the first 80% of history and is tested on the last 20%, compared against a "price stays the same" baseline. The app tells you when the model fails to beat it.
- **Long-term outlook (1-5 years)** as a range of outcomes (bear, median, bull) based on the stock's own volatility and a growth assumption you can tune. Includes a reality check of how often past ranges held.
- **Flexible price chart:** range buttons (1M, 3M, 6M, 1Y, 2Y, All), Area or Line styles, a "today" marker, hidden market holidays, and a price axis fitted to the data. Moving averages and Bollinger bands can be toggled.
- **Technical indicators:** RSI, MACD, moving averages, Bollinger bands.
- **Model drivers:** which signals the models rely on most.
- **Live ticker bar** with NIFTY 50, SENSEX and large Indian stocks.
- **CSV downloads** for forecasts and outlooks.
- **New listings handled:** stocks with 5-199 trading days of history get a volatility range instead of a trained model (very rough under 20 days).

## Quick start

Requires Python 3.10 or newer and an internet connection (for NSE and Yahoo Finance data).

```bash
# 1. Get into the project folder
cd stock_forecast

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens at http://localhost:8501.

## How to use it

1. Choose a list in the sidebar (NSE stocks, Recently listed IPOs or Upcoming IPOs), then click the stock box and type a name or symbol, or use the "Or type any symbol" box.
2. Choose a **Prediction range**:
   - **Next 1-30 days:** pick the training history (default 5 years), the horizon and the model, then press **Run forecast**.
   - **Next 1-5 years:** pick the years ahead (default 5) and how much weight to give the stock's own past growth.
3. Read the forecast chart and the tabs below it (Backtest, Indicators, What drives it, Data).

## How it works

| Part | Approach |
| --- | --- |
| Data | Daily prices from Yahoo Finance via `yfinance`, adjusted for splits and dividends |
| Features | Returns, moving-average distance, MACD, RSI, Bollinger position, volatility, volume ratio, daily range |
| Short-term model | Predicts the log return over the horizon at a few checkpoints, then interpolates a path |
| Uncertainty | Error spread measured on the unseen test period, widened with the square root of time |
| Long-term outlook | Lognormal cone: blended growth assumption plus historical volatility |

## Project structure

```
stock_forecast/
├── app.py             # the whole app (UI, data, models)
├── requirements.txt   # Python dependencies
└── README.md
```

## Data requirements

| History available | What you get |
| --- | --- |
| 200+ trading days | Full model forecast and backtest |
| 5-199 trading days | Volatility range only, no trend, no backtest (very rough under 20 days) |
| Under 5 days | A message that there is too little data. Use Upcoming IPOs for scenario prices |
| About 2+ years | Needed for the 1-5 year outlook |

## Deploy online

The quickest way is [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push this folder to a GitHub repository.
2. Create a new app on Streamlit Community Cloud and point it at `app.py`.
3. Share the link. On a phone, open it in the browser and choose **Add to Home Screen** to use it like an app.

## Troubleshooting

- **"No price data found":** check the symbol. NSE symbols end in `.NS` (added automatically in the stock box). Delisted, suspended and some SME stocks are not available on Yahoo Finance.
- **"NSE's full list couldn't be downloaded":** the app falls back to a built-in list of about 130 major NSE stocks. You can still type any symbol. The IPO list needs NSE's full list for listing dates. Try again later or check your connection.
- **"NSE's IPO calendar couldn't be loaded":** NSE sometimes blocks automated requests. Enter the company and price band by hand instead.
- **Slow or empty results:** Yahoo Finance sometimes rate-limits requests. Wait a minute and run again, and update `yfinance` with `pip install -U yfinance`.
- **First forecast takes a few seconds:** models are trained on demand. Results are cached for 15 minutes.

## Limitations

- Predictions use only past prices and volume. News, earnings, interest rates and company-specific events are not included.
- Backtest results are on a single holdout period and can look better or worse than future performance.
- Data comes from a free, unofficial source and may contain gaps or delays.

## License and disclaimer

This project is for learning and research. Nothing it shows is investment advice. Do your own research or consult a SEBI-registered advisor before making financial decisions.

