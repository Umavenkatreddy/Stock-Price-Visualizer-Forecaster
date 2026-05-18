"""
Unit tests for Stock Price Visualizer & Forecaster.

Tests import the real Stock modules with heavy dependencies patched at the
sys.modules level BEFORE import, so coverage tools count actual source lines.
"""

import sys
import os
import types
import unittest
from unittest.mock import MagicMock, patch, call
from datetime import date, timedelta

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Patch heavy dependencies BEFORE importing any Stock module so that:
#   1. No network calls happen during tests.
#   2. coverage.py still instruments the real source files.
# ---------------------------------------------------------------------------

def _install_stubs():
    # --- yfinance ---
    yf = MagicMock()
    sys.modules["yfinance"] = yf

    # --- plotly ---
    px = MagicMock()
    go = MagicMock()
    # go.Figure() must return something with add_trace / update_layout
    mock_fig = MagicMock()
    go.Figure.return_value = mock_fig
    go.Scatter = MagicMock()

    plotly_mod = types.ModuleType("plotly")
    plotly_mod.express = px
    plotly_mod.graph_objs = go
    sys.modules["plotly"] = plotly_mod
    sys.modules["plotly.express"] = px
    sys.modules["plotly.graph_objs"] = go

    # --- dash ---
    dash_mod = types.ModuleType("dash")
    dash_mod.Dash = MagicMock(return_value=MagicMock())
    dash_mod.html = MagicMock()
    dash_mod.dcc = MagicMock()

    dep_mod = types.ModuleType("dash.dependencies")
    dep_mod.Input = MagicMock()
    dep_mod.Output = MagicMock()
    dep_mod.State = MagicMock()

    exc_mod = types.ModuleType("dash.exceptions")

    class _PreventUpdate(Exception):
        pass

    exc_mod.PreventUpdate = _PreventUpdate

    dash_mod.dependencies = dep_mod
    dash_mod.exceptions = exc_mod
    sys.modules["dash"] = dash_mod
    sys.modules["dash.dependencies"] = dep_mod
    sys.modules["dash.exceptions"] = exc_mod
    sys.modules["dash_core_components"] = MagicMock()
    sys.modules["dash_html_components"] = MagicMock()

    return yf, px, go


_yf_stub, _px_stub, _go_stub = _install_stubs()

# Add project root to path so `from Stock.app import ...` works
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Now import the real modules (stubs already in place)
from Stock.app import get_stock_price_fig, get_more  # noqa: E402
from Stock.app import update_data, stock_price, indicators, forecast  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv_df(n=60):
    dates = pd.date_range(end=date.today(), periods=n, freq="B")
    close = np.linspace(100, 200, n)
    return pd.DataFrame({
        "Date": dates,
        "Open": close - 1,
        "High": close + 2,
        "Low": close - 2,
        "Close": close,
        "Volume": np.ones(n, dtype=int) * 1_000_000,
    })


# ---------------------------------------------------------------------------
# Tests: get_stock_price_fig
# ---------------------------------------------------------------------------

class TestGetStockPriceFig(unittest.TestCase):

    def test_returns_figure(self):
        df = _make_ohlcv_df(10)
        fig = get_stock_price_fig(df)
        # px.line was called (stub returns a MagicMock)
        self.assertTrue(_px_stub.line.called)

    def test_called_with_close_and_open(self):
        _px_stub.line.reset_mock()
        df = _make_ohlcv_df(10)
        get_stock_price_fig(df)
        _, kwargs = _px_stub.line.call_args
        self.assertIn("Close", kwargs.get("y", []))
        self.assertIn("Open", kwargs.get("y", []))

    def test_x_axis_is_date(self):
        _px_stub.line.reset_mock()
        df = _make_ohlcv_df(10)
        get_stock_price_fig(df)
        _, kwargs = _px_stub.line.call_args
        self.assertEqual(kwargs.get("x"), "Date")


# ---------------------------------------------------------------------------
# Tests: get_more (EMA)
# ---------------------------------------------------------------------------

class TestGetMore(unittest.TestCase):

    def test_ema_column_added(self):
        df = _make_ohlcv_df(30)
        get_more(df)
        self.assertIn("EMA_20", df.columns)

    def test_ema_values_finite(self):
        df = _make_ohlcv_df(30)
        get_more(df)
        self.assertTrue(np.all(np.isfinite(df["EMA_20"].values)))

    def test_returns_figure(self):
        df = _make_ohlcv_df(30)
        fig = get_more(df)
        self.assertTrue(_px_stub.scatter.called)

    def test_ema_span_20(self):
        df = _make_ohlcv_df(30)
        get_more(df)
        # EMA_20 first value should equal the first Close (seed of ewm)
        self.assertAlmostEqual(df["EMA_20"].iloc[0], df["Close"].iloc[0], places=5)


# ---------------------------------------------------------------------------
# Tests: update_data callback
# ---------------------------------------------------------------------------

class TestUpdateData(unittest.TestCase):

    def test_returns_default_when_no_click(self):
        result = update_data(None, None)
        # Should return the welcome string, placeholder image url, title, and three Nones
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 6)
        self.assertIsNone(result[3])
        self.assertIsNone(result[4])
        self.assertIsNone(result[5])

    def test_raises_when_click_but_no_ticker(self):
        from dash.exceptions import PreventUpdate
        with self.assertRaises(PreventUpdate):
            update_data(1, None)

    def test_fetches_ticker_info_on_submit(self):
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "logo_url": "http://logo",
            "shortName": "ACME Corp",
            "longBusinessSummary": "Makes everything.",
        }
        _yf_stub.Ticker.return_value = mock_ticker

        result = update_data(1, "ACME")
        self.assertEqual(result[0], "Makes everything.")
        self.assertEqual(result[1], "http://logo")
        self.assertEqual(result[2], "ACME Corp")


# ---------------------------------------------------------------------------
# Tests: stock_price callback
# ---------------------------------------------------------------------------

class TestStockPrice(unittest.TestCase):

    def test_returns_empty_when_no_click(self):
        result = stock_price(None, None, None, None)
        self.assertEqual(result, [""])

    def test_raises_when_click_but_no_val(self):
        from dash.exceptions import PreventUpdate
        with self.assertRaises(PreventUpdate):
            stock_price(1, None, None, None)

    def test_downloads_and_returns_graph(self):
        df = _make_ohlcv_df(20)
        _yf_stub.download.return_value = df.drop(columns=["Date"]).set_index("Date").reset_index()
        # Patch reset_index to be a no-op that keeps Date column
        mock_df = df.copy()
        _yf_stub.download.return_value = mock_df.set_index("Date")

        # We need a df with Date column after reset_index
        real_df = _make_ohlcv_df(20)
        _yf_stub.download.return_value = real_df.set_index("Date")

        with patch("Stock.app.get_stock_price_fig", return_value=MagicMock()) as mock_fig_fn:
            result = stock_price(1, "2024-01-01", "2024-06-01", "AAPL")
            self.assertTrue(mock_fig_fn.called)
            self.assertEqual(len(result), 1)


# ---------------------------------------------------------------------------
# Tests: indicators callback
# ---------------------------------------------------------------------------

class TestIndicators(unittest.TestCase):

    def test_returns_empty_when_no_click(self):
        result = indicators(None, None, None, None)
        self.assertEqual(result, [""])

    def test_returns_empty_when_no_val(self):
        result = indicators(1, None, None, None)
        self.assertEqual(result, [""])

    def test_downloads_and_returns_graph(self):
        real_df = _make_ohlcv_df(30)
        _yf_stub.download.return_value = real_df.set_index("Date")
        with patch("Stock.app.get_more", return_value=MagicMock()) as mock_more:
            result = indicators(1, None, None, "AAPL")
            self.assertTrue(mock_more.called)
            self.assertEqual(len(result), 1)


# ---------------------------------------------------------------------------
# Tests: forecast callback
# ---------------------------------------------------------------------------

class TestForecast(unittest.TestCase):

    def test_returns_empty_when_no_click(self):
        result = forecast(None, None, None)
        self.assertEqual(result, [""])

    def test_raises_when_click_but_no_val(self):
        from dash.exceptions import PreventUpdate
        with self.assertRaises(PreventUpdate):
            forecast(1, "5", None)

    def test_calls_prediction_and_returns_graph(self):
        with patch("Stock.app.prediction", return_value=MagicMock()) as mock_pred:
            result = forecast(1, "5", "AAPL")
            mock_pred.assert_called_once_with("AAPL", 6)
            self.assertEqual(len(result), 1)


# ---------------------------------------------------------------------------
# Tests: SVR internals (imported directly, not via prediction which needs network)
# ---------------------------------------------------------------------------

class TestSVR(unittest.TestCase):

    def _build_data(self, n=54):
        X = [[i] for i in range(n)]
        Y = np.linspace(100, 150, n).tolist()
        return X, Y

    def test_svr_fit_predict(self):
        from sklearn.svm import SVR
        X, Y = self._build_data()
        model = SVR(kernel="rbf", C=100, epsilon=0.01, gamma=0.1)
        model.fit(X, Y)
        preds = model.predict([[54], [55]])
        self.assertEqual(len(preds), 2)
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_svr_plausible_range(self):
        from sklearn.svm import SVR
        X, Y = self._build_data()
        model = SVR(kernel="rbf", C=100, epsilon=0.01, gamma=0.1)
        model.fit(X, Y)
        pred = model.predict([[54]])[0]
        self.assertGreater(pred, 50)
        self.assertLess(pred, 250)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
