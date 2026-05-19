"""
Unit tests for Stock Price Visualizer & Forecaster.

Imports real Stock modules with heavy dependencies patched at sys.modules BEFORE
import so coverage instruments the actual source files.
"""

import sys
import os
import types
import unittest
from unittest.mock import MagicMock, patch
from datetime import date, timedelta

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Install stubs BEFORE importing any Stock module
# ---------------------------------------------------------------------------

def _install_stubs():
    # model (app.py does `from model import prediction` at module level)
    model_mod = types.ModuleType("model")
    model_mod.prediction = MagicMock(return_value=MagicMock())
    sys.modules["model"] = model_mod

    # yfinance
    yf = MagicMock()
    sys.modules["yfinance"] = yf

    # plotly
    px = MagicMock()
    go = MagicMock()
    mock_fig = MagicMock()
    go.Figure.return_value = mock_fig
    go.Scatter = MagicMock()
    plotly_mod = types.ModuleType("plotly")
    plotly_mod.express = px
    plotly_mod.graph_objs = go
    sys.modules["plotly"] = plotly_mod
    sys.modules["plotly.express"] = px
    sys.modules["plotly.graph_objs"] = go

    # dash - make @app.callback a pass-through decorator
    real_dash_app = MagicMock()
    real_dash_app.callback = lambda *a, **kw: (lambda f: f)  # pass-through
    real_dash_app.server = MagicMock()
    real_dash_app.layout = None

    dash_mod = types.ModuleType("dash")
    dash_mod.Dash = MagicMock(return_value=real_dash_app)
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

    return yf, px, go, model_mod


_yf_stub, _px_stub, _go_stub, _model_stub = _install_stubs()

# Add project root so Stock package is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import real functions after stubs are in place
from Stock.app import get_stock_price_fig, get_more  # noqa: E402
from Stock.app import update_data, stock_price, indicators, forecast  # noqa: E402
from Stock.app import APP_VERSION  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv_df(n=30):
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

    def setUp(self):
        _px_stub.reset_mock()

    def test_calls_px_line(self):
        df = _make_ohlcv_df()
        get_stock_price_fig(df)
        self.assertTrue(_px_stub.line.called)

    def test_uses_close_and_open_columns(self):
        df = _make_ohlcv_df()
        get_stock_price_fig(df)
        _, kwargs = _px_stub.line.call_args
        self.assertIn("Close", kwargs.get("y", []))
        self.assertIn("Open", kwargs.get("y", []))

    def test_x_axis_is_date(self):
        df = _make_ohlcv_df()
        get_stock_price_fig(df)
        _, kwargs = _px_stub.line.call_args
        self.assertEqual(kwargs.get("x"), "Date")

    def test_returns_figure(self):
        df = _make_ohlcv_df()
        fig = get_stock_price_fig(df)
        self.assertIsNotNone(fig)


# ---------------------------------------------------------------------------
# Tests: get_more (EMA)
# ---------------------------------------------------------------------------

class TestGetMore(unittest.TestCase):

    def setUp(self):
        _px_stub.reset_mock()

    def test_adds_ema_column(self):
        df = _make_ohlcv_df()
        get_more(df)
        self.assertIn("EMA_20", df.columns)

    def test_ema_values_are_finite(self):
        df = _make_ohlcv_df()
        get_more(df)
        self.assertTrue(np.all(np.isfinite(df["EMA_20"].values)))

    def test_calls_px_scatter(self):
        df = _make_ohlcv_df()
        get_more(df)
        self.assertTrue(_px_stub.scatter.called)

    def test_ema_first_value_equals_first_close(self):
        df = _make_ohlcv_df()
        get_more(df)
        self.assertAlmostEqual(df["EMA_20"].iloc[0], df["Close"].iloc[0], places=5)

    def test_ema_monotonically_tracks_trend(self):
        df = _make_ohlcv_df(40)
        get_more(df)
        # With a strictly increasing Close, EMA_20 should also be increasing
        ema = df["EMA_20"].values
        self.assertTrue(np.all(np.diff(ema) >= 0))


# ---------------------------------------------------------------------------
# Tests: update_data callback
# ---------------------------------------------------------------------------

class TestUpdateData(unittest.TestCase):

    def test_returns_tuple_when_no_click(self):
        result = update_data(None, None)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 6)

    def test_last_three_none_on_default(self):
        result = update_data(None, None)
        self.assertIsNone(result[3])
        self.assertIsNone(result[4])
        self.assertIsNone(result[5])

    def test_default_includes_welcome_text(self):
        result = update_data(None, None)
        self.assertIn("PLEASE Enter a legitimate stock code", result[0])

    def test_raises_prevent_update_when_click_no_ticker(self):
        from dash.exceptions import PreventUpdate
        with self.assertRaises(PreventUpdate):
            update_data(1, None)

    def test_fetches_ticker_info_on_valid_submit(self):
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "logo_url": "http://logo.png",
            "shortName": "ACME Corp",
            "longBusinessSummary": "Makes everything.",
        }
        _yf_stub.Ticker.return_value = mock_ticker
        result = update_data(1, "ACME")
        self.assertEqual(result[0], "Makes everything.")
        self.assertEqual(result[1], "http://logo.png")
        self.assertEqual(result[2], "ACME Corp")
        self.assertIsNone(result[3])
        self.assertIsNone(result[4])
        self.assertIsNone(result[5])


# ---------------------------------------------------------------------------
# Tests: stock_price callback
# ---------------------------------------------------------------------------

class TestStockPrice(unittest.TestCase):

    def test_returns_empty_when_no_click(self):
        result = stock_price(None, None, None, None)
        self.assertEqual(result, [""])

    def test_raises_prevent_update_when_click_no_val(self):
        from dash.exceptions import PreventUpdate
        with self.assertRaises(PreventUpdate):
            stock_price(1, None, None, None)

    def test_downloads_with_date_range(self):
        real_df = _make_ohlcv_df(20)
        _yf_stub.download.return_value = real_df.set_index("Date")
        with patch("Stock.app.get_stock_price_fig", return_value=MagicMock()):
            result = stock_price(1, "2024-01-01", "2024-06-01", "AAPL")
            _yf_stub.download.assert_called()
            self.assertEqual(len(result), 1)

    def test_downloads_without_date_range(self):
        real_df = _make_ohlcv_df(20)
        _yf_stub.download.return_value = real_df.set_index("Date")
        with patch("Stock.app.get_stock_price_fig", return_value=MagicMock()):
            result = stock_price(1, None, None, "AAPL")
            _yf_stub.download.assert_called()
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

    def test_downloads_with_date_range(self):
        real_df = _make_ohlcv_df(30)
        _yf_stub.download.return_value = real_df.set_index("Date")
        with patch("Stock.app.get_more", return_value=MagicMock()):
            result = indicators(1, "2024-01-01", "2024-06-01", "AAPL")
            self.assertEqual(len(result), 1)

    def test_downloads_without_date_range(self):
        real_df = _make_ohlcv_df(30)
        _yf_stub.download.return_value = real_df.set_index("Date")
        with patch("Stock.app.get_more", return_value=MagicMock()):
            result = indicators(1, None, None, "AAPL")
            self.assertEqual(len(result), 1)


# ---------------------------------------------------------------------------
# Tests: forecast callback
# ---------------------------------------------------------------------------

class TestForecast(unittest.TestCase):

    def test_returns_empty_when_no_click(self):
        result = forecast(None, None, None)
        self.assertEqual(result, [""])

    def test_raises_prevent_update_when_click_no_val(self):
        from dash.exceptions import PreventUpdate
        with self.assertRaises(PreventUpdate):
            forecast(1, "5", None)

    def test_calls_prediction_with_n_plus_one(self):
        _model_stub.prediction.reset_mock()
        _model_stub.prediction.return_value = MagicMock()
        result = forecast(1, "5", "AAPL")
        _model_stub.prediction.assert_called_once_with("AAPL", 6)
        self.assertEqual(len(result), 1)

    def test_calls_prediction_with_different_n_days(self):
        _model_stub.prediction.reset_mock()
        _model_stub.prediction.return_value = MagicMock()
        forecast(1, "10", "TSLA")
        _model_stub.prediction.assert_called_once_with("TSLA", 11)


# ---------------------------------------------------------------------------
# Tests: SVR model internals
# ---------------------------------------------------------------------------

class TestSVR(unittest.TestCase):

    def _data(self, n=54):
        X = [[i] for i in range(n)]
        Y = np.linspace(100, 150, n).tolist()
        return X, Y

    def test_svr_fit_predict(self):
        from sklearn.svm import SVR
        X, Y = self._data()
        model = SVR(kernel="rbf", C=100, epsilon=0.01, gamma=0.1)
        model.fit(X, Y)
        preds = model.predict([[54], [55]])
        self.assertEqual(len(preds), 2)
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_svr_plausible_range(self):
        from sklearn.svm import SVR
        X, Y = self._data()
        model = SVR(kernel="rbf", C=100, epsilon=0.01, gamma=0.1)
        model.fit(X, Y)
        pred = model.predict([[54]])[0]
        self.assertGreater(pred, 50)
        self.assertLess(pred, 250)

    def test_train_test_split_ratio(self):
        from sklearn.model_selection import train_test_split
        X = [[i] for i in range(60)]
        Y = list(range(60))
        x_train, x_test, _, _ = train_test_split(X, Y, test_size=0.1, shuffle=False)
        self.assertEqual(len(x_test), 6)
        self.assertEqual(len(x_train), 54)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Tests: APP_VERSION
# ---------------------------------------------------------------------------

class TestAppVersion(unittest.TestCase):

    def test_version_is_string(self):
        self.assertIsInstance(APP_VERSION, str)

    def test_version_format(self):
        parts = APP_VERSION.split(".")
        self.assertEqual(len(parts), 3)
        self.assertTrue(all(p.isdigit() for p in parts))
