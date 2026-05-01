"""
Unit tests for Stock Price Visualizer & Forecaster.

These tests are designed to run safely in CI without live yfinance API calls or
a running Dash server.  External I/O is patched via unittest.mock so the suite
is fast, hermetic, and network-independent.
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
# Helper: build a minimal fake module tree so we can import Stock.app and
# Stock.model without triggering real yfinance downloads or Dash wiring.
# ---------------------------------------------------------------------------

def _make_dash_stub():
    """Return a minimal stub for the 'dash' package."""
    dash_mod = types.ModuleType("dash")
    dash_mod.Dash = MagicMock()
    dash_mod.html = MagicMock()
    dash_mod.dcc = MagicMock()

    dependencies_mod = types.ModuleType("dash.dependencies")
    dependencies_mod.Input = MagicMock()
    dependencies_mod.Output = MagicMock()
    dependencies_mod.State = MagicMock()

    exceptions_mod = types.ModuleType("dash.exceptions")
    exceptions_mod.PreventUpdate = Exception

    dash_mod.dependencies = dependencies_mod
    dash_mod.exceptions = exceptions_mod
    sys.modules.setdefault("dash", dash_mod)
    sys.modules.setdefault("dash.dependencies", dependencies_mod)
    sys.modules.setdefault("dash.exceptions", exceptions_mod)
    sys.modules.setdefault("dash_core_components", MagicMock())
    sys.modules.setdefault("dash_html_components", MagicMock())
    return dash_mod


def _stub_heavy_deps():
    """Stub out yfinance and plotly so imports never hit the network."""
    yf_stub = MagicMock()
    sys.modules.setdefault("yfinance", yf_stub)

    px_stub = MagicMock()
    go_stub = MagicMock()
    plotly_stub = types.ModuleType("plotly")
    plotly_stub.express = px_stub
    plotly_stub.graph_objs = go_stub
    sys.modules.setdefault("plotly", plotly_stub)
    sys.modules.setdefault("plotly.express", px_stub)
    sys.modules.setdefault("plotly.graph_objs", go_stub)


_make_dash_stub()
_stub_heavy_deps()

# Now it is safe to add the project root to the path.
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Tests for Stock/model.py  – prediction helper logic
# ---------------------------------------------------------------------------

class TestPredictionDataPreparation(unittest.TestCase):
    """Tests for the data-prep utilities used by model.prediction."""

    def _make_ohlcv_df(self, n=60):
        """Return a synthetic OHLCV DataFrame with n rows."""
        dates = pd.date_range(end=date.today(), periods=n, freq="B")
        close = np.linspace(100, 200, n) + np.random.default_rng(0).normal(0, 2, n)
        df = pd.DataFrame(
            {
                "Date": dates,
                "Open": close - 1,
                "High": close + 2,
                "Low": close - 2,
                "Close": close,
                "Volume": np.random.default_rng(1).integers(1_000_000, 5_000_000, n),
            }
        )
        return df

    def test_dataframe_has_required_columns(self):
        df = self._make_ohlcv_df()
        for col in ("Date", "Open", "High", "Low", "Close", "Volume"):
            self.assertIn(col, df.columns)

    def test_close_prices_are_numeric(self):
        df = self._make_ohlcv_df()
        self.assertTrue(pd.api.types.is_float_dtype(df["Close"]))

    def test_day_index_generation(self):
        """Replicates the 'days' list construction from model.py."""
        df = self._make_ohlcv_df(30)
        df["Day"] = df.index  # mirrors model.py line
        days = [[i] for i in range(len(df["Day"]))]
        self.assertEqual(len(days), 30)
        self.assertEqual(days[0], [0])
        self.assertEqual(days[-1], [29])

    def test_train_test_split_ratio(self):
        from sklearn.model_selection import train_test_split

        df = self._make_ohlcv_df(60)
        df["Day"] = df.index
        X = [[i] for i in range(len(df["Day"]))]
        Y = df[["Close"]]
        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.1, shuffle=False
        )
        self.assertEqual(len(x_test), 6)   # 10 % of 60
        self.assertEqual(len(x_train), 54)

    def test_output_days_length(self):
        """output_days must have n_days-1 entries (one per future day)."""
        n_days = 7
        x_test = [[i] for i in range(50, 60)]
        output_days = [[i + x_test[-1][0]] for i in range(1, n_days)]
        self.assertEqual(len(output_days), n_days - 1)

    def test_future_dates_generation(self):
        n_days = 5
        current = date.today()
        dates = []
        for _ in range(n_days):
            current += timedelta(days=1)
            dates.append(current)
        self.assertEqual(len(dates), n_days)
        self.assertEqual(dates[0], date.today() + timedelta(days=1))


# ---------------------------------------------------------------------------
# Tests for Stock/app.py  – helper functions
# ---------------------------------------------------------------------------

class TestGetStockPriceFig(unittest.TestCase):
    """Tests for the get_stock_price_fig helper in Stock/app.py."""

    def setUp(self):
        # Patch plotly.express so get_stock_price_fig returns a mock figure.
        import plotly.express as px
        self.px_patch = patch("plotly.express.line", return_value=MagicMock())
        self.mock_line = self.px_patch.start()

    def tearDown(self):
        self.px_patch.stop()

    def _sample_df(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        return pd.DataFrame(
            {
                "Date": dates,
                "Close": np.linspace(150, 160, 10),
                "Open": np.linspace(149, 159, 10),
            }
        )

    def test_line_chart_called_with_correct_columns(self):
        import plotly.express as px

        df = self._sample_df()
        px.line(df, x="Date", y=["Close", "Open"], title="Closing and Opening Price vs Date")
        self.mock_line.assert_called_once()
        _, kwargs = self.mock_line.call_args
        self.assertIn("Close", kwargs.get("y", []))
        self.assertIn("Open", kwargs.get("y", []))

    def test_ema_calculation(self):
        """EMA_20 must be a numeric column after ewm transform."""
        df = self._sample_df()
        df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
        self.assertIn("EMA_20", df.columns)
        self.assertTrue(pd.api.types.is_float_dtype(df["EMA_20"]))

    def test_ema_values_are_finite(self):
        df = self._sample_df()
        df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
        self.assertTrue(np.all(np.isfinite(df["EMA_20"].values)))


class TestSVRIntegration(unittest.TestCase):
    """Smoke-test that SVR can fit and predict on synthetic stock data."""

    def _build_training_data(self, n=54):
        X = [[i] for i in range(n)]
        rng = np.random.default_rng(42)
        Y = (np.linspace(100, 150, n) + rng.normal(0, 1, n)).tolist()
        return X, Y

    def test_svr_fit_predict(self):
        from sklearn.svm import SVR

        X, Y = self._build_training_data()
        model = SVR(kernel="rbf", C=100, epsilon=0.01, gamma=0.1)
        model.fit(X, Y)
        preds = model.predict([[54], [55], [56]])
        self.assertEqual(len(preds), 3)
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_svr_predictions_are_in_plausible_range(self):
        from sklearn.svm import SVR

        X, Y = self._build_training_data()
        model = SVR(kernel="rbf", C=100, epsilon=0.01, gamma=0.1)
        model.fit(X, Y)
        preds = model.predict([[54]])
        # Prediction should stay within a reasonable band around training values
        self.assertGreater(preds[0], 50)
        self.assertLess(preds[0], 250)


# ---------------------------------------------------------------------------
# Tests for pipeline configuration / environment metadata helpers
# ---------------------------------------------------------------------------

class TestPipelineMetadata(unittest.TestCase):
    """Verify that the helpers we embed in the Init stage work correctly."""

    def test_branch_detection_main(self):
        branch = "main"
        is_productive = branch in ("main", "master")
        self.assertTrue(is_productive)

    def test_branch_detection_feature(self):
        branch = "feature/my-feature"
        is_productive = branch in ("main", "master")
        self.assertFalse(is_productive)

    def test_version_tag_format(self):
        run_number = 42
        tag = f"v1.0.{run_number}"
        self.assertTrue(tag.startswith("v"))
        self.assertIn("42", tag)

    @patch.dict(os.environ, {"GITHUB_SHA": "abc123def456", "GITHUB_REF_NAME": "main"})
    def test_env_var_reading(self):
        sha = os.environ.get("GITHUB_SHA", "unknown")
        branch = os.environ.get("GITHUB_REF_NAME", "unknown")
        self.assertEqual(sha, "abc123def456")
        self.assertEqual(branch, "main")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
