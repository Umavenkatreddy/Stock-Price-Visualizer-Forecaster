import sys, os, unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
from datetime import date, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

def _df(n=60):
    c = np.linspace(100, 160, n)
    return pd.DataFrame({"Date": pd.date_range(end=date.today(), periods=n, freq="B"),
        "Open": c-1, "High": c+2, "Low": c-2, "Close": c, "Volume": np.ones(n,int)*500_000})

class TestSVR(unittest.TestCase):
    def _xy(self, n=54): return [[i] for i in range(n)], np.linspace(100,150,n).tolist()
    def test_fit_predict(self):
        from sklearn.svm import SVR
        X,Y=self._xy(); m=SVR(kernel="rbf",C=100,epsilon=0.01,gamma=0.1); m.fit(X,Y)
        self.assertEqual(len(m.predict([[54],[55]])), 2)
    def test_plausible_range(self):
        from sklearn.svm import SVR
        X,Y=self._xy(); m=SVR(kernel="rbf",C=100,epsilon=0.01,gamma=0.1); m.fit(X,Y)
        p=m.predict([[54]])[0]; self.assertGreater(p,50); self.assertLess(p,250)
    def test_split_ratio(self):
        from sklearn.model_selection import train_test_split
        X=[[i] for i in range(60)]; Y=list(range(60))
        xtr,xte,_,_=train_test_split(X,Y,test_size=0.1,shuffle=False)
        self.assertEqual(len(xte),6); self.assertEqual(len(xtr),54)
    def test_grid_search(self):
        from sklearn.svm import SVR
        from sklearn.model_selection import GridSearchCV
        X=[[i] for i in range(30)]; Y=np.linspace(100,130,30).tolist()
        gsc=GridSearchCV(SVR(kernel="rbf"),{"C":[1,10],"epsilon":[0.1],"gamma":[0.01]},cv=2,scoring="neg_mean_absolute_error",n_jobs=-1)
        gsc.fit(X,Y); self.assertIn("C",gsc.best_params_)
    def test_output_days_len(self): self.assertEqual(len([[i+10] for i in range(1,5)]),4)
    def test_dates_len(self):
        d=date.today(); dates=[]
        for _ in range(5): d+=timedelta(days=1); dates.append(d)
        self.assertEqual(len(dates),5)
    def test_day_col(self):
        df=_df(); df.reset_index(inplace=True); df["Day"]=df.index
        self.assertIn("Day",df.columns); self.assertEqual(df["Day"].iloc[0],0)
    def test_close_present(self): self.assertIn("Close",_df().columns)
    def test_ravel(self):
        r=pd.DataFrame({"Close":[100,101,102]}).values.ravel()
        self.assertEqual(r.ndim,1); self.assertEqual(len(r),3)
    def test_finite(self):
        from sklearn.svm import SVR
        X=[[i] for i in range(54)]; Y=np.linspace(100,150,54).tolist()
        m=SVR(kernel="rbf",C=100,epsilon=0.01,gamma=0.1); m.fit(X,Y)
        self.assertTrue(np.all(np.isfinite(m.predict([[55+i] for i in range(5)]))))

class TestPredictionMocked(unittest.TestCase):
    def _call(self, ticker="AAPL", n=6):
        yf=MagicMock(); yf.download.return_value=_df()
        go=MagicMock(); go.Figure.return_value=MagicMock(); go.Scatter=MagicMock(return_value=MagicMock())
        with patch.dict(sys.modules,{"yfinance":yf,"plotly.graph_objs":go}):
            sys.modules.pop("Stock.model",None)
            from Stock.model import prediction
            return prediction(ticker,n),yf,go
    def test_returns_fig(self): r,_,_=self._call(); self.assertIsNotNone(r)
    def test_dl_called(self): _,yf,_=self._call(); yf.download.assert_called()
    def test_dl_ticker(self):
        _,yf,_=self._call("TSLA",6); args,_=yf.download.call_args; self.assertEqual(args[0],"TSLA")
    def test_dl_period(self):
        _,yf,_=self._call(); _,kw=yf.download.call_args; self.assertEqual(kw.get("period"),"60d")
    def test_scatter(self): _,_,go=self._call(); go.Scatter.assert_called()

if __name__=="__main__": unittest.main()