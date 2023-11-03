from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS, NBEATSx, RNN, LSTM, GRU, MLP
from neuralforecast.utils import AirPassengersDF
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import statsforecast.models as stats_models
import numpy as np
import pandas as pd
from typing import Any
import prophet


class TimeSeriesDataFrameParser:
    
    """
    converts time series to a dataframe with the following columns 
    and index can be parsed to datetime (or is already):
    [y1, y2, y3, ..., yn]
    to a dataframe with the following columns:
    [ds, unique_idx, y]
    
    EG 
    | index | y1 | y2 |
    |10-21-2023 | 1 | 2 |
    becomes
    | ds | unique_id | y |
    |10-21-2023 | 1 | 1 |
    |10-21-2023 | 2 | 2 |
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy() if not isinstance(df, pd.Series) else df.to_frame()
        self.parse_dates()
        self.parsed = None
        
    def parse_dates(self):
        self.df.index = pd.to_datetime(self.df.index)
        return self
    
    def parse_as_type(self, idx: int, type: Any)->pd.DataFrame:
        """
        Parse column to given type

        :param idx: colum index
        :type idx: int
        :param type: type to convert to
        :type type: Any
        :return: converted column
        :rtype: pd.DataFrame
        """
        return self.df.iloc[:, idx].astype(type)
    
    def parse_column_as_type(self, col_name: str, type: Any)->pd.DataFrame:
        """
        Parse column to given type

        :param col_name: column name
        :type col_name: str
        :param type: type to convert to
        :type type: Any
        :return: converted column
        :rtype: pd.DataFrame
        """
        return self.df[col_name].astype(type)
    
    def parse_as_type_inplace(self, idx: int, type: Any):
        """
        Parse column to given type in place

        :param idx: colum index
        :type idx: int
        :param type: type to convert to
        :type type: Any
        :return: self
        :rtype: self
        """
        self.df.iloc[:, idx] = self.parse_as_type(idx, type)
        return self

    def parse_one_column(self, idx: int, copy=True)->pd.DataFrame:
        values = [pd.to_datetime(self.df.index).values, np.ones(self.df.shape[0], dtype=int) * idx, self.df.iloc[:, idx].values.astype(float)]
        parsed = pd.DataFrame(values, copy=copy).T
        parsed.columns = ["ds", "unique_id", "y"]
        return parsed
    
    def parse(self)->pd.DataFrame:
        if self.parsed is None:
            self.parsed = pd.concat([self.parse_one_column(i) for i in range(len(self.df.columns))])
        return self.parsed
    
    def parse_train_test_split(self, pct_train=0.8):
        parsed = self.parse() if self.parsed is None else self.parsed
        train, test = [], []
        for id, group in parsed.groupby("unique_id"):
            train.append(group.iloc[:int(len(group) * pct_train)])
            test.append(group.iloc[int(len(group) * pct_train):])

        train = pd.concat(train)
        test = pd.concat(test)
        return train, test
    
    def id_iter(self):
        parsed = self.parse() if self.parsed is None else self.parsed
        for _, group in parsed.groupby("unique_id"):
            yield group



def get_default_stats_models():
    import statsforecast.models as stats_models
    
    models = [
        stats_models.AutoTheta(),
        stats_models.AutoARIMA(),
        stats_models.AutoETS(),
        stats_models.AutoCES(),
        stats_models.DynamicOptimizedTheta(),
    ]
    return models

def get_default_stats_vol_models():
    models = [
        stats_models.GARCH(1,1),
        stats_models.ARCH()
        ]
    return models

def stats_forecast(x, horizon, freq, models=None, confidence=95):
    models = get_default_stats_models() if models is None else models
    
    sf = StatsForecast(
        models=models,
        freq =freq
    )

    sf.fit(x)
    pred = sf.predict(h=horizon, level=[confidence])
    return sf, pred


def prophet_predict(df, pred_dates, **kwargs):
    pred = []
    models = []
    for id, group in df.groupby("unique_id"):
        proph = prophet.Prophet(**kwargs)
        proph.fit(group)
        group_pred = proph.predict(pred_dates)
        group_pred["unique_id"] = id
        pred.append(group_pred)
        models.append(proph)
    return models, pd.concat(pred)



def get_default_deep_models(horizon, max_steps):
    models = [
            NBEATS(input_size=4 * horizon,  h=horizon, max_steps=max_steps),
            NHITS(input_size=4 * horizon, h=horizon, max_steps=max_steps),
            LSTM(input_size=4 * horizon, h=horizon, max_steps=max_steps, encoder_n_layers=16, encoder_hidden_size=500, ),
            GRU(input_size=4 * horizon, h=horizon, max_steps=max_steps, encoder_n_layers=16, encoder_hidden_size=500),
            RNN(input_size=4 * horizon, h=horizon, max_steps=max_steps, encoder_n_layers=16, encoder_hidden_size=500),
            MLP(input_size=4 * horizon, h=horizon, max_steps=max_steps, hidden_size=1024, num_layers=4),
          ]
    return models

def deep_predict(df, models = None, freq="D", horizon=1, max_steps=1000, **kwargs):
    models = get_default_deep_models(horizon, max_steps) if models is None else models
    nf = NeuralForecast(models=models, freq=freq)
    nf.fit(df)
    pred = nf.predict().reset_index()
    return nf, pred

if __name__ == "__main__":
    data = pd.read_csv("AAPL.csv", index_col="Date", parse_dates=True)
    data = TimeSeriesDataFrameParser(data)
    train, test = data.parse_train_test_split()
    
    sf, stats_pred = stats_forecast(train, test.shape[0], "D", )
    print(stats_pred.head())
    
    proph_models, proph_pred = prophet_predict(train, test)
    print(proph_pred.head())
    
    deep_models, deep_pred = deep_predict(train, horizon=test.shape[0])
    print(deep_pred.head())