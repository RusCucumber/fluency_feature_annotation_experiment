from typing import Union, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyper
import torch
from torch import Tensor, nn

class ManyFacetRaschAnalyzer:
    def __init__(self, debug: bool =False):
        self._r = pyper.R(use_pandas=True, use_numpy=True, dump_stdout=debug)
        self._r('library(TAM)')

        self.__debug = debug

    def fit(self, 
            data: Union[str, Path, pd.DataFrame], 
            target_columns: Union[list, np.ndarray]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if not isinstance(target_columns, (list, np.ndarray)):
            raise ValueError(f"target_columns must be list or numpy.ndarray but {type(target_columns)} is given")

        df = self.__to_df(data)[target_columns]
        self._r.assign('df', df)
        self._r('model <- TAM::tam.mml(df, irtmodel="RSM")')

        if self.__debug:
            self._r('summary(model)')

        self._r('rater_estimates <- model$xsi')
        self._r('tam.fit(model)')
        self._r('threshold <- tam.threshold(model)')
        self._r('person_ability <- tam.wle(model)')
        self._r('df_result <- cbind(df, person_ability)')

        df_threshold = pd.DataFrame(self._r.get('threshold'), index=target_columns)
        df_result = self._r.get('df_result')
        df_item_params = self._r.get("model$item_irt")

        if df_threshold is None or df_result is None or df_item_params is None:
            raise RuntimeError(f"R processing error.")

        df_result.columns = [col.replace(" ", "") for col in df_result.columns]
        df_threshold.columns = [f"{int(c)}|{int(c) + 1}" for c in df_threshold.columns]
        df_item_params.columns = [col.replace(" ", "") for col in df_item_params.columns]
        df_item_params = df_item_params.set_index("item", drop=True)
        df_item_params.index = [str(idx).replace("b", "").replace("'", "") for idx in df_item_params.index] 

        return df_result, df_threshold, df_item_params
        
    def __to_df(self, data: Union[str, Path, pd.DataFrame]):
        if isinstance(data, (str, Path)):
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise ValueError(f"data must be path like object or pandas.DataFrame but {type(data)} is given.")

        return df

class ManyFacetRaschModel(nn.Module):
    def __init__(self, n_categories: int) -> None:
        super().__init__()

        # カテゴリ k-1 からカテゴリ k への遷移の難易度
        self.d = nn.Parameter(
            torch.randn(n_categories, dtype=torch.float)
        )

    def numerator(self, theta: Tensor, b: Tensor, beta: Tensor, k: int, x_max=0):
        nmrtr = torch.exp(
            (theta - b - beta) * (k + 1) - torch.sum(self.d[:k + 1]) - x_max
        )
        return nmrtr

    def denominator(self, theta: Tensor, b: Tensor, beta: Tensor):
        dnmntr = []

        for l in range(len(self.d)):
            dnmntr.append(
                (theta - b - beta) * (l + 1) - torch.sum(self.d[:l + 1])
            )
        
        dnmntr = torch.cat(dnmntr, dim=1)
        x_max, _ = dnmntr.max(dim=1, keepdim=True)
        dnmntr = torch.sum(
            torch.exp(dnmntr - x_max), dim=1, keepdim=True
        )

        return dnmntr, x_max

    def forward(self, theta: Tensor, b: Tensor, beta: Tensor):
        P_ijrk = []

        dnmntr, x_max = self.denominator(theta, b, beta)
        for k in range(len(self.d)):
            nmrtr = self.numerator(theta, b, beta, k, x_max)
            p = nmrtr / dnmntr
            P_ijrk.append(p)

        return torch.cat(P_ijrk, dim=1)

def visualize_item_characteristics_curves(
        ax: plt.Axes,
        df_item_params: pd.Series,
        n_categories: int,
) -> plt.Axes: #TODO: 項目特性曲線を描画する処理の作成
    mfr = ManyFacetRaschModel(n_categories)
    
    d = [-0.0] + df_item_params["tau.Cat1":].tolist()
    d = torch.nn.Parameter(
        torch.tensor(d, dtype=torch.float)
    )
    mfr.d = d

    theta = np.linspace(-10, 10, 1000).reshape(1000, 1)
    theta = torch.tensor(theta, dtype=torch.float)
    
    b = torch.zeros_like(theta)
    
    beta = torch.zeros_like(theta) + df_item_params["beta"]
    
    mfr.eval()
    for params in mfr.parameters():
        params.requires_grad = False

    with torch.no_grad():
        p = mfr(theta, b, beta)

    for k in range(n_categories):
        ax.plot(theta, p[:, k], label=f"cat={k}")
    ax.legend()

    return ax

def logit_2_rating(logits: Union[list, np.ndarray, pd.Series], threshold_path: Path) -> np.ndarray:
    threshold = pd.read_csv(threshold_path, index_col=0).mean().values

    if isinstance(logits, list):
        logits = np.array(logits)
    if isinstance(logits, pd.Series):
        logits = logits.to_numpy()
    
    rating = np.zeros_like(logits, dtype=np.int64)
    for score, (m1, m2) in enumerate(zip(threshold[:-1], threshold[1:])):
        mask = (logits > m1) & (logits <= m2)
        rating[mask] = 1 + score

    rating[logits > m2] = 2 + score

    return rating

def merge_rating(rating: np.ndarray, merge_from: int, merge_to: int) -> np.ndarray:
    merged_rating = rating.copy()

    mask = (rating == merge_from)
    merged_rating[mask] = merge_to

    return merged_rating