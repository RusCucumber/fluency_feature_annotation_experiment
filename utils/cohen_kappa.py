from typing import Union, Tuple, List

import numpy as np
import pandas as pd
import pyper

class RCohenKappa:
    def __init__(self, debug: bool =False):
        self._r = pyper.R(use_pandas=True, use_numpy=True, dump_stdout=debug)
        self._r("library(psych)")
        self._r("library(kappaSize)")

        self.__debug = debug
    
    def generate_df(
            self, 
            x: Union[list, np.ndarray],
            y: Union[list, np.ndarray],
    ) -> pd.DataFrame:
        if len(x) != len(y):
            raise RuntimeError(f"Length of x ({len(x)}) and y ({len(y)}) is different.")

        data = np.array([x, y]).T
        df = pd.DataFrame(data)

        if self.__debug:
            print(df.head())

        return df

    def power_analysis(
            self,
            kappa_null: float,
            kappa_alt: float,
            props: Union[float, List[float], np.ndarray],
            alpha: float =0.05,
            beta: float =0.8,
            raters: int =2
    ):  
        if raters > 6:
            raise ValueError(f"N raters should be 2 <= raters <= 6")
        

        args = f"kappa0={kappa_null}, kappa1={kappa_alt}, alpha={alpha}, power={beta}, raters={raters}"

        if isinstance(props, float):
            args += f", props={props}"
            self._r.run(f"res <- PowerBinary({args})")
            n = self._r.get("res$N")

            return n

        if len(props) > 5:
            raise ValueError(f"N categories should be less than 6")

        if not isinstance(props, (list, np.ndarray)):
            raise TypeError(f"props with type \"{type(props)}\" is not supported.")
            
        if sum(props) != 1.0:
            raise RuntimeError(f"Sum of props is not 1 but {sum(props)}")
        
        args_props = f", c({', '.join([str(p) for p in props])})"
        args += args_props

        if len(props) == 2:
            cmd = f"PowerBinary({args})"
        else:
            cmd = f"Power{len(props)}Cats({args})"
        
        self._r.run(f"res <- {cmd}")
        n = self._r.get("res$N")

        return n

    def cohen_kappa(
            self,
            x: Union[list, np.ndarray],
            y: Union[list, np.ndarray],
            weighted: bool =False
    ) -> Tuple[float, float, float, float]:
        df = self.generate_df(x, y)
        self._r.assign("df", df)
        self._r("kappa <- cohen.kappa(df)")
        self._r("CI <- data.frame(kappa$confid)")
        
        if weighted:
            kappa = self._r.get("kappa$weighted.kappa")
            var = self._r.get("kappa$var.weighted")
            lower = self._r.get("CI['weighted kappa', 'lower']")
            upper = self._r.get("CI['weighted kappa', 'upper']")
            return kappa, var, lower, upper

        kappa = self._r.get("kappa$kappa")
        var = self._r.get("kappa$var.kappa")
        lower = self._r.get("CI['unweighted kappa', 'lower']")
        upper = self._r.get("CI['unweighted kappa', 'upper']")
        return kappa, var, lower, upper
        
if __name__ == "__main__":
    cohen_kappa = RCohenKappa(debug=False)

    x = [0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1]
    y = [0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0]

    kappa, var, lower, upper = cohen_kappa.cohen_kappa(x, y)
    print(f"kappa = {kappa:.03f}")
    print(f"CI    = |{lower:.03f} - {upper:.03f}|")
    print(f"var   = {var:.03f}")

    x = [0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 2, 2, 2]
    y = [0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 2, 1, 0]

    kappa, var, lower, upper = cohen_kappa.cohen_kappa(x, y)
    print(f"kappa = {kappa:.03f}")
    print(f"CI    = |{lower:.03f} - {upper:.03f}|")
    print(f"var   = {var:.03f}")

    n = cohen_kappa.power_analysis(
        kappa_null=0.3, kappa_alt=0.61,
        props=0.05, alpha=0.05, beta=0.8, raters=2
    )
    print(f"Minimum N = {n}")

    n = cohen_kappa.power_analysis(
        kappa_null=0.3, kappa_alt=0.61,
        props=[0.1, 0.3, 0.6], alpha=0.05, beta=0.8, raters=2
    )
    print(f"Minimum N = {n}")