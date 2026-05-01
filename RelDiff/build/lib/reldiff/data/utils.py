from typing import Any, Literal, Tuple, Optional
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

CatEncoding = Literal["one-hot", "ordinal"]
Normalization = Literal["standard", "quantile", "minmax", "uniform"]
NaN = float("nan")


def get_category_proportions(
    x_cat: torch.LongTensor, num_classes: list, add_mask: bool = True
) -> torch.Tensor:
    proportions_list = []
    for i, num_class in enumerate(num_classes):
        x_cat_onehot = nn.functional.one_hot(
            x_cat[:, i], num_classes=num_class + 1 if add_mask else num_class
        ).float()
        proportions = x_cat_onehot.mean(0)
        if add_mask:
            assert proportions[-1] == 0.0
        proportions_list.append(proportions)

    return proportions_list


class SigmaScaler(preprocessing.StandardScaler):
    def __init__(self, sigma_data=1.0, **kwargs):
        super(SigmaScaler, self).__init__(**kwargs)
        self.sigma_data = sigma_data

    def fit(self, X, y=None):
        return super(SigmaScaler, self).fit(X, y)

    def transform(self, X, y=None):
        return super(SigmaScaler, self).transform(X, y) * self.sigma_data

    def inverse_transform(self, X, y=None):
        return super(SigmaScaler, self).inverse_transform(X / self.sigma_data, y)


class TensorDequantizer(torch.nn.Module):
    def __init__(self, eps=1e-7, loc=0.0, scale=1.0):
        super(TensorDequantizer, self).__init__()
        self.left_edges_y = nn.ParameterDict()
        self.left_edges_x = nn.ParameterDict()
        self.right_edges_y = nn.ParameterDict()
        self.right_edges_x = nn.ParameterDict()
        self.slopes = nn.ParameterDict()
        self.eps = eps
        self.normal = torch.distributions.Normal(loc, scale)

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def fit(self, x):
        assert len(x.shape) == 2
        # for each column in x, get bounds necessary to do dequantization
        for i in range(x.shape[1]):
            self.get_edges(x[:, i], str(i))

        # set all parameters as non-trainable
        for param in self.parameters():
            param.requires_grad = False

        return self

    @torch.no_grad()
    def transform(self, x):
        output = []

        for i in range(x.shape[1]):
            z = x[:, i].to(self.device)
            idx = torch.searchsorted(
                self.left_edges_x[str(i)], z.contiguous(), side="right"
            )
            idx = idx - 1
            idx[idx < 0] = 0

            # clip inputs to learned domain
            inp = z.clone()
            inp[idx == 0] = self.left_edges_x[str(i)][0]
            inp[idx == len(self.left_edges_x[str(i)]) - 1] = self.left_edges_x[str(i)][
                -1
            ]

            left_x = torch.take(self.left_edges_x[str(i)], idx)
            right_x = torch.take(self.right_edges_x[str(i)], idx)

            # add uniform noise in [0, next value)
            inp = inp + torch.rand(inp.shape[0], device=right_x.device) * (
                right_x - inp
            )

            # linearly interpolate edges
            slope = torch.take(self.slopes[str(i)], idx)
            left_y = torch.take(self.left_edges_y[str(i)], idx)
            interpolation = left_y + (inp - left_x) * slope

            # transform to normal distribution, avoiding values too close to 0.0 and 1.0
            y = self.normal.icdf(interpolation)
            clip_min = self.normal.icdf(
                torch.scalar_tensor(self.eps - np.spacing(1), device=y.device)
            )
            clip_max = self.normal.icdf(
                torch.scalar_tensor(1 - (self.eps - np.spacing(1)), device=y.device)
            )
            y = torch.clamp(y, clip_min, clip_max)
            output.append(y)

        return torch.stack(output, 1)

    @torch.no_grad()
    def inverse_transform(self, x):
        output = []
        for i in range(x.shape[1]):
            y = self.normal.cdf(x[:, i].to(self.device))
            idx = torch.searchsorted(self.left_edges_y[str(i)], y, side="right")
            idx = idx - 1
            idx[idx < 0] = 0
            output.append(self.left_edges_x[str(i)][idx])

        return torch.stack(output, 1)

    def get_edges(self, x, i):
        z, _ = torch.sort(x)
        nobs = len(z)
        vals, counts = torch.unique(z, return_counts=True)
        empirical_pdf = counts / nobs

        self.left_edges_x[i] = vals
        self.right_edges_x[i] = torch.cat((vals[1:], (vals[-1] + 1.0)[None]))
        self.right_edges_y[i] = empirical_pdf.cumsum(0)
        self.left_edges_y[i] = torch.cat(
            (torch.zeros((1,)), self.right_edges_y[i][:-1])
        )
        self.slopes[i] = (self.right_edges_y[i] - self.left_edges_y[i]) / (
            self.right_edges_x[i] - self.left_edges_x[i]
        )

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def forward(self, x, inverse=False):
        if inverse:
            return self.inverse_transform(x)
        else:
            return self.transform(x)


def transform_datetime(
    df: pd.DataFrame, column: str, min_datetime: np.datetime64
) -> Tuple[pd.DataFrame, list[str]]:
    nulls = df[column].isnull()
    df[column] = pd.to_datetime(df[column], errors="coerce")
    diff: np.timedelta64 = df[column] - min_datetime
    df[f"{column}_date"] = diff.dt.days
    df.loc[nulls, f"{column}_date"] = NaN
    datetime_columns = [f"{column}_date"]
    # check if time is needed
    if df[column].dt.hour.sum() > 0:
        df[f"{column}_time"] = diff.dt.seconds
        df.loc[nulls, f"{column}_time"] = NaN
        datetime_columns.append(f"{column}_time")
    return df.drop(columns=[column]), datetime_columns


def transform_datetime_old(
    df: pd.DataFrame, column: str
) -> Tuple[pd.DataFrame, list[str]]:
    nulls = df[column].isnull()
    df[f"{column}_Year"] = df[column].dt.year
    df[f"{column}_Month"] = df[column].dt.month
    df[f"{column}_Day"] = df[column].dt.day
    df.loc[nulls, f"{column}_Year"] = NaN
    df.loc[nulls, f"{column}_Month"] = NaN
    df.loc[nulls, f"{column}_Day"] = NaN
    datetime_columns = [f"{column}_Year", f"{column}_Month", f"{column}_Day"]
    # check if hours, minutes, seconds are needed
    if df[column].dt.hour.sum() > 0:
        df[f"{column}_Hour"] = df[column].dt.hour
        df[f"{column}_Minute"] = df[column].dt.minute
        df[f"{column}_Second"] = df[column].dt.second
        df.loc[nulls, f"{column}_Hour"] = NaN
        df.loc[nulls, f"{column}_Minute"] = NaN
        df.loc[nulls, f"{column}_Second"] = NaN
        datetime_columns.extend(
            [f"{column}_Hour", f"{column}_Minute", f"{column}_Second"]
        )
    return df.drop(columns=[column]), datetime_columns


# adapted from https://github.com/amazon-science/tabsyn/blob/main/src/data.py#L211
def normalize(
    X: pd.DataFrame,
    X_missing: np.ndarray,
    normalization: Normalization,
    seed: Optional[int],
    return_normalizer: bool = False,
    standardize: bool = False,
    sigma_data: float = 1.0,
) -> Tuple[pd.DataFrame, Optional[Any]]:
    if normalization == "standard":
        normalizer = preprocessing.StandardScaler()
    elif normalization == "minmax":
        normalizer = preprocessing.MinMaxScaler()
    elif normalization == "quantile":
        normalizer = preprocessing.QuantileTransformer(
            output_distribution="normal",
            n_quantiles=max(min(X.shape[0] // 30, 1000), 10),
            subsample=int(1e9),
            random_state=seed,
        )
    elif normalization == "uniform":
        normalizer = preprocessing.QuantileTransformer(
            output_distribution="uniform",
            n_quantiles=max(min(X.shape[0] // 30, 1000), 10),
            subsample=int(1e9),
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown normalization {normalization}")

    # normalizer
    if normalization != "standard" and standardize:
        # Ensure the data has unit variance across all columns

        normalizer = Pipeline(
            [
                ("scaler", normalizer),
                ("standard", SigmaScaler(sigma_data=sigma_data)),
            ]
        )

    tmp_missing = X[X_missing]
    X[X_missing] = NaN
    normalizer.fit(X)
    X[X_missing] = tmp_missing
    X_transformed = normalizer.transform(X)
    if standardize:
        # Set the missing values to zero (mean) explicitly.
        # This keeps the mean 0 but skews the variance.
        # However the non-missing data is properly standardized.
        X_transformed[X_missing] = 0.0
    if return_normalizer:
        return X_transformed, normalizer
    return X_transformed


def cat_encode(
    X: pd.DataFrame,
    encoding: CatEncoding,
    return_encoder: bool = False,
) -> Tuple[pd.DataFrame, Optional[Any]]:
    # Step 1. Map strings to 0-based ranges

    if encoding == "ordinal":
        encoder = preprocessing.OrdinalEncoder(
            handle_unknown="error",
            dtype="int64",
        ).fit(X)
        encoder.fit(X)

        if return_encoder:
            return encoder.transform(X), encoder
        return encoder.transform(X)

    # Step 2. Encode.

    elif encoding == "one-hot":
        encoder = preprocessing.OneHotEncoder(
            handle_unknown="error", sparse=False, dtype=np.float32
        )
        encoder.fit(X)

    if return_encoder:
        return X.transform(X), encoder
    return X.transform(X)


def encode_data(
    X_num: pd.DataFrame,
    X_cat: pd.DataFrame,
    X_missing: np.ndarray,
    normalization: Optional[Normalization] = "quantile",
    cat_encoding: CatEncoding = "ordinal",
    seed: Optional[int] = None,
    standardize: bool = False,
    sigma_data: float = 1.0,
):
    if X_num.shape[1] == 0:
        X_num, num_transform = X_num, None
    else:
        X_num, num_transform = normalize(
            X_num,
            X_missing,
            normalization,
            seed,
            return_normalizer=True,
            standardize=standardize,
            sigma_data=sigma_data,
        )

    if X_cat.shape[1] == 0:
        X_cat, cat_transform = X_cat, None
    else:
        X_cat, cat_transform = cat_encode(
            X_cat,
            cat_encoding,
            return_encoder=True,
        )

    assert cat_encoding == "ordinal", "One-hot encoding not supported yet"

    return X_num, X_cat, num_transform, cat_transform


def get_categories(X_cat: np.ndarray) -> list[int]:
    return [len(set(X_cat[:, i])) for i in range(X_cat.shape[1])]


### Moved from TabDiff.trainer.py
@torch.no_grad()
def split_num_cat_target(syn_data, info, num_inverse, int_inverse, cat_inverse):
    task_type = info["task_type"]

    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]

    n_num_feat = len(num_col_idx)
    n_cat_feat = len(cat_col_idx)

    if task_type == "regression":
        n_num_feat += len(target_col_idx)
    else:
        n_cat_feat += len(target_col_idx)

    syn_num = syn_data[:, :n_num_feat]
    syn_cat = syn_data[:, n_num_feat:]

    syn_num = num_inverse(syn_num).astype(np.float32)
    syn_num = int_inverse(syn_num).astype(np.float32)
    syn_cat = cat_inverse(syn_cat)

    if info["task_type"] == "regression":
        syn_target = syn_num[:, : len(target_col_idx)]
        syn_num = syn_num[:, len(target_col_idx) :]

    else:
        print(syn_cat.shape)
        syn_target = syn_cat[:, : len(target_col_idx)]
        syn_cat = syn_cat[:, len(target_col_idx) :]

    return syn_num, syn_cat, syn_target


def recover_data(syn_num, syn_cat, syn_target, info):
    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]
    column_info = info["column_info"]

    idx_mapping = info["idx_mapping"]
    idx_mapping = {int(key): value for key, value in idx_mapping.items()}

    syn_df = pd.DataFrame()

    if info["task_type"] == "regression":
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]]
                syn_df[i] = syn_df[i].round(column_info[str(i)]["decimals"])
                subtype = column_info[str(i)]["subtype"]
                if subtype == "float":
                    syn_df[i] = syn_df[i].astype(np.float32)
                elif subtype == "int":
                    syn_df[i] = syn_df[i].astype(np.int64)
                else:
                    raise ValueError(f"Unknown subtype {subtype} for column {i}")
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[
                    :, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)
                ]

    else:
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]]
                syn_df[i] = syn_df[i].round(column_info[str(i)]["decimals"])
                subtype = column_info[str(i)]["subtype"]
                if subtype == "float":
                    syn_df[i] = syn_df[i].astype(np.float32)
                elif subtype == "int":
                    syn_df[i] = syn_df[i].astype(np.int64)
                else:
                    raise ValueError(f"Unknown subtype {subtype} for column {i}")
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[
                    :, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)
                ]

    return syn_df


def get_decimals(column: pd.Series) -> int:
    num_decimals = (
        column.astype("str")
        .str.split(".", expand=True)[1]
        .apply(lambda x: len(x) if x else 0)
        .max()
    )
    if num_decimals == 1:
        # check if all decimals are 0
        non_nan = column[~column.isna()]
        if (non_nan == non_nan.round()).all():
            num_decimals = 0
    return int(num_decimals)
