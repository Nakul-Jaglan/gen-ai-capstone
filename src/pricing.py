from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from src.config import Settings

DROP_COLUMNS = [
    "No",
    "UnitPrice",
    "PricePerTsubo",
    "Period",
    "Remarks",
    "Renovation",
    "FloorPlan",
    "Purpose",
    "Use",
    "MunicipalityCode",
    "TimeToNearestStation",
    "MaxTimeToNearestStation",
    "FrontageIsGreaterFlag",
    "AreaIsGreaterFlag",
    "TotalFloorAreaIsGreaterFlag",
    "Prefecture",
]

REQUIRED_COLUMNS = [
    "Region",
    "DistrictName",
    "NearestStation",
    "MinTimeToNearestStation",
    "LandShape",
    "Frontage",
    "TotalFloorArea",
    "BuildingYear",
    "Structure",
    "Classification",
    "Breadth",
    "CityPlanning",
    "CoverageRatio",
    "FloorAreaRatio",
    "Direction",
]

CATEGORICAL_COLUMNS = [
    "Type",
    "Region",
    "Municipality",
    "DistrictName",
    "NearestStation",
    "LandShape",
    "Structure",
    "Classification",
    "CityPlanning",
    "Direction",
]

NUMERICAL_COLUMNS = [
    "Frontage",
    "TotalFloorArea",
    "BuildingYear",
    "Breadth",
    "CoverageRatio",
    "FloorAreaRatio",
    "MinTimeToNearestStation",
    "Area",
]


@dataclass
class PredictionResult:
    price_jpy: float
    warnings: list[str]


@dataclass
class AnalyticsResult:
    handled: bool
    answer: str
    source: str


class PricingEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.loaded = False

        self.model: Any = None
        self.analysis_data: pd.DataFrame | None = None

        self.label_encoders: dict[str, LabelEncoder] = {}
        self.encoder_maps: dict[str, dict[str, int]] = {}
        self.fallback_class_idx: dict[str, int] = {}

        self.scaler_1 = MinMaxScaler()
        self.scaler_2 = MinMaxScaler()

        self.feature_cols: list[str] = []
        self.num_cols_ext = NUMERICAL_COLUMNS + ["AgeOfBuilding"]

        self.unique_values: dict[str, list[Any]] = {}
        self.numeric_bounds: dict[str, tuple[float, float, float]] = {}
        self.text_lookup: dict[str, list[tuple[str, str]]] = {}

    def load(self) -> None:
        if self.loaded:
            return

        data = pd.read_csv(self.settings.data_path, low_memory=False)
        data = data.drop(columns=DROP_COLUMNS, errors="ignore")
        data = data[data["Type"] != "Agricultural Land"].copy()
        data = data.dropna(subset=REQUIRED_COLUMNS)

        self.analysis_data = data.copy()

        for col in CATEGORICAL_COLUMNS:
            values = sorted(data[col].dropna().astype(str).unique().tolist())
            self.unique_values[col] = values

        text_cols = [
            "Type",
            "Region",
            "Municipality",
            "DistrictName",
            "Structure",
            "CityPlanning",
            "Direction",
        ]
        for col in text_cols:
            lookup: list[tuple[str, str]] = []
            for value in self.unique_values.get(col, []):
                key = str(value).strip().lower()
                if len(key) >= 3:
                    lookup.append((key, str(value)))
            lookup.sort(key=lambda x: len(x[0]), reverse=True)
            self.text_lookup[col] = lookup

        for col in NUMERICAL_COLUMNS + ["Year", "Quarter", "PrewarBuilding"]:
            series = data[col].astype(float)
            q01, q99 = np.quantile(series, [0.01, 0.99])
            median = float(np.median(series))
            self.numeric_bounds[col] = (float(q01), float(q99), median)

        for col in CATEGORICAL_COLUMNS:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            self.label_encoders[col] = le
            self.encoder_maps[col] = {str(cls): int(i) for i, cls in enumerate(le.classes_)}
            mode_encoded = int(data[col].mode(dropna=True).iloc[0])
            self.fallback_class_idx[col] = mode_encoded

        self.scaler_1.fit(data[NUMERICAL_COLUMNS])
        data[NUMERICAL_COLUMNS] = self.scaler_1.transform(data[NUMERICAL_COLUMNS])

        data["AgeOfBuilding"] = data["Year"] - data["BuildingYear"]
        self.scaler_2.fit(data[self.num_cols_ext])
        data[self.num_cols_ext] = self.scaler_2.transform(data[self.num_cols_ext])

        self.feature_cols = [col for col in data.columns if col != "TradePrice"]

        self.model = joblib.load(self.settings.model_path)
        self.loaded = True

    def default_payload(self) -> dict[str, Any]:
        self.load()

        defaults = {
            "Type": self.unique_values["Type"][0],
            "Region": self.unique_values["Region"][0],
            "Municipality": self.unique_values["Municipality"][0],
            "DistrictName": self.unique_values["DistrictName"][0],
            "NearestStation": self.unique_values["NearestStation"][0],
            "LandShape": self.unique_values["LandShape"][0],
            "Structure": self.unique_values["Structure"][0],
            "Classification": self.unique_values["Classification"][0],
            "CityPlanning": self.unique_values["CityPlanning"][0],
            "Direction": self.unique_values["Direction"][0],
            "MinTimeToNearestStation": float(self.numeric_bounds["MinTimeToNearestStation"][2]),
            "Area": float(self.numeric_bounds["Area"][2]),
            "Frontage": float(self.numeric_bounds["Frontage"][2]),
            "TotalFloorArea": float(self.numeric_bounds["TotalFloorArea"][2]),
            "BuildingYear": int(round(self.numeric_bounds["BuildingYear"][2])),
            "PrewarBuilding": int(round(self.numeric_bounds["PrewarBuilding"][2])),
            "Breadth": float(self.numeric_bounds["Breadth"][2]),
            "CoverageRatio": float(self.numeric_bounds["CoverageRatio"][2]),
            "FloorAreaRatio": float(self.numeric_bounds["FloorAreaRatio"][2]),
            "Year": int(round(self.numeric_bounds["Year"][2])),
            "Quarter": int(round(self.numeric_bounds["Quarter"][2])),
        }
        return defaults

    def predict(self, raw_payload: dict[str, Any]) -> PredictionResult:
        self.load()

        payload = raw_payload.copy()
        payload["Quarter"] = int(np.clip(int(payload["Quarter"]), 1, 4))
        payload["PrewarBuilding"] = int(np.clip(int(payload["PrewarBuilding"]), 0, 1))

        df = pd.DataFrame([payload])
        warnings: list[str] = []

        for col in CATEGORICAL_COLUMNS:
            key = str(df.at[0, col])
            mapped = self.encoder_maps[col].get(key)
            if mapped is None:
                warnings.append(
                    f"Unseen category for '{col}' replaced with training-mode category."
                )
                mapped = self.fallback_class_idx[col]
            df[col] = mapped

        df[NUMERICAL_COLUMNS] = self.scaler_1.transform(df[NUMERICAL_COLUMNS])
        df["AgeOfBuilding"] = df["Year"] - df["BuildingYear"]
        df[self.num_cols_ext] = self.scaler_2.transform(df[self.num_cols_ext])
        model_input = df[self.feature_cols]

        log_price = float(self.model.predict(model_input)[0])
        price = float(np.expm1(log_price))

        return PredictionResult(price_jpy=price, warnings=warnings)

    def top_feature_importance(self, top_n: int = 12) -> pd.DataFrame:
        self.load()
        imp = pd.DataFrame(
            {
                "Feature": self.feature_cols,
                "Importance": self.model.feature_importances_,
            }
        )
        return imp.sort_values("Importance", ascending=False).head(top_n)

    def _match_category(self, question_l: str, col: str) -> str | None:
        for key, original in self.text_lookup.get(col, []):
            if key in question_l:
                return original
        return None

    def payload_from_text(self, question: str) -> tuple[dict[str, Any] | None, list[str]]:
        """Infer a pricing payload from natural language. Returns payload and extracted fields."""
        self.load()
        question_l = question.lower()
        payload = self.default_payload()
        extracted_fields: list[str] = []

        for col in [
            "Type",
            "Region",
            "Municipality",
            "DistrictName",
            "Structure",
            "CityPlanning",
            "Direction",
        ]:
            value = self._match_category(question_l, col)
            if value is not None:
                payload[col] = value
                extracted_fields.append(col)

        patterns: list[tuple[str, str, Any]] = [
            ("TotalFloorArea", r"(?:total\s*floor\s*area|floor\s*area)\s*(?:is|=|of)?\s*(\d+(?:\.\d+)?)", float),
            ("Area", r"(?:land\s*area|plot\s*area|site\s*area)\s*(?:is|=|of)?\s*(\d+(?:\.\d+)?)", float),
            ("BuildingYear", r"(?:building\s*year|year\s*built|built\s*in)\s*(?:is|=|of|in)?\s*(19\d{2}|20\d{2})", int),
            ("Frontage", r"(?:frontage|road\s*frontage)\s*(?:is|=|of)?\s*(\d+(?:\.\d+)?)", float),
            ("Breadth", r"(?:road\s*width|breadth)\s*(?:is|=|of)?\s*(\d+(?:\.\d+)?)", float),
            ("CoverageRatio", r"(?:coverage\s*ratio)\s*(?:is|=|of)?\s*(\d+(?:\.\d+)?)", float),
            ("FloorAreaRatio", r"(?:floor\s*area\s*ratio|far)\s*(?:is|=|of)?\s*(\d+(?:\.\d+)?)", float),
            ("MinTimeToNearestStation", r"(?:walk|walking|station).{0,20}?(\d{1,3})\s*(?:min|minute)", int),
            ("MinTimeToNearestStation", r"(\d{1,3})\s*(?:min|minute).{0,20}?(?:walk|walking|station)", int),
            ("Year", r"(?:transaction\s*year|sale\s*year)\s*(?:is|=|of)?\s*(19\d{2}|20\d{2})", int),
            ("Quarter", r"(?:quarter|q)\s*([1-4])", int),
        ]

        for field, pattern, caster in patterns:
            match = re.search(pattern, question_l)
            if not match:
                continue
            value_raw = match.group(1)
            try:
                value = caster(value_raw)
            except ValueError:
                continue
            payload[field] = value
            if field not in extracted_fields:
                extracted_fields.append(field)

        if "old building" in question_l or "prewar" in question_l:
            payload["PrewarBuilding"] = 1
            extracted_fields.append("PrewarBuilding")
        elif "new building" in question_l or "not prewar" in question_l:
            payload["PrewarBuilding"] = 0
            extracted_fields.append("PrewarBuilding")

        pricing_intent = any(
            token in question_l
            for token in ["price", "predict", "valuation", "estimate", "worth", "cost"]
        )

        if len(set(extracted_fields)) >= 2:
            return payload, sorted(set(extracted_fields))
        if pricing_intent and len(set(extracted_fields)) >= 1:
            return payload, sorted(set(extracted_fields))
        return None, sorted(set(extracted_fields))

    def answer_market_query(self, question: str) -> AnalyticsResult:
        """Answer common market analysis questions directly from dataset statistics."""
        self.load()
        if self.analysis_data is None or self.analysis_data.empty:
            return AnalyticsResult(
                handled=False,
                answer="",
                source="02.csv",
            )

        q = question.lower()
        df = self.analysis_data

        def top_by(column: str, ascending: bool = False) -> pd.DataFrame:
            grouped = (
                df.groupby(column)["TradePrice"]
                .agg(["mean", "median", "count"])
                .reset_index()
            )
            filtered = grouped[grouped["count"] >= 25]
            if filtered.empty:
                filtered = grouped
            return filtered.sort_values("mean", ascending=ascending)

        def safe_norm(series: pd.Series) -> pd.Series:
            s = series.astype(float)
            s_min = float(s.min())
            s_max = float(s.max())
            if s_max - s_min < 1e-12:
                return pd.Series(np.zeros(len(s), dtype=float), index=s.index)
            return (s - s_min) / (s_max - s_min)

        def investment_rankings(column: str) -> pd.DataFrame:
            yearly = (
                df.groupby([column, "Year"])["TradePrice"]
                .agg(mean="mean", count="count")
                .reset_index()
            )
            if yearly.empty:
                return pd.DataFrame()

            latest_year = int(yearly["Year"].max())
            latest = yearly[yearly["Year"] == latest_year][[column, "mean", "count"]].rename(
                columns={
                    "mean": "recent_price",
                    "count": "recent_count",
                }
            )

            history = yearly[yearly["Year"] < latest_year]
            if history.empty:
                return pd.DataFrame()

            baseline = history.groupby(column).agg(
                base_price=("mean", "mean"),
                base_count=("count", "sum"),
            ).reset_index()

            merged = latest.merge(baseline, on=column, how="inner")
            merged = merged[
                (merged["recent_count"] >= 20)
                & (merged["base_count"] >= 30)
                & (merged["base_price"] > 0)
            ]

            if merged.empty:
                merged = latest.merge(baseline, on=column, how="inner")
                merged = merged[
                    (merged["recent_count"] >= 10)
                    & (merged["base_count"] >= 10)
                    & (merged["base_price"] > 0)
                ]
            if merged.empty:
                return pd.DataFrame()

            merged = merged.copy()
            merged["growth_rate"] = (merged["recent_price"] - merged["base_price"]) / merged["base_price"]
            merged["growth_n"] = safe_norm(merged["growth_rate"])
            merged["liquidity_n"] = safe_norm(merged["recent_count"])
            merged["affordability_n"] = safe_norm(1.0 / merged["recent_price"].clip(lower=1.0))

            # Composite score balances upside momentum, transaction activity, and entry affordability.
            merged["investment_score"] = (
                0.55 * merged["growth_n"]
                + 0.30 * merged["liquidity_n"]
                + 0.15 * merged["affordability_n"]
            )

            return merged.sort_values("investment_score", ascending=False)

        high_words = ["highest", "most expensive", "maximum", "max", "top", "peak"]
        low_words = ["lowest", "cheapest", "least expensive", "minimum", "min"]
        is_high = any(w in q for w in high_words)
        is_low = any(w in q for w in low_words)

        group_col = None
        group_label = "area"
        if any(w in q for w in ["municipality", "city", "town"]):
            group_col = "Municipality"
            group_label = "municipality"
        elif any(w in q for w in ["district", "neighborhood", "neighbourhood"]):
            group_col = "DistrictName"
            group_label = "district"
        elif "station" in q:
            group_col = "NearestStation"
            group_label = "station area"
        elif any(w in q for w in ["region", "area", "zone"]):
            group_col = "Region"
            group_label = "region"

        investment_words = [
            "invest",
            "investment",
            "profit",
            "return",
            "roi",
            "appreciation",
            "undervalued",
            "growth potential",
            "opportunity",
        ]
        if any(w in q for w in investment_words):
            if group_col is None:
                group_col = "Region"
                group_label = "region"

            ranked = investment_rankings(group_col)
            if ranked.empty:
                return AnalyticsResult(False, "", "02.csv")

            latest_year = int(df["Year"].max())
            top3 = ranked.head(3)
            leader = top3.iloc[0]
            best_growth = float(top3["growth_rate"].max())
            summary = "; ".join(
                (
                    f"{row[group_col]} (score {row['investment_score']:.2f}, "
                    f"growth {row['growth_rate'] * 100:.1f}%, "
                    f"latest avg JPY {row['recent_price']:,.0f})"
                )
                for _, row in top3.iterrows()
            )

            if best_growth <= 0:
                answer = (
                    f"Using historical transaction momentum up to {latest_year}, the latest period shows broad non-positive growth across the leading candidates. "
                    f"In this environment, {leader[group_col]} ranks as the strongest relative option based on resilience, liquidity, and affordability rather than outright upside. "
                    f"Top {group_label} options: {summary}. "
                    "This is a data-driven investment signal from past trades, not a guaranteed profit forecast."
                )
            else:
                answer = (
                    f"Using historical transaction momentum up to {latest_year}, the strongest investment opportunity in this dataset is "
                    f"{leader[group_col]} based on a composite score combining price growth, transaction liquidity, and affordability. "
                    f"Top {group_label} opportunities: {summary}. "
                    "This is a data-driven investment signal from past trades, not a guaranteed profit forecast."
                )
            return AnalyticsResult(True, answer, "02.csv")

        if group_col is not None and (is_high or is_low or "average" in q or "avg" in q):
            ranked = top_by(group_col, ascending=is_low)
            if ranked.empty:
                return AnalyticsResult(False, "", "02.csv")

            leader = ranked.iloc[0]
            top3 = ranked.head(3)
            top3_text = "; ".join(
                f"{row[group_col]} (JPY {row['mean']:,.0f})" for _, row in top3.iterrows()
            )

            if is_high:
                answer = (
                    f"Based on the transaction dataset, the {group_label} with the highest average trade price is "
                    f"{leader[group_col]} at approximately JPY {leader['mean']:,.0f} "
                    f"(median JPY {leader['median']:,.0f}, {int(leader['count'])} transactions). "
                    f"Top {group_label}s by average price: {top3_text}."
                )
            elif is_low:
                answer = (
                    f"Based on the transaction dataset, the {group_label} with the lowest average trade price is "
                    f"{leader[group_col]} at approximately JPY {leader['mean']:,.0f} "
                    f"(median JPY {leader['median']:,.0f}, {int(leader['count'])} transactions)."
                )
            else:
                answer = (
                    f"Average trade prices by {group_label} were computed from the dataset. "
                    f"The highest current average is {leader[group_col]} at JPY {leader['mean']:,.0f}. "
                    f"Top entries: {top3_text}."
                )
            return AnalyticsResult(True, answer, "02.csv")

        if any(w in q for w in ["overall average", "average price", "mean price", "median price"]):
            mean_price = float(df["TradePrice"].mean())
            median_price = float(df["TradePrice"].median())
            answer = (
                f"Across the full dataset, the average trade price is JPY {mean_price:,.0f} "
                f"and the median trade price is JPY {median_price:,.0f}."
            )
            return AnalyticsResult(True, answer, "02.csv")

        if any(w in q for w in ["trend", "by year", "over year", "yearwise", "year-wise"]):
            yearly = (
                df.groupby("Year")["TradePrice"]
                .mean()
                .reset_index()
                .sort_values("Year")
            )
            if yearly.empty:
                return AnalyticsResult(False, "", "02.csv")
            peak = yearly.sort_values("TradePrice", ascending=False).iloc[0]
            trough = yearly.sort_values("TradePrice", ascending=True).iloc[0]
            answer = (
                f"Yearly price trend analysis shows the peak average year as {int(peak['Year'])} "
                f"at JPY {peak['TradePrice']:,.0f}, while the lowest average year is {int(trough['Year'])} "
                f"at JPY {trough['TradePrice']:,.0f}."
            )
            return AnalyticsResult(True, answer, "02.csv")

        return AnalyticsResult(False, "", "02.csv")
