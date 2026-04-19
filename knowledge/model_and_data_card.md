# Model and Data Card

## Dataset
- File: `02.csv`
- Domain: Japanese residential real estate transactions
- Sample size: ~52k rows used in current local project file
- Target: `TradePrice` (JPY)

## Features Used in Pricing Tool
Categorical:
- Type, Region, Municipality, DistrictName, NearestStation
- LandShape, Structure, Classification, CityPlanning, Direction

Numerical:
- Frontage, TotalFloorArea, BuildingYear, Breadth
- CoverageRatio, FloorAreaRatio, MinTimeToNearestStation, Area
- Year, Quarter, PrewarBuilding

Engineered:
- AgeOfBuilding = Year - BuildingYear

## Model
- Algorithm: RandomForestRegressor (serialized in `rf_model_new.joblib`)
- Target transform: log1p during training, expm1 at inference output stage

## Known Limitations
- Old model artifact version mismatch warning may appear if sklearn versions differ.
- Input categories unseen in training are fallback-mapped to mode values.
- Retrieval answers are constrained by available local corpus quality.

## Responsible Usage
- Prediction is decision support, not legal valuation advice.
- For high-stakes transactions, combine with certified appraisal and live market comps.
