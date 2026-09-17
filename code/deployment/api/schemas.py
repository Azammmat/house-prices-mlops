"""
Pydantic schemas for the House Price prediction API.

Only the 12 key fields are exposed to the client. Every other feature
the model was trained on is filled in server-side from
models/feature_defaults.json (see model_loader.py).
"""

from pydantic import BaseModel, Field


class HouseFeaturesInput(BaseModel):
    """The user-editable subset of features used for prediction."""

    OverallQual: int = Field(
        ..., ge=1, le=10, description="Overall material and finish quality (1-10)"
    )
    GrLivArea: float = Field(
        ..., gt=0, description="Above grade (ground) living area, square feet"
    )
    GarageCars: int = Field(..., ge=0, le=5, description="Garage size in car capacity")
    TotalBsmtSF: float = Field(..., ge=0, description="Total basement area, square feet")
    FullBath: int = Field(..., ge=0, le=5, description="Full bathrooms above grade")
    YearBuilt: int = Field(..., ge=1800, le=2100, description="Original construction year")
    Neighborhood: str = Field(..., description="Physical location within Ames city limits")
    BedroomAbvGr: int = Field(..., ge=0, le=15, description="Bedrooms above grade")
    LotArea: float = Field(..., gt=0, description="Lot size, square feet")
    KitchenQual: str = Field(..., description="Kitchen quality: Ex, Gd, TA, Fa, Po")
    HouseStyle: str = Field(..., description="Style of dwelling, e.g. 1Story, 2Story")
    CentralAir: str = Field(..., description="Central air conditioning: Y or N")

    class Config:
        json_schema_extra = {
            "example": {
                "OverallQual": 7,
                "GrLivArea": 1710,
                "GarageCars": 2,
                "TotalBsmtSF": 856,
                "FullBath": 2,
                "YearBuilt": 2003,
                "Neighborhood": "CollgCr",
                "BedroomAbvGr": 3,
                "LotArea": 8450,
                "KitchenQual": "Gd",
                "HouseStyle": "2Story",
                "CentralAir": "Y",
            }
        }


class PredictionResponse(BaseModel):
    """The predicted sale price."""

    predicted_price: float = Field(..., description="Predicted SalePrice in USD")
