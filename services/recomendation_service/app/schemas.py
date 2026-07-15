"""Pydantic v2 request and response schemas for the recommendation API."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

PRODUCT_NAMES: tuple[str, ...] = (
    "ind_ahor_fin_ult1",
    "ind_aval_fin_ult1",
    "ind_cco_fin_ult1",
    "ind_cder_fin_ult1",
    "ind_cno_fin_ult1",
    "ind_ctju_fin_ult1",
    "ind_ctma_fin_ult1",
    "ind_ctop_fin_ult1",
    "ind_ctpp_fin_ult1",
    "ind_deco_fin_ult1",
    "ind_deme_fin_ult1",
    "ind_dela_fin_ult1",
    "ind_ecue_fin_ult1",
    "ind_fond_fin_ult1",
    "ind_hip_fin_ult1",
    "ind_plan_fin_ult1",
    "ind_pres_fin_ult1",
    "ind_reca_fin_ult1",
    "ind_tjcr_fin_ult1",
    "ind_valo_fin_ult1",
    "ind_viv_fin_ult1",
    "ind_nomina_ult1",
    "ind_nom_pens_ult1",
    "ind_recibo_ult1",
)

ProductName = Literal[
    "ind_ahor_fin_ult1",
    "ind_aval_fin_ult1",
    "ind_cco_fin_ult1",
    "ind_cder_fin_ult1",
    "ind_cno_fin_ult1",
    "ind_ctju_fin_ult1",
    "ind_ctma_fin_ult1",
    "ind_ctop_fin_ult1",
    "ind_ctpp_fin_ult1",
    "ind_deco_fin_ult1",
    "ind_deme_fin_ult1",
    "ind_dela_fin_ult1",
    "ind_ecue_fin_ult1",
    "ind_fond_fin_ult1",
    "ind_hip_fin_ult1",
    "ind_plan_fin_ult1",
    "ind_pres_fin_ult1",
    "ind_reca_fin_ult1",
    "ind_tjcr_fin_ult1",
    "ind_valo_fin_ult1",
    "ind_viv_fin_ult1",
    "ind_nomina_ult1",
    "ind_nom_pens_ult1",
    "ind_recibo_ult1",
]


class ApiModel(BaseModel):
    """Shared strict configuration for public API schemas."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class PredictionRequest(ApiModel):
    """A client snapshot used to recommend newly acquired products."""

    fecha_dato: date = Field(description="Snapshot date.")
    age: int = Field(ge=0, le=120, description="Client age in years.")
    antiguedad: int = Field(
        ge=0,
        le=1_500,
        description="Client tenure in months.",
    )
    renta: float = Field(
        ge=0,
        allow_inf_nan=False,
        description="Household gross income.",
    )

    # The remaining profile fields mirror the source Santander schema. They are
    # optional because some online callers may only have a partial profile.
    ncodpers: int | None = Field(default=None, gt=0)
    ind_empleado: str | None = None
    pais_residencia: str | None = Field(default=None, min_length=2, max_length=2)
    sexo: str | None = Field(default=None, min_length=1, max_length=1)
    fecha_alta: date | None = None
    ind_nuevo: Literal[0, 1] | None = None
    indrel: int | None = None
    ult_fec_cli_1t: date | None = None
    indrel_1mes: str | int | None = None
    tiprel_1mes: str | None = None
    indresi: str | None = Field(default=None, min_length=1, max_length=1)
    indext: str | None = Field(default=None, min_length=1, max_length=1)
    conyuemp: str | None = Field(default=None, min_length=1, max_length=1)
    canal_entrada: str | None = None
    indfall: str | None = Field(default=None, min_length=1, max_length=1)
    tipodom: int | None = None
    cod_prov: int | None = None
    nomprov: str | None = None
    ind_actividad_cliente: Literal[0, 1] | None = None
    segmento: str | None = None

    current_products: list[ProductName] = Field(
        default_factory=list,
        max_length=len(PRODUCT_NAMES),
        description="Products already owned by the client.",
    )
    top_k: int = Field(
        default=7,
        ge=1,
        le=len(PRODUCT_NAMES),
        description="Maximum number of new products to return.",
    )

    @field_validator("current_products")
    @classmethod
    def products_must_be_unique(cls, products: list[ProductName]) -> list[ProductName]:
        """Reject duplicate ownership entries instead of silently hiding them."""

        if len(products) != len(set(products)):
            raise ValueError("current_products must not contain duplicates")
        return products


class BatchPredictionRequest(ApiModel):
    """A bounded collection of prediction requests."""

    requests: list[PredictionRequest] = Field(min_length=1, max_length=100)

    @model_validator(mode="before")
    @classmethod
    def accept_list_or_items_alias(cls, value: Any) -> Any:
        """Accept a raw list and the legacy `items` key as convenience forms."""

        if isinstance(value, list):
            return {"requests": value}
        if isinstance(value, dict) and "items" in value and "requests" not in value:
            normalized = dict(value)
            normalized["requests"] = normalized.pop("items")
            return normalized
        return value


class Recommendation(ApiModel):
    """A single ranked product recommendation."""

    product: ProductName
    score: float = Field(allow_inf_nan=False)
    rank: int = Field(ge=1, le=len(PRODUCT_NAMES))


class PredictionResponse(ApiModel):
    """Recommendations returned for one client snapshot."""

    customer_id: int | None = None
    model_version: str
    recommendations: list[Recommendation]


class BatchPredictionResponse(ApiModel):
    """Ordered prediction results for a batch request."""

    predictions: list[PredictionResponse]


class HealthResponse(ApiModel):
    """Readiness information for the loaded model artifact."""

    status: Literal["ok"]
    model_version: str
    product_count: int
