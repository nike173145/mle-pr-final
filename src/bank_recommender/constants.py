"""Shared column definitions and deterministic project defaults."""

from __future__ import annotations

RANDOM_SEED = 42
ID_COLUMN = "ncodpers"
DATE_COLUMN = "fecha_dato"

PRODUCT_COLUMNS = [
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

PRODUCT_DESCRIPTIONS_RU = {
    "ind_ahor_fin_ult1": "Сберегательный счёт",
    "ind_aval_fin_ult1": "Банковская гарантия",
    "ind_cco_fin_ult1": "Текущий счёт",
    "ind_cder_fin_ult1": "Деривативный счёт",
    "ind_cno_fin_ult1": "Зарплатный проект",
    "ind_ctju_fin_ult1": "Детский счёт",
    "ind_ctma_fin_ult1": "Особый счёт 3",
    "ind_ctop_fin_ult1": "Особый счёт",
    "ind_ctpp_fin_ult1": "Особый счёт 2",
    "ind_deco_fin_ult1": "Краткосрочный депозит",
    "ind_deme_fin_ult1": "Среднесрочный депозит",
    "ind_dela_fin_ult1": "Долгосрочный депозит",
    "ind_ecue_fin_ult1": "Цифровой счёт",
    "ind_fond_fin_ult1": "Инвестиционный фонд",
    "ind_hip_fin_ult1": "Ипотека",
    "ind_plan_fin_ult1": "Пенсионный план",
    "ind_pres_fin_ult1": "Кредит",
    "ind_reca_fin_ult1": "Налоговый счёт",
    "ind_tjcr_fin_ult1": "Кредитная карта",
    "ind_valo_fin_ult1": "Ценные бумаги",
    "ind_viv_fin_ult1": "Счёт домохозяйства",
    "ind_nomina_ult1": "Зарплатный счёт",
    "ind_nom_pens_ult1": "Пенсионные обязательства",
    "ind_recibo_ult1": "Счёт прямого дебета",
}

NUMERIC_PROFILE_COLUMNS = [
    "age",
    "ind_nuevo",
    "antiguedad",
    "indrel",
    "tipodom",
    "ind_actividad_cliente",
    "renta",
]

CATEGORICAL_PROFILE_COLUMNS = [
    "ind_empleado",
    "pais_residencia",
    "sexo",
    "indrel_1mes",
    "tiprel_1mes",
    "indresi",
    "indext",
    "canal_entrada",
    "indfall",
    "cod_prov",
    "nomprov",
    "segmento",
]

RAW_FEATURE_COLUMNS = [
    DATE_COLUMN,
    *NUMERIC_PROFILE_COLUMNS,
    *CATEGORICAL_PROFILE_COLUMNS,
    *PRODUCT_COLUMNS,
]

MODEL_NUMERIC_COLUMNS = [
    *NUMERIC_PROFILE_COLUMNS,
    *PRODUCT_COLUMNS,
    "snapshot_year",
    "snapshot_month",
    "product_count",
    "tenure_years",
    "log_income",
]

MODEL_CATEGORICAL_COLUMNS = CATEGORICAL_PROFILE_COLUMNS
MODEL_FEATURE_COLUMNS = MODEL_NUMERIC_COLUMNS + MODEL_CATEGORICAL_COLUMNS
