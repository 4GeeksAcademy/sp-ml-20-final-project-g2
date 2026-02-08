"""
Streamlit app: Weekly demand forecasting by product/group.

Notes:
- Historical base data (df_base) is assumed to be from 2025 if 'year' is missing.
- Uploaded CSV data is shifted +1 ISO year (e.g., 2026-W03 -> 2027-W03) so it does
  not affect current-year forecasts.
- If uploaded data overlaps (product, year, week), uploaded values replace base ones.
"""

from __future__ import annotations

import json
import os
import pickle
import re
from typing import Any, Iterable, List, Optional

import pandas as pd
import streamlit as st


# =============================================================================
# Streamlit configuration
# =============================================================================
st.set_page_config(page_title="📦 Demand Forecast", layout="wide")
st.title("📦 Forecast de Demanda por Producto / Grupo")


# =============================================================================
# Load model and base data
# =============================================================================
MODEL_PATH = "models/73_Cat_Boost_Regressor.pkl"
BASE_DATA_PATH = "data/processed/df.pkl"
BASE_YEAR_FALLBACK = 2025
N_LAGS = 8

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(BASE_DATA_PATH, "rb") as f:
    df_base = pickle.load(f)

# Ensure year/week_start in df_base (historical data is assumed to be 2025)
if "year" not in df_base.columns:
    df_base["year"] = BASE_YEAR_FALLBACK

if "week_start" not in df_base.columns:
    df_base["week_start"] = pd.to_datetime(
        df_base["year"].astype(str)
        + "-W"
        + df_base["num_semana"].astype(str).str.zfill(2)
        + "-1",
        format="%G-W%V-%u",
        errors="coerce",
    )
    df_base["week_end"] = df_base["week_start"] + pd.Timedelta(days=6)

# Build absolute JSON path relative to this script
base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, "..", "models", "category_keywords.json")

if not os.path.exists(json_path):
    st.error(f"❌ No encuentro el JSON en: {json_path}")
    st.stop()

with open(json_path, "r", encoding="utf-8") as f:
    category_keywords = json.load(f)


# =============================================================================
# Utilities: model expected features
# =============================================================================
def get_model_features(model_obj: Any, df_fallback: pd.DataFrame) -> List[str]:
    """Return the feature list expected by the model, with robust fallbacks."""
    if hasattr(model_obj, "feature_names_"):
        try:
            feats = list(model_obj.feature_names_)
            if feats:
                return feats
        except Exception:
            pass

    if hasattr(model_obj, "feature_names_in_"):
        return list(model_obj.feature_names_in_)

    drop_cols = {"y", "week_start", "week_end", "group", "groups", "year"}
    return [c for c in df_fallback.columns if c not in drop_cols]


MODEL_FEATURES = get_model_features(model, df_base)


# =============================================================================
# Robust CSV read
# =============================================================================
def read_csv_robust(uploaded_file) -> pd.DataFrame:
    """Read a CSV trying common encodings and separators."""
    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "ISO-8859-1", "latin1"]
    last_error: Optional[Exception] = None

    for enc in encodings_to_try:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(
                uploaded_file,
                encoding=enc,
                sep=None,
                engine="python",
                on_bad_lines="skip",
            )

            # If separator autodetect fails, retry with common delimiters
            if df.shape[1] == 1:
                uploaded_file.seek(0)
                df = pd.read_csv(
                    uploaded_file,
                    encoding=enc,
                    sep=";",
                    engine="python",
                    on_bad_lines="skip",
                )

                if df.shape[1] == 1:
                    uploaded_file.seek(0)
                    df = pd.read_csv(
                        uploaded_file,
                        encoding=enc,
                        sep=",",
                        engine="python",
                        on_bad_lines="skip",
                    )

            return df
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    if last_error is not None:
        raise last_error
    raise ValueError("Unable to read CSV (no error details available).")


# =============================================================================
# Preprocessing
# =============================================================================
def clean_text(value: Any) -> str:
    """Normalize product text to uppercase without Spanish diacritics."""
    if pd.isna(value):
        return "UNKNOWN"

    text = str(value).strip().upper()
    text = (
        text.replace("Á", "A")
        .replace("É", "E")
        .replace("Í", "I")
        .replace("Ó", "O")
        .replace("Ú", "U")
        .replace("Ñ", "N")
    )
    return text


def assign_category(name: Any, category_kw: dict) -> str:
    """Assign a group based on substring keyword matching from a JSON mapping."""
    s = str(name).lower()
    for category, keywords in category_kw.items():
        for kw in keywords:
            if kw in s:
                return category
    return "Otros"


def normalize_raw_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Standardize common column names from raw pharmacy exports."""
    df = df_raw.copy()
    df.columns = df.columns.str.strip()

    rename_map = {
        # Quantity
        "Cant.": "Cant",
        "cant.": "Cant",
        "cantidad": "Cant",
        "Cantidad": "Cant",
        "CANTIDAD": "Cant",
        "CANT": "Cant",
        # Product
        "producto": "Producto",
        "PRODUCTO": "Producto",
        # Group (accept plural and variants)
        "grupo": "Group",
        "GRUPO": "Group",
        "group": "Group",
        "GROUP": "Group",
        "groups": "Group",
        "GROUPS": "Group",
        "Groups": "Group",
        # Date
        "fecha": "Fecha",
        "FECHA": "Fecha",
    }
    return df.rename(columns=rename_map)


def preprocess_raw_data_min(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Minimal preprocessing: convert to weekly aggregated demand."""
    df = df_raw.copy()

    required_cols = {"Fecha", "Producto", "Cant"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Faltan columnas obligatorias: {missing}")

    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True)
    df["Cant"] = pd.to_numeric(df["Cant"], errors="coerce")
    df = df.dropna(subset=["Fecha", "Producto", "Cant"])

    df["product"] = df["Producto"].apply(clean_text)

    # Week starts on Monday
    df["week_start"] = (
        df["Fecha"] - pd.to_timedelta(df["Fecha"].dt.weekday, unit="D")
    ).dt.normalize()

    iso = df["week_start"].dt.isocalendar()
    df["year"] = iso.year.astype(int)
    df["num_semana"] = iso.week.astype(int)

    weekly = (
        df.groupby(["product", "week_start"], as_index=False)["Cant"]
        .sum()
        .rename(columns={"Cant": "y"})
        .sort_values(["product", "week_start"])
        .reset_index(drop=True)
    )

    iso2 = weekly["week_start"].dt.isocalendar()
    weekly["year"] = iso2.year.astype(int)
    weekly["num_semana"] = iso2.week.astype(int)
    weekly["week_end"] = weekly["week_start"] + pd.Timedelta(days=6)

    return weekly


def shift_iso_year(weekly: pd.DataFrame, year_offset: int = 1) -> pd.DataFrame:
    """Shift ISO-year weekly data forward by `year_offset` years (default +1)."""
    w = weekly.copy()

    if "year" not in w.columns or "num_semana" not in w.columns:
        iso = pd.to_datetime(w["week_start"], errors="coerce").dt.isocalendar()
        w["year"] = iso.year.astype(int)
        w["num_semana"] = iso.week.astype(int)

    w["year"] = w["year"].astype(int) + int(year_offset)

    w["week_start"] = pd.to_datetime(
        w["year"].astype(str)
        + "-W"
        + w["num_semana"].astype(str).str.zfill(2)
        + "-1",
        format="%G-W%V-%u",
        errors="coerce",
    )
    w["week_end"] = w["week_start"] + pd.Timedelta(days=6)
    return w


def merge_upload_into_base(
    df_base_in: pd.DataFrame,
    df_weekly_new: pd.DataFrame,
    year_offset: int = 1,
) -> pd.DataFrame:
    """
    Merge weekly uploaded data into base.

    Rules:
    - Uploaded data is shifted by +1 ISO year (default) so it does not affect the
      current year's forecast.
    - If (product, year, num_semana) duplicates exist, uploaded values replace base.
    """
    base = df_base_in.copy()
    new = df_weekly_new.copy()

    if "week_start" not in base.columns:
        base["week_start"] = pd.to_datetime(
            base["year"].astype(str)
            + "-W"
            + base["num_semana"].astype(str).str.zfill(2)
            + "-1",
            format="%G-W%V-%u",
            errors="coerce",
        )
        base["week_end"] = base["week_start"] + pd.Timedelta(days=6)

    new = shift_iso_year(new, year_offset=year_offset)

    base_min = base[
        ["product", "year", "num_semana", "week_start", "week_end", "y"]
    ].copy()
    new_min = new[
        ["product", "year", "num_semana", "week_start", "week_end", "y"]
    ].copy()

    combined = pd.concat(
        [base_min.assign(_src=0), new_min.assign(_src=1)],
        ignore_index=True,
    )

    # Sort so uploaded rows win for the same (product, year, week)
    combined = combined.sort_values(
        ["product", "year", "num_semana", "_src"]
    ).reset_index(drop=True)

    combined = combined.drop_duplicates(
        subset=["product", "year", "num_semana"],
        keep="last",
    ).drop(columns=["_src"])

    return combined.sort_values(
        ["product", "year", "num_semana"]
    ).reset_index(drop=True)


def create_lags_fill(df_weekly: pd.DataFrame, n_lags: int = N_LAGS) -> pd.DataFrame:
    """Create lag features and fill missing early lags with expanding mean."""
    df = df_weekly.copy()

    if "week_start" not in df.columns:
        df["week_start"] = pd.to_datetime(
            df["year"].astype(str)
            + "-W"
            + df["num_semana"].astype(str).str.zfill(2)
            + "-1",
            format="%G-W%V-%u",
            errors="coerce",
        )
        df["week_end"] = df["week_start"] + pd.Timedelta(days=6)

    df = df.sort_values(["product", "week_start"]).copy()

    for lag in range(1, n_lags + 1):
        df[f"y_lag{lag}"] = df.groupby("product")["y"].shift(lag)

    lag_cols = [f"y_lag{lag}" for lag in range(1, n_lags + 1)]
    past_mean = df.groupby("product")["y"].apply(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    past_mean = past_mean.reset_index(level=0, drop=True)

    for col in lag_cols:
        df[col] = df[col].fillna(past_mean)

    df[lag_cols] = df[lag_cols].fillna(0)
    return df.reset_index(drop=True)


# =============================================================================
# Final df_model normalization
# =============================================================================
def normalize_df_model(df_model: pd.DataFrame) -> pd.DataFrame:
    """Ensure consistent column naming and derive year/week when possible."""
    df = df_model.copy()

    # Normalize group column naming
    if "groups" in df.columns and "group" not in df.columns:
        df = df.rename(columns={"groups": "group"})
    if "Group" in df.columns and "group" not in df.columns:
        df = df.rename(columns={"Group": "group"})

    # If product/group are missing, return as-is
    if "product" not in df.columns or "group" not in df.columns:
        return df

    # Ensure year/week
    if ("year" not in df.columns) or ("num_semana" not in df.columns):
        if "week_start" in df.columns:
            ws = pd.to_datetime(df["week_start"], errors="coerce")
            iso = ws.dt.isocalendar()
            df["year"] = iso.year.astype("Int64")
            df["num_semana"] = iso.week.astype("Int64")
        elif "Fecha" in df.columns:
            dates = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True)
            iso = dates.dt.isocalendar()
            df["year"] = iso.year.astype("Int64")
            df["num_semana"] = iso.week.astype("Int64")

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "num_semana" in df.columns:
        df["num_semana"] = pd.to_numeric(df["num_semana"], errors="coerce").astype("Int64")

    return df


# =============================================================================
# Forecasting
# =============================================================================
def ensure_model_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Add missing model features with zeros and reorder columns."""
    for col in MODEL_FEATURES:
        if col not in X.columns:
            X[col] = 0
    return X[MODEL_FEATURES]


def forecast_weeks(
    df_model: pd.DataFrame,
    model_obj: Any,
    product: str,
    group: str,
    n_weeks: int,
) -> pd.DataFrame:
    """Forecast the next `n_weeks` weeks for a given product/group."""
    df_pg = df_model[
        (df_model["product"] == product) & (df_model["group"] == group)
    ].copy()

    if len(df_pg) < 1:
        raise ValueError(
            "No hay suficiente historial model-ready (tras lags) para este producto/grupo."
        )

    sort_cols = ["num_semana"]
    if "year" in df_pg.columns:
        sort_cols = ["year", "num_semana"]
    df_pg = df_pg.sort_values(sort_cols).reset_index(drop=True)

    last_row = df_pg.iloc[-1].copy()
    last_week_start = pd.to_datetime(
        last_row.get("week_start", pd.Timestamp.today().normalize())
    )

    future_rows: List[dict] = []
    current_week_start = last_week_start

    for _ in range(n_weeks):
        current_week_start = current_week_start + pd.Timedelta(weeks=1)
        current_week_end = current_week_start + pd.Timedelta(days=6)

        iso = current_week_start.isocalendar()
        year_next = int(iso.year)
        week_next = int(iso.week)

        new_row = {
            "product": product,
            "group": group,
            "year": year_next,
            "num_semana": week_next,
            "y_lag1": float(last_row["y"]),
            "y_lag2": float(last_row["y_lag1"]),
            "y_lag3": float(last_row["y_lag2"]),
            "y_lag4": float(last_row["y_lag3"]),
            "y_lag5": float(last_row["y_lag4"]),
            "y_lag6": float(last_row["y_lag5"]),
            "y_lag7": float(last_row["y_lag6"]),
            "y_lag8": float(last_row["y_lag7"]),
        }

        X = ensure_model_columns(pd.DataFrame([new_row]))

        y_pred = model_obj.predict(X)[0]
        y_pred = max(round(float(y_pred)), 0)

        new_row["y"] = y_pred
        new_row["week_start"] = current_week_start.date()
        new_row["week_end"] = current_week_end.date()

        future_rows.append(new_row)
        last_row = new_row

    return pd.DataFrame(future_rows)


def forecast_all_products(
    df_model: pd.DataFrame,
    model_obj: Any,
    n_weeks: int,
    group_filter: Optional[str] = None,
) -> pd.DataFrame:
    """Forecast for all product/group combinations (optionally filtered by group)."""
    df_iter = df_model.copy()
    if group_filter is not None:
        df_iter = df_iter[df_iter["group"] == group_filter]

    combos = (
        df_iter[["product", "group"]]
        .drop_duplicates()
        .sort_values(["group", "product"])
        .itertuples(index=False, name=None)
    )

    all_forecasts: List[pd.DataFrame] = []

    for prod, grp in combos:
        df_pg = df_model[(df_model["product"] == prod) & (df_model["group"] == grp)]
        if len(df_pg) < 1:
            continue

        try:
            fc = forecast_weeks(df_model, model_obj, prod, grp, n_weeks)
            all_forecasts.append(fc)
        except Exception:  # noqa: BLE001
            continue

    if not all_forecasts:
        raise ValueError(
            "No hay combinaciones producto/grupo con suficiente historial (tras lags) para predecir."
        )

    result = pd.concat(all_forecasts, ignore_index=True)
    return result.sort_values(["group", "product", "year", "num_semana"]).reset_index(drop=True)


# =============================================================================
# Sidebar: raw CSV upload
# =============================================================================
st.sidebar.header("📂 Datos de entrada")

with st.sidebar.expander("ℹ️ Instrucciones para subir el CSV", expanded=False):
    st.markdown(
        """
### 📄 Formato del archivo
- Tipo: **.csv**
- Separador: **coma (,)** o **punto y coma (;)**
- Fechas: formato recomendado **dd/mm/aaaa** (ej: 12/02/2026)

---

### ✅ Columnas necesarias (el archivo debe contenerlas)
El CSV debe incluir **al menos** estas columnas (el orden no importa):

- **Fecha** → fecha de la operación / venta
- **Producto** → nombre del producto
- **Rubro** → grupo o categoría del producto
- **Cant.** → cantidad vendida (número)

👉 El resto de columnas pueden estar presentes, pero **no son obligatorias** para el forecast.

---

### 📋 Columnas habituales admitidas
Tu archivo puede contener, entre otras, las siguientes columnas (se aceptan sin problema):

- Tipo Mov.
- Fac. Tipo
- Fac. Suc.
- Fac. Nun.
- Fisc. Numero
- Tipo Pago
- Precio
- Sub. Total
- Cobertura
- Ajustes
- Desc. Adic.
- Total. Cliente
- IVA
- Tasa Iva
- Total Gravado
- Total sin Gravar

---

### 🧠 Recomendaciones
- Una fila puede representar una venta o un movimiento diario.
- Puede haber varias filas del mismo producto y día (se agregan automáticamente).
- Evita valores vacíos en **Fecha**, **Producto**, **Rubro** y **Cant.**.
- No pasa nada si hay columnas adicionales o información contable.

---

### ⚠️ Problemas comunes
- Si al cargar el archivo todo aparece en una sola columna, revisa el **separador** (coma vs punto y coma).
- Si la predicción falla, revisa que los nombres de las columnas estén bien escritos.
        """
    )

uploaded_file = st.sidebar.file_uploader(
    "Sube tu CSV de ventas (raw)",
    type=["csv"],
    key="ventas_csv_uploader",
)

if uploaded_file is not None:
    try:
        df_raw = read_csv_robust(uploaded_file)
        df_raw = normalize_raw_columns(df_raw)

        with st.expander("🔎 Vista previa del CSV cargado"):
            st.write("Columnas detectadas:", df_raw.columns.tolist())
            st.dataframe(df_raw.head())

        weekly_new = preprocess_raw_data_min(df_raw)

        # Uploaded data is applied to the next ISO year (e.g., 2026 -> 2027)
        weekly_combined = merge_upload_into_base(df_base, weekly_new, year_offset=1)

        # Create lag features for the model
        df_model = create_lags_fill(weekly_combined, n_lags=N_LAGS)

        st.sidebar.success("Datos crudos procesados correctamente ✅")
    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ Error al leer/procesar el CSV: {exc}")
        st.stop()
else:
    df_model = df_base.copy()

df_model = normalize_df_model(df_model)

# Always generate 'groups' from JSON, and keep 'group' for backward compatibility
df_model["groups"] = df_model["product"].apply(
    lambda x: assign_category(x, category_keywords)
)
df_model["group"] = df_model["groups"]

# Ensure week_start/week_end exist (defensive; should already be present)
if "week_start" not in df_model.columns:
    df_model["week_start"] = pd.to_datetime(
        df_model["year"].astype(str)
        + "-W"
        + df_model["num_semana"].astype(str).str.zfill(2)
        + "-1",
        format="%G-W%V-%u",
        errors="coerce",
    )
    df_model["week_end"] = df_model["week_start"] + pd.Timedelta(days=6)


# =============================================================================
# Main UI
# =============================================================================
col1, col2 = st.columns(2)

with col1:
    mode = st.radio(
        "Modo de predicción",
        ["Un producto", "Todos los productos"],
        horizontal=True,
    )

with col2:
    weeks = st.selectbox("Horizonte de predicción (semanas)", [1, 2, 4, 8])

ALL_GROUPS_LABEL = "✅ Todos los grupos"

product: Optional[str] = None
group: Optional[str] = None
group_filter_all: Optional[str] = None

if mode == "Un producto":
    group_options = [ALL_GROUPS_LABEL] + sorted(df_model["group"].dropna().unique())
    selected_group = st.selectbox(
        "Filtrar por group (opcional)",
        group_options,
        key="group_filter_single",
    )
    selected_group_filter = None if selected_group == ALL_GROUPS_LABEL else selected_group

    if selected_group_filter is None:
        product_options = sorted(df_model["product"].dropna().unique())
    else:
        product_options = sorted(
            df_model[df_model["group"] == selected_group_filter]["product"]
            .dropna()
            .unique()
        )

    if len(product_options) == 0:
        st.error("No hay productos para el group seleccionado.")
        st.stop()

    product = st.selectbox(
        "Selecciona producto",
        product_options,
        key="product_select_single",
    )

    if selected_group_filter is not None:
        group = selected_group_filter
        st.caption(f"Group seleccionado: **{group}**")
    else:
        groups_for_product = sorted(
            df_model[df_model["product"] == product]["group"].dropna().unique()
        )
        if len(groups_for_product) == 0:
            st.error("Este producto no tiene groups asociados.")
            st.stop()

        group = st.selectbox(
            "Selecciona group",
            groups_for_product,
            key="group_select_single",
        )
else:
    group_options = [ALL_GROUPS_LABEL] + sorted(df_model["group"].dropna().unique())
    selected_group_all = st.selectbox(
        "Filtrar por group (opcional)",
        group_options,
        key="group_filter_all",
    )
    group_filter_all = None if selected_group_all == ALL_GROUPS_LABEL else selected_group_all


# =============================================================================
# Prediction + results
# =============================================================================
if st.button("🔮 Predecir demanda"):
    try:
        if mode == "Un producto":
            if product is None or group is None:
                st.error("Selecciona producto y group.")
                st.stop()

            forecast = forecast_weeks(df_model, model, product, group, weeks)

            st.subheader(f"📊 Predicción para {product} | {group}")

            table_show_single = (
                forecast[["year", "num_semana", "week_start", "week_end", "y"]]
                .rename(
                    columns={
                        "year": "Año",
                        "num_semana": "Semana",
                        "week_start": "Desde",
                        "week_end": "Hasta",
                        "y": "Demanda estimada",
                    }
                )
                .reset_index(drop=True)
            )

            st.dataframe(table_show_single, hide_index=True)

            # KPIs
            kpi_total = int(forecast["y"].sum())
            kpi_mean = float(forecast["y"].mean())
            idx_max = int(forecast["y"].idxmax())

            from_max = pd.to_datetime(forecast.loc[idx_max, "week_start"]).strftime(
                "%d/%m/%Y"
            )
            to_max = pd.to_datetime(forecast.loc[idx_max, "week_end"]).strftime(
                "%d/%m/%Y"
            )
            max_value = int(forecast.loc[idx_max, "y"])

            st.caption("Total horizonte")
            st.markdown(f"### {kpi_total}")

            st.caption("Media semanal")
            st.markdown(f"### {kpi_mean:.1f}")

            st.caption("Semana pico")
            st.markdown(f"### {from_max} → {to_max} • {max_value}")

            # Weekly bars
            bars_df = forecast[["week_start", "week_end", "y"]].copy()
            ws = pd.to_datetime(bars_df["week_start"])
            we = pd.to_datetime(bars_df["week_end"])

            bars_df["periodo"] = ws.dt.strftime("%d") + "–" + we.dt.strftime("%d %b")
            bars_df = bars_df.set_index("periodo")[["y"]]

            st.subheader("📊 Forecast semanal")
            st.bar_chart(bars_df)

            # CSV download
            csv_bytes = table_show_single.to_csv(index=False).encode("utf-8")

            safe_product = re.sub(r"[^A-Za-z0-9._-]+", "_", str(product))
            safe_group = re.sub(r"[^A-Za-z0-9._-]+", "_", str(group))
            filename = f"forecast_{safe_product}_{safe_group}_{weeks}w.csv"

            st.download_button(
                label="⬇️ Descargar forecast (CSV)",
                data=csv_bytes,
                file_name=filename,
                mime="text/csv",
            )

        else:
            forecast_all = forecast_all_products(
                df_model,
                model,
                n_weeks=weeks,
                group_filter=group_filter_all,
            )

            label_group = "TODOS" if group_filter_all is None else group_filter_all
            st.subheader(f"📦 Predicción ({label_group}) – {weeks} semanas")

            table_all = (
                forecast_all[
                    ["product", "group", "year", "num_semana", "week_start", "week_end", "y"]
                ]
                .rename(
                    columns={
                        "product": "Producto",
                        "group": "Group",
                        "year": "Año",
                        "num_semana": "Semana",
                        "week_start": "Desde",
                        "week_end": "Hasta",
                        "y": "Demanda estimada",
                    }
                )
                .reset_index(drop=True)
            )

            st.dataframe(table_all, hide_index=True)

            # Top products chart
            top_n = 10
            top_products = (
                forecast_all.groupby("product", as_index=False)["y"]
                .sum()
                .sort_values("y", ascending=False)
                .head(top_n)
                .rename(
                    columns={
                        "product": "Producto",
                        "y": f"Demanda total ({weeks} semanas)",
                    }
                )
            )

            st.subheader(f"📊 Top {top_n} productos por demanda total")
            st.bar_chart(top_products.set_index("Producto"))

            # Ranking table
            ranking = (
                forecast_all.groupby(["product", "group"])["y"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
                .rename(
                    columns={
                        "product": "Producto",
                        "group": "Group",
                        "y": f"Demanda total ({weeks} semanas)",
                    }
                )
            )
            st.subheader("🔥 Ranking (Producto + Group) por demanda total")
            st.dataframe(ranking.head(25).reset_index(drop=True), hide_index=True)

            # CSV download
            csv_all_bytes = table_all.to_csv(index=False).encode("utf-8")
            suffix_raw = "ALL" if group_filter_all is None else f"GROUP_{group_filter_all}"
            suffix_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(suffix_raw))
            filename_all = f"forecast_all_{suffix_safe}_{weeks}w.csv"

            st.download_button(
                label="⬇️ Descargar forecast completo (CSV)",
                data=csv_all_bytes,
                file_name=filename_all,
                mime="text/csv",
            )

    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))