import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd
import pickle
import re
import json
import os

# =====================================================
# Streamlit configuration
# =====================================================
st.set_page_config(page_title="📦 Demand Forecast", layout="wide")
st.title("📦 Forecast de Demanda por Producto / Grupo")

st.markdown(
    """
    <style>
    /* Chips del multiselect */
    div[data-baseweb="tag"] {
        background-color: #00796B !important;   /* verde farmacia */
        color: white !important;
    }

    div[data-baseweb="tag"] span {
        color: white !important;
    }

    /* Botón X del chip */
    div[data-baseweb="tag"] svg {
        fill: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# Load model and base data
# =====================================================
model = pickle.load(open("models/73_Cat_Boost_Regressor.pkl", "rb"))
df_base = pickle.load(open("data/processed/df.pkl", "rb"))

# Ensure year/week_start in df_base (historical data is assumed to be 2025)
if "year" not in df_base.columns:
    # History corresponds to 2025 (based on our data), we set year=2025
    df_base["year"] = 2025

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


# =====================================================
# Utilities: model expected features
# =====================================================
def get_model_features(model, df_fallback):
    if hasattr(model, "feature_names_"):
        try:
            feats = list(model.feature_names_)
            if feats:
                return feats
        except Exception:
            pass

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    drop_cols = {"y", "week_start", "week_end", "group", "groups", "year"}
    return [c for c in df_fallback.columns if c not in drop_cols]


MODEL_FEATURES = get_model_features(model, df_base)

# =====================================================
# Robust CSV read
# =====================================================
def read_csv_robust(uploaded_file):
    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "ISO-8859-1", "latin1"]
    last_error = None

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
        except Exception as e:
            last_error = e

    raise last_error


# =====================================================
# Preprocessing
# =====================================================
def clean_text(x):
    if pd.isna(x):
        return "UNKNOWN"

    x = str(x).strip().upper()
    x = (
        x.replace("Á", "A")
        .replace("É", "E")
        .replace("Í", "I")
        .replace("Ó", "O")
        .replace("Ú", "U")
        .replace("Ñ", "N")
    )
    return x


def assign_category(name: str, category_keywords: dict) -> str:
    s = str(name).lower()
    for category, keywords in category_keywords.items():
        for kw in keywords:
            if kw in s:
                return category
    return "Otros"


def normalize_raw_columns(df_raw):
    df_raw = df_raw.copy()
    df_raw.columns = df_raw.columns.str.strip()

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
        # Group (accepts plural and variants)
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
    return df_raw.rename(columns=rename_map)


def preprocess_raw_data_min(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    required_cols = {"Fecha", "Producto", "Cant"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Faltan columnas obligatorias: {missing}")

    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True)
    df["Cant"] = pd.to_numeric(df["Cant"], errors="coerce")
    df = df.dropna(subset=["Fecha", "Producto", "Cant"])

    df["product"] = df["Producto"].apply(clean_text)

    df["week_start"] = (
        df["Fecha"] - pd.to_timedelta(df["Fecha"].dt.weekday, unit="D")
    ).dt.normalize()

    # Year and week_number for informational purposes only
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

    # Reattach year/week number for UI
    iso2 = weekly["week_start"].dt.isocalendar()
    weekly["year"] = iso2.year.astype(int)
    weekly["num_semana"] = iso2.week.astype(int)

    return weekly


def shift_iso_year(weekly: pd.DataFrame, year_offset: int = 1) -> pd.DataFrame:
    """Shift ISO-year weekly data forward by `year_offset` years (default +1)."""
    w = weekly.copy()

    # Ensure year/week number
    if "year" not in w.columns or "num_semana" not in w.columns:
        iso = pd.to_datetime(w["week_start"], errors="coerce").dt.isocalendar()
        w["year"] = iso.year.astype(int)
        w["num_semana"] = iso.week.astype(int)

    w["year"] = w["year"].astype(int) + int(year_offset)

    # Rebuild week_start from ISO year+week (Monday)
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
    df_base: pd.DataFrame,
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
    base = df_base.copy()
    new = df_weekly_new.copy()

    # Ensure week_start/year/num_week in base
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

    # Move upload to the following year
    new = shift_iso_year(new, year_offset=year_offset)

    # Normalize minimum columns
    base_min = base[
        ["product", "year", "num_semana", "week_start", "week_end", "y"]
    ].copy()
    new_min = new[
        ["product", "year", "num_semana", "week_start", "week_end", "y"]
    ].copy()

    # Replace duplicates: base first, upload last -> keep="last"
    combined = pd.concat(
        [base_min.assign(_src=0), new_min.assign(_src=1)],
        ignore_index=True,
    )

    # Sort so upload wins if keys match
    combined = combined.sort_values(
        ["product", "year", "num_semana", "_src"]
    ).reset_index(drop=True)

    combined = combined.drop_duplicates(
        subset=["product", "year", "num_semana"],
        keep="last",
    )
    combined = combined.drop(columns=["_src"])

    # Final ordering
    combined = combined.sort_values(["product", "year", "num_semana"]).reset_index(
        drop=True
    )
    return combined


def create_lags_fill(df_weekly: pd.DataFrame, n_lags: int = 8) -> pd.DataFrame:
    df = df_weekly.sort_values(["product", "week_start"]).copy()

    for lag in range(1, n_lags + 1):
        df[f"y_lag{lag}"] = df.groupby("product")["y"].shift(lag)

    # Fill initial gaps (so as not to lose new products)
    lag_cols = [f"y_lag{l}" for l in range(1, n_lags + 1)]
    past_mean = df.groupby("product")["y"].apply(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    past_mean = past_mean.reset_index(level=0, drop=True)

    for c in lag_cols:
        df[c] = df[c].fillna(past_mean)

    df[lag_cols] = df[lag_cols].fillna(0)

    return df.reset_index(drop=True)


# =====================================================
# Final df_model normalization (KEY FOR ERROR 'year')
# =====================================================
def normalize_df_model(df_model: pd.DataFrame) -> pd.DataFrame:
    df = df_model.copy()

    # Normalize group name
    if "groups" in df.columns and "group" not in df.columns:
        df = df.rename(columns={"groups": "group"})
    if "Group" in df.columns and "group" not in df.columns:
        df = df.rename(columns={"Group": "group"})

    # Ensure product/group exist
    if "product" not in df.columns or "group" not in df.columns:
        return df

    # Ensure year/week_number if missing (from week_start or date)
    if ("year" not in df.columns) or ("num_semana" not in df.columns):
        if "week_start" in df.columns:
            ws = pd.to_datetime(df["week_start"], errors="coerce")
            iso = ws.dt.isocalendar()
            df["year"] = iso.year.astype("Int64")
            df["num_semana"] = iso.week.astype("Int64")
        elif "Fecha" in df.columns:
            f = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True)
            iso = f.dt.isocalendar()
            df["year"] = iso.year.astype("Int64")
            df["num_semana"] = iso.week.astype("Int64")

    # Convert to int types if possible
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "num_semana" in df.columns:
        df["num_semana"] = pd.to_numeric(df["num_semana"], errors="coerce").astype(
            "Int64"
        )

    return df


# =====================================================
# Forecasting
# =====================================================
def _ensure_model_columns(X: pd.DataFrame) -> pd.DataFrame:
    for col in MODEL_FEATURES:
        if col not in X.columns:
            X[col] = 0
    return X[MODEL_FEATURES]


def forecast_weeks(df_model, model, product, group, n_weeks):
    df_pg = df_model[
        (df_model["product"] == product) & (df_model["group"] == group)
    ].copy()

    # With lags already created, one row is sufficient to predict forward
    if len(df_pg) < 1:
        raise ValueError(
            "No hay suficiente historial model-ready (tras lags) para este producto/grupo."
        )

    # Sort robustly even if year is missing (but it should be normalized)
    sort_cols = ["num_semana"]
    if "year" in df_pg.columns:
        sort_cols = ["year", "num_semana"]
    df_pg = df_pg.sort_values(sort_cols).reset_index(drop=True)

    last_row = df_pg.iloc[-1].copy()

    last_week_start = pd.to_datetime(
        last_row.get("week_start", pd.Timestamp.today().normalize())
    )

    today = pd.Timestamp.today().normalize()
    this_monday = (today - pd.to_timedelta(today.weekday(), unit="D")).normalize()

    anchor_start = max(last_week_start, this_monday)

    future = []
    current_week_start = anchor_start

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

        X = pd.DataFrame([new_row])
        X = _ensure_model_columns(X)

        y_pred = model.predict(X)[0]
        y_pred = max(round(float(y_pred)), 0)

        new_row["y"] = y_pred
        new_row["week_start"] = current_week_start.date()
        new_row["week_end"] = current_week_end.date()

        future.append(new_row)
        last_row = new_row

    return pd.DataFrame(future)


def forecast_all_products(df_model, model, n_weeks, group_filter=None):
    all_forecasts = []

    df_iter = df_model.copy()
    if group_filter is not None:
        if isinstance(group_filter, (list, tuple, set)):
            if len(group_filter) > 0:
                df_iter = df_iter[df_iter["group"].isin(group_filter)]
            else:
                # Lista vacía = no hay nada seleccionado -> no hay resultados
                df_iter = df_iter.iloc[0:0]
        else:
            df_iter = df_iter[df_iter["group"] == group_filter]

    combos = (
        df_iter[["product", "group"]]
        .drop_duplicates()
        .sort_values(["group", "product"])
        .itertuples(index=False, name=None)
    )

    for product, group in combos:
        df_pg = df_model[(df_model["product"] == product) & (df_model["group"] == group)]
        if len(df_pg) < 1:
            continue

        try:
            fc = forecast_weeks(df_model, model, product, group, n_weeks)
            all_forecasts.append(fc)
        except Exception:
            continue

    if not all_forecasts:
        raise ValueError(
            "No hay combinaciones producto/grupo con suficiente historial (tras lags) para predecir."
        )

    result = pd.concat(all_forecasts, ignore_index=True)
    return result.sort_values(["group", "product", "year", "num_semana"]).reset_index(
        drop=True
    )


# =====================================================
# Sidebar: raw CSV upload
# =====================================================
st.sidebar.header("📂 Datos de entrada")

with st.sidebar.expander("ℹ️ Instrucciones para subir el CSV", expanded=False):
    st.markdown(
        """
### 📄 Formato del archivo
- Tipo: **.csv**
- Fechas: formato recomendado **dd/mm/aaaa** (ej: 12/02/2026)

---

### ✅ Columnas necesarias (obligatorias)
El CSV debe incluir **al menos** estas columnas:

- **Fecha** → fecha de la operación / venta  
- **Producto** → nombre del producto  
- **Cant.** → cantidad vendida (número)

---

### 🗓️ Regla importante (no afecta el forecast del año actual)
- Los datos que subas se **desplazan automáticamente al año siguiente (ISO)**.  
  Ejemplo: ventas en **enero 2026** se aplican a **enero 2027** (mismas semanas ISO).
- Si hay registros repetidos para la misma combinación **(Producto, Año ISO, Semana ISO)**, los datos del archivo **reemplazan** a los del histórico.

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
- Puede haber varias filas del mismo producto y día: se agregan automáticamente a nivel semanal.
- Evita valores vacíos en **Fecha**, **Producto** y **Cant.**.

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

        # Rule: uploaded data is applied to the following year (e.g., 2026 -> 2027)
        weekly_combined = merge_upload_into_base(df_base, weekly_new, year_offset=1)

        # Create lags for the model
        df_model = create_lags_fill(weekly_combined, n_lags=8)

        st.sidebar.success("Datos crudos procesados correctamente ✅")

    except Exception as e:
        st.error(f"❌ Error al leer/procesar el CSV: {e}")
        st.stop()
else:
    df_model = df_base.copy()

# Always normalize (year/week_start + group names)
df_model = normalize_df_model(df_model)


# =====================================================
# GROUPS (from JSON) - ALWAYS generate
# =====================================================
df_model["groups"] = df_model["product"].apply(
    lambda x: assign_category(x, category_keywords)
)
df_model["group"] = df_model["groups"]  # alias to avoid breaking existing code

# If "group" does not exist (because df_base does not have it), create it via JSON
if "group" not in df_model.columns:
    df_model["group"] = df_model["product"].apply(
        lambda x: assign_category(x, category_keywords)
    )

if "week_start" not in df_model.columns:
    current_year = pd.Timestamp.today().year
    df_model["year"] = current_year
    df_model["week_start"] = pd.to_datetime(
        df_model["year"].astype(str)
        + "-W"
        + df_model["num_semana"].astype(str).str.zfill(2)
        + "-1",
        format="%G-W%V-%u",
        errors="coerce",
    )
    df_model["week_end"] = df_model["week_start"] + pd.Timedelta(days=6)


# =====================================================
# Main UI
# =====================================================

ALL_GROUPS_LABEL = "✅ Todos los grupos"

col1, col2 = st.columns(2)

with col1:
    mode = st.radio(
        "Modo de predicción",
        ["Un producto", "Todos los productos"],
        horizontal=True,
    )

with col2:
    weeks = st.selectbox("Horizonte de predicción (semanas)", [1, 2, 4, 8])

# Avoid NameError
product = None
group = None
group_filter_all = None

if mode == "Un producto":
    # Single-group filter (optional) + "All groups" option
    group_options = [ALL_GROUPS_LABEL] + sorted(
        df_model["group"].dropna().unique().tolist()
    )

    selected_group = st.selectbox(
        "Filtrar por group (opcional)",
        group_options,
        key="group_filter_single",
    )
    selected_group_filter = (
        None if selected_group == ALL_GROUPS_LABEL else selected_group
    )

    # Product list depends on selected group (or all)
    if selected_group_filter is None:
        product_options = sorted(df_model["product"].dropna().unique().tolist())
    else:
        product_options = sorted(
            df_model[df_model["group"] == selected_group_filter]["product"]
            .dropna()
            .unique()
            .tolist()
        )

    if not product_options:
        st.error("No hay productos para el group seleccionado.")
        st.stop()

    product = st.selectbox(
        "Selecciona producto",
        product_options,
        key="product_select_single",
    )

    # If user filtered by group, lock group to that filter
    if selected_group_filter is not None:
        group = selected_group_filter
        st.caption(f"Grupo seleccionado: **{group}**")
    else:
        # Otherwise allow choosing among product's groups
        groups_for_product = sorted(
            df_model[df_model["product"] == product]["group"]
            .dropna()
            .unique()
            .tolist()
        )
        if not groups_for_product:
            st.error("Este producto no tiene groups asociados.")
            st.stop()

        group = st.selectbox(
            "Tipo de categoría",
            groups_for_product,
            key="group_select_single",
        )

else:
    # Multi-group selection ONLY for "All products"
    group_options = sorted(df_model["group"].dropna().unique().tolist())

    select_all_groups = st.checkbox(
        "✅ Todos los grupos",
        value=True,
        key="select_all_groups_all_mode",
    )

    selected_groups_all = st.multiselect(
        "Filtrar por group (opcional)",
        options=group_options,
        default=[],
        disabled=select_all_groups,
        key="group_filter_all_multi",
        help="Si marcás 'Todos los grupos', se ignora esta selección.",
    )

    group_filter_all = None if select_all_groups else selected_groups_all

# =====================================================
# Prediction + results
# =====================================================
if "week_start" not in df_model.columns:
    current_year = pd.Timestamp.today().year
    df_model["year"] = current_year
    df_model["week_start"] = pd.to_datetime(
        df_model["year"].astype(str)
        + "-W"
        + df_model["num_semana"].astype(str).str.zfill(2)
        + "-1",
        format="%G-W%V-%u",
        errors="coerce",
    )
    df_model["week_end"] = df_model["week_start"] + pd.Timedelta(days=6)

if st.button("🔮 Predecir demanda"):
    try:
        if mode == "Un producto":
            forecast = forecast_weeks(df_model, model, product, group, weeks)

            st.subheader(f"📊 Predicción para {product} | {group}")

            # Table to display (UI only)
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

            # KPIs + chart (works the same with 1 or 8 weeks)
            kpi_total = int(forecast["y"].sum())
            kpi_media = float(forecast["y"].mean())
            idx_max = forecast["y"].idxmax()

            desde_max = pd.to_datetime(forecast.loc[idx_max, "week_start"]).strftime(
                "%d/%m/%Y"
            )
            hasta_max = pd.to_datetime(forecast.loc[idx_max, "week_end"]).strftime(
                "%d/%m/%Y"
            )
            valor_max = int(forecast.loc[idx_max, "y"])

            st.caption("Total horizonte")
            st.markdown(f"### {kpi_total}")

            st.caption("Media semanal")
            st.markdown(f"### {kpi_media:.1f}")

            st.caption("Semana pico")
            st.markdown(f"### {desde_max} → {hasta_max} • {valor_max}")

            # Bars per week (readable even with 1 week)
            bars_df = forecast[["week_start", "week_end", "y"]].copy()

            ws = pd.to_datetime(bars_df["week_start"])
            we = pd.to_datetime(bars_df["week_end"])

            bars_df["periodo"] = ws.dt.strftime("%d") + "–" + we.dt.strftime("%d %b")
            bars_df = bars_df.set_index("periodo")[["y"]]

            st.subheader("📊 Forecast semanal")

            if len(bars_df) == 1:
                left, center, right = st.columns([5, 2, 5])

                with center:
                    periodo_unico = bars_df.index[0]
                    valor_unico = int(bars_df["y"].iloc[0])

                    desde = pd.to_datetime(
                        forecast["week_start"].iloc[0]
                    ).strftime("%d/%m/%Y")
                    hasta = pd.to_datetime(
                        forecast["week_end"].iloc[0]
                    ).strftime("%d/%m/%Y")

                    st.metric(
                        label=f"Forecast ({desde} → {hasta})",
                        value=valor_unico,
                    )
                    st.bar_chart(bars_df)
            else:
                st.bar_chart(bars_df)

            st.subheader("📈 Histórico vs Forecast")

            hist = df_model[(df_model["product"] == product) & (df_model["group"] == group)].copy()

            # Nos quedamos con algo presentable
            hist = hist[["week_start", "y"]].copy()
            hist["week_start"] = pd.to_datetime(hist["week_start"], errors="coerce")
            hist = hist.dropna().sort_values("week_start")

            fc = forecast[["week_start", "y"]].copy()
            fc["week_start"] = pd.to_datetime(fc["week_start"], errors="coerce")
            fc = fc.dropna().sort_values("week_start")

            hist["tipo"] = "Histórico"
            fc["tipo"] = "Forecast"

            plot_df = pd.concat([hist, fc], ignore_index=True)

            # Opción A: Plotly (recomendado)
            fig = px.line(
                plot_df,
                x="week_start",
                y="y",
                color="tipo",
                markers=True,
                title=f"{product} • {group}",
                labels={"week_start": "Semana", "y": "Cantidad", "tipo": ""},
            )
            st.plotly_chart(fig, use_container_width=True)

            # Mini-métrica: variación vs última semana histórica
            if len(hist) > 0 and len(fc) > 0:
                last_hist = float(hist["y"].iloc[-1])
                first_fc = float(fc["y"].iloc[0])
                delta = first_fc - last_hist
                st.metric("Forecast 1ª semana vs última histórica", value=int(first_fc), delta=int(delta))


            # CSV (separate from what is displayed)
            csv_bytes = table_show_single.copy().to_csv(index=False).encode("utf-8")

            # Secure filename (avoid .html due to unusual characters)
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

            # UI table
            st.dataframe(table_all, hide_index=True)

            # Top products by total demand (bars)
            TOP_N = 10

            top_products = (
                forecast_all.groupby("product", as_index=False)["y"]
                .sum()
                .sort_values("y", ascending=False)
                .head(TOP_N)
            )

            top_products = top_products.rename(
                columns={
                    "product": "Producto",
                    "y": f"Demanda total ({weeks} semanas)",
                }
            )

            st.subheader(f"📊 Top {TOP_N} productos por demanda total")
            st.bar_chart(top_products.set_index("Producto"))

            # Ranking
            resumen = (
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
            st.dataframe(resumen.head(25).reset_index(drop=True), hide_index=True)

            # Graph showing demand participation by group
            by_group = (
                forecast_all.groupby("group", as_index=False)["y"]
                .sum()
                .sort_values("y", ascending=False)
            )

            if by_group.shape[0] > 1:
                st.subheader("🍰 Participación de demanda por grupo")
                fig = px.pie(
                    by_group,
                    names="group",
                    values="y",
                    title=f"Demanda total por grupo ({weeks} semanas)",
                )
                st.plotly_chart(fig, use_container_width=True)

                # A bar graph
                st.caption("📊 Demanda por grupo")
                st.bar_chart(by_group.set_index("group")[["y"]])

            # Separate CSV
            csv_all_bytes = table_all.copy().to_csv(index=False).encode("utf-8")
            suffix_raw = (
                "ALL"
                if group_filter_all is None
                else f"GROUP_{group_filter_all}"
            )
            suffix_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(suffix_raw))
            filename_all = f"forecast_all_{suffix_safe}_{weeks}w.csv"

            st.download_button(
                label="⬇️ Descargar forecast completo (CSV)",
                data=csv_all_bytes,
                file_name=filename_all,
                mime="text/csv",
            )

    except Exception as e:
        st.error(str(e))