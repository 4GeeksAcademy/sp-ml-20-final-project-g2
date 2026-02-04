import streamlit as st
import pandas as pd
import pickle

# =====================================================
# CONFIGURACIÓN STREAMLIT
# =====================================================
st.set_page_config(page_title="📦 Demand Forecast", layout="wide")
st.title("📦 Forecast de Demanda por Producto / Grupo")

# =====================================================
# CARGA DE MODELO Y DATA BASE (fallback)
# =====================================================
model = pickle.load(open("models/72_Cat_Boost_Regressor.pkl", "rb"))
df_base = pickle.load(open("data/processed/df_72_catboost_extra_column.pkl", "rb"))

# =====================================================
# UTILIDADES: FEATURES QUE ESPERA EL MODELO
# =====================================================
def get_model_features(model, df_fallback):
    # CatBoost
    if hasattr(model, "feature_names_"):
        try:
            feats = list(model.feature_names_)
            if feats:
                return feats
        except Exception:
            pass

    # sklearn-style
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    # fallback: inferir del df_base (evitando columnas típicas no-feature)
    drop_cols = {"y", "week_start", "week_end"}
    return [c for c in df_fallback.columns if c not in drop_cols]

MODEL_FEATURES = get_model_features(model, df_base)

# =====================================================
# LECTURA ROBUSTA CSV (encoding + separador)
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
                on_bad_lines="skip"
            )

            if df.shape[1] == 1:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=enc, sep=";", engine="python", on_bad_lines="skip")

                if df.shape[1] == 1:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=enc, sep=",", engine="python", on_bad_lines="skip")

            return df

        except Exception as e:
            last_error = e

    raise last_error

# =====================================================
# PREPROCESAMIENTO (RAW -> WEEKLY)
# =====================================================
def clean_text(x):
    if pd.isna(x):
        return "UNKNOWN"
    x = str(x).strip().upper()
    x = (
        x.replace("Á", "A").replace("É", "E").replace("Í", "I")
         .replace("Ó", "O").replace("Ú", "U").replace("Ñ", "N")
    )
    return x

def normalize_raw_columns(df_raw):
    df_raw = df_raw.copy()
    df_raw.columns = df_raw.columns.str.strip()

    rename_map = {
        # Cantidad
        "Cant.": "Cant",
        "cant.": "Cant",
        "cantidad": "Cant",
        "Cantidad": "Cant",
        "CANTIDAD": "Cant",
        "CANT": "Cant",

        # Producto
        "producto": "Producto",
        "PRODUCTO": "Producto",

        # Grupo (acepta plural y variantes)
        "grupo": "Group",
        "GRUPO": "Group",
        "group": "Group",
        "GROUP": "Group",
        "groups": "Group",
        "GROUPS": "Group",
        "Groups": "Group",

        # Fecha
        "fecha": "Fecha",
        "FECHA": "Fecha",
    }
    return df_raw.rename(columns=rename_map)

def preprocess_raw_data(df_raw):
    df = df_raw.copy()

    required_cols = {"Fecha", "Producto", "Group", "Cant"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Faltan columnas obligatorias: {missing}")

    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True)
    df["Cant"] = pd.to_numeric(df["Cant"], errors="coerce")
    df = df.dropna(subset=["Fecha", "Producto", "Group", "Cant"])

    df["product"] = df["Producto"].apply(clean_text)
    df["group"] = df["Group"].apply(clean_text)

    iso = df["Fecha"].dt.isocalendar()
    df["year"] = iso.year.astype(int)
    df["num_semana"] = iso.week.astype(int)

    df["week_start"] = (df["Fecha"] - pd.to_timedelta(df["Fecha"].dt.weekday, unit="D")).dt.normalize()
    df["week_end"] = df["week_start"] + pd.Timedelta(days=6)

    df_weekly = (
        df.groupby(["product", "group", "year", "num_semana"], as_index=False)
          .agg(
              y=("Cant", "sum"),
              week_start=("week_start", "min"),
              week_end=("week_end", "max"),
          )
          .sort_values(["product", "group", "year", "num_semana"])
          .reset_index(drop=True)
    )
    return df_weekly

def create_lags(df_weekly, n_lags=8):
    df = df_weekly.sort_values(["product", "group", "year", "num_semana"]).copy()
    for lag in range(1, n_lags + 1):
        df[f"y_lag{lag}"] = df.groupby(["product", "group"])["y"].shift(lag)
    return df.dropna().reset_index(drop=True)

# =====================================================
# FORECAST
# =====================================================
def _ensure_model_columns(X: pd.DataFrame) -> pd.DataFrame:
    for col in MODEL_FEATURES:
        if col not in X.columns:
            X[col] = 0
    return X[MODEL_FEATURES]

def forecast_weeks(df_model, model, product, group, n_weeks):
    df_pg = df_model[(df_model["product"] == product) & (df_model["group"] == group)].copy()

    if len(df_pg) < 10:
        raise ValueError("No hay suficiente historial para esta combinación producto/grupo (mínimo 10 semanas).")

    df_pg = df_pg.sort_values(["year", "num_semana"]).reset_index(drop=True)
    last_row = df_pg.iloc[-1].copy()

    last_week_start = pd.to_datetime(last_row.get("week_start", pd.Timestamp.today().normalize()))
    future = []
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

def forecast_all_products(df_model, model, n_weeks, min_hist_weeks=10):
    all_forecasts = []

    combos = (
        df_model[["product", "group"]]
        .drop_duplicates()
        .sort_values(["group", "product"])
        .itertuples(index=False, name=None)
    )

    for product, group in combos:
        df_pg = df_model[(df_model["product"] == product) & (df_model["group"] == group)]
        if len(df_pg) < min_hist_weeks:
            continue

        try:
            fc = forecast_weeks(df_model, model, product, group, n_weeks)
            all_forecasts.append(fc)
        except Exception:
            continue

    if not all_forecasts:
        raise ValueError("No hay combinaciones producto/grupo con suficiente historial para predecir.")

    result = pd.concat(all_forecasts, ignore_index=True)
    return result.sort_values(["group", "product", "year", "num_semana"]).reset_index(drop=True)

# =====================================================
# SIDEBAR – CARGA CSV CRUDO
# =====================================================
st.sidebar.header("📂 Datos de entrada")
uploaded_file = st.sidebar.file_uploader("Sube tu CSV de ventas (raw)", type=["csv"])

if uploaded_file is not None:
    try:
        df_raw = read_csv_robust(uploaded_file)
        df_raw = normalize_raw_columns(df_raw)

        with st.expander("🔎 Vista previa del CSV cargado"):
            st.write("Columnas detectadas:", df_raw.columns.tolist())
            st.dataframe(df_raw.head())

        df_weekly = preprocess_raw_data(df_raw)
        df_model = create_lags(df_weekly)

        st.sidebar.success("Datos crudos procesados correctamente ✅")

    except Exception as e:
        st.error(f"❌ Error al leer/procesar el CSV: {e}")
        st.stop()
else:
    df_model = df_base.copy()

    # normalizar nombre de columna en df_base
    if "groups" in df_model.columns and "group" not in df_model.columns:
        df_model = df_model.rename(columns={"groups": "group"})
    if "Group" in df_model.columns and "group" not in df_model.columns:
        df_model = df_model.rename(columns={"Group": "group"})

# =====================================================
# UI PRINCIPAL
# =====================================================
col1, col2 = st.columns(2)

with col1:
    mode = st.radio(
        "Modo de predicción",
        ["Un producto", "Todos los productos"],
        horizontal=True,
        key="mode_radio"
    )

with col2:
    weeks = st.selectbox(
        "Horizonte de predicción (semanas)",
        [1, 2, 4, 8],
        key="weeks_select"
    )

# ---- Selector opcional de grupo (aplica solo en modo "Un producto")
ALL_GROUPS_LABEL = "✅ Todos los grupos"

selected_group_filter = None
if mode == "Un producto":
    if "group" not in df_model.columns:
        st.error("El dataset cargado no contiene la columna 'group' (o 'groups'). Revisa nombres de columnas.")
        st.stop()

    group_options = [ALL_GROUPS_LABEL] + sorted(df_model["group"].dropna().unique().tolist())
    selected_group = st.selectbox("Filtrar por group (opcional)", group_options, key="group_filter_select")

    if selected_group != ALL_GROUPS_LABEL:
        selected_group_filter = selected_group

    # productos filtrados por group si aplica
    if selected_group_filter is None:
        product_options = sorted(df_model["product"].dropna().unique().tolist())
    else:
        product_options = sorted(df_model[df_model["group"] == selected_group_filter]["product"].dropna().unique().tolist())

    if not product_options:
        st.error("No hay productos para el group seleccionado.")
        st.stop()

    product = st.selectbox("Selecciona producto", product_options, key="product_select_single")

    # si hay filtro de group, el group es único (ese); si no, hay que elegir group para ese producto
    if selected_group_filter is not None:
        group = selected_group_filter
        st.caption(f"Grupo seleccionado: **{group}**")
    else:
        groups_for_product = sorted(df_model[df_model["product"] == product]["group"].dropna().unique().tolist())
        if not groups_for_product:
            st.error("Este producto no tiene groups asociados en el dataset.")
            st.stop()

        group = st.selectbox("Pertenece al grupo", groups_for_product, key="group_select_single")
else:
    product = None
    group = None

# =====================================================
# PREDICCIÓN + RESULTADOS
# =====================================================
if st.button("🔮 Predecir demanda"):
    try:
        if mode == "Un producto":
            forecast = forecast_weeks(df_model, model, product, group, weeks)

            st.subheader(f"📊 Predicción para {product} | {group}")

            table = (
                forecast[["year", "num_semana", "week_start", "week_end", "y"]]
                .rename(columns={
                    "year": "Año",
                    "num_semana": "Semana",
                    "week_start": "Desde",
                    "week_end": "Hasta",
                    "y": "Demanda estimada"
                })
            )

            st.dataframe(table)
            st.line_chart(forecast.set_index(["year", "num_semana"])["y"])

            csv = table.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Descargar forecast (CSV)",
                csv,
                f"forecast_{product}_{group}_{weeks}w.csv",
                "text/csv"
            )

        else:
            forecast_all = forecast_all_products(df_model, model, n_weeks=weeks)

            st.subheader(f"📦 Predicción para TODOS los productos/grupos ({weeks} semanas)")

            table = (
                forecast_all[["product", "group", "year", "num_semana", "week_start", "week_end", "y"]]
                .rename(columns={
                    "product": "Producto",
                    "group": "Group",
                    "year": "Año",
                    "num_semana": "Semana",
                    "week_start": "Desde",
                    "week_end": "Hasta",
                    "y": "Demanda estimada"
                })
            )

            st.dataframe(table)

            resumen = (
                forecast_all.groupby(["product", "group"])["y"].sum()
                .sort_values(ascending=False)
                .reset_index()
                .rename(columns={"product": "Producto", "group": "Group", "y": f"Demanda total ({weeks} semanas)"})
            )

            st.subheader("🔥 Ranking (Producto + Group) por demanda total")
            st.dataframe(resumen.head(25))

            csv = table.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Descargar forecast completo (CSV)",
                csv,
                f"forecast_all_products_groups_{weeks}w.csv",
                "text/csv"
            )

    except Exception as e:
        st.error(str(e))