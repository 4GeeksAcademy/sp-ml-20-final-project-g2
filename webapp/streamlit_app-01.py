import streamlit as st
import pandas as pd
import pickle

# =====================================================
# CONFIGURACIÓN STREAMLIT
# =====================================================
st.set_page_config(page_title="📦 Demand Forecast", layout="wide")
st.title("📦 Forecast de Demanda por Producto")

# =====================================================
# CARGA DE MODELO Y DATA BASE (fallback)
# =====================================================
model = pickle.load(open("models/71_random_forest_regressor.pkl", "rb"))
df_base = pickle.load(open("data/processed/df-nuevo.pkl", "rb"))

# =====================================================
# LECTURA ROBUSTA CSV (encoding + separador)
# =====================================================
def read_csv_robust(uploaded_file):
    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "ISO-8859-1", "latin1"]

    last_error = None

    for enc in encodings_to_try:
        try:
            uploaded_file.seek(0)

            # 1) Intento: autodetectar separador (python engine)
            df = pd.read_csv(
                uploaded_file,
                encoding=enc,
                sep=None,              # autodetecta , ; \t |
                engine="python",
                on_bad_lines="skip"    # salta líneas rotas
            )

            # Si se leyó como 1 sola columna gigante, intentamos separadores comunes manualmente
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
# FUNCIONES DE PREPROCESAMIENTO (CSV CRUDO)
# =====================================================
def clean_product_name(name):
    if pd.isna(name):
        return "UNKNOWN"
    name = str(name).upper()
    name = (
        name.replace("Á", "A").replace("É", "E").replace("Í", "I")
            .replace("Ó", "O").replace("Ú", "U").replace("Ñ", "N")
    )
    return name.strip()

def normalize_raw_columns(df_raw):
    # limpiar espacios en headers
    df_raw.columns = df_raw.columns.str.strip()

    # mapeo típico (por si viene Cant. / FECHA / etc.)
    rename_map = {
        "Cant.": "Cant",
        "cant.": "Cant",
        "cantidad": "Cant",
        "Cantidad": "Cant",
        "CANTIDAD": "Cant",

        "producto": "Producto",
        "PRODUCTO": "Producto",

        "fecha": "Fecha",
        "FECHA": "Fecha",
    }
    df_raw = df_raw.rename(columns=rename_map)
    return df_raw

def preprocess_raw_data(df_raw):
    df = df_raw.copy()

    # Validar columnas mínimas
    required_cols = {"Fecha", "Producto", "Cant"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Faltan columnas obligatorias: {missing}")

    # Fecha
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Fecha", "Producto", "Cant"])

    # Cant numérica
    df["Cant"] = pd.to_numeric(df["Cant"], errors="coerce")
    df = df.dropna(subset=["Cant"])

    # Producto limpio
    df["product"] = df["Producto"].apply(clean_product_name)

    # Semana ISO
    df["num_semana"] = df["Fecha"].dt.isocalendar().week.astype(int)

    # Agregado semanal
    df_weekly = (
        df.groupby(["product", "num_semana"], as_index=False)
          .agg(y=("Cant", "sum"))
          .sort_values(["product", "num_semana"])
    )

    return df_weekly

def create_lags(df, n_lags=8):
    df = df.sort_values("num_semana").copy()
    for lag in range(1, n_lags + 1):
        df[f"y_lag{lag}"] = df["y"].shift(lag)
    return df.dropna().reset_index(drop=True)

# =====================================================
# FUNCIÓN DE FORECAST
# =====================================================
def forecast_weeks(df, model, product, n_weeks):

    df_prod = df[df["product"] == product].copy()

    if len(df_prod) < 10:
        raise ValueError("No hay suficiente historial para este producto (mínimo 10 semanas).")

    # Orden robusto: con year si existe, si no solo por num_semana
    sort_cols = ["num_semana"]
    if "year" in df_prod.columns:
        sort_cols = ["year", "num_semana"]
    df_prod = df_prod.sort_values(sort_cols)

    last_row = df_prod.iloc[-1].copy()

    # Fechas: usar week_start si existe; si no, fecha estimada
    if "week_start" in df_prod.columns:
        last_week_start = pd.to_datetime(df_prod["week_start"].max())
    else:
        last_week_start = pd.Timestamp.today().normalize()

    future = []

    for i in range(n_weeks):
        new_row = {
            "product": product,
            "num_semana": int(last_row["num_semana"]) + 1,
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
        y_pred = model.predict(X)[0]
        y_pred = max(round(float(y_pred)), 0)

        start_date = last_week_start + pd.Timedelta(weeks=i+1)
        end_date = start_date + pd.Timedelta(days=6)

        new_row["y"] = y_pred
        new_row["week_start"] = start_date.date()
        new_row["week_end"] = end_date.date()

        future.append(new_row)
        last_row = new_row

    return pd.DataFrame(future)

def forecast_all_products(df, model, n_weeks, min_hist_weeks=10):
    all_forecasts = []

    for product in sorted(df["product"].unique()):
        df_prod = df[df["product"] == product]

        # saltar productos con poco historial
        if len(df_prod) < min_hist_weeks:
            continue

        try:
            fc = forecast_weeks(df, model, product, n_weeks)
            fc["product"] = product
            all_forecasts.append(fc)
        except Exception:
            continue

    if not all_forecasts:
        raise ValueError("No hay productos con suficiente historial para predecir.")

    result = pd.concat(all_forecasts, ignore_index=True)

    # ordenar
    sort_cols = ["product", "num_semana"]
    if "year" in df.columns:
        sort_cols = ["product", "year", "num_semana"]

    return result.sort_values(sort_cols).reset_index(drop=True)

# =====================================================
# SIDEBAR – CARGA CSV CRUDO
# =====================================================
st.sidebar.header("📂 Datos de entrada")

uploaded_file = st.sidebar.file_uploader(
    "Sube tu CSV de ventas (raw)",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df_raw = read_csv_robust(uploaded_file)
        df_raw = normalize_raw_columns(df_raw)

        # (Opcional) mostrar preview para debug
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
    df_model = df_base

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

if mode == "Un producto":
    product = st.selectbox(
        "Selecciona producto",
        sorted(df_model["product"].unique()),
        key="product_select_single"
    )
else:
    product = None

# =====================================================
# PREDICCIÓN + RESULTADOS
# =====================================================
if st.button("🔮 Predecir demanda"):
    try:
        if mode == "Un producto":
            forecast = forecast_weeks(df_model, model, product, weeks)

            st.subheader(f"📊 Predicción para {product}")

            table = (
                forecast[["num_semana", "week_start", "week_end", "y"]]
                .rename(columns={
                    "num_semana": "Semana",
                    "week_start": "Desde",
                    "week_end": "Hasta",
                    "y": "Demanda estimada"
                })
            )
            st.dataframe(table)

            st.line_chart(forecast.set_index("num_semana")["y"])

            csv = table.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Descargar forecast (CSV)",
                csv,
                f"forecast_{product}_{weeks}w.csv",
                "text/csv"
            )

        else:
            forecast_all = forecast_all_products(df_model, model, n_weeks=weeks)

            st.subheader(f"📦 Predicción para TODOS los productos ({weeks} semanas)")

            table = (
                forecast_all[["product", "num_semana", "week_start", "week_end", "y"]]
                .rename(columns={
                    "product": "Producto",
                    "num_semana": "Semana",
                    "week_start": "Desde",
                    "week_end": "Hasta",
                    "y": "Demanda estimada"
                })
            )

            st.dataframe(table)

            # Resumen útil: demanda total por producto en el horizonte
            resumen = (
                forecast_all.groupby("product")["y"].sum()
                .sort_values(ascending=False)
                .reset_index()
                .rename(columns={"product": "Producto", "y": f"Demanda total ({weeks} semanas)"})
            )

            st.subheader("🔥 Ranking de productos por demanda total")
            st.dataframe(resumen.head(25))

            csv = table.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Descargar forecast completo (CSV)",
                csv,
                f"forecast_all_products_{weeks}w.csv",
                "text/csv"
            )

    except ValueError as e:
        st.error(str(e))

        # =========================
        # DESCARGA CSV
        # =========================
        csv = forecast[["num_semana", "y"]].to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Descargar forecast en CSV",
            data=csv,
            file_name=f"forecast_{product}.csv",
            mime="text/csv"
        )

    except ValueError as e:
        st.error(str(e))