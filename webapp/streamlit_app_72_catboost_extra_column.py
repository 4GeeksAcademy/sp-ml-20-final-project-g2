import streamlit as st
import pandas as pd
import pickle
import re
import json
import os

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

# Construir ruta absoluta basada en la ubicación del script
base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, "..", "models", "category_keywords.json")

if not os.path.exists(json_path):
    st.error(f"❌ No encuentro el JSON en: {json_path}")
    st.stop()

with open(json_path, "r", encoding="utf-8") as f:
    category_keywords = json.load(f)

# =====================================================
# MAPPING product -> groups (desde df_base, sin tocar EDA)
# =====================================================
product_to_group = (
    df_base[["product", "groups"]]
    .dropna()
    .drop_duplicates(subset=["product"])
    .set_index("product")["groups"]
    .to_dict()
)

# =====================================================
# UTILIDADES: FEATURES QUE ESPERA EL MODELO
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

    drop_cols = {"y", "week_start", "week_end"}
    return [c for c in df_fallback.columns if c not in drop_cols]

MODEL_FEATURES = get_model_features(model, df_base)

# =====================================================
# LECTURA ROBUSTA CSV
# =====================================================
def read_csv_robust(uploaded_file):
    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "ISO-8859-1", "latin1"]
    last_error = None

    for enc in encodings_to_try:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=enc, sep=None, engine="python", on_bad_lines="skip")

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
# PREPROCESAMIENTO
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
        # Cantidad
        "Cant.": "Cant", "cant.": "Cant", "cantidad": "Cant",
        "Cantidad": "Cant", "CANTIDAD": "Cant", "CANT": "Cant",

        # Producto
        "producto": "Producto", "PRODUCTO": "Producto",

        # Grupo (acepta plural y variantes)
        "grupo": "Group", "GRUPO": "Group",
        "group": "Group", "GROUP": "Group",
        "groups": "Group", "GROUPS": "Group", "Groups": "Group",

        # Fecha
        "fecha": "Fecha", "FECHA": "Fecha",
    }
    return df_raw.rename(columns=rename_map)

def preprocess_raw_data(df_raw):
    df = df_raw.copy()

    # ✅ Solo columnas que vienen del raw
    required_cols = {"Fecha", "Producto", "Cant"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Faltan columnas obligatorias: {missing}")

    # Tipos
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True)
    df["Cant"] = pd.to_numeric(df["Cant"], errors="coerce")

    # ✅ NUNCA pedir Group aquí
    df = df.dropna(subset=["Fecha", "Producto", "Cant"])

    # Producto limpio (para el modelo)
    df["product"] = df["Producto"].apply(clean_text)

    # ✅ group derivado desde Producto usando el JSON category_keywords
    df["group"] = df["Producto"].apply(lambda x: assign_category(x, category_keywords))
    df["groups"] = df["group"]  # por compatibilidad si algo usa 'groups'

    # Semana ISO
    iso = df["Fecha"].dt.isocalendar()
    df["year"] = iso.year.astype(int)
    df["num_semana"] = iso.week.astype(int)

    # week_start / week_end
    df["week_start"] = (df["Fecha"] - pd.to_timedelta(df["Fecha"].dt.weekday, unit="D")).dt.normalize()
    df["week_end"] = df["week_start"] + pd.Timedelta(days=6)

    # Agregado semanal
    df_weekly = (
        df.groupby(["product", "group", "year", "num_semana"], as_index=False)
          .agg(
              y=("Cant", "sum"),
              week_start=("week_start", "min"),
              week_end=("week_end", "max"),
          )
          .sort_values(["group", "product", "year", "num_semana"])
          .reset_index(drop=True)
    )

    return df_weekly

def create_lags(df_weekly, n_lags=8):
    df = df_weekly.sort_values(["group", "product", "year", "num_semana"]).copy()
    for lag in range(1, n_lags + 1):
        df[f"y_lag{lag}"] = df.groupby(["product", "group"])["y"].shift(lag)
    return df.dropna().reset_index(drop=True)

# =====================================================
# NORMALIZACIÓN FINAL DEL DF_MODEL (CLAVE PARA EL ERROR 'year')
# =====================================================
def normalize_df_model(df_model: pd.DataFrame) -> pd.DataFrame:
    df = df_model.copy()

    # normalizar nombre group
    if "groups" in df.columns and "group" not in df.columns:
        df = df.rename(columns={"groups": "group"})
    if "Group" in df.columns and "group" not in df.columns:
        df = df.rename(columns={"Group": "group"})

    # asegurar product/group
    if "product" not in df.columns or "group" not in df.columns:
        return df

    # asegurar year/num_semana si faltan (de week_start o Fecha)
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

    # tipos a int si se puede
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "num_semana" in df.columns:
        df["num_semana"] = pd.to_numeric(df["num_semana"], errors="coerce").astype("Int64")

    return df

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

    # ✅ con lags ya creados, con 1 fila basta para predecir hacia adelante
    if len(df_pg) < 1:
        raise ValueError("No hay suficiente historial model-ready (tras lags) para este producto/grupo.")

    # ordenar robusto aunque falte year (pero ya lo normalizamos)
    sort_cols = ["num_semana"]
    if "year" in df_pg.columns:
        sort_cols = ["year", "num_semana"]
    df_pg = df_pg.sort_values(sort_cols).reset_index(drop=True)

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

def forecast_all_products(df_model, model, n_weeks, group_filter=None):
    all_forecasts = []

    df_iter = df_model.copy()
    if group_filter is not None:
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
        raise ValueError("No hay combinaciones producto/grupo con suficiente historial (tras lags) para predecir.")

    result = pd.concat(all_forecasts, ignore_index=True)
    return result.sort_values(["group", "product", "year", "num_semana"]).reset_index(drop=True)

# =====================================================
# SIDEBAR – CARGA CSV CRUDO
# =====================================================
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

Estas columnas **se ignoran automáticamente** si no son necesarias para la predicción.

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

# ✅ normalizar siempre (arregla el 'year' y nombres de group)
df_model = normalize_df_model(df_model)

# Validación mínima
required_cols = {"product", "group"}
missing = required_cols - set(df_model.columns)
if missing:
    st.error(f"El dataset no contiene columnas necesarias: {missing}")
    st.stop()

# =====================================================
# UI PRINCIPAL
# =====================================================
col1, col2 = st.columns(2)

with col1:
    mode = st.radio("Modo de predicción", ["Un producto", "Todos los productos"], horizontal=True)

with col2:
    weeks = st.selectbox("Horizonte de predicción (semanas)", [1, 2, 4, 8])

ALL_GROUPS_LABEL = "✅ Todos los grupos"

# evitar NameError
product = None
group = None
group_filter_all = None

if mode == "Un producto":
    group_options = [ALL_GROUPS_LABEL] + sorted(df_model["group"].dropna().unique().tolist())
    selected_group = st.selectbox("Filtrar por group (opcional)", group_options, key="group_filter_single")
    selected_group_filter = None if selected_group == ALL_GROUPS_LABEL else selected_group

    if selected_group_filter is None:
        product_options = sorted(df_model["product"].dropna().unique().tolist())
    else:
        product_options = sorted(df_model[df_model["group"] == selected_group_filter]["product"].dropna().unique().tolist())

    if not product_options:
        st.error("No hay productos para el group seleccionado.")
        st.stop()

    product = st.selectbox("Selecciona producto", product_options, key="product_select_single")

    if selected_group_filter is not None:
        group = selected_group_filter
        st.caption(f"Group seleccionado: **{group}**")
    else:
        groups_for_product = sorted(df_model[df_model["product"] == product]["group"].dropna().unique().tolist())
        if not groups_for_product:
            st.error("Este producto no tiene groups asociados.")
            st.stop()
        group = st.selectbox("Selecciona group", groups_for_product, key="group_select_single")

else:
    group_options = [ALL_GROUPS_LABEL] + sorted(df_model["group"].dropna().unique().tolist())
    selected_group_all = st.selectbox("Filtrar por group (opcional)", group_options, key="group_filter_all")
    group_filter_all = None if selected_group_all == ALL_GROUPS_LABEL else selected_group_all

# =====================================================
# PREDICCIÓN + RESULTADOS
# =====================================================
if st.button("🔮 Predecir demanda"):
    try:
        if mode == "Un producto":
            forecast = forecast_weeks(df_model, model, product, group, weeks)

            st.subheader(f"📊 Predicción para {product} | {group}")

            # 1) Tabla para mostrar (solo UI)
            table_show_single = (
                forecast[["year", "num_semana", "week_start", "week_end", "y"]]
                .rename(columns={
                    "year": "Año",
                    "num_semana": "Semana",
                    "week_start": "Desde",
                    "week_end": "Hasta",
                    "y": "Demanda estimada"
                })
                .reset_index(drop=True)
            )

            st.dataframe(table_show_single, hide_index=True)

            # 2) DF para gráfico (separado) - índice simple
            # --- KPIs + gráfico bonito (funciona igual con 1 o 8 semanas)
            kpi_total = int(forecast["y"].sum())
            kpi_media = float(forecast["y"].mean())
            idx_max = forecast["y"].idxmax()
            kpi_max = int(forecast.loc[idx_max, "y"])
            idx_max = forecast["y"].idxmax()

            desde_max = pd.to_datetime(forecast.loc[idx_max, "week_start"]).strftime("%d/%m/%Y")
            hasta_max = pd.to_datetime(forecast.loc[idx_max, "week_end"]).strftime("%d/%m/%Y")
            valor_max = int(forecast.loc[idx_max, "y"])

            st.caption("Total horizonte")
            st.markdown(f"### {kpi_total}")

            st.caption("Media semanal")
            st.markdown(f"### {kpi_media:.1f}")

            st.caption("Semana pico")
            st.markdown(f"### {desde_max} → {hasta_max} • {valor_max}")

            # Barras por semana (muy legible, incluso con 1 semana)
            bars_df = forecast[["week_start", "week_end", "y"]].copy()

            ws = pd.to_datetime(bars_df["week_start"])
            we = pd.to_datetime(bars_df["week_end"])

            bars_df["periodo"] = (
                ws.dt.strftime("%d")
                + "–"
                + we.dt.strftime("%d %b")
            )

            bars_df = bars_df.set_index("periodo")[["y"]]

            st.subheader("📊 Forecast semanal")

            if len(bars_df) == 1:
                left, center, right = st.columns([5, 2, 5])

                with center:
                    periodo_unico = bars_df.index[0]
                    valor_unico = int(bars_df["y"].iloc[0])

                    desde = pd.to_datetime(forecast["week_start"].iloc[0]).strftime("%d/%m/%Y")
                    hasta = pd.to_datetime(forecast["week_end"].iloc[0]).strftime("%d/%m/%Y")
                    st.metric(label=f"Forecast ({desde} → {hasta})", value=valor_unico)
                    st.bar_chart(bars_df)

            else:
                st.bar_chart(bars_df)

            # 3) CSV (separado de lo mostrado)
            csv_bytes = table_show_single.copy().to_csv(index=False).encode("utf-8")

            # 🔒 filename seguro (evita .html por caracteres raros)
            safe_product = re.sub(r"[^A-Za-z0-9._-]+", "_", str(product))
            safe_group = re.sub(r"[^A-Za-z0-9._-]+", "_", str(group))
            filename = f"forecast_{safe_product}_{safe_group}_{weeks}w.csv"

            st.download_button(
                label="⬇️ Descargar forecast (CSV)",
                data=csv_bytes,
                file_name=filename,
                mime="text/csv"
            )

        else:
            forecast_all = forecast_all_products(
                df_model, model, n_weeks=weeks, group_filter=group_filter_all
            )

            label_group = "TODOS" if group_filter_all is None else group_filter_all
            st.subheader(f"📦 Predicción ({label_group}) – {weeks} semanas")

            table_all = (
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
                .reset_index(drop=True)
            )

            # Tabla UI
            st.dataframe(table_all, hide_index=True)

            # 📊 Top productos por demanda total (barras)
            TOP_N = 10

            top_products = (
                forecast_all
                .groupby("product", as_index=False)["y"]
                .sum()
                .sort_values("y", ascending=False)
                .head(TOP_N)
            )

            top_products = top_products.rename(columns={
                "product": "Producto",
                "y": f"Demanda total ({weeks} semanas)"
            })

            st.subheader(f"📊 Top {TOP_N} productos por demanda total")
            st.bar_chart(
                top_products.set_index("Producto")
            )

            # Ranking (si lo quieres)
            resumen = (
                forecast_all.groupby(["product", "group"])["y"].sum()
                .sort_values(ascending=False)
                .reset_index()
                .rename(columns={
                    "product": "Producto",
                    "group": "Group",
                    "y": f"Demanda total ({weeks} semanas)"
                })
            )
            st.subheader("🔥 Ranking (Producto + Group) por demanda total")
            st.dataframe(resumen.head(25).reset_index(drop=True), hide_index=True)

            # CSV separado
            csv_all_bytes = table_all.copy().to_csv(index=False).encode("utf-8")
            suffix_raw = "ALL" if group_filter_all is None else f"GROUP_{group_filter_all}"
            suffix_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(suffix_raw))
            filename_all = f"forecast_all_{suffix_safe}_{weeks}w.csv"

            st.download_button(
                label="⬇️ Descargar forecast completo (CSV)",
                data=csv_all_bytes,
                file_name=filename_all,
                mime="text/csv"
            )

    except Exception as e:
        st.error(str(e))