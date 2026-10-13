"""
Laboratorio de Recomendación — App de Streamlit (Actividad 3, Grupo 7)
Construida a partir de streamlit_starter.py.

Permite a alguien de negocio explorar los recomendadores del carrusel "Para ti":
elegir un cliente y un método, ver sus 3 recomendaciones con el "porque",
comparar los métodos (hit rate, precisión y cobertura en el grupo control)
y revisar el resultado del experimento A/B.

Correr con:  streamlit run app_recomendador.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Recomendador — Laboratorio", page_icon="☕", layout="wide")

VERDE = "#00704A"
NARANJA = "#C4512F"
GRIS = "#9A988E"
DATA = Path(__file__).parent / "data"   # funciona igual local y en Streamlit Cloud


@st.cache_data
def cargar():
    catalogo = pd.read_csv(DATA / "catalogo.csv")
    hist = pd.read_csv(DATA / "ordenes_historial.csv")
    hold = pd.read_csv(DATA / "ordenes_holdout.csv")
    exp = pd.read_csv(DATA / "experimento.csv")
    clientes = pd.read_csv(DATA / "clientes.csv")
    productos = list(catalogo["item"])
    M = (pd.crosstab(hist["customer_id"], hist["item"]) > 0).astype(int).reindex(
        index=clientes["customer_id"], columns=productos, fill_value=0)
    X = M.values.astype(np.float32)
    H = (pd.crosstab(hold["customer_id"], hold["item"]) > 0).astype(int).reindex(
        index=M.index, columns=productos, fill_value=0).values
    NUEVO = H * (1 - X)                                   # probó algo NUEVO en el holdout
    NUEVO[:, productos.index("Agua")] = 0
    es_control = (exp.set_index("customer_id").loc[M.index, "grupo"] == "control").values

    # Ingreso y margen de 90 días por cliente (para la pestaña del experimento)
    precio = catalogo.set_index("item")["precio"]
    margen = catalogo.set_index("item")["margen"]
    hold["ingreso"] = hold["item"].map(precio)
    hold["margen_usd"] = hold["ingreso"] * hold["item"].map(margen)
    por_cliente = hold.groupby("customer_id")[["ingreso", "margen_usd"]].sum().reindex(
        clientes["customer_id"]).fillna(0)
    ab = exp.set_index("customer_id").join(por_cliente)
    return productos, M.index.values, X, NUEVO, es_control, ab


productos, ids, X, NUEVO, es_control, ab = cargar()
COL_AGUA = productos.index("Agua")
popularidad = X.sum(axis=0)

# TODO 1: tabla de similitudes producto × producto (diagonal en 0), calculada UNA vez
S_it = cosine_similarity(X.T)
np.fill_diagonal(S_it, 0)


@st.cache_data
def svd_puntajes(n_factores):
    svd = TruncatedSVD(n_components=n_factores, random_state=42)
    Z = svd.fit_transform(X)
    return (Z @ svd.components_).astype(np.float32)


@st.cache_data
def vecinos_todos(K, bloque=1000):
    """Los K vecinos más parecidos de TODOS los clientes (por bloques para no usar 256 MB de RAM)."""
    idx = np.zeros((len(X), K), dtype=np.int32)
    sim = np.zeros((len(X), K), dtype=np.float32)
    for ini in range(0, len(X), bloque):
        S = cosine_similarity(X[ini:ini + bloque], X)
        S[np.arange(len(S)), np.arange(ini, ini + len(S))] = 0   # uno mismo no cuenta
        top = np.argpartition(-S, K, axis=1)[:, :K]
        idx[ini:ini + bloque] = top
        sim[ini:ini + bloque] = np.take_along_axis(S, top, axis=1)
    return idx, sim


def puntajes(metodo, x_vec, K, n_factores, fila):
    """Puntajes de los 29 productos para UN cliente (x_vec = su vector de 0/1)."""
    if metodo == "Popularidad":
        return popularidad.astype(float), None
    if metodo == "Filtrado por ítem":
        # TODO 2: suma de las similitudes de cada producto con lo que el cliente ya compró
        return x_vec @ S_it, None
    if metodo == "SVD (factores ocultos)":
        return svd_puntajes(n_factores)[fila], None
    # TODO 3: filtrado por cliente: similitud con TODOS, los K vecinos más parecidos
    #         y la suma de sus compras ponderada por esa similitud
    sims = cosine_similarity(x_vec[None, :], X)[0]
    sims[fila] = 0
    vec = np.argsort(-sims)[:K]
    return sims[vec] @ X[vec], vec


def top3(scores, x_vec):
    s = np.where(x_vec > 0, -np.inf, scores).astype(float)   # no recomendar lo ya comprado
    s[COL_AGUA] = -np.inf                                     # ni el Agua
    return np.argsort(-s)[:3]


def porque(metodo, j, x_vec, vec, K):
    """El 'porque' de recomendar el producto j."""
    if metodo == "Popularidad":
        return f"es de lo más comprado ({int(popularidad[j]):,} clientes ya lo compraron)"
    if metodo == "Filtrado por ítem":
        comprados = np.where(x_vec > 0)[0]
        p = comprados[np.argmax(S_it[comprados, j])]
        return f"porque compró {productos[p]} (similitud {S_it[p, j]:.2f})"
    if metodo == "Filtrado por cliente":
        n = int(X[vec, j].sum())
        return f"{n} de sus {K} vecinos más parecidos lo compraron"
    return "clientes con gustos parecidos a los suyos (según los factores ocultos del SVD) lo compran"


# --- Título ---
st.title("☕ Recomendador de productos — carrusel \"Para ti\"")
st.caption("Ayuda a decidir **qué recomendador** poner en el carrusel de la app y **qué productos** "
           "mostrarle a cada cliente para aumentar el ARPU (venta cruzada).")

# TODO 4: barra lateral con los controles interactivos
METODOS = ["Filtrado por ítem", "Filtrado por cliente", "SVD (factores ocultos)", "Popularidad"]
with st.sidebar:
    st.header("Parámetros")
    metodo = st.selectbox("Método de recomendación", METODOS)
    K = st.slider("Vecinos K (filtrado por cliente)", min_value=5, max_value=200, value=30, step=5)
    n_factores = st.slider("Factores del SVD", min_value=2, max_value=20, value=8)
    respaldo = st.toggle("Usar popularidad para clientes SIN historial (arranque en frío)", value=True)
    st.divider()
    st.caption("Datos: 8,000 clientes · ~10 meses de historial · experimento de 90 días. "
               "Los efectos del dataset están plantados por el generador del curso.")

tab1, tab2, tab3 = st.tabs(["Recomendar", "Comparar métodos", "Experimento A/B"])

with tab1:
    cid = st.number_input("ID de cliente", min_value=int(ids.min()), max_value=int(ids.max()), value=14, step=1)
    fila = int(np.where(ids == cid)[0][0])
    x_vec = X[fila].copy()
    compro = [productos[j] for j in np.where(x_vec > 0)[0]]
    grupo = "control (no vio el carrusel)" if es_control[fila] else "tratamiento (vio el carrusel)"
    c1, c2 = st.columns([2, 1])
    c1.write(f"**Ya compró ({len(compro)}):** " + (", ".join(compro) if compro else "— sin historial —"))
    c2.write(f"**Grupo del experimento:** {grupo}")

    # TODO 5: las 3 recomendaciones con su puntaje y su "porque"
    metodo_usado = metodo
    if x_vec.sum() == 0 and metodo != "Popularidad":
        if respaldo:
            metodo_usado = "Popularidad"
            st.warning("Cliente **sin historial** (arranque en frío): los métodos colaborativos darían "
                       "puntaje 0 a todo, así que se usa la **popularidad** como respaldo.")
        else:
            st.warning("Cliente **sin historial**: los métodos colaborativos dan puntaje 0 a todos los "
                       "productos, así que la recomendación es prácticamente al azar. Activen el respaldo.")

    scores, vec = puntajes(metodo_usado, x_vec, K, n_factores, fila)
    recs = top3(scores, x_vec)
    st.subheader(f"Top 3 para el cliente {cid} · {metodo_usado}")
    cols = st.columns(3)
    for col, (pos, j) in zip(cols, enumerate(recs, start=1)):
        with col:
            st.metric(f"#{pos}", productos[j], f"puntaje {scores[j]:.2f}", delta_color="off")
            st.caption(porque(metodo_usado, j, x_vec, vec, K))

    probo = [productos[j] for j in np.where(NUEVO[fila] > 0)[0]]
    acierto = [productos[j] for j in recs if NUEVO[fila, j] > 0]
    with st.expander("¿Qué compró de nuevo en los 90 días del experimento? (holdout)"):
        st.write("Productos nuevos que probó:", ", ".join(probo) if probo else "ninguno")
        if acierto:
            st.success(f"✅ Acierto: {', '.join(acierto)} estaba en el top 3.")
        elif probo:
            st.info("Ninguna de las 3 recomendaciones fue un producto que probó.")

with tab2:
    # TODO 6: tabla de comparación (hit rate@3, precisión@3 y cobertura) en el grupo CONTROL
    st.markdown("Evaluado con el **holdout de 90 días**, solo en clientes del **grupo control** que "
                "probaron algo nuevo (sus compras no fueron influidas por el carrusel).")
    mask_eval = es_control & (NUEVO.sum(axis=1) > 0)
    n_max = int(mask_eval.sum())
    n_eval = st.slider("¿Cuántos clientes de control evaluar?", min_value=200, max_value=n_max,
                       value=n_max, step=100)
    filas_eval = np.where(mask_eval)[0][:n_eval]

    def top_k_matriz(S, K_top=3):
        s = np.where(X > 0, -np.inf, S).astype(np.float64)
        s[:, COL_AGUA] = -np.inf
        return np.argsort(-s, axis=1)[:, :K_top]

    with st.spinner("Calculando los recomendadores para los 8,000 clientes…"):
        idx, sim = vecinos_todos(K)
        S_user = np.einsum("uk,ukp->up", sim, X[idx])
        S_item = X @ S_it
        S_pop = np.tile(popularidad, (len(X), 1)).astype(np.float32)
        tiene_hist = (X.sum(axis=1) > 0)[:, None]
        matrices = {
            "Popularidad": S_pop,
            "Filtrado por ítem": S_item,
            f"Filtrado por cliente (K={K})": S_user,
            f"SVD ({n_factores} factores)": svd_puntajes(n_factores),
            "Final (ítem + respaldo popularidad)": np.where(tiene_hist, S_item, S_pop),
        }
        filas_tabla = {}
        for nombre, S in matrices.items():
            top = top_k_matriz(S)
            hits = NUEVO[np.arange(len(X))[:, None], top]
            filas_tabla[nombre] = {
                "Hit rate@3 (control)": hits[filas_eval].sum(axis=1).astype(bool).mean(),
                "Precisión@3": hits[filas_eval].mean(),
                "Cobertura (de 28)": len(np.unique(top)),
            }
        res = pd.DataFrame(filas_tabla).T

    st.dataframe(res.style.format({"Hit rate@3 (control)": "{:.3f}", "Precisión@3": "{:.3f}",
                                   "Cobertura (de 28)": "{:.0f}"})
                 .highlight_max(subset=["Hit rate@3 (control)"], color="#cfe8dc"),
                 width="stretch")
    st.bar_chart(res["Hit rate@3 (control)"] * 100, color=VERDE, horizontal=True,
                 x_label="Hit rate@3 en control (%)")
    mejor = res["Hit rate@3 (control)"].idxmax()
    base = res.loc["Popularidad", "Hit rate@3 (control)"]
    st.info(f"**{mejor}** acierta {res.loc[mejor, 'Hit rate@3 (control)']:.1%} contra {base:.1%} de la "
            f"popularidad. Ojo: un **acierto no es causa** — en el control esas compras ocurrieron sin "
            f"ninguna recomendación. El ingreso extra solo se mide con un experimento.")

with tab3:
    st.markdown("Experimento de 90 días: a la mitad de los clientes (al azar) la app les mostró un carrusel "
                "con los **3 productos más populares** que aún no compraban. La otra mitad (control) no vio nada.")
    costo = st.number_input("Costo del carrusel (US$ por cliente tratado en 90 días)", 0.0, 2.0, 0.20, 0.05)
    trat = ab[ab["grupo"] == "tratamiento"]
    ctrl = ab[ab["grupo"] == "control"]
    d_arpu = trat["ingreso"].mean() - ctrl["ingreso"].mean()
    ee_arpu = np.sqrt(trat["ingreso"].var() / len(trat) + ctrl["ingreso"].var() / len(ctrl))
    d_margen = trat["margen_usd"].mean() - ctrl["margen_usd"].mean() - costo
    ee_margen = np.sqrt(trat["margen_usd"].var() / len(trat) + ctrl["margen_usd"].var() / len(ctrl))
    factor_vida = 0.78 / (1 + 0.03 - 0.78)
    m1, m2, m3 = st.columns(3)
    m1.metric("ΔARPU (90 días)", f"${d_arpu:+.2f}",
              f"IC 95% {d_arpu - 1.96 * ee_arpu:+.2f} a {d_arpu + 1.96 * ee_arpu:+.2f}", delta_color="off")
    m2.metric("Δmargen neto por cliente", f"${d_margen:+.2f}",
              f"IC 95% {d_margen - 1.96 * ee_margen:+.2f} a {d_margen + 1.96 * ee_margen:+.2f}", delta_color="off")
    m3.metric("ΔCLV por cliente", f"${d_margen * factor_vida:+.2f}",
              f"factor de vida {factor_vida:.2f} trimestres (r=0.78, d=0.03)", delta_color="off")
    st.caption(f"ARPU control ${ctrl['ingreso'].mean():.2f} · tratamiento ${trat['ingreso'].mean():.2f} · "
               f"ΔCLV para los {len(trat):,} tratados ≈ ${d_margen * factor_vida * len(trat):+,.0f}")
