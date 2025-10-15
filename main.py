# main.py
import random
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.spatial import cKDTree
import plotly.express as px
import math

# ==============================================
# 1. LEITURA E PREPARO DA BASE
# ==============================================
df = pd.read_csv('base_enderecos.csv')
df = df[df['NOME_UF'].isin([
    'São Paulo', 'Rio de Janeiro', 'Espírito Santo', 'Bahia',
    'Santa Catarina', 'Rio Grande do Sul', 'Paraná', 'Goiás'
])].reset_index(drop=True)
df['lat'] = df['lat'].astype(float).round(6)
df['lon'] = df['lon'].astype(float).round(6)

# ==============================================
# 2. PARÂMETROS DEFAULT POR ESTADO
# ==============================================
estado_defaults = {estado: {'RAIO_KM':100, 'META_COBERTURA':90} for estado in df['NOME_UF'].unique()}

# ==============================================
# 3. FUNÇÕES DO ALGORITMO
# ==============================================
def greedy_coverage(df, tree, raio_rad, meta_frac):
    n = len(df)
    covered = np.zeros(n, dtype=bool)
    agentes = []
    coberturas = []

    while covered.mean() < meta_frac:
        uncovered_idx = np.where(~covered)[0]
        if len(uncovered_idx) == 0:
            break

        best_point = None
        best_cover = []
        best_size = 0
        candidates = random.sample(list(uncovered_idx), min(200, len(uncovered_idx)))

        for i in candidates:
            neighbors = tree.query_ball_point(tree.data[i], r=raio_rad)
            uncovered_neighbors = [n for n in neighbors if not covered[n]]
            if len(uncovered_neighbors) > best_size:
                best_size = len(uncovered_neighbors)
                best_cover = uncovered_neighbors
                best_point = i

        if best_point is None:
            break

        covered[best_cover] = True
        agentes.append(best_point)
        coberturas.append(best_size)

    return agentes, covered, coberturas

def algoritmo_guloso(df, RAIO_KM=150, META_COBERTURA=90):
    coords = df[['lat','lon']].to_numpy()
    tree = cKDTree(np.radians(coords))
    raio_rad = RAIO_KM / 6371.0
    meta_frac = META_COBERTURA / 100.0

    agentes_idx, covered_mask, coberturas = greedy_coverage(df, tree, raio_rad, meta_frac)
    df_agentes = df.iloc[agentes_idx].reset_index(drop=True)
    df_agentes['cobertos'] = coberturas
    df_agentes = df_agentes.sort_values('cobertos', ascending=False).reset_index(drop=True)
    return df_agentes, covered_mask, agentes_idx

def associar_pontos_a_agentes(df, df_agentes):
    tree_agentes = cKDTree(df_agentes[['lat','lon']].to_numpy())
    dists, idxs = tree_agentes.query(df[['lat','lon']].to_numpy())
    df['agente_idx'] = df_agentes.index[idxs].values
    df['dist_to_agent_km'] = dists * 6371
    return df

def hex_to_rgba(hex_color, alpha=0.15):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2],16)
    g = int(hex_color[2:4],16)
    b = int(hex_color[4:6],16)
    return f'rgba({r},{g},{b},{alpha})'

def plotar_mapa_com_circulos(df, df_agentes, covered_mask, RAIO_KM=100, mostrar_circulos=True, n_points_circle=50):
    cores = px.colors.qualitative.Plotly
    cores_agentes = {idx: cores[i % len(cores)] for i, idx in enumerate(df_agentes.index)}

    fig = go.Figure()

    # Pontos cobertos
    fig.add_trace(go.Scattermapbox(
        lat=df.loc[covered_mask,'lat'],
        lon=df.loc[covered_mask,'lon'],
        mode='markers',
        marker=dict(
            size=6,
            color=[cores_agentes[i] for i in df.loc[covered_mask,'agente_idx']],
            opacity=0.7
        ),
        name='Cobertos',
        hovertemplate="Agente idx: %{customdata[0]}<br>Lat: %{lat}<br>Lon: %{lon}",
        customdata=np.array([df.loc[covered_mask,'agente_idx']]).T
    ))

    # Pontos não cobertos
    fig.add_trace(go.Scattermapbox(
        lat=df.loc[~covered_mask,'lat'],
        lon=df.loc[~covered_mask,'lon'],
        mode='markers',
        marker=dict(size=6,color='gray',opacity=0.8,symbol='circle'),
        name='Não cobertos',
        hovertemplate="Agente mais próximo: %{customdata[0]}<br>Distância (km): %{customdata[1]:.2f}<br>Lat: %{lat}<br>Lon: %{lon}",
        customdata=np.array([df.loc[~covered_mask,'agente_idx'], df.loc[~covered_mask,'dist_to_agent_km']]).T
    ))

    # Agentes
    fig.add_trace(go.Scattermapbox(
        lat=df_agentes['lat'],
        lon=df_agentes['lon'],
        mode='markers+text',
        marker=dict(size=18,color=[cores_agentes[i] for i in df_agentes.index],symbol='star'),
        text=[str(i) for i in df_agentes.index],
        name='Agentes',
        hovertemplate="Agente idx: %{text}<br>Lat: %{lat}<br>Lon: %{lon}<br>Cobertos: %{customdata[0]}",
        customdata=np.array([df_agentes['cobertos']]).T
    ))

    # Círculos de cobertura
    if mostrar_circulos:
        for idx, agente in df_agentes.iterrows():
            lats,lons = [],[]
            for theta in np.linspace(0,2*math.pi,n_points_circle):
                d_lat = (RAIO_KM/6371.0)*math.cos(theta)
                d_lon = (RAIO_KM/6371.0)*math.sin(theta)/math.cos(math.radians(agente['lat']))
                lats.append(agente['lat'] + math.degrees(d_lat))
                lons.append(agente['lon'] + math.degrees(d_lon))
            fig.add_trace(go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode='lines',
                fill='toself',
                fillcolor=hex_to_rgba(cores[idx % len(cores)], alpha=0.15),
                line=dict(color=cores[idx % len(cores)], width=1),
                showlegend=False,
                opacity=0.6
            ))

    fig.update_layout(
        mapbox=dict(style='carto-positron', center=dict(lat=df['lat'].mean(),lon=df['lon'].mean()),zoom=5),
        margin=dict(l=0,r=0,t=0,b=0),
        height=800,
        showlegend=True
    )
    return fig

# ==============================================
# 4. DASHBOARD STREAMLIT
# ==============================================
st.set_page_config(page_title="Mapa de Cobertura", layout="wide")
st.title("📍 Mapa de Cobertura por Estado")
st.markdown("Ajuste os parâmetros abaixo e visualize o impacto no mapa em tempo real.")

# Layout de sliders e checkbox
col1,col2,col3,col4 = st.columns([2,1,1,1])
with col1:
    estado = st.selectbox("Estado", list(estado_defaults.keys()), index=0)
with col2:
    raio = st.slider("Raio (km)",10,1000,estado_defaults[estado]['RAIO_KM'],step=10)
with col3:
    meta = st.slider(
        "Meta de cobertura (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(estado_defaults[estado]['META_COBERTURA']),
        step=0.1
    )
with col4:
    mostrar_circulos = st.checkbox("Mostrar círculos de cobertura", value=True)

# Inicializa cache por estado
if "agentes_por_estado" not in st.session_state:
    st.session_state["agentes_por_estado"] = {}
if "covered_mask_por_estado" not in st.session_state:
    st.session_state["covered_mask_por_estado"] = {}

# Filtrar estado
df_estado = df[df['NOME_UF']==estado].copy()

if df_estado.empty:
    st.warning("⚠️ Nenhum dado disponível para este estado.")
else:
    df_agentes, covered_mask, agentes_idx = algoritmo_guloso(df_estado, RAIO_KM=raio, META_COBERTURA=meta)
    df_estado = associar_pontos_a_agentes(df_estado, df_agentes)

    # Salvar no session_state
    st.session_state["agentes_por_estado"][estado] = df_agentes
    st.session_state["covered_mask_por_estado"][estado] = covered_mask

    # Cobertura real
    cobertura_real = round(covered_mask.mean() * 100,1)

    # Plotar mapa
    fig = plotar_mapa_com_circulos(df_estado, df_agentes, covered_mask, RAIO_KM=raio, mostrar_circulos=mostrar_circulos)
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

    st.success(f"✅ {len(df_agentes)} agentes necessários para {cobertura_real:.1f}% de cobertura no estado {estado}.")
    st.markdown(f"**Parâmetros selecionados:** Raio = {raio} km | Meta de cobertura = {meta:.1f}%")
    st.markdown("Passe o mouse sobre os pontos sem cobertura para visualizar o agente mais próximo.")

    # =================================================
    # 5. TABELAS LADO A LADO
    # =================================================
    col1_table, col2_table = st.columns(2)

    # Tabela 1: agentes do estado selecionado
    with col1_table:
        st.subheader(f"🏆 Agentes em {estado}")
        df_agentes_resumo = df_agentes.copy()
        df_agentes_resumo['percent_cobertura'] = (df_agentes_resumo['cobertos']/len(df_estado)*100).round(1)
        st.dataframe(df_agentes_resumo[['lat','lon','cobertos','percent_cobertura']])

    # Tabela 2: resumo de todos os estados
    with col2_table:
        st.subheader("📊 Resumo por Estado")
        estados_resumo = []
        for est, df_ag in st.session_state["agentes_por_estado"].items():
            mask = st.session_state["covered_mask_por_estado"][est]
            cobertura = mask.mean()*100
            estados_resumo.append({'Estado': est, 'Qtd Agentes': len(df_ag), 'Cobertura Total (%)': round(cobertura,1)})
        df_estados_resumo = pd.DataFrame(estados_resumo).sort_values('Estado')
        st.dataframe(df_estados_resumo)
