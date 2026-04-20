import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata

# Sayfa Yapılandırması
st.set_page_config(page_title="Filyos 3D Geoteknik SaaS", layout="wide")
st.title("🏗️ Filyos Sıvılaşma & Zemin İyileştirme Platformu (PRO+)")

# --- 1. SİDEBAR (KONTROLLER) ---
st.sidebar.header("📂 1. Veri Yükleme")
yuklenen_dosya = st.sidebar.file_uploader("Sondaj Verisi Yükle (CSV)", type=['csv'])

st.sidebar.header("⚙️ 2. Deprem Senaryosu")
pga = st.sidebar.slider("Deprem İvmesi (PGA)", 0.10, 0.60, 0.30, 0.05)
mw = st.sidebar.slider("Deprem Büyüklüğü (Mw)", 6.0, 8.0, 7.5, 0.1)

st.sidebar.header("🚜 3. Zemin İyileştirme (SİMÜLATÖR)")
iyilestirme_n = st.sidebar.slider("İyileştirme Etkisi (+N Vuruş)", 0, 30, 0, 1)

st.sidebar.header("🗺️ 4. Zemin Isı Haritası")
hedef_derinlik = st.sidebar.slider("Enterpolasyon Derinliği (m)", 1.0, 30.0, 10.0, 0.5)

# --- 2. VERİ YÜKLEME VE MOTOR ---
@st.cache_data
def veri_isle(dosya_nesnesi):
    if dosya_nesnesi is not None:
        try:
            return pd.read_csv(dosya_nesnesi, sep=';')
        except:
            return pd.read_csv(dosya_nesnesi, sep=',') # Virgül ayırıcı denemesi
    return pd.DataFrame()

try:
    df = veri_isle(yuklenen_dosya)
    
    if not df.empty:
        # Veri Temizleme (Sayısal dönüştürme)
        for col in ['Enlem', 'Boylam', 'Derinlik_m', 'GYS_m', 'N_arazi', 'FC', 'PI']:
            if col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].str.replace(',', '.').astype(float)
        
        # Koordinat Sistemi Hesaplama
        min_y, min_x = df['Enlem'].min(), df['Boylam'].min()
        is_utm = abs(min_y) > 90 or abs(min_x) > 180

        if is_utm:
            df['Y_m'] = df['Enlem'] - min_y
            df['X_m'] = df['Boylam'] - min_x
        else:
            mean_lat = df['Enlem'].mean()
            df['Y_m'] = (df['Enlem'] - min_y) * 111320.0
            df['X_m'] = (df['Boylam'] - min_x) * (111320.0 * np.cos(np.radians(mean_lat)))

        # Mühendislik Hesapları (Sıvılaşma FS)
        df['N_hesap'] = df['N_arazi'] + iyilestirme_n
        gamma, pa = 19.0, 100.0
        df['Sigma_v'] = df['Derinlik_m'] * gamma
        df['u'] = np.where(df['Derinlik_m'] > df['GYS_m'], (df['Derinlik_m'] - df['GYS_m']) * 9.81, 0)
        df['Sigma_ve'] = (df['Sigma_v'] - df['u']).replace(0, 0.1)
        
        rd = np.exp(-1.012 - 1.126 * np.sin(df['Derinlik_m']/11.73 + 5.133) + (0.106 + 0.118 * np.sin(df['Derinlik_m']/11.28 + 5.142)) * mw)
        csr = 0.65 * pga * (df['Sigma_v'] / df['Sigma_ve']) * rd
        cn = np.minimum(2.0, (pa / df['Sigma_ve'])**0.5)
        n160cs = (df['N_hesap'] * cn) + np.exp(1.63 + 9.7/(df['FC']+0.1) - (15.7/(df['FC']+0.1))**2)
        crr = np.exp((n160cs/14.1) + (n160cs/126)**2 - (n160cs/23.6)**3 + (n160cs/23.6)**4 - 2.8) * (6.9 * np.exp(-mw/4) - 0.058)
        df['FS'] = (crr / csr).fillna(2.0)

        # GÜNCELLEME 1: Renk Atama (Griler Yeşil Oldu)
        def get_color(fs, depth, pi):
            if depth > 20.0 or pi > 12 or fs > 2.0: return 'green' # Güvenli bölgeler
            if fs < 1.0: return 'red'
            if fs < 1.2: return 'orange'
            return 'green'
        df['Renk'] = [get_color(f, d, p) for f, d, p in zip(df['FS'], df['Derinlik_m'], df['PI'])]

        tab1, tab_gis, tab2, tab3 = st.tabs(["🌍 3D Analiz", "🗺️ Uydu Haritası", "📏 2D Kesit & Lens", "📊 Veri Tablosu"])

        with tab1:
            fig3d = go.Figure()
            # Sondaj Çubukları
            for s_no in df['Sondaj_No'].unique():
                temp = df[df['Sondaj_No'] == s_no]
                fig3d.add_trace(go.Scatter3d(
                    x=temp['X_m'], y=temp['Y_m'], z=-temp['Derinlik_m'],
                    mode='lines+markers', name=str(s_no),
                    marker=dict(size=5, color=temp['Renk'], line=dict(width=1, color='black')),
                    line=dict(color='white', width=3)
                ))

            # Isı Haritası (Yüzey)
            tolerans = 2.0
            dilim = df[(df['Derinlik_m'] >= hedef_derinlik-tolerans) & (df['Derinlik_m'] <= hedef_derinlik+tolerans)]
            
            if len(dilim['Sondaj_No'].unique()) >= 3:
                # Enterpolasyon
                isi_veri = dilim.sort_values('Derinlik_m').groupby('Sondaj_No').first().reset_index()
                grid_x, grid_y = np.meshgrid(np.linspace(df['X_m'].min()-10, df['X_m'].max()+10, 50),
                                             np.linspace(df['Y_m'].min()-10, df['Y_m'].max()+10, 50))
                grid_z = griddata(isi_veri[['X_m', 'Y_m']].values, isi_veri['FS'].values, (grid_x, grid_y), method='nearest')
                
                fig3d.add_trace(go.Surface(
                    x=grid_x, y=grid_y, z=np.full(grid_x.shape, -hedef_derinlik),
                    surfacecolor=grid_z, colorscale='RdYlGn', cmin=0.5, cmax=1.5,
                    opacity=0.7, name='Sıvılaşma Riski'
                ))

            fig3d.update_layout(scene=dict(aspectmode='data', zaxis_title='Derinlik (m)'), 
                                template="plotly_dark", height=700, margin=dict(l=0,r=0,b=0,t=0))
            st.plotly_chart(fig3d, use_container_width=True)

        with tab_gis:
            try:
                import pyproj
                yuzey = df.sort_values('Derinlik_m').groupby('Sondaj_No').first().reset_index()
                lat, lon = yuzey['Enlem'], yuzey['Boylam']
                if is_utm:
                    trans = pyproj.Transformer.from_crs("EPSG:32636", "EPSG:4326", always_xy=True)
                    lon, lat = trans.transform(yuzey['Boylam'].values, yuzey['Enlem'].values)
                
                fig_map = go.Figure(go.Scattermapbox(lat=lat, lon=lon, mode='markers+text',
                    marker=dict(size=12, color=yuzey['Renk']), text=yuzey['Sondaj_No'], textposition="top right"))
                fig_map.update_layout(mapbox_style="open-street-map", mapbox=dict(center=dict(lat=lat.mean(), lon=lon.mean()), zoom=15),
                                      margin=dict(l=0,r=0,b=0,t=0), height=600)
                st.plotly_chart(fig_map, use_container_width=True)
            except Exception as e:
                st.error(f"Harita hatası: {e}")

        # GÜNCELLEME 2: 2D Kesit ve Sıvılaşma Lensi
        with tab2:
            st.markdown("### 📏 2D Stratigrafik Kesit ve Sıvılaşma Lensi")
            tum_kuyular = list(df['Sondaj_No'].unique())
            secili_kuyular = st.multiselect("Kesit Hattındaki Kuyular:", tum_kuyular, default=tum_kuyular[:3] if len(tum_kuyular)>=3 else tum_kuyular)

            if len(secili_kuyular) >= 2:
                fig2d = go.Figure()
                kesit_df = df[df['Sondaj_No'].isin(secili_kuyular)].copy()
                kuyu_konumları = kesit_df.groupby('Sondaj_No')['X_m'].mean().sort_values()
                sirali_kuyular = kuyu_konumları.index.tolist()

                # --- 1. SIVILAŞMA DÜZLEMİ (2D ENTERPOLASYON) ---
                points_x, points_z, values_fs = [], [], []
                
                for kuyu in sirali_kuyular:
                    temp = kesit_df[kesit_df['Sondaj_No'] == kuyu]
                    for _, row in temp.iterrows():
                        points_x.append(kuyu_konumları[kuyu])
                        points_z.append(-row['Derinlik_m'])
                        values_fs.append(row['FS'])

                if len(points_x) > 4:
                    # Kesit alanını küçük piksellere (grid) böl
                    grid_x, grid_z = np.meshgrid(
                        np.linspace(min(points_x), max(points_x), 100),
                        np.linspace(min(points_z), 0, 100) # 0 yüzeyden en dibe kadar
                    )
                    
                    # Lineer Enterpolasyon
                    grid_fs = griddata((points_x, points_z), values_fs, (grid_x, grid_z), method='linear')
                    
                    # Isı haritasını (Contour) arkaya ekle
                    fig2d.add_trace(go.Contour(
                        x=np.linspace(min(points_x), max(points_x), 100),
                        y=np.linspace(min(points_z), 0, 100),
                        z=grid_fs,
                        colorscale=[[0, 'red'], [0.2, 'orange'], [0.5, 'green'], [1.0, 'darkgreen']],
                        cmin=0.5, cmax=2.0,
                        opacity=0.45, 
                        showscale=True,
                        colorbar=dict(title="FS (Risk)", len=0.5, x=1.05),
                        name="Sıvılaşma Lensi",
                        hoverinfo='skip'
                    ))

                # --- 2. KUYU ÇUBUKLARI (Ön Planda) ---
                for kuyu in sirali_kuyular:
                    temp = kesit_df[kesit_df['Sondaj_No'] == kuyu]
                    fig2d.add_trace(go.Scatter(
                        x=[kuyu_konumları[kuyu]]*len(temp), y=-temp['Derinlik_m'],
                        mode='lines+markers+text', 
                        marker=dict(size=14, symbol='square', color=temp['Renk'], line=dict(width=2, color='black')),
                        line=dict(color='white', width=4), 
                        text=temp['Zemin_Sinifi'], textposition="middle right", textfont=dict(color='white', size=11),
                        name=kuyu,
                        hovertemplate="Sondaj: %{name}<br>Derinlik: %{y}m<br>FS: %{customdata[0]:.2f}<extra></extra>", 
                        customdata=temp[['FS']]
                    ))

                fig2d.update_layout(
                    template="plotly_dark", height=600, 
                    xaxis=dict(title='Doğu-Batı Ekseni (Metre)', tickvals=kuyu_konumları.values, ticktext=sirali_kuyular), 
                    yaxis=dict(title='Derinlik (m)'), 
                    showlegend=False,
                    title="Yeraltı Sıvılaşma Lensi ve Kuyu Korelasyonu"
                )
                st.plotly_chart(fig2d, use_container_width=True)
            else:
                st.info("Kesit oluşturmak için lütfen yukarıdan en az 2 kuyu seçin.")

        with tab3:
            st.dataframe(df)

    else:
        st.info("Lütfen sol menüden CSV formatında sondaj verilerinizi yükleyin.")
except Exception as e:
    st.error(f"⚠️ Kritik Bir Hata Oluştu: {e}")
