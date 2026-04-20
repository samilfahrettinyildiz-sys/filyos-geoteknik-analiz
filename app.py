import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata

# Sayfa Yapılandırması
st.set_page_config(page_title="Filyos 3D Geoteknik SaaS", layout="wide")
st.title("🏗️ Filyos Sıvılaşma & Zemin İyileştirme Platformu (V3 - Final)")

# --- 1. SİDEBAR (KONTROLLER) ---
st.sidebar.header("📂 1. Veri Yükleme")
yuklenen_dosya = st.sidebar.file_uploader("Sondaj Verisi Yükle (CSV)", type=['csv'])

st.sidebar.header("⚙️ 2. Deprem Senaryosu")
pga = st.sidebar.slider("Deprem İvmesi (PGA)", 0.10, 0.60, 0.30, 0.05)
mw = st.sidebar.slider("Deprem Büyüklüğü (Mw)", 6.0, 8.0, 7.5, 0.1)

st.sidebar.header("🚜 3. Zemin İyileştirme (SİMÜLATÖR)")
iyilestirme_n = st.sidebar.slider("İyileştirme Etkisi (+N Vuruş)", 0, 30, 0, 1)

st.sidebar.header("🗺️ 4. 3D Isı Haritası Ayarı")
hedef_derinlik = st.sidebar.slider("Yüzey Derinliği (m)", 1.0, 30.0, 10.0, 0.5)

# --- 2. VERİ YÜKLEME VE HESAPLAMA MOTORU ---
@st.cache_data
def veri_isle(dosya_nesnesi):
    if dosya_nesnesi is not None:
        try:
            return pd.read_csv(dosya_nesnesi, sep=';')
        except:
            return pd.read_csv(dosya_nesnesi, sep=',')
    return pd.DataFrame()

try:
    df = veri_isle(yuklenen_dosya)
    
    if not df.empty:
        # Veri Temizleme ve Sayısal Dönüştürme
        sayisal_sutunlar = ['Enlem', 'Boylam', 'Derinlik_m', 'GYS_m', 'N_arazi', 'FC', 'PI']
        for col in sayisal_sutunlar:
            if col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
        
        # Koordinat Dönüşümü
        min_y, min_x = df['Enlem'].min(), df['Boylam'].min()
        is_utm = abs(min_y) > 90 or abs(min_x) > 180

        if is_utm:
            df['Y_m'] = df['Enlem'] - min_y
            df['X_m'] = df['Boylam'] - min_x
        else:
            mean_lat = df['Enlem'].mean()
            df['Y_m'] = (df['Enlem'] - min_y) * 111320.0
            df['X_m'] = (df['Boylam'] - min_x) * (111320.0 * np.cos(np.radians(mean_lat)))

        # Idriss & Boulanger (2008) Analiz Motoru
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

        # HOCANIN İSTEĞİ 1: Renklendirme Mantığı (Güvenli Alanlar Yeşil)
        def get_color(fs, depth, pi):
            if depth > 20.0 or pi > 12 or fs > 2.0: return 'green'
            if fs < 1.0: return 'red'
            if fs < 1.2: return 'orange'
            return 'green'
        df['Renk'] = [get_color(f, d, p) for f, d, p in zip(df['FS'], df['Derinlik_m'], df['PI'])]

        # Sekmeli Yapı
        tab1, tab_gis, tab2, tab3 = st.tabs(["🌍 3D Model", "🗺️ Uydu Haritası", "📏 2D Kesit & Lens", "📊 Veri Tablosu"])

        with tab1:
            fig3d = go.Figure()
            # Sondaj Çubukları
            for s_no in df['Sondaj_No'].unique():
                temp = df[df['Sondaj_No'] == s_no]
                fig3d.add_trace(go.Scatter3d(
                    x=temp['X_m'], y=temp['Y_m'], z=-temp['Derinlik_m'],
                    mode='lines+markers', name=str(s_no),
                    marker=dict(size=6, color=temp['Renk'], line=dict(width=1, color='black')),
                    line=dict(color='lightgray', width=4)
                ))

            # 3D Isı Yüzeyi
            tol = 1.5
            dilim = df[(df['Derinlik_m'] >= hedef_derinlik-tol) & (df['Derinlik_m'] <= hedef_derinlik+tol)]
            if len(dilim['Sondaj_No'].unique()) >= 3:
                isi_veri = dilim.sort_values('Derinlik_m').groupby('Sondaj_No').first().reset_index()
                gx, gy = np.meshgrid(np.linspace(df['X_m'].min()-5, df['X_m'].max()+5, 50),
                                     np.linspace(df['Y_m'].min()-5, df['Y_m'].max()+5, 50))
                gz = griddata(isi_veri[['X_m', 'Y_m']].values, isi_veri['FS'].values, (gx, gy), method='linear')
                
                fig3d.add_trace(go.Surface(
                    x=gx, y=gy, z=np.full(gx.shape, -hedef_derinlik),
                    surfacecolor=gz, colorscale='RdYlGn', cmin=0.5, cmax=1.5,
                    opacity=0.6, name='Sıvılaşma Yüzeyi'
                ))

            fig3d.update_layout(scene=dict(aspectmode='data', zaxis_title='Derinlik (m)'), 
                                template="plotly_dark", height=750)
            st.plotly_chart(fig3d, use_container_width=True)

        with tab_gis:
            try:
                yuzey = df.sort_values('Derinlik_m').groupby('Sondaj_No').first().reset_index()
                fig_map = go.Figure(go.Scattermapbox(
                    lat=yuzey['Enlem'], lon=yuzey['Boylam'], mode='markers+text',
                    marker=dict(size=15, color=yuzey['Renk']), text=yuzey['Sondaj_No'], textposition="top right"
                ))
                fig_map.update_layout(mapbox_style="open-street-map", mapbox=dict(center=dict(lat=yuzey['Enlem'].mean(), lon=yuzey['Boylam'].mean()), zoom=16),
                                      margin=dict(l=0,r=0,b=0,t=0), height=600)
                st.plotly_chart(fig_map, use_container_width=True)
            except:
                st.warning("Harita çizimi için UTM koordinatları yerine Enlem/Boylam gereklidir.")

        # HOCANIN İSTEĞİ 2: 2D Kesit ve Sıvılaşma Lensi (Düzeltilmiş)
        with tab2:
            st.markdown("### 📏 2D Stratigrafik Kesit ve Sıvılaşma Lensi")
            tum_kuyular = sorted(list(df['Sondaj_No'].unique()))
            secili_kuyular = st.multiselect("Kesit Hattı İçin Kuyuları Seçin:", tum_kuyular, default=tum_kuyular[:3])

            if len(secili_kuyular) >= 2:
                fig2d = go.Figure()
                kesit_df = df[df['Sondaj_No'].isin(secili_kuyular)].copy()
                kuyu_x = kesit_df.groupby('Sondaj_No')['X_m'].mean().sort_values()
                sirali_liste = kuyu_x.index.tolist()

                # --- 2D LENS ENTERPOLASYONU ---
                px, pz, vfs = [], [], []
                for k in sirali_liste:
                    temp_k = kesit_df[kesit_df['Sondaj_No'] == k]
                    for _, r in temp_k.iterrows():
                        px.append(kuyu_x[k])
                        pz.append(-r['Derinlik_m'])
                        vfs.append(r['FS'])

                if len(px) > 4:
                    gx, gz = np.meshgrid(np.linspace(min(px), max(px), 100), np.linspace(min(pz), 0, 100))
                    gfs = griddata((px, pz), vfs, (gx, gz), method='linear')
                    
                    # Contour arkada (ZMIN/ZMAX Düzeltmesi Burada)
                    fig2d.add_trace(go.Contour(
                        x=np.linspace(min(px), max(px), 100), y=np.linspace(min(pz), 0, 100), z=gfs,
                        colorscale=[[0, 'red'], [0.3, 'orange'], [0.6, 'green'], [1.0, 'darkgreen']],
                        zmin=0.5, zmax=2.0, # Hata veren cmin yerine zmin kullanıldı
                        opacity=0.4, showscale=True, colorbar=dict(title="FS")
                    ))

                # Kuyu Çubukları önde
                for k in sirali_liste:
                    temp_k = kesit_df[kesit_df['Sondaj_No'] == k]
                    fig2d.add_trace(go.Scatter(
                        x=[kuyu_x[k]]*len(temp_k), y=-temp_k['Derinlik_m'],
                        mode='lines+markers+text', name=str(k),
                        marker=dict(size=14, symbol='square', color=temp_k['Renk'], line=dict(width=2, color='white')),
                        line=dict(color='white', width=3),
                        text=temp_k['Zemin_Sinifi'], textposition="middle left"
                    ))

                fig2d.update_layout(template="plotly_dark", height=600, xaxis_title="Mesafe (m)", yaxis_title="Derinlik (m)", showlegend=False)
                st.plotly_chart(fig2d, use_container_width=True)

        with tab3:
            st.dataframe(df.style.background_gradient(subset=['FS'], cmap='RdYlGn_r'))

    else:
        st.info("Lütfen sol menüden Filyos CSV dosyanızı yükleyin.")
except Exception as e:
    st.error(f"⚠️ Bir Hata Oluştu: {e}")
