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
st.sidebar.markdown("Sahaya Taş Kolon veya Jet-Grout uygulayarak SPT-N vuruşlarını artırın.")
iyilestirme_n = st.sidebar.slider("İyileştirme Etkisi (+N Vuruş)", 0, 30, 0, 1)

st.sidebar.header("🗺️ 4. Zemin Isı Haritası")
hedef_derinlik = st.sidebar.slider("Enterpolasyon Derinliği (m)", 1.0, 20.0, 5.0, 0.5)

# --- 2. VERİ YÜKLEME VE MOTOR ---
@st.cache_data
def veri_isle(dosya_nesnesi):
    if dosya_nesnesi is not None:
        return pd.read_csv(dosya_nesnesi, sep=';')
    else:
        # Hata vermemesi için boş şablon
        return pd.DataFrame(columns=['Sondaj_No', 'Enlem', 'Boylam', 'Derinlik_m', 'Zemin_Sinifi', 'GYS_m', 'N_arazi', 'FC', 'PI'])

try:
    df = veri_isle(yuklenen_dosya)
    
    if len(df) > 0:
        # Temizlik
        for col in ['Enlem', 'Boylam', 'Derinlik_m', 'GYS_m', 'N_arazi', 'FC', 'PI']:
            if col in df.columns and df[col].dtype == object:
                df[col] = df[col].str.replace(',', '.').astype(float)

        # --- YENİ: OTOMATİK KOORDİNAT SİSTEMİ (UTM vs DERECE) ---
        min_y = df['Enlem'].min()
        min_x = df['Boylam'].min()

        # Enlem değeri 90'dan büyükse, bu kesinlikle UTM (Metre) koordinatıdır.
        if abs(min_y) > 90 or abs(min_x) > 180:
            # UTM Modu: Değerler zaten metre, sadece sahayı 0 noktasına taşı (Çarpan Yok)
            df['Y_Koordinat_m'] = df['Enlem'] - min_y
            df['X_Koordinat_m'] = df['Boylam'] - min_x
        else:
            # Coğrafi Derece Modu: Değerleri metreye çevirmek için 111.320 ile çarp
            mean_lat = df['Enlem'].mean()
            df['Y_Koordinat_m'] = (df['Enlem'] - min_y) * 111320.0
            df['X_Koordinat_m'] = (df['Boylam'] - min_x) * (111320.0 * np.cos(np.radians(mean_lat)))

        # --- SİMÜLATÖR: Zemin İyileştirme Etkisi ---
        df['N_arazi_hesap'] = df['N_arazi'] + iyilestirme_n

        # --- MÜHENDİSLİK HESAPLAMALARI (I&B 2008) ---
        gamma, pa = 19.0, 100.0 
        df['Sigma_v'] = df['Derinlik_m'] * gamma
        df['u'] = np.where(df['Derinlik_m'] > df['GYS_m'], (df['Derinlik_m'] - df['GYS_m']) * 9.81, 0)
        df['Sigma_ve'] = df['Sigma_v'] - df['u']
        
        z_derinlik = df['Derinlik_m']
        df['rd'] = np.exp(-1.012 - 1.126 * np.sin(z_derinlik/11.73 + 5.133) + (0.106 + 0.118 * np.sin(z_derinlik/11.28 + 5.142)) * mw)
        
        df['Sigma_ve'] = df['Sigma_ve'].replace(0, 0.001) 
        df['CSR'] = 0.65 * pga * (df['Sigma_v'] / df['Sigma_ve']) * df['rd']
        df['CN'] = np.minimum(2.0, (pa / df['Sigma_ve'])**0.5)
        
        df['N160'] = df['N_arazi_hesap'] * df['CN'] 
        delta_n = np.exp(1.63 + 9.7/(df['FC']+0.1) - (15.7/(df['FC']+0.1))**2)
        df['N160cs'] = df['N160'] + delta_n
        
        n = df['N160cs']
        df['CRR'] = np.exp((n/14.1) + (n/126)**2 - (n/23.6)**3 + (n/23.6)**4 - 2.8) * (6.9 * np.exp(-mw/4) - 0.058)
        
        df['CSR'] = df['CSR'].replace(0, 0.001)
        df['FS'] = df['CRR'] / df['CSR']

        # Kritik Filtreler
        kil_sarti = (df['PI'] > 12) | (df['Zemin_Sinifi'].str.contains('CH|CL', case=False, na=False))
        kuru_sart = (df['Derinlik_m'] <= df['GYS_m'])
        derinlik_sarti = (df['Derinlik_m'] > 20.0)
        df.loc[kil_sarti | kuru_sart | derinlik_sarti, 'FS'] = 2.0
        
        def renk_ata(row):
            if pd.isna(row['FS']): return 'gray'
            if row['FS'] == 2.0: return 'gray'
            if row['FS'] < 1.0: return 'red'
            if row['FS'] < 1.2: return 'orange'
            return 'green'
        df['Renk'] = df.apply(renk_ata, axis=1)

        # --- SEKMELİ ARAYÜZ (TABS) ---
        tab1, tab2, tab3 = st.tabs(["🌍 3D Saha & Isı Haritası", "📏 2D Geoteknik Kesit", "📊 Rapor & Çıktı"])

        # === SEKME 1: 3D HARİTA ===
        with tab1:
            fig3d = go.Figure()
            
            # Dinamik Kenar Boşlukları ve Sınır Koruması
            max_x_raw, max_y_raw = df['X_Koordinat_m'].max(), df['Y_Koordinat_m'].max()
            
            # Güvenlik Kontrolü: Koordinatlarda hala ekstrem bir durum varsa uyar.
            if max_x_raw > 150000 or max_y_raw > 150000:
                st.error("⚠️ KORDİNAT UYARISI: Veriniz UTM olarak algılandı ancak saha genişliği 150 km'den fazla çıkıyor! CSV dosyanızda yanlış girilmiş (örn. bir sıfırı eksik/fazla) uçuk bir sondaj koordinatı olabilir.")

            max_extent = max(max_x_raw, max_y_raw)
            if max_extent == 0: max_extent = 100
            
            max_x = max(max_x_raw, max_extent * 0.4)
            max_y = max(max_y_raw, max_extent * 0.4)
            pad = max(10.0, max_extent * 0.05)

            # 0 Kotu Zemin Plakası
            fig3d.add_trace(go.Mesh3d(
                x=[-pad, max_x+pad, max_x+pad, -pad], 
                y=[-pad, -pad, max_y+pad, max_y+pad], 
                z=[0, 0, 0, 0],
                i=[0, 0], j=[1, 2], k=[2, 3],
                opacity=0.1, color='white', hoverinfo='none', name='Zemin'
            ))

            # Sondajlar
            for sondaj in df['Sondaj_No'].unique():
                temp = df[df['Sondaj_No'] == sondaj]
                fig3d.add_trace(go.Scatter3d(
                    x=temp['X_Koordinat_m'], y=temp['Y_Koordinat_m'], z=-temp['Derinlik_m'],
                    mode='lines+markers',
                    marker=dict(size=7, color=temp['Renk'], line=dict(width=1, color='black')),
                    line=dict(color='rgba(255,255,255,0.4)', width=5),
                    text=[f"Sondaj: {s}<br>Zemin: {z}<br>FS: {f:.2f}<br>N: {n}" for s,z,f,n in zip(temp['Sondaj_No'], temp['Zemin_Sinifi'], temp['FS'], temp['N_arazi_hesap'])],
                    hoverinfo='text', name=sondaj
                ))

            arama_toleransi = 2.5 
            dilim_df = df[(df['Derinlik_m'] >= hedef_derinlik - arama_toleransi) & (df['Derinlik_m'] <= hedef_derinlik + arama_toleransi)].copy()
            
            dilim_df['FS'] = pd.to_numeric(dilim_df['FS'], errors='coerce')
            dilim_df = dilim_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['X_Koordinat_m', 'Y_Koordinat_m', 'FS'])
            
            if len(dilim_df['Sondaj_No'].unique()) >= 3: 
                dilim_df['Fark'] = abs(dilim_df['Derinlik_m'] - hedef_derinlik)
                isi_veri = dilim_df.sort_values('Fark').groupby('Sondaj_No').first().reset_index()
                
                # Matris Uyuşmazlığı Çözümü (2D Grid)
                xi = np.linspace(-pad, max_x+pad, 50)
                yi = np.linspace(-pad, max_y+pad, 50)
                
                X_grid, Y_grid = np.meshgrid(xi, yi)
                points = isi_veri[['X_Koordinat_m', 'Y_Koordinat_m']].values
                values = isi_veri['FS'].values
                
                grid_z = griddata(points, values, (X_grid, Y_grid), method='nearest')
                
                Z_yuzeyi = np.full((50, 50), -hedef_derinlik, dtype=float) + np.sin(X_grid/5.0)*0.01 + np.cos(Y_grid/5.0)*0.01
                colorscale = [[0, 'red'], [0.25, 'orange'], [0.5, 'green'], [1.0, 'gray']]
                
                fig3d.add_trace(go.Surface(
                    x=X_grid,       
                    y=Y_grid,       
                    z=Z_yuzeyi, 
                    surfacecolor=grid_z, 
                    colorscale=colorscale, 
                    cmin=0.5, cmax=2.0,
                    opacity=0.6, 
                    name=f'{hedef_derinlik}m Isı Haritası', 
                    showscale=True,
                    colorbar=dict(title='Risk (FS)', len=0.5, x=1.1)
                ))
            else:
                kuyu_sayisi = len(dilim_df['Sondaj_No'].unique())
                st.warning(f"⚠️ {hedef_derinlik}m derinliğinde harita çizilemiyor. Bu derinlik aralığında sadece {kuyu_sayisi} kuyu bulundu (Min. 3 kuyu lazım).")

            fig3d.update_layout(
                template="plotly_dark", margin=dict(l=0, r=0, b=0, t=0), height=700,
                scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Derinlik (m)', aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.5))
            )
            st.plotly_chart(fig3d, use_container_width=True)

        # === SEKME 2: 2D KESİT ÇIKARICI ===
        with tab2:
            st.markdown("### 2D Stratigrafik Kesit Hattı")
            
            tum_kuyular = list(df['Sondaj_No'].unique())
            secili_kuyular = st.multiselect("Kesit Hattındaki Kuyular:", tum_kuyular, default=tum_kuyular[:3] if len(tum_kuyular)>=3 else tum_kuyular)

            if len(secili_kuyular) >= 2:
                fig2d = go.Figure()
                kesit_df = df[df['Sondaj_No'].isin(secili_kuyular)].copy()
                kuyu_konumları = kesit_df.groupby('Sondaj_No')['X_Koordinat_m'].mean().sort_values()
                sirali_kuyular = kuyu_konumları.index.tolist()

                for kuyu in sirali_kuyular:
                    temp = kesit_df[kesit_df['Sondaj_No'] == kuyu]
                    x_pos = kuyu_konumları[kuyu]
                    
                    fig2d.add_trace(go.Scatter(
                        x=[x_pos]*len(temp), y=-temp['Derinlik_m'],
                        mode='lines+markers+text',
                        marker=dict(size=18, symbol='square', color=temp['Renk'], line=dict(width=1, color='white')),
                        line=dict(color='gray', width=4),
                        text=temp['Zemin_Sinifi'],
                        textposition="middle right",
                        name=kuyu,
                        hovertemplate="Sondaj: %{name}<br>Derinlik: %{y}m<br>FS: %{customdata[0]:.2f}<extra></extra>",
                        customdata=temp[['FS']]
                    ))

                fig2d.update_layout(
                    template="plotly_dark", height=600,
                    xaxis=dict(title='Doğu-Batı Ekseni (Metre)', tickvals=kuyu_konumları.values, ticktext=sirali_kuyular),
                    yaxis=dict(title='Derinlik (m)'),
                    showlegend=False
                )
                st.plotly_chart(fig2d, use_container_width=True)
            else:
                st.info("Kesit oluşturmak için lütfen yukarıdan en az 2 kuyu seçin.")

        # === SEKME 3: TABLO VE ÇIKTI ===
        with tab3:
            st.markdown("### 📊 Hesaplama Çıktıları")
            gosterilecek_df = df[['Sondaj_No', 'Derinlik_m', 'Zemin_Sinifi', 'N_arazi', 'N_arazi_hesap', 'FS', 'Renk']]
            st.dataframe(gosterilecek_df)
            
            csv_cikti = df.to_csv(index=False, sep=';').encode('utf-8')
            st.download_button(
                label="📥 Tüm Sonuçları Excel/CSV Olarak İndir",
                data=csv_cikti, file_name='Filyos_Analiz_Raporu.csv', mime='text/csv',
            )
    else:
        st.info("Lütfen sol taraftan bir CSV verisi yükleyin.")

except Exception as e:
    st.error(f"⚠️ Kritik Hata: {e}")