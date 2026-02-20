import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(page_title="BIST PRO Tarayıcı", layout="wide", page_icon="📈")

# ─────────────────────────────────────────────────────────────────────────────
# TASARIM
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .stApp { background: #0e1117; color: #ffffff; }
    .stButton>button {
        background: #00ff41; color: #000; font-weight: bold;
        border: none; padding: 15px 30px; border-radius: 10px; width: 100%;
    }
    .stock-card { background: #1f2937; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #00ff41; }
    .metric-box { background: #1f2937; padding: 15px; border-radius: 10px; border: 1px solid #00ff41; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HİSSE LİSTESİ (50 ADET)
# ─────────────────────────────────────────────────────────────────────────────
TICKERS = [
    "THYAO.IS", "ASELS.IS", "GARAN.IS", "AKBNK.IS", "EREGL.IS", "TUPRS.IS",
    "SASA.IS", "KCHOL.IS", "SAHOL.IS", "BIMAS.IS", "MGROS.IS", "FROTO.IS",
    "TOASO.IS", "TCELL.IS", "TTKOM.IS", "HEKTS.IS", "ALARK.IS", "DOHOL.IS",
    "ISCTR.IS", "YKBNK.IS", "HALKB.IS", "VAKBN.IS", "KOZAL.IS", "SOKM.IS",
    "CCOLA.IS", "ANSGR.IS", "PGSUS.IS", "ULKER.IS", "TAVHL.IS", "ISGYO.IS",
    "EKGYO.IS", "VESBE.IS", "BRISA.IS", "DEVA.IS", "GUBRF.IS", "POLHO.IS",
    "CIMSA.IS", "NUHOL.IS", "KARSN.IS", "DOAS.IS", "TTRAK.IS", "MAVI.IS",
    "AEFES.IS", "LOGO.IS", "NETAS.IS", "IHLGM.IS", "OYAKC.IS", "SELEC.IS",
    "FENER.IS", "GSRAY.IS"
]

# ─────────────────────────────────────────────────────────────────────────────
# FONKSİYONLAR
# ─────────────────────────────────────────────────────────────────────────────

def calculate_rsi(close, period=14):
    """RSI hesapla - DÜZELTİLMİŞ"""
    close = np.array(close)
    if len(close) < period + 1:
        return 50.0
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.mean(gain[:period]) if len(gain) >= period else 0
    avg_loss = np.mean(loss[:period]) if len(loss) >= period else 1
    
    for i in range(period, len(gain)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 100
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_fundamentals(ticker):
    """Temel verileri çek"""
    try:
        info = yf.Ticker(ticker).info
        return {
            'pe': info.get('trailingPE'),
            'pb': info.get('priceToBook'),
            'sector': info.get('sector', 'Genel'),
            'market_cap': info.get('marketCap'),
            'earnings_growth': info.get('earningsQuarterlyGrowth'),
            'dividend_yield': info.get('dividendYield')
        }
    except:
        return {'pe': None, 'pb': None, 'sector': 'Genel', 'market_cap': None, 
                'earnings_growth': None, 'dividend_yield': None}

def score_ticker(ticker):
    """Hisse analizi - PROFESYONEL PUANLAMA (Maks 100)"""
    try:
        # Teknik veri
        df = yf.download(ticker, period="6mo", progress=False, timeout=10)
        
        if df is None or len(df) < 60:
            return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if 'Close' not in df.columns:
            return None
        
        df = df.dropna(subset=['Close'])
        if len(df) < 60:
            return None
        
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        vol = df['Volume'].values
        
        current_price = float(close[-1])
        
        # ── TEKNİK ANALİZ (60 Puan) ──────────────────────────────────────────
        teknik_skor = 0
        
        # MA50 (15 puan)
        ma50 = float(np.mean(close[-50:])) if len(close) >= 50 else current_price
        if current_price > ma50:
            teknik_skor += 15
        
        # RSI (15 puan)
        rsi = calculate_rsi(close, 14)
        if 50 <= rsi <= 60:
            teknik_skor += 15  # İdeal bölge
        elif 45 <= rsi < 50:
            teknik_skor += 12
        elif 60 < rsi <= 70:
            teknik_skor += 10
        elif 35 <= rsi < 45:
            teknik_skor += 8
        else:
            teknik_skor += 5
        
        # MACD (15 puan)
        exp1 = pd.Series(close).ewm(span=12, adjust=False).mean()
        exp2 = pd.Series(close).ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_val = float(macd.iloc[-1])
        signal_val = float(signal.iloc[-1])
        hist_val = float((macd - signal).iloc[-1])
        hist_prev = float((macd - signal).iloc[-2]) if len(macd) > 2 else 0
        
        if macd_val > signal_val and hist_val > hist_prev:
            teknik_skor += 15  # Güçlü alı
        elif macd_val > signal_val:
            teknik_skor += 10
        elif hist_val > 0:
            teknik_skor += 5
        
        # Momentum (15 puan)
        momentum = 0
        if len(close) >= 22:
            momentum = ((close[-1] / close[-21]) - 1) * 100
        
        if momentum > 10:
            teknik_skor += 15
        elif momentum > 5:
            teknik_skor += 12
        elif momentum > 0:
            teknik_skor += 8
        elif momentum > -5:
            teknik_skor += 4
        
        teknik_skor = min(teknik_skor, 60)
        
        # ── TEMEL ANALİZ (40 Puan) ───────────────────────────────────────────
        temel_skor = 0
        fundamentals = get_fundamentals(ticker)
        
        pe = fundamentals['pe']
        pb = fundamentals['pb']
        sector = fundamentals['sector']
        earnings_growth = fundamentals['earnings_growth']
        dividend_yield = fundamentals['dividend_yield']
        
        # F/K (15 puan)
        if pe and pe > 0:
            if pe < 10:
                temel_skor += 15
            elif pe < 15:
                temel_skor += 12
            elif pe < 20:
                temel_skor += 8
            elif pe < 30:
                temel_skor += 4
            else:
                temel_skor += 2
        else:
            temel_skor += 5  # Veri yoksa nötr
        
        # PD/DD (15 puan)
        if pb and pb > 0:
            if pb < 2:
                temel_skor += 15
            elif pb < 3:
                temel_skor += 12
            elif pb < 5:
                temel_skor += 8
            elif pb < 8:
                temel_skor += 4
            else:
                temel_skor += 2
        else:
            temel_skor += 5
        
        # Kar Büyümesi (10 puan)
        if earnings_growth:
            if earnings_growth > 0.5:
                temel_skor += 10
            elif earnings_growth > 0.25:
                temel_skor += 8
            elif earnings_growth > 0.1:
                temel_skor += 6
            elif earnings_growth > 0:
                temel_skor += 4
            else:
                temel_skor += 0
        else:
            temel_skor += 3
        
        temel_skor = min(temel_skor, 40)
        
        # ── TOPLAM ───────────────────────────────────────────────────────────
        toplam_skor = teknik_skor + temel_skor
        
        # MACD Label
        if macd_val > signal_val and hist_val > hist_prev:
            macd_label = "🔥 Güçlü Alı"
        elif macd_val > signal_val:
            macd_label = "✅ Pozitif"
        else:
            macd_label = "❌ Negatif"
        
        # Hacim kontrolü
        vol_5d = np.mean(vol[-5:]) if len(vol) >= 5 else 0
        vol_20d = np.mean(vol[-20:]) if len(vol) >= 20 else 0
        volume_ok = vol_5d > vol_20d if vol_20d > 0 else False
        
        return {
            'Hisse': ticker.replace('.IS', ''),
            'Fiyat': round(current_price, 2),
            'Toplam Skor': round(toplam_skor, 1),
            'Teknik Skor': round(teknik_skor, 1),
            'Temel Skor': round(temel_skor, 1),
            'RSI': round(rsi, 1),
            'MACD': macd_label,
            'Momentum%': round(momentum, 2),
            'F/K': round(pe, 1) if pe else 'N/A',
            'PD/DD': round(pb, 1) if pb else 'N/A',
            'Sektör': sector,
            'Kar Büyümesi': f"{earnings_growth*100:.1f}%" if earnings_growth else 'N/A',
            'Hacim': '✅' if volume_ok else '❌',
            'MA50 Üstü': '✅' if current_price > ma50 else '❌'
        }
    except Exception as e:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# ANA UYGULAMA
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.title("🚀 BIST PRO Hisse Tarayıcı")
    st.markdown("**Temel + Teknik Analiz | Maks 100 Puan | 70+ = AL**")
    st.info("⚠️ Yatırım Tavsiyesi Değildir. Veriler 15 dk gecikmeli olabilir.")
    
    st.sidebar.header("⚙️ Ayarlar")
    min_score = st.sidebar.slider("Minimum AL Skoru", 50, 90, 70, 5)
    max_stocks = st.sidebar.slider("Taranacak Hisse Sayısı", 20, 50, 50, 5)
    show_all = st.sidebar.checkbox("Tüm Hisseleri Göster", value=False)
    
    st.divider()
    
    if st.button("🚀 TARAMAYI BAŞLAT"):
        with st.spinner('⏳ Taranıyor... (2-3 dakika)'):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            error_count = 0
            
            for i, ticker in enumerate(TICKERS[:max_stocks]):
                status_text.text(f"🔍 {ticker} ({i+1}/{max_stocks})")
                result = score_ticker(ticker)
                if result:
                    results.append(result)
                else:
                    error_count += 1
                progress_bar.progress((i + 1) / max_stocks)
            
            status_text.empty()
            progress_bar.empty()
            
            st.divider()
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🔍 Taranan", max_stocks)
            c2.metric("✅ Bulunan", len(results))
            c3.metric("🚀 AL Listesi", len([r for r in results if r['Toplam Skor'] >= min_score]))
            c4.metric("⚠️ Hata", error_count)
            
            if not results:
                st.error("⚠️ Hiç veri alınamadı. 2 dakika bekleyip tekrar deneyin.")
                st.stop()
            
            df = pd.DataFrame(results)
            df = df.sort_values('Toplam Skor', ascending=False).reset_index(drop=True)
            df_al = df[df['Toplam Skor'] >= min_score]
            
            if len(df_al) < 3 and not show_all:
                df_al = df.head(10)
            
            st.divider()
            
            if not df_al.empty:
                st.subheader(f"🏆 En İyi 5 Hisse ({min_score}+ Puan)")
                cols = st.columns(min(5, len(df_al)))
                for idx, (_, row) in enumerate(df_al.head(5).iterrows()):
                    with cols[idx]:
                        emoji = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉" if idx == 2 else "⭐"
                        st.markdown(f"""
                        <div class="stock-card">
                            <h4>{emoji} {row['Hisse']}</h4>
                            <b>Fiyat:</b> {row['Fiyat']} ₺<br>
                            <b>Toplam Skor:</b> <span style="color:#00ff41;font-size:18px">{row['Toplam Skor']}/100</span><br>
                            <b>Teknik:</b> {row['Teknik Skor']}/60 | <b>Temel:</b> {row['Temel Skor']}/40<br>
                            <b>RSI:</b> {row['RSI']} | <b>MACD:</b> {row['MACD']}<br>
                            <b>F/K:</b> {row['F/K']} | <b>PD/DD:</b> {row['PD/DD']}<br>
                            <b>Sektör:</b> {row['Sektör']}<br>
                            <b>Momentum:</b> %{row['Momentum%']}
                        </div>
                        """, unsafe_allow_html=True)
                
                st.divider()
                st.subheader("📊 Tüm AL Listesi")
                display_cols = ['Hisse', 'Fiyat', 'Toplam Skor', 'Teknik Skor', 'Temel Skor',
                               'RSI', 'MACD', 'F/K', 'PD/DD', 'Sektör', 'Momentum%', 'Kar Büyümesi', 'Hacim']
                st.dataframe(df_al[display_cols], use_container_width=True, hide_index=True)
                
                csv = df_al[display_cols].to_csv(index=False, encoding='utf-8-sig')
                st.download_button("📥 CSV İndir", csv, "bist_pro_tarama.csv", "text/csv")
                
                # Grafik
                st.divider()
                st.subheader("📈 Skor Dağılımı (İlk 15)")
                df_chart = df_al.head(15)
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df_chart['Hisse'], y=df_chart['Teknik Skor'], 
                                    name='Teknik (60)', marker_color='#4A90D9'))
                fig.add_trace(go.Bar(x=df_chart['Hisse'], y=df_chart['Temel Skor'], 
                                    name='Temel (40)', marker_color='#F4A83A'))
                fig.add_hline(y=min_score, line_dash='dash', line_color='red', 
                             annotation_text=f"AL Eşiği ({min_score})")
                fig.update_layout(template='plotly_dark', barmode='stack', 
                                 title='Teknik + Temel Skor Dağılımı', height=400,
                                 xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"✅ Tarama Tamamlandı! {len(df_al)} hisse {min_score}+ puan aldı.")
            else:
                st.warning(f"⚠️ {min_score}+ puan alan bulunamadı. Skoru düşürün.")
    
    else:
        st.info("⬅️ Ayarları yapıp **Taramayı Başlat** butonuna basın.")
        
        with st.expander("📖 Puanlama Sistemi Detayları"):
            st.markdown("""
            ### 📊 Toplam 100 Puan Üzerinden
            
            #### 🔵 Temel Analiz (40 Puan)
            | Kriter | Maks Puan | Detay |
            |--------|-----------|-------|
            | F/K Oranı | 15 | <10 = 15 puan, >30 = 2 puan |
            | PD/DD | 15 | <2 = 15 puan, >8 = 2 puan |
            | Kar Büyümesi | 10 | >%50 = 10 puan, negatif = 0 |
            
            #### 🟠 Teknik Analiz (60 Puan)
            | Kriter | Maks Puan | Detay |
            |--------|-----------|-------|
            | MA50 Üzeri | 15 | Fiyat > MA50 |
            | RSI | 15 | 50-60 arası ideal (15 puan) |
            | MACD | 15 | Crossover + histogram büyüme |
            | Momentum | 15 | 1 aylık getiri >%10 = tam puan |
            
            ### 🎯 AL Sinyali: 70+ Puan
            - **80+**: Çok Güçlü Al
            - **70-79**: Güçlü Al
            - **60-69**: İzle
            - **<60**: Bekle
            """)

if __name__ == "__main__":
    main()
