import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(page_title="BIST Swing Trade", layout="wide", page_icon="📈")

# ─────────────────────────────────────────────────────────────────────────────
# TASARIM (CSS)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #0e1117 0%, #1a1d24 100%); color: #ffffff; }
    .stButton>button {
        background: linear-gradient(135deg, #00ff41 0%, #00cc33 100%);
        color: #000; font-weight: bold; border: none;
        padding: 15px 30px; border-radius: 10px; width: 100%;
        box-shadow: 0 4px 15px rgba(0, 255, 65, 0.3);
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0, 255, 65, 0.5); }
    .metric-card { background: #1f2937; padding: 20px; border-radius: 10px; border: 1px solid #00ff41; }
    .stock-card { background: #1f2937; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #00ff41; }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HİSSE LİSTESİ (100 ADET)
# ─────────────────────────────────────────────────────────────────────────────
BIST_TICKERS = [
    "THYAO.IS","EREGL.IS","GARAN.IS","AKBNK.IS","YKBNK.IS","ISCTR.IS","KCHOL.IS",
    "SASA.IS","BIMAS.IS","FROTO.IS","TUPRS.IS","ASELS.IS","TOASO.IS","PGSUS.IS",
    "HALKB.IS","VAKBN.IS","TKFEN.IS","ENKAI.IS","KOZAL.IS","KRDMD.IS","PETKM.IS",
    "TTKOM.IS","TAVHL.IS","OTKAR.IS","SAHOL.IS","ARCLK.IS","VESTL.IS","MGROS.IS",
    "EKGYO.IS","ULKER.IS","TCELL.IS","SISE.IS","DOHOL.IS","AEFES.IS","LOGO.IS",
    "MAVI.IS","NETAS.IS","BRISA.IS","CCOLA.IS","IHLGM.IS","ALARK.IS","AKSEN.IS",
    "AYGAZ.IS","TSKB.IS","KLNMA.IS","ISGYO.IS","SODA.IS","CIMSA.IS","OYAKC.IS",
    "ADANA.IS","HEKTS.IS","DOAS.IS","TTRAK.IS","KARSN.IS","BSOKE.IS","GUBRF.IS",
    "SELEC.IS","ISDMR.IS","FENER.IS","GSRAY.IS","BJKAS.IS","TRKCM.IS","AKGRT.IS",
    "ANSGR.IS","RAYSG.IS","ALKIM.IS","KUTPO.IS","ERBOS.IS","DMSAS.IS","YATAS.IS",
    "DENGE.IS","ODAS.IS","KFEIN.IS","GRSEL.IS","VKGYO.IS","RYGYO.IS","TRGYO.IS",
    "OZGYO.IS","ALGYO.IS","PKART.IS","SOKM.IS","CELHA.IS","DYOBY.IS","EGEEN.IS",
    "EKIZ.IS","GOODY.IS","HATEK.IS","IHLAS.IS","KORDS.IS","LIDER.IS","OSMEN.IS",
    "PENGD.IS","QNBFB.IS","SKBNK.IS","TATGD.IS","TEBNK.IS","TMPOL.IS","VERUS.IS",
    "YAPRK.IS","YESIL.IS","YGGYO.IS","ZEDUR.IS","AKFGY.IS","AKPAZ.IS","ALTNY.IS"
]

# ─────────────────────────────────────────────────────────────────────────────
# FONKSİYONLAR
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_sector_stats():
    stats = {}
    sample = BIST_TICKERS[:50]
    for ticker in sample:
        try:
            info = yf.Ticker(ticker).info
            sector = info.get('sector', 'Genel')
            pe = info.get('trailingPE')
            pb = info.get('priceToBook')
            if sector not in stats:
                stats[sector] = {'pe': [], 'pb': []}
            if pe and pe > 0:
                stats[sector]['pe'].append(pe)
            if pb and pb > 0:
                stats[sector]['pb'].append(pb)
        except Exception:
            pass
    
    result = {}
    for sector, data in stats.items():
        result[sector] = {
            'pe_mean': np.mean(data['pe']) if data['pe'] else 15,
            'pb_mean': np.mean(data['pb']) if data['pb'] else 3
        }
    return result

def calculate_rsi(close, period=14):
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[:period]) if len(gain) >= period else 0
    avg_loss = np.mean(loss[:period]) if len(loss) >= period else 1
    rsi = 50
    for i in range(period, len(close)):
        avg_gain = (avg_gain * (period-1) + gain[i]) / period
        avg_loss = (avg_loss * (period-1) + loss[i]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else 100
        rsi = 100 - (100 / (1 + rs))
    return rsi

def score_ticker(ticker, sector_stats):
    try:
        df = yf.download(ticker, period="1y", progress=False)
        if df is None or len(df) < 60:
            return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.dropna(subset=['Close', 'High', 'Low', 'Volume'])
        if len(df) < 60:
            return None
        
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        vol = df['Volume'].values
        
        current_price = float(close[-1])
        ma50 = float(np.mean(close[-50:])) if len(close) >= 50 else current_price
        ma200 = float(np.mean(close[-200:])) if len(close) >= 200 else None
        
        above_ma50 = current_price > ma50
        above_ma200 = ma200 is None or current_price > ma200
        
        if not (above_ma50 and above_ma200):
            return None
        
        rsi = calculate_rsi(close, 14)
        
        exp1 = pd.Series(close).ewm(span=12, adjust=False).mean()
        exp2 = pd.Series(close).ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_val = float(macd.iloc[-1])
        signal_val = float(signal.iloc[-1])
        hist_val = float((macd - signal).iloc[-1])
        hist_prev = float((macd - signal).iloc[-2]) if len(macd) > 2 else 0
        
        macd_cross = (float(macd.iloc[-2]) < float(signal.iloc[-2])) and (macd_val > signal_val)
        hist_growing = hist_val > 0 and hist_val > hist_prev
        
        vol_5d = np.mean(vol[-5:]) if len(vol) >= 5 else 0
        vol_20d = np.mean(vol[-20:]) if len(vol) >= 20 else 0
        volume_ok = vol_5d > vol_20d if vol_20d > 0 else False
        
        tr = np.mean(high[-14:] - low[-14:]) if len(high) >= 14 else current_price * 0.05
        atr_pct = (tr / current_price) * 100
        
        # Temel Veriler
        pb = None
        pe = None
        sector = 'Genel'
        earnings_growth = None
        try:
            info = yf.Ticker(ticker).info
            pb = info.get('priceToBook')
            pe = info.get('trailingPE') or info.get('forwardPE')
            sector = info.get('sector', 'Genel')
            earnings_growth = info.get('earningsQuarterlyGrowth')
        except Exception:
            pass
        
        # PUANLAMA
        temel_skor = 0
        teknik_skor = 0
        
        # PD/DD (15 puan)
        if pb and pb > 0:
            sector_pb = sector_stats.get(sector, {}).get('pb_mean', 3)
            if pb < sector_pb * 0.5:
                temel_skor += 15
            elif pb < sector_pb * 0.75:
                temel_skor += 12
            elif pb < sector_pb:
                temel_skor += 8
            else:
                temel_skor += 4
        else:
            temel_skor += 5
        
        # F/K (15 puan)
        if pe and pe > 0:
            sector_pe = sector_stats.get(sector, {}).get('pe_mean', 15)
            if pe < sector_pe * 0.5:
                temel_skor += 15
            elif pe < sector_pe * 0.75:
                temel_skor += 12
            elif pe < sector_pe:
                temel_skor += 8
            elif pe < sector_pe * 1.5:
                temel_skor += 4
            else:
                temel_skor += 1
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
            temel_skor += 3
        
        temel_skor = min(temel_skor, 40)
        
        # RSI (20 puan)
        if 50 <= rsi <= 60:
            teknik_skor += 20
        elif 40 <= rsi < 50:
            teknik_skor += 12
        elif 60 < rsi <= 70:
            teknik_skor += 15
        elif rsi < 40:
            teknik_skor += 8
        else:
            teknik_skor += 7
        
        # MACD (20 puan)
        if macd_cross:
            teknik_skor += 20
        elif hist_growing and macd_val > 0:
            teknik_skor += 16
        elif hist_growing:
            teknik_skor += 10
        elif hist_val > 0:
            teknik_skor += 8
        elif macd_val > signal_val:
            teknik_skor += 5
        
        # Hacim (10 puan)
        if volume_ok:
            vol_ratio = vol_5d / vol_20d if vol_20d > 0 else 1
            if vol_ratio > 2:
                teknik_skor += 10
            elif vol_ratio > 1.5:
                teknik_skor += 8
            elif vol_ratio > 1.2:
                teknik_skor += 6
            else:
                teknik_skor += 4
        
        # ATR (10 puan)
        if 1.5 <= atr_pct <= 3:
            teknik_skor += 10
        elif 3 < atr_pct <= 4.5:
            teknik_skor += 7
        elif 0.8 <= atr_pct < 1.5:
            teknik_skor += 4
        else:
            teknik_skor += 1
        
        # Bonuslar
        if ma200 and ma50 > ma200:
            teknik_skor += 5
        ma50_dist = ((current_price - ma50) / ma50) * 100
        if 2 <= ma50_dist <= 8:
            teknik_skor += 5
        elif 8 < ma50_dist <= 15:
            teknik_skor += 2
        else:
            teknik_skor += 3
        
        teknik_skor = min(teknik_skor, 60)
        toplam_skor = temel_skor + teknik_skor
        
        # MACD Label
        if macd_cross:
            macd_label = "🔥 Crossover"
        elif hist_growing:
            macd_label = "📈 Büyüyor"
        elif hist_val > 0:
            macd_label = "✅ Pozitif"
        else:
            macd_label = "❌ Negatif"
        
        return {
            'Ticker': ticker,
            'Fiyat': round(current_price, 2),
            'Sektör': sector,
            'Toplam Skor': round(toplam_skor, 1),
            'Temel Skor': round(temel_skor, 1),
            'Teknik Skor': round(teknik_skor, 1),
            'RSI': round(rsi, 1),
            'MACD': macd_label,
            'Hacim': '✅' if volume_ok else '❌',
            'ATR%': round(atr_pct, 2),
            'PD/DD': round(pb, 2) if pb else 'N/A',
            'F/K': round(pe, 2) if pe else 'N/A'
        }
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# ANA UYGULAMA
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.title("📈 BIST Swing Trade Tarayıcı")
    st.markdown("**100 Puan Üzerinden Değerlendirme | 70+ Puan = AL Sinyali**")
    st.info("⚠️ Yatırım Tavsiyesi Değildir. Veriler 15 dk gecikmeli olabilir.")
    
    st.sidebar.header("⚙️ Ayarlar")
    min_score = st.sidebar.slider("Minimum AL Skoru", 50, 90, 70, 5)
    max_tickers = st.sidebar.slider("Taranacak Hisse Sayısı", 30, 100, 80, 10)
    
    st.divider()
    
    if st.button("🚀 TARAMAYI BAŞLAT"):
        with st.spinner('⏳ Taranıyor... (2-3 dakika)'):
            sector_stats = get_sector_stats()
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            scan_list = BIST_TICKERS[:max_tickers]
            
            for i, ticker in enumerate(scan_list):
                status_text.text(f"🔍 {ticker} ({i+1}/{len(scan_list)})")
                result = score_ticker(ticker, sector_stats)
                if result:
                    results.append(result)
                progress_bar.progress((i + 1) / len(scan_list))
            
            status_text.empty()
            progress_bar.empty()
            
            if not results:
                st.error("⚠️ Hiç sonuç alınamadı. Lütfen tekrar deneyin.")
                st.stop()
            
            df = pd.DataFrame(results)
            df = df.sort_values('Toplam Skor', ascending=False).reset_index(drop=True)
            df_al = df[df['Toplam Skor'] >= min_score]
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🔍 Taranan", len(scan_list))
            c2.metric("✅ Bulunan", len(df))
            c3.metric("🚀 AL Listesi", len(df_al))
            c4.metric("📊 Min Skor", min_score)
            
            st.divider()
            
            st.subheader(f"🚀 AL Listesi ({len(df_al)} Hisse)")
            
            if not df_al.empty:
                st.markdown("### 🏆 En İyi 5 Hisse")
                cols = st.columns(min(5, len(df_al)))
                for idx, (_, row) in enumerate(df_al.head(5).iterrows()):
                    with cols[idx]:
                        emoji = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉" if idx == 2 else "⭐"
                        st.markdown(f"""
                        <div class="stock-card">
                            <h4>{emoji} {row['Ticker']}</h4>
                            <b>Fiyat:</b> {row['Fiyat']} ₺<br>
                            <b>Skor:</b> {row['Toplam Skor']}/100<br>
                            <b>RSI:</b> {row['RSI']}<br>
                            <b>MACD:</b> {row['MACD']}<br>
                            <b>Sektör:</b> {row['Sektör']}
                        </div>
                        """, unsafe_allow_html=True)
                
                st.divider()
                
                st.subheader("📊 Tüm AL Listesi")
                display_cols = ['Ticker', 'Fiyat', 'Sektör', 'Toplam Skor', 'Temel Skor', 
                               'Teknik Skor', 'RSI', 'MACD', 'Hacim', 'ATR%', 'PD/DD', 'F/K']
                st.dataframe(df_al[display_cols], use_container_width=True, hide_index=True)
                
                csv = df_al[display_cols].to_csv(index=False, encoding='utf-8-sig')
                st.download_button("📥 CSV İndir", csv, "bist_al_listesi.csv", "text/csv")
                
                st.divider()
                st.subheader("📈 Skor Dağılımı (İlk 20)")
                df_chart = df_al.head(20)
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df_chart['Ticker'], y=df_chart['Temel Skor'], 
                                    name='Temel', marker_color='#4A90D9'))
                fig.add_trace(go.Bar(x=df_chart['Ticker'], y=df_chart['Teknik Skor'], 
                                    name='Teknik', marker_color='#F4A83A'))
                fig.add_hline(y=min_score, line_dash='dash', line_color='red')
                fig.update_layout(template='plotly_dark', barmode='stack', 
                                 title='Temel + Teknik Skor', height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.warning(f"⚠️ {min_score}+ puan alan hisse bulunamadı. Skoru düşürün.")
            
            st.success("✅ Tarama Tamamlandı!")
    
    else:
        st.info("⬅️ Sol panelden ayarları yapıp **Taramayı Başlat** butonuna basın.")
        
        with st.expander("📖 Puanlama Sistemi"):
            st.markdown("""
            ### Temel Analiz (40 Puan)
            - PD/DD: 15 puan
            - F/K: 15 puan
            - Kar Büyümesi: 10 puan
            
            ### Teknik Analiz (60 Puan)
            - Trend (MA50+MA200): Zorunlu
            - RSI: 20 puan
            - MACD: 20 puan
            - Hacim: 10 puan
            - ATR: 10 puan
            - Bonuslar: 10 puan
            
            ### AL Sinyali: 70+ Puan
            """)

if __name__ == "__main__":
    main()
