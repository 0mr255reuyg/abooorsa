import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(page_title="BIST Tarayıcı", layout="wide", page_icon="📈")

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
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HİSSE LİSTESİ (50 ADET - HIZLI)
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
# FONKSİYONLAR (SADELEŞTİRİLMİŞ)
# ─────────────────────────────────────────────────────────────────────────────

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

def score_ticker(ticker):
    try:
        # Tek istek - hızlı
        df = yf.download(ticker, period="6mo", progress=False, timeout=5)
        
        if df is None or len(df) < 50:
            return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if 'Close' not in df.columns:
            return None
        
        df = df.dropna(subset=['Close'])
        if len(df) < 50:
            return None
        
        close = df['Close'].values
        current_price = float(close[-1])
        
        # MA50
        ma50 = float(np.mean(close[-50:])) if len(close) >= 50 else current_price
        above_ma50 = current_price > ma50
        
        # RSI
        rsi = calculate_rsi(close, 14)
        
        # MACD (basit)
        exp1 = pd.Series(close).ewm(span=12, adjust=False).mean()
        exp2 = pd.Series(close).ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_val = float(macd.iloc[-1])
        signal_val = float(signal.iloc[-1])
        
        # Momentum (1 ay)
        momentum = 0
        if len(close) >= 21:
            momentum = ((close[-1] / close[-21]) - 1) * 100
        
        # PUANLAMA (ÇOK BASİT)
        score = 50  # Başlangıç
        
        if above_ma50:
            score += 20
        if 45 <= rsi <= 65:
            score += 15
        elif 35 <= rsi < 45:
            score += 10
        elif 65 < rsi <= 75:
            score += 8
        if macd_val > signal_val:
            score += 15
        if momentum > 0:
            score += 10
        elif momentum > -5:
            score += 5
        
        # MACD Label
        if macd_val > signal_val:
            macd_label = "✅ Pozitif"
        else:
            macd_label = "❌ Negatif"
        
        return {
            'Hisse': ticker.replace('.IS', ''),
            'Fiyat': round(current_price, 2),
            'Skor': score,
            'RSI': round(rsi, 1),
            'MACD': macd_label,
            'Momentum%': round(momentum, 2),
            'MA50 Üstü': '✅' if above_ma50 else '❌'
        }
    except Exception as e:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# ANA UYGULAMA
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.title("📈 BIST Hisse Tarayıcı")
    st.markdown("**Hızlı Teknik Analiz | 60+ Puan = AL**")
    st.info("⚠️ Yatırım Tavsiyesi Değildir")
    
    # ── TEST BAĞLANTI ────────────────────────────────────────────────────────
    st.sidebar.header("⚙️ Ayarlar")
    
    # Önce Yahoo çalışıyor mu test et
    if 'connection_tested' not in st.session_state:
        with st.sidebar:
            st.write("🔌 Bağlantı testi...")
            try:
                test = yf.download("THYAO.IS", period="5d", progress=False)
                if test is not None and len(test) > 0:
                    st.session_state['connection_tested'] = True
                    st.success("✅ Yahoo çalışıyor!")
                else:
                    st.session_state['connection_tested'] = False
                    st.error("❌ Yahoo bağlantı hatası!")
            except:
                st.session_state['connection_tested'] = False
                st.error("❌ Yahoo bağlantı hatası!")
    
    min_score = st.sidebar.slider("Min Skor", 40, 80, 60, 5)
    max_stocks = st.sidebar.slider("Hisse Sayısı", 20, 50, 40, 5)
    
    st.divider()
    
    if st.button("🚀 TARAMAYI BAŞLAT"):
        if not st.session_state.get('connection_tested', False):
            st.error("❌ Yahoo Finance bağlantısı başarısız. Lütfen 1 dakika bekleyip tekrar deneyin.")
            st.stop()
        
        with st.spinner('⏳ Taranıyor... (1-2 dakika)'):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, ticker in enumerate(TICKERS[:max_stocks]):
                status_text.text(f"🔍 {ticker} ({i+1}/{max_stocks})")
                result = score_ticker(ticker)
                if result:
                    results.append(result)
                progress_bar.progress((i + 1) / max_stocks)
            
            status_text.empty()
            progress_bar.empty()
            
            if not results:
                st.error("⚠️ Hiç veri alınamadı. Yahoo Finance şu an yoğun olabilir. 2 dakika bekleyip tekrar deneyin.")
                st.stop()
            
            df = pd.DataFrame(results)
            df = df.sort_values('Skor', ascending=False).reset_index(drop=True)
            df_al = df[df['Skor'] >= min_score]
            
            # Az hisse varsa en yüksekleri göster
            if len(df_al) < 3:
                df_al = df.head(10)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("🔍 Taranan", max_stocks)
            c2.metric("✅ Bulunan", len(df))
            c3.metric("🚀 AL Listesi", len(df_al))
            
            st.divider()
            
            if not df_al.empty:
                st.subheader("🏆 En İyi 5 Hisse")
                cols = st.columns(min(5, len(df_al)))
                for idx, (_, row) in enumerate(df_al.head(5).iterrows()):
                    with cols[idx]:
                        emoji = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉" if idx == 2 else "⭐"
                        st.markdown(f"""
                        <div class="stock-card">
                            <h4>{emoji} {row['Hisse']}</h4>
                            <b>Fiyat:</b> {row['Fiyat']} ₺<br>
                            <b>Skor:</b> {row['Skor']}/100<br>
                            <b>RSI:</b> {row['RSI']}<br>
                            <b>MACD:</b> {row['MACD']}<br>
                            <b>Momentum:</b> %{row['Momentum%']}
                        </div>
                        """, unsafe_allow_html=True)
                
                st.divider()
                st.subheader("📊 Tüm Sonuçlar")
                st.dataframe(df_al, use_container_width=True, hide_index=True)
                
                csv = df_al.to_csv(index=False, encoding='utf-8-sig')
                st.download_button("📥 CSV İndir", csv, "bist_tarama.csv", "text/csv")
            else:
                st.warning(f"⚠️ {min_score}+ puan alan bulunamadı.")
            
            st.success("✅ Tamamlandı!")
    
    else:
        st.info("⬅️ Ayarları yapıp **Taramayı Başlat** butonuna basın.")

if __name__ == "__main__":
    main()
    
