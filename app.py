import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import warnings
import traceback

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
    .error-box { background: #2d1a1a; border-left: 4px solid #ff4444; padding: 15px; margin: 10px 0; border-radius: 8px; }
    .success-box { background: #1a2d1a; border-left: 4px solid #00ff41; padding: 15px; margin: 10px 0; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HİSSE LİSTESİ (10 ADET - TEST İÇİN)
# ─────────────────────────────────────────────────────────────────────────────
TEST_TICKERS = [
    "THYAO.IS", "ASELS.IS", "GARAN.IS", "AKBNK.IS", "EREGL.IS",
    "SASA.IS", "BIMAS.IS", "FROTO.IS", "TUPRS.IS", "KCHOL.IS"
]

# ─────────────────────────────────────────────────────────────────────────────
# FONKSİYONLAR (HATA LOG'LU)
# ─────────────────────────────────────────────────────────────────────────────

def test_yahoo_connection():
    """Yahoo Finance bağlantısını test et"""
    errors = []
    try:
        st.write("🔌 Test 1: yfinance import...")
        import yfinance
        st.write(f"✅ yfinance versiyon: {yfinance.__version__}")
    except Exception as e:
        errors.append(f"❌ yfinance import hatası: {str(e)}")
    
    try:
        st.write("🔌 Test 2: THYAO.IS veri çekme...")
        df = yf.download("THYAO.IS", period="5d", progress=False, timeout=10)
        if df is not None and len(df) > 0:
            st.write(f"✅ Veri alındı: {len(df)} satır")
            st.write(f"📊 Sütunlar: {list(df.columns)}")
            return True, errors
        else:
            errors.append("❌ Veri boş geldi")
            st.write("❌ Veri boş geldi")
    except Exception as e:
        errors.append(f"❌ Download hatası: {str(e)}")
        st.write(f"❌ Hata: {str(e)}")
    
    try:
        st.write("🔌 Test 3: Ticker info...")
        ticker = yf.Ticker("THYAO.IS")
        info = ticker.info
        if info:
            st.write(f"✅ Info alındı: {len(info)} alan")
        else:
            errors.append("❌ Info boş")
    except Exception as e:
        errors.append(f"❌ Info hatası: {str(e)}")
    
    return False, errors

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

def score_ticker(ticker, log_errors=False):
    """Hisse analizi - detaylı hata logu"""
    try:
        df = yf.download(ticker, period="6mo", progress=False, timeout=10)
        
        if df is None:
            if log_errors:
                st.write(f"❌ {ticker}: Veri None")
            return None
        
        if len(df) < 50:
            if log_errors:
                st.write(f"❌ {ticker}: Yetersiz veri ({len(df)} satır)")
            return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if 'Close' not in df.columns:
            if log_errors:
                st.write(f"❌ {ticker}: Close sütunu yok. Sütunlar: {list(df.columns)}")
            return None
        
        df = df.dropna(subset=['Close'])
        if len(df) < 50:
            if log_errors:
                st.write(f"❌ {ticker}: Dropna sonrası yetersiz veri")
            return None
        
        close = df['Close'].values
        current_price = float(close[-1])
        
        ma50 = float(np.mean(close[-50:])) if len(close) >= 50 else current_price
        above_ma50 = current_price > ma50
        
        rsi = calculate_rsi(close, 14)
        
        exp1 = pd.Series(close).ewm(span=12, adjust=False).mean()
        exp2 = pd.Series(close).ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_val = float(macd.iloc[-1])
        signal_val = float(signal.iloc[-1])
        
        momentum = 0
        if len(close) >= 21:
            momentum = ((close[-1] / close[-21]) - 1) * 100
        
        score = 50
        if above_ma50: score += 20
        if 45 <= rsi <= 65: score += 15
        elif 35 <= rsi < 45: score += 10
        elif 65 < rsi <= 75: score += 8
        if macd_val > signal_val: score += 15
        if momentum > 0: score += 10
        elif momentum > -5: score += 5
        
        macd_label = "✅ Pozitif" if macd_val > signal_val else "❌ Negatif"
        
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
        if log_errors:
            st.write(f"❌ {ticker}: {str(e)}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# ANA UYGULAMA
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.title("📈 BIST Hisse Tarayıcı (Debug)")
    st.markdown("**Sorun Tespit Modu**")
    st.info("⚠️ Bu versiyon hata ayıklama içindir.")
    
    st.divider()
    
    # ── BAĞLANTI TESTİ ────────────────────────────────────────────────────────
    st.subheader("🔌 Adım 1: Yahoo Finance Bağlantı Testi")
    
    if st.button("🧪 Bağlantıyı Test Et"):
        with st.spinner('Test ediliyor...'):
            success, errors = test_yahoo_connection()
            
            if success:
                st.markdown('<div class="success-box">✅ Yahoo Finance çalışıyor! Taramaya geçebilirsiniz.</div>', unsafe_allow_html=True)
                st.session_state['yahoo_ok'] = True
            else:
                st.markdown('<div class="error-box">❌ Yahoo Finance bağlantı sorunu!</div>', unsafe_allow_html=True)
                st.session_state['yahoo_ok'] = False
                
                if errors:
                    st.write("### 📋 Hata Detayları:")
                    for err in errors:
                        st.write(err)
                
                st.warning("""
                ### 🔧 Olası Çözümler:
                1. **Streamlit Cloud IP Blok**: Yahoo Finance, Streamlit Cloud IP'lerini blokluyor olabilir.
                2. **Çözüm**: Bilgisayarınızda çalıştırın (`streamlit run app.py`)
                3. **Alternatif**: TradingView hisse tarama kullanın
                4. **Alternatif**: Farklı hosting (Render, Railway) deneyin
                """)
    
    st.divider()
    
    # ── TARAMA ────────────────────────────────────────────────────────────────
    st.subheader("🔍 Adım 2: Hisse Tarama")
    
    if not st.session_state.get('yahoo_ok', False):
        st.warning("⚠️ Önce bağlantı testini başarılı yapın!")
    else:
        min_score = st.slider("Min Skor", 40, 80, 60, 5)
        show_logs = st.checkbox("Hata Loglarını Göster", value=True)
        
        if st.button("🚀 TARAMAYI BAŞLAT"):
            with st.spinner('⏳ Taranıyor...'):
                results = []
                progress_bar = st.progress(0)
                
                for i, ticker in enumerate(TEST_TICKERS):
                    if show_logs:
                        st.write(f"🔍 {ticker}...")
                    
                    result = score_ticker(ticker, log_errors=show_logs)
                    if result:
                        results.append(result)
                        if show_logs:
                            st.write(f"✅ {ticker}: Skor {result['Skor']}")
                    
                    progress_bar.progress((i + 1) / len(TEST_TICKERS))
                
                progress_bar.empty()
                
                st.divider()
                st.write(f"### 📊 Sonuç: {len(results)}/{len(TEST_TICKERS)} hisse")
                
                if results:
                    df = pd.DataFrame(results)
                    df = df.sort_values('Skor', ascending=False)
                    df_al = df[df['Skor'] >= min_score]
                    
                    st.write(f"✅ {len(df_al)} hisse {min_score}+ puan aldı")
                    st.dataframe(df_al, use_container_width=True)
                    
                    csv = df_al.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button("📥 CSV İndir", csv, "bist_tarama.csv", "text/csv")
                else:
                    st.error("❌ Hiç hisse bulunamadı. Yahoo Finance veri çekemiyor.")
    
    st.divider()
    
    # ── BİLGİ ─────────────────────────────────────────────────────────────────
    st.subheader("ℹ️ Sorun Nedir?")
    st.markdown("""
    ### 🚨 Streamlit Cloud + Yahoo Finance Sorunu
    
    Yahoo Finance, **bulut IP adreslerini** (Streamlit Cloud, AWS, Google Cloud vb.) sık sık **blokluyor** veya **rate-limit** uyguluyor.
    
    ### ✅ Çözüm Önerileri:
    
    | Yöntem | Açıklama | Zorluk |
    |--------|----------|--------|
    | **Bilgisayarda Çalıştır** | `pip install streamlit yfinance` → `streamlit run app.py` | ⭐ Kolay |
    | **TradingView** | Ücretsiz hisse tarama ekranı | ⭐ Kolay |
    | **Render.com** | Farklı hosting, IP farklı olabilir | ⭐⭐ Orta |
    | **Matriks/İdeal Data** | Ücretli ama garantili BIST verisi | ⭐⭐⭐ Zor |
    
    ### 💻 Bilgisayarda Çalıştırma (ÖNERİLEN):
    
    ```bash
    # 1. Python kur (python.org)
    # 2. Terminal aç
    pip install streamlit yfinance pandas numpy plotly
    
    # 3. app.py'yi çalıştır
    streamlit run app.py
    
    # 4. Tarayıcıda açılır (http://localhost:8501)
    ```
    
    Bu şekilde **sınırsız ve kesintisiz** çalışır!
    """)

if __name__ == "__main__":
    main()
