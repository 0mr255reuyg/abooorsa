import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import warnings
import plotly.graph_objects as go
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# BIST HİSSE LİSTESİ
# ─────────────────────────────────────────────────────────────────────────────
BIST_TICKERS = [
    "THYAO.IS","EREGL.IS","GARAN.IS","AKBNK.IS","YKBNK.IS","ISCTR.IS","KCHOL.IS",
    "SASA.IS","BIMAS.IS","FROTO.IS","TUPRS.IS","ASELS.IS","TOASO.IS","PGSUS.IS",
    "HALKB.IS","VAKBN.IS","TKFEN.IS","ENKAI.IS","KOZAL.IS","KRDMD.IS","PETKM.IS",
    "TTKOM.IS","TAVHL.IS","OTKAR.IS","SAHOL.IS","ARCLK.IS","VESTL.IS","MGROS.IS",
    "EKGYO.IS","ULKER.IS","TCELL.IS","SISE.IS","DOHOL.IS","AEFES.IS","LOGO.IS",
    "MAVI.IS","NETAS.IS","KOZA1.IS","BRISA.IS","CCOLA.IS","IHLGM.IS","ALARK.IS",
    "ZOREN.IS","AKSEN.IS","AYGAZ.IS","GOLTS.IS","TSKB.IS","KLNMA.IS","ISGYO.IS"
]

# Tekrar edenleri temizle
BIST_TICKERS = list(dict.fromkeys(BIST_TICKERS))

# ─────────────────────────────────────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────────────────────────────────────
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calculate_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

# ─────────────────────────────────────────────────────────────────────────────
# ANA PUANLAMA FONKSİYONU
# ─────────────────────────────────────────────────────────────────────────────
def score_ticker(ticker: str, sector_stats: dict):
    try:
        raw = yf.download(ticker, period="1y", interval="1d", auto_adjust=True, progress=False)
        if raw is None or len(raw) < 60:
            return None

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw = raw.dropna(subset=["Close", "High", "Low", "Volume"])
        if len(raw) < 60:
            return None

        close = raw["Close"].squeeze()
        high = raw["High"].squeeze()
        low = raw["Low"].squeeze()
        vol = raw["Volume"].squeeze()

        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()
        current_price = float(close.iloc[-1])
        ma50_val = float(ma50.iloc[-1])
        ma200_val = float(ma200.iloc[-1]) if not np.isnan(ma200.iloc[-1]) else None

        above_ma50 = current_price > ma50_val
        above_ma200 = (ma200_val is None) or (current_price > ma200_val)

        trend_ok = above_ma50 and above_ma200
        if not trend_ok:
            return {
                "Ticker": ticker, "Fiyat": round(current_price, 2),
                "Toplam Skor": 0, "Temel Skor": 0, "Teknik Skor": 0,
                "RSI": None, "MACD Sinyal": "-", "Hacim OK": False,
                "MA50 Üzeri": above_ma50, "MA200 Üzeri": above_ma200,
                "Elendi": "Trend Altı"
            }

        rsi_series = calculate_rsi(close, 14)
        rsi_val = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0

        macd_line, signal_line, histogram = calculate_macd(close)
        macd_val = float(macd_line.iloc[-1])
        signal_val = float(signal_line.iloc[-1])
        hist_val = float(histogram.iloc[-1])
        hist_prev = float(histogram.iloc[-2]) if len(histogram) > 1 else 0.0

        macd_cross = (float(macd_line.iloc[-2]) < float(signal_line.iloc[-2])) and (macd_val > signal_val)
        hist_growing = hist_val > 0 and hist_val > hist_prev

        vol_5d = float(vol.iloc[-5:].mean())
        vol_20d = float(vol.iloc[-20:].mean())
        volume_ok = vol_5d > vol_20d

        atr_series = calculate_atr(high, low, close, 14)
        atr_val = float(atr_series.iloc[-1])
        atr_pct = (atr_val / current_price) * 100

        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info or {}
        except Exception:
            info = {}

        pb_ratio = info.get("priceToBook", None)
        pe_ratio = info.get("trailingPE", None) or info.get("forwardPE", None)
        earnings_growth = info.get("earningsQuarterlyGrowth", None)
        sector = info.get("sector", "Unknown")

        temel_skor = 0
        teknik_skor = 0

        # Temel Puanlama
        if pb_ratio and pb_ratio > 0: temel_skor += 10
        if pe_ratio and pe_ratio > 0: temel_skor += 10
        if earnings_growth and earnings_growth > 0: temel_skor += 10

        # Teknik Puanlama
        if 40 <= rsi_val <= 65: teknik_skor += 15
        if macd_cross: teknik_skor += 20
        elif hist_growing: teknik_skor += 10
        if volume_ok: teknik_skor += 10
        if 1.5 <= atr_pct <= 4.5: teknik_skor += 10

        toplam_skor = min(temel_skor + teknik_skor, 100)

        macd_label = "🔥 Crossover" if macd_cross else "📈 Hist. Büyüyor" if hist_growing else "✅ Pozitif" if hist_val > 0 else "❌ Negatif"

        return {
            "Ticker": ticker,
            "Fiyat": round(current_price, 2),
            "Sektör": sector,
            "Toplam Skor": round(toplam_skor, 1),
            "Temel Skor": round(temel_skor, 1),
            "Teknik Skor": round(teknik_skor, 1),
            "RSI": round(rsi_val, 1),
            "MACD Sinyal": macd_label,
            "Hacim OK": volume_ok,
            "MA50 Üzeri": above_ma50,
            "MA200 Üzeri": above_ma200,
            "ATR%": round(atr_pct, 2),
            "Elendi": None
        }

    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT ARAYÜZÜ
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="BIST Swing Trade Tarayıcı", page_icon="📈", layout="wide")

st.title("📈 BIST Swing Trade Tarama & Puanlama Sistemi")
st.markdown("**Sistem Mantığı:** Temel ve Teknik Analiz harmanlanarak 100 üzerinden puanlama yapılır.")
st.divider()

with st.sidebar:
    st.header("⚙️ Tarama Ayarları")
    min_score = st.slider("Minimum AL Skoru", 30, 90, 50, 5)
    # Varsayılan hisse sayısını 20'ye düşürdük (çökmemesi için)
    max_tickers = st.slider("Taranacak Hisse Sayısı", 5, len(BIST_TICKERS), 20, 5)
    # Gecikmeyi 0.5 yaptık (Yahoo banlamasın diye)
    delay = st.slider("İstekler Arası Gecikme (sn)", 0.1, 2.0, 0.5, 0.1)
    
    start_button = st.button("🚀 Taramayı Başlat", type="primary", use_container_width=True)

scan_list = BIST_TICKERS[:max_tickers]

if start_button:
    st.info(f"🔍 {len(scan_list)} hisse taranıyor... Lütfen bekleyin.")
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, ticker in enumerate(scan_list):
        status_text.text(f"⏳ Taranan: {ticker} ({i+1}/{len(scan_list)})")
        res = score_ticker(ticker, {})
        if res: results.append(res)
        progress_bar.progress((i + 1) / len(scan_list))
        time.sleep(delay)

    progress_bar.empty()
    status_text.empty()

    if not results:
        st.error("Veri alınamadı. İnternet veya Yahoo Finance kaynaklı bir sorun olabilir.")
        st.stop()

    df_all = pd.DataFrame(results).sort_values("Toplam Skor", ascending=False)
    df_al = df_all[(df_all["Elendi"].isna()) & (df_all["Toplam Skor"] >= min_score)]

    st.subheader(f"🚀 AL Listesi ({min_score}+ Puan)")
    if df_al.empty:
        st.warning("Eşiği geçen hisse bulunamadı.")
    else:
        st.dataframe(df_al, use_container_width=True)

else:
    st.info("⬅️ Sol panelden **Taramayı Başlat** butonuna basın.")
