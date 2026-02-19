import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
import time

# -----------------------------------------------------------------------------
# 1. SAYFA AYARLARI & CSS TASARIM
# -----------------------------------------------------------------------------
st.set_page_config(page_title="BIST PRO Portföy", layout="wide", page_icon="🚀")

st.markdown("""
    <style>
    .stApp { 
        background: linear-gradient(135deg, #0e1117 0%, #1a1d24 100%); 
        color: #ffffff; 
    }
    .stButton>button {
        background: linear-gradient(135deg, #00ff41 0%, #00cc33 100%); 
        color: #000; font-weight: bold; border: none; 
        padding: 15px 30px; border-radius: 10px; width: 100%; 
        font-size: 16px; box-shadow: 0 4px 15px rgba(0, 255, 65, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover { 
        transform: translateY(-2px); 
        box-shadow: 0 6px 20px rgba(0, 255, 65, 0.5); 
    }
    .stock-card {
        background: linear-gradient(135deg, #1f2937 0%, #2d3748 100%);
        border-radius: 15px; padding: 20px; margin: 15px 0;
        border: 1px solid #374151; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        animation: slideIn 0.5s ease-out;
    }
    .stock-card:hover { 
        transform: translateY(-5px); 
        border-color: #00ff41; 
        box-shadow: 0 8px 25px rgba(0, 255, 65, 0.2);
    }
    .sl-box { 
        background: rgba(255, 68, 68, 0.1); 
        border-left: 4px solid #ff4444; 
        padding: 10px; margin: 10px 0; border-radius: 8px; 
    }
    .tp-box { 
        background: rgba(0, 255, 65, 0.1); 
        border-left: 4px solid #00ff41; 
        padding: 10px; margin: 10px 0; border-radius: 8px; 
    }
    .ai-box { 
        background: rgba(0, 204, 255, 0.1); 
        border-left: 4px solid #00ccff; 
        padding: 15px; margin: 10px 0; border-radius: 8px; 
    }
    .metric-box {
        background: #1f2937; padding: 15px; border-radius: 10px;
        border: 1px solid #00ff41; text-align: center;
    }
    @keyframes slideIn { 
        from { opacity: 0; transform: translateY(20px); } 
        to { opacity: 1; transform: translateY(0); } 
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. VERİ FONKSİYONLARI
# -----------------------------------------------------------------------------

def get_bist_tickers():
    """BIST 100 hisseleri (Veri gelme ihtimali yüksek olanlar)"""
    return [
        "THYAO.IS", "ASELS.IS", "GARAN.IS", "AKBNK.IS", "EREGL.IS", "TUPRS.IS", 
        "SASA.IS", "KCHOL.IS", "SAHOL.IS", "BIMAS.IS", "MGROS.IS", "FROTO.IS", 
        "TOASO.IS", "TCELL.IS", "TTKOM.IS", "HEKTS.IS", "ALARK.IS", "DOHOL.IS",
        "ISCTR.IS", "YKBNK.IS", "HALKB.IS", "VAKBN.IS", "KOZAL.IS", "SOKM.IS",
        "CCOLA.IS", "ANSGR.IS", "PGSUS.IS", "ULKER.IS", "TAVHL.IS", "ISGYO.IS",
        "EKGYO.IS", "VESBE.IS", "BRISA.IS", "FLO.IS", "DEVA.IS", "GUBRF.IS",
        "POLHO.IS", "CIMSA.IS", "NUHOL.IS", "BOLUC.IS", "KARTN.IS", "TRKCM.IS",
        "LOGO.IS", "YATAS.IS", "USAK.IS", "DENGE.IS", "ALTNY.IS", "KFEIN.IS",
        "BIZIM.IS", "CATAS.IS", "CRDFA.IS", "DAGI.IS", "DERIM.IS", "DESA.IS",
        "DMSAS.IS", "DOAS.IS", "ECILC.IS", "EDATA.IS", "EGEEN.IS", "EMKEL.IS",
        "ERBOS.IS", "ERSU.IS", "ETLER.IS", "FENER.IS", "FINBN.IS", "FKORE.IS",
        "GOODY.IS", "GRHOL.IS", "GSYO.IS", "HATEK.IS", "HUBVC.IS", "ICBCT.IS",
        "IHLAS.IS", "IHLGM.IS", "INDES.IS", "ISDMR.IS", "ISKUR.IS", "IZMDC.IS",
        "KARSN.IS", "KAYSE.IS", "KLBRN.IS", "KONTR.IS", "KUTPO.IS", "LKNDY.IS",
        "MAVI.IS", "MEBRS.IS", "MENSA.IS", "MERTG.IS", "NETAS.IS", "OYAKC.IS",
        "OTKAR.IS", "OZRDN.IS", "PAGYO.IS", "PEKGI.IS", "PNLSN.IS", "PRKAB.IS"
    ]

def fetch_data(ticker):
    """Hisse verisini çeker"""
    try:
        df = yf.download(ticker, period="1y", progress=False)
        if df.empty or len(df) < 60:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if 'Close' not in df.columns:
            return None
        return df
    except:
        return None

def calculate_indicators(df):
    """Teknik indikatörleri hesaplar (Numpy ile hızlı)"""
    try:
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        
        # RSI (14)
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gain[:14]) if len(gain) >= 14 else 0
        avg_loss = np.mean(loss[:14]) if len(loss) >= 14 else 1
        rsi = 50
        for i in range(14, len(close)):
            avg_gain = (avg_gain * 13 + gain[i]) / 14
            avg_loss = (avg_loss * 13 + loss[i]) / 14
            rs = avg_gain / avg_loss if avg_loss != 0 else 100
            rsi = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = pd.Series(close).ewm(span=12, adjust=False).mean()
        exp2 = pd.Series(close).ewm(span=26, adjust=False).mean()
        macd = (exp1 - exp2).iloc[-1]
        signal = macd.ewm(span=9, adjust=False).mean().iloc[-1] if len(exp1) > 9 else macd
        
        # ATR & SMA50
        tr = np.mean(high[-14:] - low[-14:]) if len(high) >= 14 else close[-1] * 0.05
        sma50 = np.mean(close[-50:]) if len(close) >= 50 else close[-1]
        
        return {
            'rsi': rsi,
            'macd': macd,
            'signal': signal,
            'atr': tr,
            'close': close[-1],
            'sma50': sma50
        }
    except:
        return None

def get_fundamentals(ticker):
    """Temel verileri çeker (Boş gelirse varsayılan değer)"""
    try:
        info = yf.Ticker(ticker).info
        pe = info.get('trailingPE')
        pb = info.get('priceToBook')
        sector = info.get('sector', 'Genel')
        return pe if pe else 15, pb if pb else 2, sector
    except:
        return 15, 2, 'Genel'

def calculate_sl_tp(price, atr):
    """Stop-Loss ve Take-Profit hesapla"""
    sl = price - (atr * 2.5)
    tp1 = price + (atr * 3)
    tp2 = price + (atr * 6)
    return max(sl, price * 0.90), tp1, tp2

def generate_ai_comment(rsi, macd, signal, price, sma50):
    """AI tarzı yorum üretir"""
    c = []
    if rsi > 70:
        c.append("⚠️ RSI aşırı alımda, düzeltme gelebilir.")
    elif rsi < 30:
        c.append("✅ RSI aşırı satımda, tepki yükselebilir.")
    else:
        c.append("📊 RSI nötr bölgede.")
    
    if macd > signal:
        c.append("📈 MACD alı sinyali veriyor.")
    else:
        c.append("📉 MACD satı sinyali veriyor.")
    
    if price > sma50:
        c.append("📈 Fiyat 50 günlük ortalamanın üstünde.")
    else:
        c.append("📉 Fiyat baskı altında.")
    
    return " ".join(c)

# -----------------------------------------------------------------------------
# 3. ANALİZ MOTORU (GARANTİLİ 5 HİSSE)
# -----------------------------------------------------------------------------

def scan_market():
    """Piyasayı tarar ve en iyi 5 hisseyi seçer"""
    tickers = get_bist_tickers()
    candidates = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ticker in enumerate(tickers):
        status_text.text(f"🔍 Taranıyor: {ticker} ({i+1}/{len(tickers)})")
        
        df = fetch_data(ticker)
        if df is None:
            progress_bar.progress((i + 1) / len(tickers))
            continue
        
        ind = calculate_indicators(df)
        if ind is None:
            progress_bar.progress((i + 1) / len(tickers))
            continue
        
        pe, pb, sector = get_fundamentals(ticker)
        
        # PUANLAMA SİSTEMİ (Esnek)
        score = 50  # Başlangıç puanı
        if ind['rsi'] > 45: score += 15
        if ind['macd'] > ind['signal']: score += 20
        if ind['close'] > ind['sma50']: score += 15
        if pe < 30 and pe > 0: score += 10
        if pb < 5 and pb > 0: score += 10
        
        # Momentum
        momentum = 0
        if len(df) >= 30:
            momentum = (ind['close'] / df['Close'].iloc[-30] - 1) * 100
            if momentum > 0: score += 10
        
        sl, tp1, tp2 = calculate_sl_tp(ind['close'], ind['atr'])
        ai_comment = generate_ai_comment(ind['rsi'], ind['macd'], ind['signal'], ind['close'], ind['sma50'])
        
        candidates.append({
            'Hisse': ticker,
            'Fiyat': round(ind['close'], 2),
            'Puan': score,
            'RSI': round(ind['rsi'], 1),
            'F/K': round(pe, 1),
            'PD/DD': round(pb, 1),
            'Sektör': sector,
            'Stop': round(sl, 2),
            'TP1': round(tp1, 2),
            'TP2': round(tp2, 2),
            'AI': ai_comment,
            'Momentum': round(momentum, 2)
        })
        
        progress_bar.progress((i + 1) / len(tickers))
    
    status_text.text("✅ Tarama Tamamlandı!")
    
    if not candidates:
        return pd.DataFrame()
    
    df = pd.DataFrame(candidates)
    df = df.sort_values(by='Puan', ascending=False)
    top5 = df.head(5)
    
    return top5

# -----------------------------------------------------------------------------
# 4. PORTFÖY YÖNETİMİ
# -----------------------------------------------------------------------------

PORTFOLIO_FILE = 'portfoy.json'

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None
    return None

def save_portfolio(data):
    with open(PORTFOLIO_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def delete_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        os.remove(PORTFOLIO_FILE)

# -----------------------------------------------------------------------------
# 5. GÖRSEL BİLEŞENLER
# -----------------------------------------------------------------------------

def display_stock_card(stock, index):
    """Animasyonlu hisse kartı"""
    st.markdown(f"""
    <div class="stock-card" style="animation-delay: {index * 0.1}s;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <h3 style="margin: 0; color: #00ff41;">📈 {stock['Hisse']}</h3>
            <h3 style="margin: 0; color: #fff;">{stock['Fiyat']:.2f} ₺</h3>
        </div>
        <div style="display: flex; gap: 10px; margin: 15px 0; flex-wrap: wrap;">
            <span style="background: #374151; padding: 5px 12px; border-radius: 5px;">
                🏆 Puan: <b style="color:#00ff41">{stock['Puan']}</b>
            </span>
            <span style="background: #374151; padding: 5px 12px; border-radius: 5px;">
                📊 RSI: <b>{stock['RSI']}</b>
            </span>
            <span style="background: #374151; padding: 5px 12px; border-radius: 5px;">
                💰 F/K: <b>{stock['F/K']}</b>
            </span>
            <span style="background: #374151; padding: 5px 12px; border-radius: 5px;">
                🚀 Momentum: <b style="color:{'#00ff41' if stock['Momentum']>0 else '#ff4444'}">%{stock['Momentum']}</b>
            </span>
            <span style="background: #374151; padding: 5px 12px; border-radius: 5px;">
                🏢 {stock['Sektör']}
            </span>
        </div>
        <div class="sl-box">
            🛑 <b>Stop-Loss:</b> {stock['Stop']:.2f} ₺ 
            <span style="color: #aaa; font-size: 12px;">(Risk Yönetimi)</span>
        </div>
        <div class="tp-box">
            ✅ <b>TP1 (Hedef 1):</b> {stock['TP1']:.2f} ₺ 
            &nbsp;|&nbsp; 
            ✅ <b>TP2 (Hedef 2):</b> {stock['TP2']:.2f} ₺
        </div>
        <div class="ai-box">
            🤖 <b>AI Analiz:</b> {stock['AI']}
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. ANA UYGULAMA
# -----------------------------------------------------------------------------

def main():
    st.title("🚀 BIST PRO Portföy Yöneticisi")
    st.markdown("### 🇹🇷 AI Destekli | Stop-Loss | 30 Gün Kilitli")
    st.info("⚠️ **Yatırım Tavsiyesi Değildir.** Bu uygulama sadece teknik analiz ve eğitim amaçlıdır.")
    
    st.sidebar.header("⚙️ Menü")
    page = st.sidebar.radio("Sayfa Seç", ["💼 Portföy", "🏆 Piyasa Tarama", "ℹ️ Hakkında"])
    
    current_date = datetime.now()
    portfolio = load_portfolio()
    days_left = 0
    is_locked = False
    
    if portfolio:
        try:
            start_date = datetime.strptime(portfolio['start_date'], '%Y-%m-%d')
            days_left = 30 - (current_date - start_date).days
            if days_left > 0:
                is_locked = True
        except:
            pass
    
    # --- PORTFÖY SAYFASI ---
    if page == "💼 Portföy":
        c1, c2, c3 = st.columns(3)
        c1.metric("Durum", "🔒 KİLİTLİ" if is_locked else "✅ AÇIK")
        c2.metric("Kalan Gün", max(0, days_left))
        c3.metric("Hisse Sayısı", len(portfolio.get('stocks', [])) if portfolio else 0)
        
        st.divider()
        
        if not is_locked:
            st.subheader("🔍 Yeni Portföy Oluştur")
            st.write("""
            - ✅ 100+ hisse taranacak
            - ✅ En iyi 5 hisse seçilecek
            - ✅ Stop-Loss ve Take-Profit belirlenecek
            - ✅ 30 gün boyunca kilitlenecek
            """)
            
            if st.button("🚀 PİYASAYI TARA VE PORTFÖY OLUŞTUR"):
                with st.spinner('⏳ 100+ Hisse analiz ediliyor... (2-3 dakika sürebilir)'):
                    df = scan_market()
                    
                    if not df.empty:
                        top5 = df.to_dict('records')
                        save_portfolio({
                            'start_date': current_date.strftime('%Y-%m-%d'),
                            'stocks': top5
                        })
                        st.success("🎉 Portföy başarıyla oluşturuldu ve kilitlendi!")
                        st.balloons()
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error("⚠️ Veri çekilemedi. Lütfen tekrar deneyin.")
        else:
            st.subheader("🔒 Aktif Portföyünüz")
            stocks = portfolio.get('stocks', [])
            
            if stocks:
                for i, stock in enumerate(stocks):
                    display_stock_card(stock, i)
                
                # Performans Grafiği
                st.divider()
                st.subheader("📈 Portföy Performansı")
                try:
                    tickers = [s['Hisse'] for s in stocks]
                    data = yf.download(tickers, period="1mo", progress=False)['Close']
                    fig = go.Figure()
                    for col in data.columns:
                        fig.add_trace(go.Scatter(x=data.index, y=data[col], name=col, line=dict(width=2)))
                    fig.update_layout(
                        template='plotly_dark',
                        title="Son 1 Ay Performansı",
                        xaxis_title="Tarih",
                        yaxis_title="Fiyat (₺)",
                        height=400,
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.info("Grafik verisi şu anda yüklenemiyor.")
            else:
                st.warning("Portföyünüz boş.")
    
    # --- PİYASA TARAMA SAYFASI ---
    elif page == "🏆 Piyasa Tarama":
        st.subheader("🏆 Tüm Piyasa Sıralaması")
        st.write("100+ hisse teknik ve temel analizle puanlandı.")
        
        if st.button("🔄 Taramayı Yenile"):
            with st.spinner('⏳ 100+ Hisse taranıyor...'):
                df = scan_market()
                st.session_state['market_data'] = df
                st.success(f"✅ {len(df)} hisse analiz edildi!")
        
        if 'market_data' in st.session_state:
            df = st.session_state['market_data']
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            csv = df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button("📥 Excel Olarak İndir", csv, "bist_analiz.csv", "text/csv")
    
    # --- HAKKINDA SAYFASI ---
    elif page == "ℹ️ Hakkında":
        st.subheader("ℹ️ Uygulama Hakkında")
        st.markdown("""
        ### 🎯 Özellikler
        - ✅ 100+ BIST hissesi taraması
        - ✅ Otomatik 5 hisse seçimi
        - ✅ Stop-Loss ve Take-Profit hesaplama
        - ✅ AI tarzı teknik yorum
        - ✅ 30 günlük portföy kilidi
        - ✅ Performans grafikleri
        - ✅ Excel çıktı alma
        
        ### ⚠️ Önemli Uyarılar
        1. **Yatırım Tavsiyesi Değildir:** Bu uygulama sadece eğitim amaçlıdır.
        2. **Veri Gecikmesi:** Ücretsiz API kullanıldığı için veriler 15 dk gecikmeli olabilir.
        3. **Cloud Limiti:** Streamlit Cloud ücretsiz sürümünde dosya kilidi zaman zaman sıfırlanabilir.
        4. **Sorumluluk:** Tüm yatırım kararlarınız kendi sorumluluğunuzdadır.
        
        ### 📊 Strateji Detayları
        - **RSI:** 45 üstü tercih edilir
        - **MACD:** Alı sinyali aranır
        - **SMA50:** Fiyat ortalamanın üstünde olmalı
        - **F/K:** 30 altı tercih edilir
        - **PD/DD:** 5 altı tercih edilir
        - **Momentum:** Son 1 ay pozitif getiri
        
        ### 🛠️ Teknik Altyapı
        - Python + Streamlit
        - Yahoo Finance API
        - Plotly Grafikler
        - Pandas + Numpy Analiz
        """)

if __name__ == "__main__":
    main()
