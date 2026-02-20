import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(page_title="BIST Hisse Tarayıcı", layout="wide", page_icon="📈")

# ─────────────────────────────────────────────────────────────────────────────
# TASARIM
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .stApp { background: #0e1117; color: #ffffff; }
    .stButton>button {
        background: #00ff41; color: #000; font-weight: bold;
        border: none; padding: 15px 30px; border-radius: 10px; width: 100%;
        font-size: 18px;
    }
    .stButton>button:hover { background: #00cc33; }
    .stock-card { 
        background: #1f2937; 
        padding: 20px; 
        border-radius: 12px; 
        margin: 15px 0; 
        border-left: 5px solid #00ff41;
        box-shadow: 0 4px 15px rgba(0,255,65,0.1);
    }
    .stock-card:hover {
        border-left-color: #00ff41;
        transform: translateX(5px);
    }
    .metric-box { 
        background: #1f2937; 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #00ff41; 
        text-align: center; 
    }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HİSSE LİSTESİ (100 ADET - BIST 100)
# ─────────────────────────────────────────────────────────────────────────────
BIST100_TICKERS = [
    "ACSEL.IS","ADEL.IS","ADNAC.IS","AKBNK.IS","AKCNS.IS","AKFGY.IS","AKFYE.IS","AKSA.IS","AKSEN.IS","AKSGY.IS",
    "AKTAE.IS","ALARK.IS","ALBRK.IS","ALFAS.IS","ALGYO.IS","ALKIM.IS","ALKLC.IS","ANELE.IS","ANHYT.IS","ARCLK.IS",
    "ARDYZ.IS","ASELS.IS","ASGYO.IS","ASTOR.IS","ATAKP.IS","ATATP.IS","AYDEM.IS","AYGAZ.IS","BAGFS.IS","BANVT.IS",
    "BERA.IS","BIENY.IS","BIMAS.IS","BIZIM.IS","BJKAS.IS","BKENT.IS","BRISA.IS","BRYAT.IS","BSOKE.IS","BTCIM.IS",
    "BUCIM.IS","CANTE.IS","CCOLA.IS","CEMTS.IS","CIMSA.IS","CLEBI.IS","CWENE.IS","DESA.IS","DOHOL.IS","DYOBY.IS",
    "ECILC.IS","EGEEN.IS","EGERB.IS","EKGYO.IS","ENERU.IS","ENJSA.IS","ENKAI.IS","EREGL.IS","ESCOM.IS","EUPWR.IS",
    "EUREN.IS","FENER.IS","FLAP.IS","FMIZP.IS","FROTO.IS","GARAN.IS","GENIL.IS","GESAN.IS","GLYHO.IS","GOLTS.IS",
    "GUBRF.IS","GWIND.IS","HALKB.IS","HATEK.IS","HEKTS.IS","HLGYO.IS","HRKET.IS","HTTBT.IS","HUNER.IS","ICBCT.IS",
    "IHLGM.IS","IHLAS.IS","ISGSY.IS","ISCTR.IS","ISKUR.IS","ISMEN.IS","ISYAT.IS","IZFAS.IS","IZMDC.IS","JANTS.IS",
    "KAPLM.IS","KAREL.IS","KARSN.IS","KATMR.IS","KCAER.IS","KCHOL.IS","KENT.IS","KLNMA.IS","KMPUR.IS","KNFRT.IS",
    "KONYA.IS","KORDS.IS","KOZAA.IS","KOZAL.IS","KRDMD.IS","KRGYO.IS","KRONT.IS","KSTUR.IS","KTLEV.IS","KUTPO.IS",
    "LOGO.IS","LKMNH.IS","MAALT.IS","MAVI.IS","MEPET.IS","MGROS.IS","MIATK.IS","MIPAZ.IS","MPARK.IS","NETAS.IS",
    "NTHOL.IS","NTTUR.IS","NUGYO.IS","NUHCM.IS","ODAS.IS","ONCSM.IS","ORCAY.IS","OTKAR.IS","OYAKC.IS","OYLUM.IS",
    "OZGYO.IS","OZKGY.IS","PAPIL.IS","PARSN.IS","PCILT.IS","PEKGY.IS","PENGD.IS","PETKM.IS","PGSUS.IS","PINSU.IS",
    "PKENT.IS","POLHO.IS","PRKAB.IS","PRKME.IS","PTOFS.IS","RAYSG.IS","RODRG.IS","ROYAL.IS","RTALB.IS","RYSAS.IS",
    "SAHOL.IS","SASA.IS","SELEC.IS","SELGD.IS","SISE.IS","SKBNK.IS","SMART.IS","SMRTG.IS","SNPAM.IS","SOKM.IS",
    "SUMAS.IS","SUNTK.IS","SUPRS.IS","TAVHL.IS","TBMAN.IS","TCELL.IS","TGSAS.IS","THYAO.IS","TKFEN.IS","TKNSA.IS",
    "TOASO.IS","TRGYO.IS","TRILC.IS","TSKB.IS","TTKOM.IS","TTRAK.IS","TUKAS.IS","TUPRS.IS","TURSG.IS","ULUFA.IS",
    "ULUSE.IS","UNCRD.IS","UYUM.IS","VAKBN.IS","VAKFN.IS","VERUS.IS","VESBE.IS","VESTL.IS","VKGYO.IS","VRGYO.IS",
    "YKBNK.IS","YATAS.IS","YEOTK.IS","YKSLN.IS","YUNSA.IS","ZOREN.IS","ZRGYO.IS"
]

# ─────────────────────────────────────────────────────────────────────────────
# FONKSİYONLAR (BASİT VE GÜVENİLİR)
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

def score_ticker(ticker):
    """
    Hisse analizi - SADECE TEKNİK (hızlı ve güvenilir)
    Maksimum 100 puan
    """
    try:
        df = yf.download(ticker, period="1y", progress=False, timeout=10)
        
        if df is None or len(df) < 100:
            return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if 'Close' not in df.columns:
            return None
        
        df = df.dropna(subset=['Close'])
        if len(df) < 100:
            return None
        
        close = df['Close'].values
        vol = df['Volume'].values if 'Volume' in df.columns else None
        current_price = float(close[-1])
        
        # ── PUANLAMA (Maks 100) ──────────────────────────────────────────────
        score = 0
        
        # 1. TREND - MA50 (20 puan)
        ma50 = float(np.mean(close[-50:])) if len(close) >= 50 else current_price
        ma200 = float(np.mean(close[-200:])) if len(close) >= 200 else None
        
        if current_price > ma50:
            score += 20
        elif current_price > ma50 * 0.95:
            score += 10
        
        if ma200 and current_price > ma200:
            score += 10
        
        # 2. RSI (20 puan)
        rsi = calculate_rsi(close, 14)
        
        if 50 <= rsi <= 65:
            score += 20  # İdeal bölge
        elif 45 <= rsi < 50:
            score += 15
        elif 65 < rsi <= 70:
            score += 12
        elif 35 <= rsi < 45:
            score += 8
        elif 70 < rsi <= 80:
            score += 5
        else:
            score += 3
        
        # 3. MACD (20 puan)
        exp1 = pd.Series(close).ewm(span=12, adjust=False).mean()
        exp2 = pd.Series(close).ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_val = float(macd.iloc[-1])
        signal_val = float(signal.iloc[-1])
        hist_val = float((macd - signal).iloc[-1])
        hist_prev = float((macd - signal).iloc[-2]) if len(macd) > 2 else 0
        
        if macd_val > signal_val and hist_val > hist_prev and hist_val > 0:
            score += 20  # Güçlü alı
        elif macd_val > signal_val:
            score += 15
        elif hist_val > 0:
            score += 10
        elif macd_val > signal_val * 0.95:
            score += 5
        
        # 4. MOMENTUM - 1 AY (20 puan)
        momentum_1m = 0
        momentum_3m = 0
        
        if len(close) >= 22:
            momentum_1m = ((close[-1] / close[-22]) - 1) * 100
        
        if len(close) >= 66:
            momentum_3m = ((close[-1] / close[-66]) - 1) * 100
        
        if momentum_1m > 15:
            score += 20
        elif momentum_1m > 10:
            score += 17
        elif momentum_1m > 5:
            score += 14
        elif momentum_1m > 0:
            score += 10
        elif momentum_1m > -5:
            score += 5
        elif momentum_1m > -10:
            score += 2
        
        # 5. HACİM (10 puan)
        if vol is not None and len(vol) >= 20:
            vol_5d = np.mean(vol[-5:])
            vol_20d = np.mean(vol[-20:])
            if vol_20d > 0:
                vol_ratio = vol_5d / vol_20d
                if vol_ratio > 2.0:
                    score += 10
                elif vol_ratio > 1.5:
                    score += 8
                elif vol_ratio > 1.2:
                    score += 6
                elif vol_ratio > 1.0:
                    score += 4
                else:
                    score += 2
        else:
            score += 3
        
        # 6. VOLATİLİTE - ATR (10 puan)
        if len(close) >= 14:
            high = df['High'].values
            low = df['Low'].values
            tr = np.mean(high[-14:] - low[-14:])
            atr_pct = (tr / current_price) * 100
            
            if 1.5 <= atr_pct <= 5.0:
                score += 10  # İdeal swing volatilitesi
            elif 0.8 <= atr_pct < 1.5:
                score += 6
            elif 5.0 < atr_pct <= 8.0:
                score += 5
            else:
                score += 2
        
        # Maksimum 100 puan
        score = min(score, 100)
        
        # MACD Label
        if macd_val > signal_val and hist_val > hist_prev:
            macd_label = "🔥 Güçlü Alı"
        elif macd_val > signal_val:
            macd_label = "✅ Pozitif"
        elif hist_val > 0:
            macd_label = "⚠️ Zayıf"
        else:
            macd_label = "❌ Negatif"
        
        # Trend Durumu
        if ma200 and current_price > ma200:
            trend = "📈 Yükseliş"
        elif current_price > ma50:
            trend = "➡️ Yatay"
        else:
            trend = "📉 Düşüş"
        
        return {
            'Hisse': ticker.replace('.IS', ''),
            'Fiyat': round(current_price, 2),
            'Toplam Skor': round(score, 1),
            'RSI': round(rsi, 1),
            'MACD': macd_label,
            'Momentum 1A': f"%{momentum_1m:.1f}",
            'Momentum 3A': f"%{momentum_3m:.1f}",
            'Trend': trend,
            'MA50 Üstü': '✅' if current_price > ma50 else '❌',
            'MA200 Üstü': '✅' if ma200 and current_price > ma200 else '➖'
        }
    
    except Exception as e:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# ANA UYGULAMA
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.title("🚀 BIST Hisse Tarayıcı")
    st.markdown("### 📊 100 Puan Üzerinden Teknik Analiz | 70+ = AL Sinyali")
    st.info("⚠️ Yatırım Tavsiyesi Değildir. Veriler 15 dakika gecikmeli olabilir.")
    
    st.sidebar.header("⚙️ Tarama Ayarları")
    
    min_score = st.sidebar.slider("Minimum AL Skoru", 50, 90, 70, 5)
    max_stocks = st.sidebar.slider("Taranacak Hisse Sayısı", 30, 100, 100, 10)
    show_all = st.sidebar.checkbox("Tüm Sonuçları Göster", value=False)
    
    st.divider()
    
    # Başlat butonu
    if st.button("🚀 TARAMAYI BAŞLAT"):
        with st.spinner('⏳ 100 hisse taranıyor... (2-3 dakika sürebilir)'):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            error_count = 0
            
            for i, ticker in enumerate(BIST100_TICKERS[:max_stocks]):
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
            
            # Özet metrikler
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🔍 Taranan", max_stocks)
            c2.metric("✅ Veri Alınan", len(results))
            c3.metric("🚀 AL Listesi", len([r for r in results if r['Toplam Skor'] >= min_score]))
            c4.metric("⚠️ Hata", error_count)
            
            if not results:
                st.error("⚠️ Hiç veri alınamadı. Yahoo Finance yoğun olabilir. 2 dakika bekleyip tekrar deneyin.")
                st.stop()
            
            # Sırala ve filtrele
            df = pd.DataFrame(results)
            df = df.sort_values('Toplam Skor', ascending=False).reset_index(drop=True)
            df_al = df[df['Toplam Skor'] >= min_score]
            
            # Az sonuç varsa en yüksekleri göster
            if len(df_al) < 5 and not show_all:
                df_al = df.head(15)
            
            st.divider()
            
            if not df_al.empty:
                # En iyi 5 kart
                st.subheader("🏆 En İyi 5 Hisse")
                cols = st.columns(min(5, len(df_al)))
                
                for idx, (_, row) in enumerate(df_al.head(5).iterrows()):
                    with cols[idx]:
                        emoji = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉" if idx == 2 else "⭐"
                        
                        score_color = "#00ff41" if row['Toplam Skor'] >= 80 else "#f4a83a" if row['Toplam Skor'] >= 70 else "#ff4444"
                        
                        st.markdown(f"""
                        <div class="stock-card">
                            <h3 style="margin:0;color:#00ff41">{emoji} {row['Hisse']}</h3>
                            <p style="margin:5px 0;font-size:20px"><b>{row['Fiyat']} ₺</b></p>
                            <p style="margin:5px 0"><b>Skor:</b> <span style="color:{score_color};font-size:22px">{row['Toplam Skor']}/100</span></p>
                            <p style="margin:5px 0"><b>RSI:</b> {row['RSI']} | <b>MACD:</b> {row['MACD']}</p>
                            <p style="margin:5px 0"><b>Trend:</b> {row['Trend']}</p>
                            <p style="margin:5px 0"><b>1A:</b> {row['Momentum 1A']} | <b>3A:</b> {row['Momentum 3A']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.divider()
                
                # Tüm tablo
                st.subheader(f"📊 AL Listesi ({len(df_al)} Hisse)")
                
                display_cols = ['Hisse', 'Fiyat', 'Toplam Skor', 'RSI', 'MACD', 'Trend', 
                               'Momentum 1A', 'Momentum 3A', 'MA50 Üstü', 'MA200 Üstü']
                
                st.dataframe(df_al[display_cols], use_container_width=True, hide_index=True)
                
                # CSV İndir
                csv = df_al[display_cols].to_csv(index=False, encoding='utf-8-sig')
                st.download_button("📥 CSV Olarak İndir", csv, "bist_tarama.csv", "text/csv")
                
                st.divider()
                
                # Grafik
                st.subheader("📈 Skor Dağılımı (İlk 20)")
                df_chart = df.head(20)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df_chart['Hisse'],
                    y=df_chart['Toplam Skor'],
                    name='Toplam Skor',
                    marker_color='#00ff41'
                ))
                fig.add_hline(y=min_score, line_dash='dash', line_color='red', 
                             annotation_text=f"AL Eşiği ({min_score})")
                fig.update_layout(
                    template='plotly_dark',
                    title='Hisse Bazında Skorlar',
                    xaxis_title='Hisse',
                    yaxis_title='Puan',
                    yaxis_range=[0, 100],
                    height=400,
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"✅ Tarama Tamamlandı! {len(df_al)} hisse {min_score}+ puan aldı.")
            
            else:
                st.warning(f"⚠️ {min_score}+ puan alan hisse bulunamadı. Skoru 60'a düşürmeyi deneyin.")
                st.info("💡 İpucu: Piyasa düşüşteyse daha az hisse yüksek puan alır.")
    
    else:
        st.info("⬅️ Sol panelden ayarları yapıp **Taramayı Başlat** butonuna basın.")
        
        with st.expander("📖 Puanlama Sistemi Nasıl Çalışır?"):
            st.markdown("""
            ### 📊 Toplam 100 Puan Üzerinden Değerlendirme
            
            | Kriter | Maks Puan | Açıklama |
            |--------|-----------|----------|
            | **MA50 Trend** | 20 | Fiyat 50 günlük ortalamanın üzerinde mi? |
            | **MA200 Trend** | 10 | Fiyat 200 günlük ortalamanın üzerinde mi? |
            | **RSI** | 20 | 50-65 arası ideal (20 puan) |
            | **MACD** | 20 | Crossover + histogram büyüme = tam puan |
            | **Momentum 1A** | 20 | 1 aylık getiri >%15 = tam puan |
            | **Hacim** | 10 | 5 günlük hacim > 20 günlük hacim |
            | **Volatilite** | 10 | ATR %1.5-5 arası ideal |
            
            ### 🎯 Sinyal Seviyeleri
            
            | Puan | Sinyal | Aksiyon |
            |------|--------|---------|
            | 80-100 | 🟢 Çok Güçlü | Hemen değerlendir |
            | 70-79 | 🟡 Güçlü | AL listesi |
            | 60-69 | 🟠 Orta | İzleme listesine al |
            | <60 | 🔴 Zayıf | Bekle |
            
            ### ⚠️ Önemli Notlar
            
            1. Bu sistem **sadece teknik analiz** kullanır
            2. Temel analiz (F/K, PD/DD) eklenmedi çünkü Yahoo Finance Cloud'da sık hata veriyor
            3. Veriler **15 dakika gecikmeli** olabilir
            4. **Yatırım tavsiyesi değildir**, kendi araştırmanızı yapın
            """)

if __name__ == "__main__":
    main()
