import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import numpy as np
from datetime import datetime
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from data_processing import calc_cv, BusinessAnalyzer, process_data
# ============================================================================
# 外部ライブラリ代替実装（scipy、sklearn不要）
# ============================================================================


# ページ設定
st.set_page_config(page_title="経営ダッシュボード", layout="wide", initial_sidebar_state="expanded")

# 最適化されたCSS
css_path = Path(__file__).parent / "assets" / "style.css"
st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

if 'dark_mode' in st.session_state:
    if st.session_state['dark_mode']:
        st.markdown(
            """
            <script>
            document.body.classList.add('dark-mode');
            </script>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <script>
            document.body.classList.remove('dark-mode');
            </script>
            """,
            unsafe_allow_html=True,
        )

st.markdown('<div class="nav-bar">🏢 経営統合管理</div>', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">📊 経営統合ダッシュボード</h1>', unsafe_allow_html=True)

if 'loaded_once' not in st.session_state:
    with st.spinner("Loading Dashboard..."):
        time.sleep(1)
    st.balloons()
    st.session_state['loaded_once'] = True

# ============================================================================
# 共通関数・ユーティリティ（最適化済み）
# ============================================================================

# 数値フォーマット関数
fmt_num = lambda n: f"{int(n):,}" if pd.notna(n) and n != 0 else "0"
fmt_dec = lambda x: f"{x:,.2f}" if pd.notna(x) else "0.00"
fmt_pct = lambda x: f"{x:.2f}%" if pd.notna(x) else "0.00%"

def create_unified_graph(x, y, name, color, title, x_title, y_title, chart_type="bar", 
                        y2_data=None, y2_name=None, y2_color=None, height=400):
    """統一グラフ作成関数（機能拡張・最適化済み）"""
    fig = go.Figure()
    
    if chart_type == "bar":
        fig.add_trace(go.Bar(x=x, y=y, name=name, marker_color=color))
    elif chart_type == "line":
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=name, line=dict(color=color, width=3)))
    elif chart_type == "scatter":
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name=name, marker=dict(color=color, size=8)))
    
    if y2_data is not None:
        fig.add_trace(go.Scatter(x=x, y=y2_data, mode="lines+markers", name=y2_name, 
                               line=dict(color=y2_color, width=3), yaxis="y2"))
        fig.update_layout(yaxis2=dict(title=y2_name, side="right", overlaying="y", showgrid=False))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, weight="bold")),
        xaxis_title=x_title, yaxis_title=y_title, height=height,
        plot_bgcolor="white", paper_bgcolor="white",
        hovermode="x unified", showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig

def show_metrics(metrics_data, cols=None):
    """メトリクス表示関数（レスポンシブ対応）"""
    if cols is None:
        cols = min(len(metrics_data), 5)  # 最大5列
    
    columns = st.columns(cols)
    for i, (label, value) in enumerate(metrics_data.items()):
        with columns[i % cols]:
            if isinstance(value, tuple):  # デルタ値付き
                st.metric(label=label, value=value[0], delta=value[1])
            else:
                st.metric(label=label, value=value)

def download_csv(df, filename, label):
    """CSV ダウンロードボタン"""
    csv = df.to_csv(index=False, encoding='utf-8-sig')
    return st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

# ============================================================================
# 高度な分析関数群
# ============================================================================

# ファイル処理・キャッシュ管理（最適化済み）
# ============================================================================

with st.sidebar:
    dark_mode = st.checkbox("🌙 ダークモード", key="dark_mode")
    st.markdown("### 📁 ファイルアップロード")
    sales_xlsx = st.file_uploader("売上 Excel ファイル", type=["xlsx"], key="sales")
    material_xlsx = st.file_uploader("原料 Excel ファイル", type=["xlsx"], key="material")
    
    # キャッシュ状態表示（簡潔化）
    st.markdown("### 🔄 システム状態")
    
    cache_info = []
    if sales_xlsx:
        cache_key = f"sales_{sales_xlsx.name}_{len(sales_xlsx.read())}"
        sales_xlsx.seek(0)
        cache_info.append(f"📈 売上: {'💾' if cache_key in st.session_state else '🔄'}")
    
    if material_xlsx:
        cache_key = f"material_{material_xlsx.name}_{len(material_xlsx.read())}"
        material_xlsx.seek(0)
        cache_info.append(f"🏭 原料: {'💾' if cache_key in st.session_state else '🔄'}")
    
    if cache_info:
        for info in cache_info:
            st.info(info)
    
    # 処理済みファイル数表示
    processed_count = sum(1 for key in st.session_state if key.startswith('meta_'))
    if processed_count > 0:
        st.success(f"💾 {processed_count} ファイル処理済み")
        if st.button("🗑️ キャッシュクリア"):
            keys_to_remove = [key for key in st.session_state if key.startswith(('sales_', 'material_', 'meta_'))]
            for key in keys_to_remove:
                del st.session_state[key]
            st.success("✅ クリア完了")
            st.rerun()

# データ処理関数（最適化済み）

def process_with_smart_cache(file_obj, data_type):
    """スマートキャッシュ処理（UI最適化済み）"""
    file_name = file_obj.name
    file_content = file_obj.read()
    file_obj.seek(0)
    
    cache_key = f"{data_type}_{file_name}_{len(file_content)}"
    meta_key = f"meta_{data_type}_{file_name}"
    
    if cache_key in st.session_state:
        # キャッシュヒット（高速表示）
        return st.session_state[cache_key]
    
    # 新規処理
    with st.spinner(f"{'📈 売上' if data_type == 'sales' else '🏭 原料'}データ処理中..."):
        result = process_data(file_content, file_name, data_type)
        
        # キャッシュ保存
        st.session_state[cache_key] = result
        st.session_state[meta_key] = {
            "file_name": file_name,
            "data_type": data_type,
            "processed_time": datetime.now().strftime("%H:%M:%S")
        }
        
        time.sleep(0.5)  # UI フィードバック
    
    return result

# データ処理実行
sales_df = daily_sales = sales_year = sales_month = None
mat_df = daily_mat = mat_year = mat_month = None

if sales_xlsx:
    try:
        sales_df, daily_sales, sales_year, sales_month = process_with_smart_cache(sales_xlsx, "sales")
    except Exception as e:
        st.error(f"❌ 売上データエラー: {str(e)}")

if material_xlsx:
    try:
        mat_df, daily_mat, mat_year, mat_month = process_with_smart_cache(material_xlsx, "material")
    except Exception as e:
        st.error(f"❌ 原料データエラー: {str(e)}")

# ビジネス分析器初期化
analyzer = None
if sales_df is not None and mat_df is not None:
    analyzer = BusinessAnalyzer(sales_df, daily_sales, mat_df, daily_mat)

# ============================================================================
# タブ構成（メインダッシュボード追加）
# ============================================================================

tabs = st.tabs([
    "🏠 メインダッシュボード", "📊 売上分析", "🏭 原料分析", "📈 比較分析", 
    "🎯 顧客分析", "📊 トレンド分析", "⚠️ リスク分析", 
    "⚡ 効率性分析", "🎲 シミュレーション", "🔍 月内最適化"
])

# ============================================================================
# メインダッシュボードタブ
# ============================================================================

with tabs[0]:
    if analyzer is not None:
        st.markdown('<h2 class="section-header">🏠 経営統合ダッシュボード</h2>', unsafe_allow_html=True)
        
        # 総合指標取得
        metrics = analyzer.get_summary_metrics()
        rankings = analyzer.get_performance_ranking()
        efficiency = analyzer.analyze_efficiency_patterns()
        anomalies = analyzer.detect_anomalies()
        
        # 上部：主要KPI
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("💰 総売上", f"¥{fmt_num(metrics['total_sales'])}")
        with col2:
            st.metric("📈 総利益", f"¥{fmt_num(metrics['total_profit'])}")
        with col3:
            st.metric("📊 利益率", f"{metrics['overall_profit_rate']:.1f}%")
        with col4:
            st.metric("⚡ 効率性", f"{efficiency['consistency_score']:.0f}点")
        with col5:
            alert_count = anomalies['anomaly_count']
            st.metric("⚠️ アラート", f"{alert_count}件", delta=f"-{alert_count}件" if alert_count > 0 else "正常")
        
        # 中部：パフォーマンス概要
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="section-header">🏆 パフォーマンス概要</h3>', unsafe_allow_html=True)
            
            # ベスト日
            best_sales = rankings["売上TOP3"].iloc[0]
            best_profit = rankings["利益TOP3"].iloc[0]
            
            st.markdown(f"""
            <div class="success-card">
                <h4>🥇 最高売上日</h4>
                <p><strong>{best_sales['日付']}</strong></p>
                <p>¥{fmt_num(best_sales['総売上額'])}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="success-card">
                <h4>💎 最高利益日</h4>
                <p><strong>{best_profit['日付']}</strong></p>
                <p>¥{fmt_num(best_profit['利益'])}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if efficiency['improvement_potential'] > 0:
                st.markdown(f"""
                <div class="info-card">
                    <h4>📈 改善ポテンシャル</h4>
                    <p>¥{fmt_num(efficiency['improvement_potential'])}</p>
                    <p>効率性向上で期待される追加利益</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown('<h3 class="section-header">⚠️ 注意事項・アラート</h3>', unsafe_allow_html=True)
            
            # リスク評価
            sales_cv = metrics['sales_cv']
            material_cv = metrics['material_cv']
            
            if sales_cv > 30:
                st.markdown(f"""
                <div class="alert-card">
                    <h4>📊 売上変動リスク</h4>
                    <p>変動係数: {sales_cv:.1f}% (高リスク)</p>
                    <p>売上の安定化施策が必要</p>
                </div>
                """, unsafe_allow_html=True)
            
            if material_cv > 30:
                st.markdown(f"""
                <div class="alert-card">
                    <h4>🏭 原料費変動リスク</h4>
                    <p>変動係数: {material_cv:.1f}% (高リスク)</p>
                    <p>原料調達の最適化が必要</p>
                </div>
                """, unsafe_allow_html=True)
            
            if anomalies['anomaly_count'] > 0:
                st.markdown(f"""
                <div class="alert-card">
                    <h4>🔍 異常値検出</h4>
                    <p>{anomalies['anomaly_count']}件の異常日を検出</p>
                    <p>詳細は「月内最適化」タブで確認</p>
                </div>
                """, unsafe_allow_html=True)
            
            if anomalies['anomaly_count'] == 0 and sales_cv < 20 and material_cv < 20:
                st.markdown("""
                <div class="success-card">
                    <h4>✅ 健全な経営状態</h4>
                    <p>リスク指標は正常範囲内</p>
                    <p>現在の運営を維持</p>
                </div>
                """, unsafe_allow_html=True)
        
        # 下部：トレンドグラフ
        st.markdown('<h3 class="section-header">📈 月内トレンド</h3>', unsafe_allow_html=True)
        
        if analyzer.comparison_df is not None:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('日別売上推移', '利益率推移', '効率スコア推移', '累積利益推移'),
                vertical_spacing=0.35,
                horizontal_spacing=0.1
            )
            
            df = analyzer.comparison_df
            
            # 売上推移
            fig.add_trace(go.Scatter(x=df['日付'], y=df['総売上額'], name='売上', 
                                   line=dict(color='#3b82f6', width=2), showlegend=False), row=1, col=1)
            # 利益率推移
            fig.add_trace(go.Scatter(x=df['日付'], y=df['利益率'], name='利益率', 
                                   line=dict(color='#10b981', width=2), showlegend=False), row=1, col=2)
            # 効率スコア推移
            fig.add_trace(go.Scatter(x=df['日付'], y=df['効率スコア'], name='効率', 
                                   line=dict(color='#f59e0b', width=2), showlegend=False), row=2, col=1)
            # 累積利益推移
            fig.add_trace(go.Scatter(x=df['日付'], y=df['利益'].cumsum(), name='累積利益', 
                                   line=dict(color='#8b5cf6', width=2), showlegend=False), row=2, col=2)
            
            # 各サブプロットの軸ラベル設定
            fig.update_xaxes(title_text="日付", row=1, col=1)
            fig.update_xaxes(title_text="日付", row=1, col=2)
            fig.update_xaxes(title_text="日付", row=2, col=1)
            fig.update_xaxes(title_text="日付", row=2, col=2)
            
            fig.update_yaxes(title_text="売上額", row=1, col=1)
            fig.update_yaxes(title_text="利益率(%)", row=1, col=2)
            fig.update_yaxes(title_text="効率スコア", row=2, col=1)
            fig.update_yaxes(title_text="累積利益", row=2, col=2)
            
            fig.update_layout(height=750, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("📤 売上と原料の両方のExcelファイルをアップロードしてください")
        
        # データなしの場合のサンプル表示
        st.markdown("""
        ### 📊 ダッシュボード機能プレビュー
        
        **主要機能:**
        - 🏆 リアルタイム経営指標
        - ⚠️ 自動アラート・リスク検知
        - 📈 改善ポテンシャル分析
        - 🎯 パフォーマンスランキング
        - 🔍 異常値自動検出
        - 💡 経営改善提案
        
        **データアップロード後に利用可能になります**
        """)

# ============================================================================
# 既存タブ（売上分析）
# ============================================================================

with tabs[1]:
    if sales_xlsx and sales_df is not None and daily_sales is not None:
        st.markdown('<h2 class="section-header">📊 売上分析</h2>', unsafe_allow_html=True)
        
        # サマリーメトリクス
        show_metrics({
            "💰 総売上額": f"¥{fmt_num(daily_sales['総売上額'].sum())}",
            "📊 日平均売上": f"¥{fmt_num(daily_sales['総売上額'].mean())}",
            "📈 最高日売上": f"¥{fmt_num(daily_sales['総売上額'].max())}",
            "📋 総取引件数": f"{len(sales_df):,} 件",
            "🎯 売上安定性": f"{100-calc_cv(daily_sales['総売上額']):.0f}点"
        })
        
        # グラフ
        fig = create_unified_graph(
            daily_sales["日付"], daily_sales["総売上額"], "日別売上", "#3b82f6",
            "日別売上推移", "日付", "売上額", 
            y2_data=daily_sales["累計売上"], y2_name="累計売上", y2_color="#ef4444"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # テーブル（コンパクト化）
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<h3 class="section-header">日別売上集計</h3>', unsafe_allow_html=True)
            download_csv(daily_sales[["日付", "総売上額", "累計売上", "移動平均_7日"]], 
                        f"日別売上_{sales_year}{sales_month}.csv", "📥 CSV")
            
            display_daily = daily_sales[["日付", "総売上額", "累計売上", "移動平均_7日"]].copy()
            for col in ["総売上額", "累計売上", "移動平均_7日"]:
                display_daily[col] = display_daily[col].apply(fmt_num)
            st.dataframe(display_daily, use_container_width=True, hide_index=True, height=350)
        
        with col2:
            st.markdown('<h3 class="section-header">売上詳細データ</h3>', unsafe_allow_html=True)
            download_csv(sales_df[["日付", "得意先コード", "得意先名", "総売上額"]], 
                        f"売上詳細_{sales_year}{sales_month}.csv", "📥 CSV")
            
            display_sales = sales_df[["日付", "得意先コード", "得意先名", "総売上額"]].copy()
            display_sales["総売上額"] = display_sales["総売上額"].apply(fmt_num)
            st.dataframe(display_sales, use_container_width=True, hide_index=True, height=350)
    else:
        st.info("📤 売上Excelファイルをアップロードしてください")

# ============================================================================
# 原料分析タブ
# ============================================================================

with tabs[2]:
    if material_xlsx and mat_df is not None and daily_mat is not None:
        st.markdown('<h2 class="section-header">🏭 原料分析</h2>', unsafe_allow_html=True)
        
        # サマリーメトリクス
        show_metrics({
            "💰 総原料費": f"¥{fmt_num(daily_mat['使用金額'].sum())}",
            "📊 日平均原料費": f"¥{fmt_num(daily_mat['使用金額'].mean())}",
            "📈 最高日原料費": f"¥{fmt_num(daily_mat['使用金額'].max())}",
            "📋 総使用記録": f"{len(mat_df):,} 件",
            "⚖️ 使用安定性": f"{100-calc_cv(daily_mat['使用金額']):.0f}点"
        })
        
        # 船名別分析（最適化済み）
        ship_analysis = mat_df.groupby("船名").agg({
            "総額": ["sum", "mean", "count"], 
            "¥/kg": "mean",
            "kg/cs": "mean",
            "cs": "sum"
        }).round(2)
        ship_analysis.columns = ["総使用額", "平均使用額", "使用回数", "平均kg単価", "平均kg/cs", "総cs数"]
        ship_analysis = ship_analysis.reset_index().sort_values("総使用額", ascending=False)
        ship_analysis["効率指標"] = ship_analysis["総使用額"] / ship_analysis["総cs数"]
        
        # グラフ
        fig = create_unified_graph(
            daily_mat["日付"], daily_mat["使用金額"], "日別原料費", "#f59e0b",
            "日別原料費推移", "日付", "原料費",
            y2_data=daily_mat["累計金額"], y2_name="累計原料費", y2_color="#8b5cf6"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # テーブル
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<h3 class="section-header">船名別効率分析</h3>', unsafe_allow_html=True)
            download_csv(ship_analysis, f"船名別分析_{mat_year}{mat_month}.csv", "📥 CSV")
            
            display_ship = ship_analysis.copy()
            for col in ["総使用額", "平均使用額", "平均kg単価", "効率指標"]:
                display_ship[col] = display_ship[col].apply(fmt_num)
            st.dataframe(display_ship, use_container_width=True, hide_index=True, height=350)
        
        with col2:
            st.markdown('<h3 class="section-header">日別原料使用</h3>', unsafe_allow_html=True)
            download_csv(daily_mat[["日付", "使用金額", "累計金額", "移動平均_7日"]], 
                        f"日別原料_{mat_year}{mat_month}.csv", "📥 CSV")
            
            display_daily_mat = daily_mat[["日付", "使用金額", "累計金額", "移動平均_7日"]].copy()
            for col in ["使用金額", "累計金額", "移動平均_7日"]:
                display_daily_mat[col] = display_daily_mat[col].apply(fmt_num)
            st.dataframe(display_daily_mat, use_container_width=True, hide_index=True, height=350)
    else:
        st.info("📤 原料Excelファイルをアップロードしてください")

# ============================================================================
# 比較分析タブ（修正済み）
# ============================================================================

with tabs[3]:
    if analyzer is not None:
        st.markdown('<h2 class="section-header">📈 比較分析</h2>', unsafe_allow_html=True)
        
        comparison_df = analyzer.comparison_df
        metrics = analyzer.get_summary_metrics()
        
        # サマリーメトリクス（修正済み）
        show_metrics({
            "💰 総売上額": f"¥{fmt_num(metrics['total_sales'])}",
            "🏭 総原料費": f"¥{fmt_num(metrics['total_material'])}",
            "📈 総利益": f"¥{fmt_num(metrics['total_profit'])}",
            "📊 利益率": f"{metrics['overall_profit_rate']:.2f}%",
            "⚖️ 原料費率": f"{metrics['overall_material_rate']:.2f}%"
        })
        
        # グラフ（統合版）
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('売上・原料費・利益の推移', '利益率・原料費率の推移'),
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )
        
        # 上段：金額推移
        fig.add_trace(go.Bar(x=comparison_df["日付"], y=comparison_df["総売上額"], 
                           name="売上額", marker_color="#3b82f6"), row=1, col=1)
        fig.add_trace(go.Bar(x=comparison_df["日付"], y=comparison_df["使用金額"], 
                           name="原料費", marker_color="#f59e0b"), row=1, col=1)
        fig.add_trace(go.Scatter(x=comparison_df["日付"], y=comparison_df["利益"], 
                               mode="lines+markers", name="利益", 
                               line=dict(color="#10b981", width=3)), row=1, col=1)
        
        # 下段：比率推移
        fig.add_trace(go.Scatter(x=comparison_df["日付"], y=comparison_df["利益率"], 
                               mode="lines+markers", name="利益率", 
                               line=dict(color="#8b5cf6", width=3)), row=2, col=1)
        fig.add_trace(go.Scatter(x=comparison_df["日付"], y=comparison_df["原料費率"], 
                               mode="lines+markers", name="原料費率", 
                               line=dict(color="#ef4444", width=3)), row=2, col=1)
        
        fig.update_layout(height=600, showlegend=True)
        fig.update_yaxes(title_text="金額 (¥)", row=1, col=1)
        fig.update_yaxes(title_text="比率 (%)", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)
        
        # 比較テーブル
        st.markdown('<h3 class="section-header">日別比較データ</h3>', unsafe_allow_html=True)
        download_csv(comparison_df[["日付", "総売上額", "使用金額", "利益", "利益率", "原料費率", "効率スコア"]], 
                    f"比較分析_{sales_year}{sales_month}.csv", "📥 CSV")
        
        display_comparison = comparison_df[["日付", "総売上額", "使用金額", "利益", "利益率", "原料費率", "効率スコア"]].copy()
        display_comparison["総売上額"] = display_comparison["総売上額"].apply(fmt_num)
        display_comparison["使用金額"] = display_comparison["使用金額"].apply(fmt_num)
        display_comparison["利益"] = display_comparison["利益"].apply(fmt_num)
        display_comparison["利益率"] = display_comparison["利益率"].apply(fmt_pct)
        display_comparison["原料費率"] = display_comparison["原料費率"].apply(fmt_pct)
        display_comparison["効率スコア"] = display_comparison["効率スコア"].apply(fmt_dec)
        
        st.dataframe(display_comparison, use_container_width=True, hide_index=True)
    else:
        st.info("📤 売上と原料の両方のExcelファイルをアップロードしてください")

# ============================================================================
# 顧客分析タブ（拡張版）
# ============================================================================

with tabs[4]:
    if analyzer is not None:
        st.markdown('<h2 class="section-header">🎯 顧客分析（拡張版）</h2>', unsafe_allow_html=True)
        
        try:
            customer_data = analyzer.customer_analysis()
            abc_analysis = customer_data["customer_stats"]
            
            # 顧客サマリー
            show_metrics({
                "👥 総顧客数": f"{customer_data['active_customers']:,} 社",
                "💰 平均顧客価値": f"¥{fmt_num(customer_data['avg_customer_value'])}",
                "🎯 上位5社集中度": f"{customer_data['top5_concentration']:.1f}%",
                "🅰️ Aランク顧客": f"{len(abc_analysis[abc_analysis['分類'] == 'A'])} 社",
                "📊 取引活発度": f"{abc_analysis['取引日数'].mean():.1f} 日"
            })
            
            # ABC分析結果詳細
            col1, col2, col3 = st.columns(3)
            for i, rank in enumerate(['A', 'B', 'C']):
                rank_data = abc_analysis[abc_analysis['分類'] == rank]
                rank_sales = rank_data['総売上'].sum()
                rank_ratio = rank_sales / abc_analysis['総売上'].sum() * 100
                
                with [col1, col2, col3][i]:
                    color = ["#10b981", "#f59e0b", "#ef4444"][i]
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {color}, {color}dd); color: white; 
                               padding: 1rem; border-radius: 8px; text-align: center;">
                        <h3>{rank}ランク顧客</h3>
                        <p><strong>{len(rank_data)}社</strong></p>
                        <p>売上: ¥{fmt_num(rank_sales)}</p>
                        <p>構成比: {rank_ratio:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # パレート図（拡張版）
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('売上パレート図（上位15社）', '顧客価値スコア分布'),
                specs=[[{"secondary_y": True}, {"secondary_y": False}]]
            )
            
            top15 = abc_analysis.head(15)
            fig.add_trace(go.Bar(x=top15["得意先名"], y=top15["総売上"], name="売上額"), row=1, col=1)
            fig.add_trace(go.Scatter(x=top15["得意先名"], y=top15["累積比率"], 
                                   mode="lines+markers", name="累積比率"), row=1, col=1, secondary_y=True)
            
            # 顧客価値分布
            fig.add_trace(go.Histogram(x=abc_analysis["顧客価値スコア"], nbinsx=20, name="顧客価値分布"), row=1, col=2)
            
            fig.update_layout(height=500)
            fig.update_yaxes(title_text="累積比率 (%)", secondary_y=True, row=1, col=1)
            st.plotly_chart(fig, use_container_width=True)
            
            # 詳細テーブル
            st.markdown('<h3 class="section-header">顧客詳細分析</h3>', unsafe_allow_html=True)
            download_csv(abc_analysis, f"顧客分析_{sales_year}{sales_month}.csv", "📥 CSV")
            
            display_abc = abc_analysis.copy()
            for col in ["総売上", "平均取引額", "顧客価値スコア"]:
                display_abc[col] = display_abc[col].apply(fmt_num)
            display_abc["累積比率"] = display_abc["累積比率"].apply(fmt_pct)
            
            st.dataframe(display_abc[["得意先名", "総売上", "平均取引額", "取引回数", "取引日数", "顧客価値スコア", "分類"]], 
                        use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"顧客分析エラー: {str(e)}")
    else:
        st.info("📤 売上Excelファイルをアップロードしてください")

# ============================================================================
# トレンド分析タブ（拡張版）
# ============================================================================

with tabs[5]:
    if sales_xlsx and daily_sales is not None:
        st.markdown('<h2 class="section-header">📊 トレンド分析（拡張版）</h2>', unsafe_allow_html=True)
        
        try:
            # 基本トレンド指標
            trend_metrics = {
                "📈 平均成長率": f"{((daily_sales['総売上額'].iloc[-1] / daily_sales['総売上額'].iloc[0]) - 1) * 100:.1f}%",
                "🎯 トレンド強度": f"{abs(daily_sales['総売上額'].corr(daily_sales['日'])) * 100:.0f}点",
                "📊 周期性指標": f"{100 - calc_cv(daily_sales['移動平均_7日']):.0f}点",
                "⚡ 勢い指標": f"{daily_sales['総売上額'].tail(7).mean() / daily_sales['総売上額'].head(7).mean():.2f}倍"
            }
            show_metrics(trend_metrics)
            
            # 曜日別分析（拡張版）
            if "曜日_jp" in daily_sales.columns:
                dow_analysis = daily_sales.groupby("曜日_jp").agg({
                    "総売上額": ["mean", "sum", "count", "std"]
                }).round(2)
                dow_analysis.columns = ["平均売上", "合計売上", "営業日数", "売上標準偏差"]
                dow_analysis = dow_analysis.reindex(['月', '火', '水', '木', '金', '土', '日'])
                dow_analysis["変動係数"] = (dow_analysis["売上標準偏差"] / dow_analysis["平均売上"] * 100).round(1)
                dow_analysis["安定性スコア"] = (100 - dow_analysis["変動係数"]).round(0)
                
                # 複合グラフ
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('曜日別平均売上', '移動平均トレンド', '曜日別安定性', '売上分布'),
                    vertical_spacing=0.25,
                    horizontal_spacing=0.12
                )
                
                # 曜日別売上
                fig.add_trace(go.Bar(x=dow_analysis.index, y=dow_analysis["平均売上"], 
                                   marker_color="#3b82f6", name="平均売上", showlegend=False), row=1, col=1)
                
                # 移動平均
                fig.add_trace(go.Scatter(x=daily_sales["日付"], y=daily_sales["総売上額"], 
                                      mode="lines", name="日別売上", opacity=0.5, 
                                      line=dict(color="#cbd5e1", width=1), showlegend=False), row=1, col=2)
                fig.add_trace(go.Scatter(x=daily_sales["日付"], y=daily_sales["移動平均_7日"], 
                                      mode="lines", name="7日移動平均", 
                                      line=dict(width=3, color="#3b82f6"), showlegend=False), row=1, col=2)
                
                # 安定性
                fig.add_trace(go.Bar(x=dow_analysis.index, y=dow_analysis["安定性スコア"], 
                                   marker_color="#10b981", name="安定性", showlegend=False), row=2, col=1)
                
                # 分布
                fig.add_trace(go.Histogram(x=daily_sales["総売上額"], nbinsx=15, 
                                         marker_color="#8b5cf6", name="売上分布", showlegend=False), row=2, col=2)
                
                # 軸ラベル設定
                fig.update_xaxes(title_text="曜日", row=1, col=1)
                fig.update_xaxes(title_text="日付", row=1, col=2)
                fig.update_xaxes(title_text="曜日", row=2, col=1)
                fig.update_xaxes(title_text="売上額", row=2, col=2)
                
                fig.update_yaxes(title_text="平均売上", row=1, col=1)
                fig.update_yaxes(title_text="売上額", row=1, col=2)
                fig.update_yaxes(title_text="安定性スコア", row=2, col=1)
                fig.update_yaxes(title_text="頻度", row=2, col=2)
                
                fig.update_layout(height=700, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # 曜日別詳細テーブル
                st.markdown('<h3 class="section-header">曜日別詳細分析</h3>', unsafe_allow_html=True)
                download_csv(dow_analysis, f"曜日別分析_{sales_year}{sales_month}.csv", "📥 CSV")
                
                display_dow = dow_analysis.copy()
                for col in ["平均売上", "合計売上", "売上標準偏差"]:
                    display_dow[col] = display_dow[col].apply(fmt_num)
                display_dow["変動係数"] = display_dow["変動係数"].apply(lambda x: f"{x:.1f}%")
                
                st.dataframe(display_dow, use_container_width=True)
                
        except Exception as e:
            st.error(f"トレンド分析エラー: {str(e)}")
    else:
        st.info("📤 売上Excelファイルをアップロードしてください")

# ============================================================================
# リスク分析タブ（拡張版）
# ============================================================================

with tabs[6]:
    if analyzer is not None:
        st.markdown('<h2 class="section-header">⚠️ リスク分析（拡張版）</h2>', unsafe_allow_html=True)
        
        try:
            metrics = analyzer.get_summary_metrics()
            anomalies = analyzer.detect_anomalies()
            customer_data = analyzer.customer_analysis()
            
            # リスク指標（拡張版）
            sales_cv = metrics['sales_cv']
            material_cv = metrics['material_cv']
            concentration_risk = customer_data['top5_concentration']
            
            def risk_level(value, thresholds):
                if value > thresholds[1]: return "高", "#ef4444"
                elif value > thresholds[0]: return "中", "#f59e0b"
                else: return "低", "#10b981"
            
            sales_risk, sales_color = risk_level(sales_cv, [15, 30])
            material_risk, material_color = risk_level(material_cv, [15, 30])
            concentration_risk_level, conc_color = risk_level(concentration_risk, [60, 80])
            
            # リスクメトリクス
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("📊 売上変動リスク", f"{sales_cv:.1f}%", delta=f"{sales_risk}リスク")
            with col2:
                st.metric("🏭 原料費変動リスク", f"{material_cv:.1f}%", delta=f"{material_risk}リスク")
            with col3:
                st.metric("🎯 顧客集中リスク", f"{concentration_risk:.1f}%", delta=f"{concentration_risk_level}リスク")
            with col4:
                st.metric("⚠️ 異常値件数", f"{anomalies['anomaly_count']}件", 
                         delta=f"{anomalies['anomaly_ratio']:.1f}%")
            with col5:
                total_risk = (sales_cv + material_cv + concentration_risk) / 3
                overall_risk, _ = risk_level(total_risk, [25, 50])
                st.metric("🔍 総合リスク", f"{total_risk:.0f}点", delta=f"{overall_risk}リスク")
            
            # リスク詳細分析
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<h3 class="section-header">🔍 リスク要因分析</h3>', unsafe_allow_html=True)
                
                # リスクレーダーチャート
                risk_categories = ['売上変動', '原料費変動', '顧客集中', '異常値頻度']
                risk_values = [sales_cv/50*100, material_cv/50*100, concentration_risk, anomalies['anomaly_ratio']*5]
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=risk_values,
                    theta=risk_categories,
                    fill='toself',
                    name='リスクレベル',
                    marker_color='rgba(239, 68, 68, 0.6)'
                ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100])
                    ),
                    title="リスク評価レーダーチャート",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                st.markdown('<h3 class="section-header">📋 リスク対策提案</h3>', unsafe_allow_html=True)
                
                # 動的リスク対策提案
                if sales_cv > 30:
                    st.markdown(f"""
                    <div class="alert-card">
                        <h4>⚠️ 売上安定化が必要</h4>
                        <p>• 新規顧客開拓でリスク分散</p>
                        <p>• 定期契約の拡大</p>
                        <p>• 季節変動の平準化</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                if concentration_risk > 80:
                    st.markdown(f"""
                    <div class="alert-card">
                        <h4>⚠️ 顧客集中リスク対策</h4>
                        <p>• 新規顧客獲得の強化</p>
                        <p>• 既存中堅顧客の育成</p>
                        <p>• 主要顧客との関係強化</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                if anomalies['anomaly_count'] > 3:
                    st.markdown(f"""
                    <div class="alert-card">
                        <h4>⚠️ 異常値頻発</h4>
                        <p>• 業務プロセスの見直し</p>
                        <p>• 予測精度の改善</p>
                        <p>• モニタリング強化</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                if sales_cv < 20 and material_cv < 20 and concentration_risk < 60:
                    st.markdown("""
                    <div class="success-card">
                        <h4>✅ 良好なリスク管理</h4>
                        <p>• 現在の運営体制を維持</p>
                        <p>• 継続的なモニタリング</p>
                        <p>• さらなる効率化の検討</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # 異常値詳細
            if anomalies['anomaly_count'] > 0:
                st.markdown('<h3 class="section-header">🔍 異常値詳細</h3>', unsafe_allow_html=True)
                
                anomaly_display = anomalies['anomaly_days'].copy()
                for col in ['総売上額', '使用金額', '利益']:
                    anomaly_display[col] = anomaly_display[col].apply(fmt_num)
                
                st.dataframe(anomaly_display, use_container_width=True, hide_index=True)
                download_csv(anomalies['anomaly_days'], f"異常値分析_{sales_year}{sales_month}.csv", "📥 異常値CSV")
                
        except Exception as e:
            st.error(f"リスク分析エラー: {str(e)}")
    else:
        st.info("📤 売上と原料の両方のExcelファイルをアップロードしてください")

# ============================================================================
# 効率性分析タブ（新機能）
# ============================================================================

with tabs[7]:
    if analyzer is not None:
        st.markdown('<h2 class="section-header">⚡ 効率性分析</h2>', unsafe_allow_html=True)
        
        try:
            efficiency_data = analyzer.analyze_efficiency_patterns()
            rankings = analyzer.get_performance_ranking()
            comparison_df = analyzer.comparison_df
            
            # 効率性メトリクス
            show_metrics({
                "🏆 最高効率スコア": f"{comparison_df['効率スコア'].max():.2f}",
                "📊 平均効率スコア": f"{comparison_df['効率スコア'].mean():.2f}",
                "🎯 効率一貫性": f"{efficiency_data['consistency_score']:.0f}点",
                "📈 改善余地": f"¥{fmt_num(efficiency_data['improvement_potential'])}",
                "⚖️ 効率安定度": f"{100-comparison_df['効率スコア'].std()*10:.0f}点"
            })
            
            # 効率性分析チャート
            col1, col2 = st.columns(2)
            
            with col1:
                # 効率スコア推移
                fig = create_unified_graph(
                    comparison_df['日付'], comparison_df['効率スコア'], 
                    "効率スコア", "#f59e0b", "日別効率スコア推移", 
                    "日付", "効率スコア", chart_type="line"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                # 効率 vs 売上散布図
                fig = px.scatter(
                    comparison_df, x='効率スコア', y='総売上額', 
                    title='効率スコア vs 売上額',
                    labels={'効率スコア': '効率スコア', '総売上額': '売上額'},
                    color='利益率', color_continuous_scale='Viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # パフォーマンスランキング
            st.markdown('<h3 class="section-header">🏆 パフォーマンスランキング</h3>', unsafe_allow_html=True)
            
            ranking_tabs = st.tabs(["🥇 売上TOP3", "💎 利益TOP3", "⚡ 効率TOP3", "📉 改善要望3"])
            
            with ranking_tabs[0]:
                top_sales = rankings["売上TOP3"]
                for i, (_, row) in enumerate(top_sales.iterrows()):
                    medal = ["🥇", "🥈", "🥉"][i]
                    st.markdown(f"""
                    <div class="success-card">
                        <h4>{medal} {row['日付']}</h4>
                        <p><strong>¥{fmt_num(row['総売上額'])}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with ranking_tabs[1]:
                top_profit = rankings["利益TOP3"]
                for i, (_, row) in enumerate(top_profit.iterrows()):
                    medal = ["🥇", "🥈", "🥉"][i]
                    st.markdown(f"""
                    <div class="success-card">
                        <h4>{medal} {row['日付']}</h4>
                        <p><strong>¥{fmt_num(row['利益'])}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with ranking_tabs[2]:
                top_efficiency = rankings["効率TOP3"]
                for i, (_, row) in enumerate(top_efficiency.iterrows()):
                    medal = ["🥇", "🥈", "🥉"][i]
                    st.markdown(f"""
                    <div class="success-card">
                        <h4>{medal} {row['日付']}</h4>
                        <p><strong>{row['効率スコア']:.2f}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with ranking_tabs[3]:
                worst_sales = rankings["売上WORST3"]
                for i, (_, row) in enumerate(worst_sales.iterrows()):
                    st.markdown(f"""
                    <div class="alert-card">
                        <h4>📉 {row['日付']}</h4>
                        <p>売上: ¥{fmt_num(row['総売上額'])}</p>
                        <p>改善余地あり</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # 効率改善提案
            st.markdown('<h3 class="section-header">💡 効率改善提案</h3>', unsafe_allow_html=True)
            
            best_day = efficiency_data['best_day']
            worst_day = efficiency_data['worst_day']
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="success-card">
                    <h4>🏆 ベストプラクティス</h4>
                    <p><strong>{best_day['日付']}</strong></p>
                    <p>効率スコア: {best_day['効率スコア']:.2f}</p>
                    <p>売上: ¥{fmt_num(best_day['総売上額'])}</p>
                    <p>原料費: ¥{fmt_num(best_day['使用金額'])}</p>
                    <p>この日のパターンを分析・再現</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                improvement_amount = (best_day['効率スコア'] - comparison_df['効率スコア'].mean()) * comparison_df['使用金額'].mean()
                st.markdown(f"""
                <div class="info-card">
                    <h4>📈 改善ポテンシャル</h4>
                    <p>全日をベスト効率に改善した場合</p>
                    <p><strong>追加利益: ¥{fmt_num(improvement_amount)}</strong></p>
                    <p>効率向上率: {((best_day['効率スコア'] / comparison_df['効率スコア'].mean()) - 1) * 100:.1f}%</p>
                    <p>実現可能な改善目標を設定</p>
                </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"効率性分析エラー: {str(e)}")
    else:
        st.info("📤 売上と原料の両方のExcelファイルをアップロードしてください")

# ============================================================================
# シミュレーション分析タブ（新機能）
# ============================================================================

with tabs[8]:
    if analyzer is not None:
        st.markdown('<h2 class="section-header">🎲 シナリオシミュレーション</h2>', unsafe_allow_html=True)
        
        try:
            simulation_data = analyzer.simulate_scenarios()
            current_profit = simulation_data['current_profit']
            
            # シミュレーション結果メトリクス
            show_metrics({
                "💰 現在利益": f"¥{fmt_num(current_profit)}",
                "🚀 最大ポテンシャル": f"¥{fmt_num(simulation_data['total_potential'])}",
                "📈 改善率": f"{(simulation_data['total_potential'] / current_profit - 1) * 100:.1f}%",
                "🎯 実現可能性": "高" if simulation_data['total_potential'] / current_profit < 1.5 else "中",
                "⚡ 効率化効果": f"¥{fmt_num(simulation_data['material_efficiency_savings'])}"
            })
            
            # シナリオ比較チャート
            scenarios = {
                '現在': current_profit,
                'ベスト効率': current_profit + simulation_data['best_efficiency_potential'],
                'ワースト改善': current_profit + simulation_data['worst_day_improvement'],
                '原料効率化': current_profit + simulation_data['material_efficiency_savings'],
                '統合最適化': current_profit + simulation_data['total_potential']
            }
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(scenarios.keys()),
                y=list(scenarios.values()),
                marker_color=['#6b7280', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6'],
                text=[f"¥{fmt_num(v)}" for v in scenarios.values()],
                textposition='auto'
            ))
            fig.update_layout(
                title="シナリオ別利益予測",
                yaxis_title="利益 (¥)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 詳細シナリオ分析
            st.markdown('<h3 class="section-header">📊 シナリオ詳細分析</h3>', unsafe_allow_html=True)
            
            scenario_tabs = st.tabs(["🚀 ベスト効率", "📈 ワースト改善", "⚙️ 原料効率化", "🎯 統合最適化"])
            
            with scenario_tabs[0]:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="info-card">
                        <h4>🚀 全日ベスト効率シナリオ</h4>
                        <p><strong>追加利益: ¥{fmt_num(simulation_data['best_efficiency_potential'])}</strong></p>
                        <p>現在比: {(simulation_data['best_efficiency_potential'] / current_profit) * 100:.1f}%向上</p>
                        <p>実現方法: 最高効率日のパターンを全日に適用</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    **実施アクション:**
                    - ベスト効率日の条件分析
                    - 成功要因の標準化
                    - 社員教育・プロセス改善
                    - KPI管理の強化
                    """)
            
            with scenario_tabs[1]:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="info-card">
                        <h4>📈 ワースト日改善シナリオ</h4>
                        <p><strong>追加利益: ¥{fmt_num(simulation_data['worst_day_improvement'])}</strong></p>
                        <p>現在比: {(simulation_data['worst_day_improvement'] / current_profit) * 100:.1f}%向上</p>
                        <p>実現方法: 低パフォーマンス日を平均レベルまで改善</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    **実施アクション:**
                    - ワースト日の要因分析
                    - 問題パターンの特定
                    - 予防策の策定
                    - 早期警告システム構築
                    """)
            
            with scenario_tabs[2]:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="info-card">
                        <h4>⚙️ 原料効率化シナリオ</h4>
                        <p><strong>コスト削減: ¥{fmt_num(simulation_data['material_efficiency_savings'])}</strong></p>
                        <p>原料費: 10%削減</p>
                        <p>実現方法: 調達最適化・使用効率向上</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    **実施アクション:**
                    - サプライヤー見直し
                    - 在庫管理最適化
                    - 使用効率の改善
                    - 代替原料の検討
                    """)
            
            with scenario_tabs[3]:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="info-card">
                        <h4>🎯 統合最適化シナリオ</h4>
                        <p><strong>総追加利益: ¥{fmt_num(simulation_data['total_potential'])}</strong></p>
                        <p>現在比: {(simulation_data['total_potential'] / current_profit) * 100:.1f}%向上</p>
                        <p>実現方法: 全シナリオの統合実施</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    improvement_ratio = simulation_data['total_potential'] / current_profit
                    difficulty = "高" if improvement_ratio > 2 else "中" if improvement_ratio > 1.5 else "低"
                    timeline = "12-18ヶ月" if improvement_ratio > 2 else "6-12ヶ月" if improvement_ratio > 1.5 else "3-6ヶ月"
                    
                    st.markdown(f"""
                    **実現性評価:**
                    - 難易度: {difficulty}
                    - 想定期間: {timeline}
                    - 投資対効果: 高
                    - リスク: 中程度
                    
                    **推奨アプローチ:**
                    段階的実施（効果の高い順番）
                    """)
            
            # 実施優先度マトリックス
            st.markdown('<h3 class="section-header">📋 実施優先度マトリックス</h3>', unsafe_allow_html=True)
            
            priority_data = {
                'シナリオ': ['ワースト改善', '原料効率化', 'ベスト効率', '統合最適化'],
                '効果': ['中', '中', '高', '超高'],
                '実現難易度': ['低', '中', '中', '高'],
                '期間': ['1-3ヶ月', '3-6ヶ月', '6-12ヶ月', '12-18ヶ月'],
                '推奨優先度': ['1位', '2位', '3位', '4位']
            }
            
            priority_df = pd.DataFrame(priority_data)
            st.dataframe(priority_df, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"シミュレーション分析エラー: {str(e)}")
    else:
        st.info("📤 売上と原料の両方のExcelファイルをアップロードしてください")

# ============================================================================
# 月内最適化分析タブ（新機能）
# ============================================================================

with tabs[9]:
    if analyzer is not None:
        st.markdown('<h2 class="section-header">🔍 月内最適化分析</h2>', unsafe_allow_html=True)
        
        try:
            comparison_df = analyzer.comparison_df
            anomalies = analyzer.detect_anomalies()
            efficiency_data = analyzer.analyze_efficiency_patterns()
            
            # 最適化指標
            show_metrics({
                "🎯 最適化度": f"{(comparison_df['効率スコア'] > comparison_df['効率スコア'].mean()).sum()}/{len(comparison_df)}日",
                "📊 安定性指数": f"{efficiency_data['consistency_score']:.0f}点",
                "⚡ 改善機会": f"{(comparison_df['効率スコア'] < comparison_df['効率スコア'].median()).sum()}日",
                "🔍 異常検出": f"{anomalies['anomaly_count']}件",
                "💡 最適化余地": f"¥{fmt_num(efficiency_data['improvement_potential'])}"
            })
            
            # クラスター分析（日別パフォーマンス分類）
            features = comparison_df[['総売上額', '使用金額', '利益', '効率スコア']].fillna(0)
            
            # データ数が少ない場合の対処
            if len(features) < 3:
                comparison_df['クラスター'] = 1  # 全て標準パフォーマンスとする
                comparison_df['パフォーマンス分類'] = '標準パフォーマンス'
            else:
                try:
                    scaler = StandardScaler()
                    features_scaled = scaler.fit_transform(features)
                    
                    kmeans = KMeans(n_clusters=min(3, len(features)), random_state=42)
                    clusters = kmeans.fit_predict(features_scaled)
                    comparison_df['クラスター'] = clusters
                    
                    # クラスター数に応じた分類名
                    if len(np.unique(clusters)) == 3:
                        cluster_names = {0: '低パフォーマンス', 1: '標準パフォーマンス', 2: '高パフォーマンス'}
                    else:
                        cluster_names = {i: f'グループ{i+1}' for i in range(len(np.unique(clusters)))}
                    
                    comparison_df['パフォーマンス分類'] = comparison_df['クラスター'].map(cluster_names)
                except Exception as e:
                    # クラスター分析に失敗した場合
                    comparison_df['クラスター'] = 1
                    comparison_df['パフォーマンス分類'] = '標準パフォーマンス'
                    st.warning("クラスター分析でエラーが発生しました。標準分類で表示します。")
            
            # クラスター可視化
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.scatter(
                    comparison_df, x='総売上額', y='効率スコア',
                    color='パフォーマンス分類',
                    title='日別パフォーマンス分類',
                    labels={'総売上額': '売上額', '効率スコア': '効率スコア'},
                    color_discrete_map={
                        '低パフォーマンス': '#ef4444',
                        '標準パフォーマンス': '#f59e0b', 
                        '高パフォーマンス': '#10b981'
                    }
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # パフォーマンス分布
                cluster_counts = comparison_df['パフォーマンス分類'].value_counts()
                fig = px.pie(
                    values=cluster_counts.values,
                    names=cluster_counts.index,
                    title='パフォーマンス分布',
                    color_discrete_map={
                        '低パフォーマンス': '#ef4444',
                        '標準パフォーマンス': '#f59e0b', 
                        '高パフォーマンス': '#10b981'
                    }
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # パフォーマンス分析詳細
            st.markdown('<h3 class="section-header">📊 パフォーマンス分類別分析</h3>', unsafe_allow_html=True)
            
            cluster_analysis = comparison_df.groupby('パフォーマンス分類').agg({
                '総売上額': ['mean', 'sum', 'count'],
                '使用金額': 'mean',
                '利益': 'mean',
                '効率スコア': 'mean'
            }).round(2)
            cluster_analysis.columns = ['平均売上', '合計売上', '日数', '平均原料費', '平均利益', '平均効率']
            
            # 数値にカンマを追加
            display_cluster = cluster_analysis.copy()
            for col in ['平均売上', '合計売上', '平均原料費', '平均利益']:
                display_cluster[col] = display_cluster[col].apply(fmt_num)
            display_cluster['平均効率'] = display_cluster['平均効率'].apply(fmt_dec)
            
            st.dataframe(display_cluster, use_container_width=True)
            
            # 改善アクションプラン
            st.markdown('<h3 class="section-header">💡 月内改善アクションプラン</h3>', unsafe_allow_html=True)
            
            action_tabs = st.tabs(["🎯 即効改善", "📈 中期改善", "🚀 長期最適化"])
            
            with action_tabs[0]:
                low_perf_days = comparison_df[comparison_df['パフォーマンス分類'] == '低パフォーマンス']
                if len(low_perf_days) > 0:
                    st.markdown(f"""
                    <div class="alert-card">
                        <h4>🎯 即効改善対象: {len(low_perf_days)}日</h4>
                        <p>平均効率スコア: {low_perf_days['効率スコア'].mean():.2f}</p>
                        <p>改善ポテンシャル: ¥{fmt_num((comparison_df['効率スコア'].median() - low_perf_days['効率スコア'].mean()) * low_perf_days['使用金額'].mean() * len(low_perf_days))}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    **即効アクション（1-2週間）:**
                    - 低パフォーマンス日の要因分析
                    - 高パフォーマンス日との比較
                    - すぐに実行可能な改善策の実施
                    - 日次チェックリストの作成
                    """)
                else:
                    st.success("✅ 即効改善が必要な低パフォーマンス日はありません")
            
            with action_tabs[1]:
                std_perf_days = comparison_df[comparison_df['パフォーマンス分類'] == '標準パフォーマンス']
                if len(std_perf_days) > 0:
                    st.markdown(f"""
                    <div class="info-card">
                        <h4>📈 中期改善対象: {len(std_perf_days)}日</h4>
                        <p>平均効率スコア: {std_perf_days['効率スコア'].mean():.2f}</p>
                        <p>高パフォーマンスまでの差: {comparison_df[comparison_df['パフォーマンス分類'] == '高パフォーマンス']['効率スコア'].mean() - std_perf_days['効率スコア'].mean():.2f}ポイント</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    **中期アクション（1-3ヶ月）:**
                    - 標準日の効率向上施策
                    - ベストプラクティスの横展開
                    - プロセス標準化の推進
                    - 継続的改善活動の実施
                    """)
            
            with action_tabs[2]:
                high_perf_days = comparison_df[comparison_df['パフォーマンス分類'] == '高パフォーマンス']
                if len(high_perf_days) > 0:
                    st.markdown(f"""
                    <div class="success-card">
                        <h4>🚀 最適化対象: {len(high_perf_days)}日</h4>
                        <p>平均効率スコア: {high_perf_days['効率スコア'].mean():.2f}</p>
                        <p>この水準を全日に拡大することが目標</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    **長期最適化（3-12ヶ月）:**
                    - 高パフォーマンスパターンの分析・標準化
                    - 全社的な効率化プログラム
                    - 技術投資・システム改善
                    - 人材育成・スキル向上
                    """)
            
            # 詳細データエクスポート
            st.markdown('<h3 class="section-header">📋 詳細分析データ</h3>', unsafe_allow_html=True)
            
            detailed_analysis = comparison_df[['日付', '総売上額', '使用金額', '利益', '効率スコア', 'パフォーマンス分類']].copy()
            detailed_analysis['改善優先度'] = detailed_analysis['パフォーマンス分類'].map({
                '低パフォーマンス': '高',
                '標準パフォーマンス': '中',
                '高パフォーマンス': '維持'
            })
            
            for col in ['総売上額', '使用金額', '利益']:
                detailed_analysis[col] = detailed_analysis[col].apply(fmt_num)
            detailed_analysis['効率スコア'] = detailed_analysis['効率スコア'].apply(fmt_dec)
            
            st.dataframe(detailed_analysis, use_container_width=True, hide_index=True)
            download_csv(detailed_analysis, f"月内最適化分析_{sales_year}{sales_month}.csv", "📥 最適化分析CSV")
            
        except Exception as e:
            st.error(f"月内最適化分析エラー: {str(e)}")
    else:
        st.info("📤 売上と原料の両方のExcelファイルをアップロードしてください")

# ============================================================================
# End of Dashboard
# ============================================================================

if __name__ == "__main__":
    pass