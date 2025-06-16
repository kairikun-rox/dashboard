import pandas as pd
import numpy as np
import scipy.stats as stats


def calculate_zscore(data):
    """Z-score calculation without scipy dependency"""
    if isinstance(data, pd.Series):
        return (data - data.mean()) / data.std()
    return (data - np.mean(data)) / np.std(data)


def simple_kmeans(X, n_clusters=3, max_iters=100, random_seed=42):
    """Lightweight K-means implementation"""
    np.random.seed(random_seed)
    n_samples, _ = X.shape
    centroids = X[np.random.choice(n_samples, n_clusters, replace=False)]

    for _ in range(max_iters):
        distances = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels


def standardize_data(data):
    """Standardize data similar to sklearn StandardScaler"""
    if isinstance(data, pd.DataFrame):
        return (data - data.mean()) / data.std()
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)


def calc_cv(data):
    """Coefficient of variation"""
    return (data.std() / data.mean()) * 100 if len(data) > 0 and data.mean() > 0 else 0


def calc_ma(df, col, window=7):
    """Moving average"""
    return df[col].rolling(window=window, min_periods=1).mean()


class BusinessAnalyzer:
    """Business analytics helper class"""

    def __init__(self, sales_df, daily_sales, mat_df, daily_mat):
        self.sales_df = sales_df
        self.daily_sales = daily_sales
        self.mat_df = mat_df
        self.daily_mat = daily_mat
        self.comparison_df = self._create_comparison_df()

    def _create_comparison_df(self):
        if self.daily_sales is None or self.daily_mat is None:
            return None
        comparison = pd.merge(
            self.daily_sales[["日", "日付", "総売上額"]],
            self.daily_mat[["日", "使用金額"]],
            on="日",
            how="outer",
        ).fillna(0)
        comparison["利益"] = comparison["総売上額"] - comparison["使用金額"]
        comparison["利益率"] = np.where(
            comparison["総売上額"] > 0,
            (comparison["利益"] / comparison["総売上額"] * 100),
            0,
        ).round(2)
        comparison["原料費率"] = np.where(
            comparison["総売上額"] > 0,
            (comparison["使用金額"] / comparison["総売上額"] * 100),
            0,
        ).round(2)
        comparison["効率スコア"] = np.where(
            comparison["使用金額"] > 0,
            comparison["総売上額"] / comparison["使用金額"],
            0,
        ).round(2)
        return comparison

    def get_summary_metrics(self):
        if self.comparison_df is None:
            return {}
        total_sales = self.comparison_df["総売上額"].sum()
        total_material = self.comparison_df["使用金額"].sum()
        total_profit = self.comparison_df["利益"].sum()
        return {
            "total_sales": total_sales,
            "total_material": total_material,
            "total_profit": total_profit,
            "overall_profit_rate": (total_profit / total_sales * 100) if total_sales > 0 else 0,
            "overall_material_rate": (total_material / total_sales * 100) if total_sales > 0 else 0,
            "avg_daily_sales": self.comparison_df["総売上額"].mean(),
            "sales_cv": calc_cv(self.comparison_df["総売上額"]),
            "material_cv": calc_cv(self.comparison_df["使用金額"]),
        }

    def get_performance_ranking(self):
        if self.comparison_df is None:
            return {}
        df = self.comparison_df.copy()
        rankings = {
            "売上TOP3": df.nlargest(3, "総売上額")[["日付", "総売上額"]],
            "利益TOP3": df.nlargest(3, "利益")[["日付", "利益"]],
            "効率TOP3": df.nlargest(3, "効率スコア")[["日付", "効率スコア"]],
            "売上WORST3": df.nsmallest(3, "総売上額")[["日付", "総売上額"]],
            "利益WORST3": df.nsmallest(3, "利益")[["日付", "利益"]],
        }
        return rankings

    def analyze_efficiency_patterns(self):
        if self.comparison_df is None:
            return {}
        df = self.comparison_df.copy()
        best_day = df.loc[df["効率スコア"].idxmax()]
        worst_day = df.loc[df["効率スコア"].idxmin()]
        median_efficiency = df["効率スコア"].median()
        below_median = df[df["効率スコア"] < median_efficiency]
        improvement_potential = (median_efficiency - below_median["効率スコア"]).sum() * below_median["使用金額"].sum()
        return {
            "best_day": best_day,
            "worst_day": worst_day,
            "improvement_potential": improvement_potential,
            "efficiency_std": df["効率スコア"].std(),
            "consistency_score": 100 - calc_cv(df["効率スコア"]),
        }

    def customer_analysis(self):
        if self.sales_df is None:
            return {}
        customer_stats = self.sales_df.groupby("得意先名").agg({
            "総売上額": ["sum", "mean", "count"],
            "日": "nunique",
        }).round(2)
        customer_stats.columns = ["総売上", "平均取引額", "取引回数", "取引日数"]
        customer_stats = customer_stats.reset_index().sort_values("総売上", ascending=False)
        customer_stats["累積売上"] = customer_stats["総売上"].cumsum()
        customer_stats["累積比率"] = customer_stats["累積売上"] / customer_stats["総売上"].sum() * 100
        customer_stats["分類"] = np.select([
            customer_stats["累積比率"] <= 80,
            customer_stats["累積比率"] <= 95,
        ], ["A", "B"], default="C")
        customer_stats["顧客価値スコア"] = (
            customer_stats["総売上"] * 0.4 +
            customer_stats["平均取引額"] * 0.3 +
            customer_stats["取引日数"] * 0.3
        )
        return {
            "customer_stats": customer_stats,
            "top5_concentration": customer_stats.head(5)["総売上"].sum() / customer_stats["総売上"].sum() * 100,
            "active_customers": len(customer_stats),
            "avg_customer_value": customer_stats["総売上"].mean(),
        }

    def detect_anomalies(self):
        if self.comparison_df is None:
            return {}
        df = self.comparison_df.copy()
        for col in ["総売上額", "使用金額", "利益"]:
            z_scores = np.abs(stats.zscore(df[col]))
            df[f"{col}_異常"] = z_scores > 2
        anomalies = df[df["総売上額_異常"] | df["使用金額_異常"] | df["利益_異常"]]
        return {
            "anomaly_days": anomalies[["日付", "総売上額", "使用金額", "利益"]],
            "anomaly_count": len(anomalies),
            "anomaly_ratio": len(anomalies) / len(df) * 100,
        }

    def simulate_scenarios(self):
        if self.comparison_df is None:
            return {}
        df = self.comparison_df.copy()
        current_total = df["利益"].sum()
        best_efficiency = df["効率スコア"].max()
        scenario1_profit = (df["使用金額"] * best_efficiency - df["使用金額"]).sum()
        median_sales = df["総売上額"].median()
        worst_days = df[df["総売上額"] < median_sales]
        scenario2_improvement = (median_sales - worst_days["総売上額"]).sum()
        scenario3_savings = df["使用金額"].sum() * 0.1
        return {
            "current_profit": current_total,
            "best_efficiency_potential": scenario1_profit,
            "worst_day_improvement": scenario2_improvement,
            "material_efficiency_savings": scenario3_savings,
            "total_potential": scenario1_profit + scenario2_improvement + scenario3_savings,
        }


def process_data(file_content, file_name, data_type):
    """Unified data loading logic"""
    import io
    xlsx_file = io.BytesIO(file_content)
    base_ym = file_name[:6]
    year, month = base_ym[:4], base_ym[4:]
    frames = []
    for sheet in [f"{i:02}" for i in range(1, 32)]:
        try:
            if data_type == "sales":
                raw = pd.read_excel(xlsx_file, sheet_name=sheet, header=4)
                if {"得意先コード", "得意先名", "総売上額"}.issubset(raw.columns):
                    df = raw[["得意先コード", "得意先名", "総売上額"]].copy()
                    df["日付"] = f"{year}/{month}/{int(sheet):02d}"
                    df["日"] = int(sheet)
                    df["総売上額"] = pd.to_numeric(df["総売上額"], errors="coerce")
                    df = df.dropna(subset=["総売上額"])
                    mask = (
                        df["得意先名"].astype(str).str.contains("合計", na=False) |
                        df["得意先コード"].astype(str).str.contains("<<|合計", na=False)
                    )
                    df = df[~mask]
                    if not df.empty:
                        frames.append(df)
            else:
                raw = pd.read_excel(xlsx_file, sheet_name=sheet, header=2)
                if {"日付", "船名", "kg/cs", "cs", "¥/kg", "総額"}.issubset(raw.columns):
                    df = raw[["日付", "船名", "kg/cs", "cs", "¥/kg", "総額"]].copy()
                    df["日付"] = pd.to_datetime(df["日付"], errors="coerce")
                    df["日"] = df["日付"].dt.day
                    df = df.dropna(subset=["日"])
                    df["総額"] = pd.to_numeric(df["総額"], errors="coerce")
                    if not df.empty:
                        frames.append(df)
        except Exception:
            continue
    if not frames:
        return None, None, year, month
    combined_df = pd.concat(frames, ignore_index=True)
    if data_type == "sales":
        daily_df = combined_df.groupby(["日", "日付"])["総売上額"].sum().reset_index()
        daily_df = daily_df.sort_values("日")
        daily_df["累計売上"] = daily_df["総売上額"].cumsum()
        daily_df["移動平均_7日"] = calc_ma(daily_df, "総売上額", 7)
        daily_df["日付_dt"] = pd.to_datetime(daily_df["日付"])
        daily_df["曜日_jp"] = daily_df["日付_dt"].dt.strftime('%A').map({
            'Monday': '月', 'Tuesday': '火', 'Wednesday': '水', 'Thursday': '木',
            'Friday': '金', 'Saturday': '土', 'Sunday': '日',
        })
    else:
        daily_df = combined_df.groupby("日")["総額"].sum().reset_index(name="使用金額")
        daily_df["日付"] = daily_df["日"].apply(lambda d: f"{year}/{month}/{d:02d}")
        daily_df = daily_df.sort_values("日")
        daily_df["累計金額"] = daily_df["使用金額"].cumsum()
        daily_df["移動平均_7日"] = calc_ma(daily_df, "使用金額", 7)
    return combined_df, daily_df, year, month
