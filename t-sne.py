import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI環境がない場合のエラー回避
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語フォント対応
import seaborn as sns
import plotly.express as px
import gower
import prince
import os
import re
import warnings
from scipy.stats import entropy
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder

# 警告の抑制 (FutureWarningなど)
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 設定・定数定義 (CONFIGURATION)
# ==============================================================================

INPUT_FILE = 'motodata_1_updated.csv'
SITE_COLUMN_NAME = '場所'

# 複数選択項目（ワンホット展開用）
MULTI_VALUE_FEATURES = [
    '口縁部_技法_沈線_特徴', '口縁部_技法_磨消縄文_沈線', 
    '胴部_技法_磨消縄文_沈線', '胴部_技法_沈線_特徴'
]
MULTI_VALUE_SEPARATOR = ', '

# 特徴量セットの定義
FEATURES_HABAHIRO = [
    '口縁部_技法_沈線_特徴', '口縁部_形状', '口縁部_主文様同士が並行/対向', '口縁部_状態_退化', 
    '口縁部_文様が開放/閉鎖', '口縁部_文様方向', '口縁部_変形', '口縁部_直下', '頸部の傾向', '胴部_新_主モチーフ',
    '胴部_技法_沈線_特徴', '胴部_技法_磨消縄文_施文順序', '胴部_技法_磨消縄文_図地', '胴部_技法_磨消縄文_沈線', '胴部_技法_磨消縄文_縄文'
]
FEATURES_HOKEI_MAKESHI = [
    '口縁部_技法_磨消縄文_施文順序','口縁部_技法_磨消縄文_図地','口縁部_技法_磨消縄文_沈線',
    '口縁部_技法_磨消縄文_縄文','口縁部_形状','口縁部_主文様同士が並行/対向','口縁部_状態_退化',
    '口縁部_文様が開放/閉鎖','口縁部_文様方向','口縁部_変形','口縁部_直下','頸部の傾向', '胴部_新_主モチーフ',
    '胴部_技法_沈線_特徴', '胴部_技法_磨消縄文_施文順序', '胴部_技法_磨消縄文_図地', '胴部_技法_磨消縄文_沈線', '胴部_技法_磨消縄文_縄文'
]
FEATURES_MUMON_MAKESHI = [
    '口縁部_形状', '口縁部_直下', '胴部_技法_磨消縄文_施文順序', '胴部_技法_磨消縄文_図地', 
    '胴部_技法_磨消縄文_沈線', '胴部_技法_磨消縄文_縄文', '胴部_新_主モチーフ'
]
FEATURES_MUMON_HABAHIRO = ['口縁部_形状','口縁部_直下','胴部_技法_幅広の沈線']

# 統合特徴量リスト
FEATURES_MUMON_UNION = list(set(FEATURES_MUMON_MAKESHI + FEATURES_MUMON_HABAHIRO))
FEATURES_UNION = list(set(FEATURES_HOKEI_MAKESHI + FEATURES_HABAHIRO))
FEATURES_FINAL_UNION = list(set(FEATURES_HOKEI_MAKESHI + FEATURES_HABAHIRO + FEATURES_MUMON_MAKESHI + FEATURES_MUMON_HABAHIRO))

# 分析シナリオ定義
ANALYSIS_SCENARIOS = {
    'Hokei_Makeshi_or_Habahiro': {
        'filter_name': 'Hokei_Makeshi_or_Habahiro_UNION',
        'query_list': [
            ("`口縁部_新_主モチーフ` == '区画文_方形・窓枠状'", "主モチーフ: '区画文_方形・窓枠状'"),
            ("`口縁部_技法_分類` in ['磨消縄文', '幅広の沈線']", "技法分類: '磨消縄文' or '幅広の沈線'")
        ],
        'custom_features': FEATURES_UNION, 
        'parts_filter_mode': 'kouen',
        'target_sites': [r'Matsu', r'Koigakubo'], 
        'clusters_settings': {
            'gower_uniform': {20: 6, 30: 5, 40: 5, 50: 5},
            'gower_weighted': {20: 6, 30: 4, 40: 6, 50: 5},
            'mca_mode': {20: 5, 30: 3, 40: 3, 50: 3},
            'mca_predictive': {20: 3, 30: 4, 40: 4, 50: 4},
        },
        'mca_n_components': 33,
        'imputer_max_iter': 20,
        'plot_title_template': "松風台・恋ヶ窪東遺跡 - 口縁部方形・窓枠状({name_base}, P={p}, k={k})" 
    },
    'Mumon_Makeshi_or_Habahiro': {
        'filter_name': 'Mumon_Makeshi_or_Habahiro_Doka_UNION',
        'query_list': [
            ("`口縁部_新_主モチーフ` == '無文'", "主モチーフ: '無文'"),
            ("`胴部_技法_分類` in ['磨消縄文', '幅広の沈線']", "技法分類: '磨消縄文' or '幅広の沈線'"),
            ("`頸部の傾向` == '同化'", "頸部の傾向: '同化'")
        ],
        'custom_features': FEATURES_MUMON_UNION, 
        'parts_filter_mode': 'kouen',
        'target_sites': [r'Matsu', r'Koigakubo'], 
        'clusters_settings': {
            'gower_uniform': {20: 3, 30: 3, 40: 3, 50: 4},
            'gower_weighted': {20: 4, 30: 4, 40: 4, 50: 3},
            'mca_mode': {20: 6, 30: 5, 40: 6, 50: 3},
            'mca_predictive': {20: 6, 30: 7, 40: 5, 50: 7},
        },
        'mca_n_components': 20,
        'imputer_max_iter': 20,
        'plot_title_template': "松風台・恋ヶ窪東遺跡 - 口縁部無文・頸部同化({name_base}, P={p}, k={k})" 
    },
    'FINAL_UNION_Hokei_vs_Mumon': {
        'filter_name': 'FINAL_UNION_Hokei_vs_Mumon',
        'query_list': [
            (
                "(`口縁部_新_主モチーフ` == '区画文_方形・窓枠状') or "
                "(`口縁部_新_主モチーフ` == '無文' and `頸部の傾向` == '同化')",
                "最終統合: ('方形・窓枠状') または ('無文' かつ '頸部同化')"
            )
        ],
        'custom_features': FEATURES_FINAL_UNION, 
        'parts_filter_mode': 'kouen',
        'target_sites': [r'Matsu', r'Koigakubo'], 
        'clusters_settings': {
            'gower_uniform': {20: 7, 30: 7, 40: 7, 50: 6},
            'gower_weighted': {20: 7, 30: 6, 40: 6, 50: 5},
            'mca_mode': {20: 3, 30: 3, 40: 4, 50: 3},
            'mca_predictive': {20: 3, 30: 4, 40: 3, 50: 3},
        },
        'mca_n_components': 40,
        'imputer_max_iter': 50,
        'plot_title_template': "松風台・恋ヶ窪東遺跡 - 統合({name_base}, P={p}, k={k})" 
    }
}

# 分析アプローチの定義
ANALYSIS_APPROACHES = [
    {'name': 'Gower_Weighted', 'type': 'gower', 'weight_strategy': 'entropy', 'cluster_key': 'gower_weighted'},
    {'name': 'Gower_Uniform', 'type': 'gower', 'weight_strategy': 'uniform', 'cluster_key': 'gower_uniform'},
    {'name': 'MCA_FAMD_Mode', 'type': 'mca', 'impute_strategy': 'mode', 'cluster_key': 'mca_mode'},
    {'name': 'MCA_FAMD_Predictive', 'type': 'mca', 'impute_strategy': 'predictive', 'cluster_key': 'mca_predictive'}
]

# ==============================================================================
# 2. データ前処理・加工関数
# ==============================================================================

def preprocess_multivalue_features(df, features_list):
    """複数選択項目をOne-Hotエンコーディングする"""
    df_processed = df.copy()
    encoded_cols = []
    
    # 分析対象の特徴量に含まれる複数選択項目のみ処理
    target_multivalue = [f for f in features_list if f in MULTI_VALUE_FEATURES and f in df.columns]
    
    # それ以外の単一選択カテゴリカル変数
    categorical_cols = [f for f in features_list if f not in MULTI_VALUE_FEATURES and f in df.columns]

    for feature in target_multivalue:
        # 欠損値を一時的な文字列に置換して処理
        temp_unknown = '___UNKNOWN___'
        series = df_processed[feature].fillna(temp_unknown).replace('不明', temp_unknown)
        
        dummies = series.str.get_dummies(sep=MULTI_VALUE_SEPARATOR)
        dummies.columns = [f"{feature}_{col}" for col in dummies.columns]
        
        # 一時的な欠損値カラムは削除
        if f"{feature}_{temp_unknown}" in dummies.columns:
            dummies.drop(columns=[f"{feature}_{temp_unknown}"], inplace=True)
            
        # 重複カラムの削除
        duplicates = [c for c in dummies.columns if c in df_processed.columns]
        if duplicates:
            df_processed.drop(columns=duplicates, inplace=True)
            
        df_processed = pd.concat([df_processed, dummies], axis=1)
        encoded_cols.extend(dummies.columns)
        
    # 元の複数選択列を削除
    df_processed.drop(columns=target_multivalue, inplace=True, errors='ignore')
    
    # エンコード済み列と重複するカテゴリカル列を除外
    categorical_cols = [c for c in categorical_cols if c not in encoded_cols]
    
    final_features = categorical_cols + encoded_cols
    return df_processed, final_features, categorical_cols, encoded_cols

def calculate_gower_weights(data, strategy='uniform'):
    """Gower距離用の重みを計算する（エントロピー or 均等）"""
    if strategy == 'uniform':
        return np.ones(len(data.columns)) / len(data.columns)
    
    # Entropy strategy
    weights = []
    for col in data.columns:
        counts = data[col].value_counts(normalize=True, dropna=True)
        if len(counts) > 1:
            weights.append(entropy(counts.values))
        else:
            weights.append(0)
            
    weights = np.array(weights)
    if np.sum(weights) > 0:
        return weights / np.sum(weights)
    else:
        return np.ones(len(data.columns)) / len(data.columns)

def impute_missing_data(data, categorical_cols, encoded_cols, strategy='mode', max_iter=20):
    """MCA用の欠損値補完を行う"""
    data_imputed = data.copy()
    data_imputed.replace('不明', np.nan, inplace=True)

    if strategy == 'mode':
        for col in data_imputed.columns:
            mode_val = data_imputed[col].mode()
            if not mode_val.empty:
                data_imputed[col] = data_imputed[col].fillna(mode_val[0])
            else:
                fill_val = 'N/A' if col in categorical_cols else 0
                data_imputed[col] = data_imputed[col].fillna(fill_val)
        return data_imputed

    elif strategy == 'predictive':
        # カテゴリカル変数の処理
        cat_data = data_imputed[categorical_cols]
        if not cat_data.empty:
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            encoded_cat = encoder.fit_transform(cat_data)
            
            imputer = IterativeImputer(max_iter=max_iter, random_state=0)
            imputed_encoded = imputer.fit_transform(encoded_cat)
            
            # 予測値を最寄りの整数に丸め、有効範囲内にクリップ
            imputed_encoded = np.round(imputed_encoded)
            for i in range(imputed_encoded.shape[1]):
                max_val = len(encoder.categories_[i]) - 1
                imputed_encoded[:, i] = np.clip(imputed_encoded[:, i], 0, max_val)
                
            imputed_cat_df = pd.DataFrame(
                encoder.inverse_transform(imputed_encoded),
                columns=categorical_cols,
                index=data.index
            )
        else:
            imputed_cat_df = pd.DataFrame(index=data.index)
            
        # Booleanデータ（One-Hot）は0/1埋め
        bool_data = data_imputed[encoded_cols].fillna(0)
        
        return pd.concat([imputed_cat_df, bool_data], axis=1)

# ==============================================================================
# 3. 可視化・出力関数
# ==============================================================================

def save_scree_plot(mca_model, output_dir, filename_suffix):
    """MCAのスクリープロット（累積寄与率）を保存"""
    try:
        if hasattr(mca_model, 'eigenvalues_'):
            explained = mca_model.eigenvalues_ / mca_model.total_inertia_
        elif hasattr(mca_model, 'explained_inertia_'):
            explained = mca_model.explained_inertia_
        else:
            return

        cumulative = np.cumsum(explained)
        n_comps = len(explained)
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, n_comps + 1), explained, alpha=0.6, label='寄与率')
        plt.plot(range(1, n_comps + 1), cumulative, 'r-o', label='累積寄与率')
        plt.axhline(y=0.8, color='c', linestyle='--', label='80%')
        plt.axhline(y=0.9, color='b', linestyle='--', label='90%')
        plt.xlabel('主成分')
        plt.ylabel('寄与率')
        plt.title(f'MCA スクリープロット ({filename_suffix})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"Scree_{filename_suffix}.png"))
        plt.close()
    except Exception as e:
        print(f"Scree plot error: {e}")

def save_cluster_estimation_plot(tsne_data, output_dir, filename_suffix):
    """エルボー法とシルエット分析のプロットを保存"""
    k_range = range(2, 11)
    sse, silhouette = [], []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(tsne_data)
        sse.append(kmeans.inertia_)
        if len(set(kmeans.labels_)) > 1:
            silhouette.append(silhouette_score(tsne_data, kmeans.labels_))
        else:
            silhouette.append(-1)
            
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(k_range, sse, 'bo-')
    ax1.set_title('エルボー法 (SSE)')
    ax1.grid(True)
    
    ax2.plot(k_range, silhouette, 'ro-')
    ax2.set_title('シルエット係数')
    ax2.grid(True)
    
    plt.suptitle(f"クラスター数推定 ({filename_suffix})")
    plt.savefig(os.path.join(output_dir, f"Estimation_{filename_suffix}.png"))
    plt.close()

def assign_plot_markers(df, scenario_name):
    """プロット用のマーカー形状とラベルを設定する"""
    df_plot = df.copy()
    
    # 胴部の有無
    has_doubu = ~df_plot['胴部'].isin(['なし', '不明', np.nan]) if '胴部' in df_plot.columns else False
    
    if scenario_name == 'FINAL_UNION_Hokei_vs_Mumon':
        # シナリオ3: 方形 vs 無文の統合
        is_hokei = df_plot['口縁部_新_主モチーフ'] == '区画文_方形・窓枠状'
        is_mumon = (df_plot['口縁部_新_主モチーフ'] == '無文') & (df_plot['頸部の傾向'] == '同化')
        
        df_plot['group'] = '不明'
        df_plot.loc[is_hokei, 'group'] = '方形'
        df_plot.loc[is_mumon, 'group'] = '無文'
        
        def get_label(row):
            g, d = row['group'], row['has_doubu']
            if g == '方形': return '方形 (胴部あり)' if d else '方形 (胴部なし)'
            if g == '無文': return '無文 (胴部あり)' if d else '無文 (胴部なし)'
            return 'その他'
            
        df_plot['has_doubu'] = has_doubu
        df_plot['marker_label'] = df_plot.apply(get_label, axis=1)
        
        symbol_map = {
            '方形 (胴部あり)': 'circle', '方形 (胴部なし)': 'circle-open',
            '無文 (胴部あり)': 'diamond', '無文 (胴部なし)': 'diamond-open',
            'その他': 'cross'
        }
        category_order = ['方形 (胴部あり)', '方形 (胴部なし)', '無文 (胴部あり)', '無文 (胴部なし)', 'その他']
        
    else:
        # その他のシナリオ
        base_label = '方形' if 'Hokei' in scenario_name else '無文'
        symbol_closed = 'circle' if 'Hokei' in scenario_name else 'diamond'
        symbol_open = 'circle-open' if 'Hokei' in scenario_name else 'diamond-open'
        
        df_plot['has_doubu'] = has_doubu
        df_plot['marker_label'] = np.where(has_doubu, f'{base_label} (胴部あり)', f'{base_label} (胴部なし)')
        
        symbol_map = {
            f'{base_label} (胴部あり)': symbol_closed,
            f'{base_label} (胴部なし)': symbol_open
        }
        category_order = [f'{base_label} (胴部あり)', f'{base_label} (胴部なし)']

    return df_plot, symbol_map, category_order

def create_cluster_profile_excel(data_analyzed, df_result, cluster_col, n_clusters, output_path, cat_cols, bool_cols):
    """Excelレポートの作成（詳細プロファイル、構成比、サマリー）"""
    print(f"   -> レポート作成中: {output_path}")
    
    target_mask = df_result['is_target']
    if not target_mask.any():
        return

    data_target = data_analyzed.loc[target_mask].copy()
    data_target['cluster'] = df_result.loc[target_mask, cluster_col]
    
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        # Sheet 1: Cluster Profiles
        for i in range(n_clusters):
            cluster_data = data_target[data_target['cluster'] == i]
            if cluster_data.empty: continue
            
            sheet_name = f'Cluster_{i}_Profile'
            start_row = 1
            
            # Categorical Profile
            cat_list = []
            for col in cat_cols:
                vc = cluster_data[col].value_counts(normalize=True).mul(100).reset_index()
                vc.columns = ['Category', 'Percentage']
                vc['Feature'] = col
                cat_list.append(vc)
            
            if cat_list:
                pd.concat(cat_list)[['Feature', 'Category', 'Percentage']].to_excel(
                    writer, sheet_name=sheet_name, index=False, startrow=start_row
                )
                
            # Boolean Profile (Top traits)
            if bool_cols:
                bool_stats = cluster_data[bool_cols].mean().mul(100).round(1)
                bool_stats = bool_stats[bool_stats > 0].sort_values(ascending=False).reset_index()
                bool_stats.columns = ['Technique', 'Prevalence(%)']
                bool_stats.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row, startcol=4)

        # Sheet 2: Composition
        comp = data_target['cluster'].value_counts().reset_index()
        comp.columns = ['Cluster', 'Count']
        comp['Percentage'] = (comp['Count'] / comp['Count'].sum() * 100).round(1)
        comp.to_excel(writer, sheet_name='Summary_Composition', index=False)

# ==============================================================================
# 4. メイン分析ループ
# ==============================================================================

def main():
    print("=== Archaeological Cluster Analysis Started ===")
    
    # データ読み込み
    try:
        df_all = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: File {INPUT_FILE} not found.")
        return

    # 1. シナリオごとのループ
    for scenario_name, settings in ANALYSIS_SCENARIOS.items():
        print(f"\n{'#'*60}\n# Scenario: {scenario_name}\n{'#'*60}")
        
        # --- ディレクトリ作成 ---
        sites_str = "_".join([s.replace('\\', '') for s in settings['target_sites']])
        output_dir = f"Result_{settings['filter_name']}_{sites_str}"
        os.makedirs(output_dir, exist_ok=True)
        
        # --- データフィルタリング ---
        query_parts = [f"({q})" for q, _ in settings['query_list']]
        if settings['parts_filter_mode'] == 'kouen':
            query_parts.append("`口縁部` != 'なし'")
            
        df_scenario = df_all.query(" and ".join(query_parts)).copy()
        print(f"Filtered Data: {len(df_scenario)} rows")
        
        if df_scenario.empty:
            print("Skipping: No data after filtering.")
            continue
            
        # --- 前処理 (One-Hot Encoding) ---
        df_processed, final_features, cat_cols, bool_cols = preprocess_multivalue_features(
            df_scenario, settings['custom_features']
        )
        
        # ターゲット遺跡フラグ作成
        regex_pattern = "|".join([f"({p})" for p in settings['target_sites']])
        df_processed['is_target'] = df_processed[SITE_COLUMN_NAME].str.contains(regex_pattern, regex=True, na=False)

        # 元データ保存 (Excel出力用)
        data_for_report = df_processed[final_features].copy()
        
        # --- アプローチごとのループ (Gower, MCA...) ---
        for approach in ANALYSIS_APPROACHES:
            approach_name = approach['name']
            cluster_key = approach['cluster_key']
            print(f"\n--- Approach: {approach_name} ---")
            
            # --- 次元圧縮 (Distance / Imputation) ---
            if approach['type'] == 'gower':
                # データ準備 ('不明'->NaN)
                data_mining = df_processed[final_features].replace('不明', np.nan)
                
                # 重み計算
                weights = calculate_gower_weights(data_mining, strategy=approach['weight_strategy'])
                
                # Gower距離行列
                dist_matrix = gower.gower_matrix(data_mining, weight=weights)
                dim_reduced_data = dist_matrix  # t-SNEには距離行列を渡す (metric='precomputed')
                tsne_metric = 'precomputed'
                
            elif approach['type'] == 'mca':
                # 欠損値補完
                data_imputed = impute_missing_data(
                    df_processed[final_features], cat_cols, bool_cols, 
                    strategy=approach['impute_strategy'], 
                    max_iter=settings.get('imputer_max_iter', 20)
                )
                
                # MCA実行
                n_comps = settings.get('mca_n_components', 10)
                mca = prince.MCA(n_components=n_comps, n_iter=3, random_state=42)
                mca.fit(data_imputed)
                
                # スクリープロット保存 (初回のみ)
                save_scree_plot(mca, output_dir, approach_name)
                
                dim_reduced_data = mca.transform(data_imputed)
                tsne_metric = 'euclidean'

            # --- Perplexity Loop ---
            perplexities = [p for p in [20, 30, 40, 50] if p < len(df_scenario)]
            if not perplexities: perplexities = [max(1, len(df_scenario)-1)]

            for p in perplexities:
                run_name = f"{approach_name}_perp{p}"
                n_clusters = settings['clusters_settings'][cluster_key].get(p, 5)
                
                print(f"Running: {run_name} (k={n_clusters})")

                # t-SNE
                tsne = TSNE(n_components=2, perplexity=p, metric=tsne_metric, init='random' if tsne_metric=='precomputed' else 'pca', random_state=0)
                tsne_coords = tsne.fit_transform(dim_reduced_data)
                
                # クラスター数推定プロット
                save_cluster_estimation_plot(tsne_coords, output_dir, run_name)
                
                # K-Means
                kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
                clusters = kmeans.fit_predict(tsne_coords)
                
                # --- 結果の統合とプロット ---
                df_result = df_scenario[['名称', '場所']].copy()
                if '口縁部_新_主モチーフ' in df_scenario.columns:
                     # 統合シナリオのマーカー判定用
                     df_result = pd.concat([df_result, df_scenario[['口縁部_新_主モチーフ', '頸部の傾向', '胴部']]], axis=1)
                
                df_result['X'] = tsne_coords[:, 0]
                df_result['Y'] = tsne_coords[:, 1]
                df_result['cluster'] = clusters
                df_result['is_target'] = df_processed['is_target'].values
                
                # マーカー設定
                df_plot, symbol_map, category_orders = assign_plot_markers(df_result, scenario_name)
                
                # グラフタイトル
                title = settings['plot_title_template'].format(
                    name_base=approach_name, p=p, k=n_clusters
                )
                
                # Plotly 描画
                fig = px.scatter(
                    df_plot, x='X', y='Y',
                    color=df_plot['cluster'].astype(str),
                    symbol='marker_label', symbol_map=symbol_map,
                    category_orders={'marker_label': category_orders},
                    hover_data=['名称', '場所'],
                    title=title,
                    opacity=0.7
                )
                
                # 背景(非ターゲット)を目立たなくする
                fig.for_each_trace(
                    lambda t: t.update(marker=dict(opacity=0.3, size=5)) 
                    if not any(target in t.name for target in category_orders) else None
                )
                
                # 保存
                html_path = os.path.join(output_dir, f"Plot_{run_name}.html")
                fig.write_html(html_path)
                
                # Excelレポート
                excel_path = os.path.join(output_dir, f"Profile_{run_name}.xlsx")
                create_cluster_profile_excel(
                    data_for_report, df_result, 'cluster', n_clusters, excel_path, cat_cols, bool_cols
                )

    print("\n=== All Analysis Completed ===")

if __name__ == "__main__":
    main()