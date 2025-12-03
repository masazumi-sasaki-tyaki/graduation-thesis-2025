import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys
from scipy.stats import chi2_contingency

# ==========================================================================
# 1. 設定セクション (USER CONFIGURATION)
# ==========================================================================

# 入出力ファイル・フォルダ設定
INPUT_FILE = 'motodata_1_updated.csv'
OUTPUT_RANKING_IMAGE = 'feature_ranking_final.png'
OUTPUT_HEATMAP_FOLDER = 'クロス集計ヒートマップ'
OUTPUT_MULTISELECT_HEATMAP_FOLDER = 'クロス集計ヒートマップ_複数選択'

# 分析対象のターゲット変数（Y軸）
TARGET_VARIABLE = '口縁部_新_主モチーフ'

# 分析対象カテゴリの絞り込み（Y軸）
# 特定のカテゴリのみを分析したい場合はリストに記述、全件対象の場合は空リスト []
TARGET_CATEGORIES_TO_USE = [
    '区画文_方形・窓枠状', '直線文_横位線', '無文', 'モチーフ不明:沈線', 'モチーフ不明:磨消縄文'
]
# TARGET_CATEGORIES_TO_USE = []  # 全件対象の場合

# 分析に使用する変数（X軸候補）のリスト
COLUMNS_TO_ANALYZE = [
    '口唇部_断面形', '口唇部_器面調整', '口唇部_装飾', 
    '口縁部直下', '口縁部_技法_縄文_特徴', '口縁部_技法_沈線_特徴', '口縁部_形状', '口縁部_方向', 
    '口縁部_技法_磨消縄文_縄文', '口縁部_技法_磨消縄文_施文順序', '口縁部_技法_磨消縄文_図地', '口縁部_新_主モチーフ',
    '口縁部_技法_磨消縄文_沈線', '口縁部_器面調整', '口縁部_状態_連続・非連続', '口縁部_状態_退化', 
    '口縁部_列数', '口縁部_文様が開放/閉鎖', '口縁部_変形', '口縁部_主文様同士が並行/対向', '口縁部_文様方向',
    '頸部の傾向', '頸部の状態',
    '胴部方向', '胴部_技法_縄文_特徴', '胴部_技法_沈線_特徴', '胴部_技法_磨消縄文_縄文',
    '胴部_技法_磨消縄文_施文順序', '胴部_技法_磨消縄文_図地', '胴部_技法_磨消縄文_沈線',
    '胴部_器面調整', '胴部_新_主モチーフ', '胴部_列数', '胴部_文様が開放', '胴部_変形', 
    '胴部_主文様同士が並行/対向', '胴部_文様方向',
    '上端連繋', '下端連繋', '横位連繫線',
    '胴部_技法_分類', '口縁部_技法_分類',
]

# クラメールのV係数のしきい値（これより高い相関のみヒートマップを出力）
CRAMER_V_THRESHOLD = 0.07

# 複数選択（ダミー変数）グループの定義
MULTI_SELECT_GROUPS = [
    {
        'group_name': '口縁部_間モチーフ',
        'prefix': '口縁部_間モチーフ_',
        'filter_col': '口縁部_間モチーフ_あり', 
        'exclude_suffixes': [
            'あり', 'なし', 'nan', 
            '区画文_区画文でない', '区画文_nan', 
            '曲線文_曲線文でない', '曲線文_nan',
            '直線文_直線文でない', '直線文_nan',
            '特殊文_特殊文でない', '特殊文_nan',
            '区画文', '曲線文', '直線文', '特殊文'
        ]
    },
    {
        'group_name': '胴部_間モチーフ',
        'prefix': '胴部_間モチーフ_',
        'filter_col': '胴部_間モチーフ_あり',
        'exclude_suffixes': [
            'あり', 'なし', 'nan', 
            '区画文_区画文でない', '区画文_nan', 
            '曲線文_曲線文でない', '曲線文_nan',
            '直線文_直線文でない', '直線文_nan',
            '特殊文_特殊文でない', '特殊文_nan',
            '区画文', '曲線文', '直線文', '特殊文'
        ]
    },
    {
        'group_name': '胴部_主モチーフ内モチーフ',
        'prefix': '胴部_主モチーフ内モチーフ_',
        'filter_col': '胴部_主モチーフ内モチーフ_あり',
        'exclude_suffixes': [
            'あり', 'なし', 'nan', 
            '区画文_区画文でない', '区画文_nan', 
            '曲線文_曲線文でない', '曲線文_nan',
            '直線文_直線文でない', '直線文_nan',
            '特殊文_特殊文でない', '特殊文_nan',
            '区画文', '曲線文', '直線文', '特殊文'
        ]
    },
    {
        'group_name': '頸部の状態',
        'prefix': '頸部の状態_',
        'filter_col': '頸部_あり',
        'exclude_suffixes': ['nan']
    }
]

# X軸（横軸）に表示するカテゴリの絞り込み設定
X_AXIS_CATEGORIES_TO_USE = {
    '単一カテゴリ': {
        '胴部_新_主モチーフ': [
            '特殊文_紡錘文', '無文', '曲線文_横に長いJ字文', '曲線文_縦に長いJ字文', 
            '区画文_方形・窓枠状', 'モチーフ不明:沈線', 'モチーフ不明:磨消縄文'
        ]
    },
    '複数選択': {
        # 例: '口縁部_間モチーフ': ['直線文', '曲線文']
    }
}

# 日本語フォントの設定
plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================================================
# 2. 関数定義 (HELPER FUNCTIONS)
# ==========================================================================

def load_data(filepath):
    """分析用CSVデータを読み込む"""
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        df = pd.read_csv(filepath)
        print("--- Loaded Analysis Data ---")
        print(f"Data Source: {filepath}")
        print(f"Row Count: {len(df)}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def cramers_v(x, y):
    """
    クラメールのV係数（連関係数）を計算する関数
    0〜1の値を取り、1に近いほど相関が強い
    """
    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.shape[0] < 2 or confusion_matrix.shape[1] < 2:
        return 0.0
    
    try:
        chi2 = chi2_contingency(confusion_matrix)[0]
    except ValueError:
        return 0.0

    if chi2 == 0:
        return 0.0
        
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    
    # 補正（Bias correction）
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    
    denominator = min((kcorr-1), (rcorr-1))
    if denominator <= 0:
        return 0.0
        
    return np.sqrt(phi2corr / denominator)

def calculate_cramer_ranking(df, target_col, analysis_cols):
    """ターゲット変数と各変数の関連性を計算し、ランキングDataFrameを返す"""
    valid_cols = [col for col in analysis_cols if col in df.columns and col != target_col]
    
    # 除外された列の警告
    missing_cols = set(analysis_cols) - set(valid_cols) - {target_col}
    if missing_cols:
        print(f"Warning: The following columns are missing and skipped: {missing_cols}")
        
    results = []
    for col in valid_cols:
        v_score = cramers_v(df[target_col], df[col])
        results.append({'Variable': col, 'CramersV': v_score})
        
    ranking_df = pd.DataFrame(results).sort_values(by='CramersV', ascending=False)
    return ranking_df

def save_ranking_plot(ranking_df, target_col, filepath):
    """変数ランキングの棒グラフを描画・保存する"""
    try:
        plt.figure(figsize=(12, max(8, len(ranking_df) * 0.4)))
        sns.barplot(x='CramersV', y='Variable', data=ranking_df, palette='coolwarm')
        
        title_suffix = " (Y軸: 指定カテゴリのみ)" if TARGET_CATEGORIES_TO_USE else ""
        plt.title(f"'{target_col}' と各変数との関連の強さ（クラメールのV係数）{title_suffix}", fontsize=16)
        plt.xlabel('クラメールのV係数（1に近いほど関連が強い）', fontsize=12)
        plt.ylabel('変数名', fontsize=12)
        plt.tight_layout()
        
        plt.savefig(filepath, dpi=600)
        print(f"\nSaved ranking plot to '{filepath}'.")
        plt.close()
    except Exception as e:
        print(f"Error saving ranking plot: {e}")

def sanitize_filename(filename):
    """ファイル名に使えない文字を置換する"""
    invalid_chars = r'[\\/:*?"<>|\' ]'
    return re.sub(invalid_chars, '_', filename)

def save_crosstab_heatmaps(df, target_col, ranking_df, v_threshold, output_dir):
    """
    V係数がしきい値を超える項目のクロス集計ヒートマップを保存する
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "="*50)
    print(f"(Single Category) Generating heatmaps for V > {v_threshold}...")
    print(f"Output Directory: {output_dir}")
    print("="*50 + "\n")
    
    high_corr_vars = ranking_df[ranking_df['CramersV'] > v_threshold]
    
    if high_corr_vars.empty:
        print(f"No variables found with V > {v_threshold}.")
        return

    for _, row in high_corr_vars.iterrows():
        variable = row['Variable']
        v_score = row['CramersV']
        print(f"Processing (V={v_score:.3f}): '{target_col}' vs '{variable}'")
        
        try:
            crosstab_detail = pd.crosstab(df[target_col], df[variable])
            
            # 行がすべて0のものを除外
            crosstab_detail = crosstab_detail.loc[(crosstab_detail.sum(axis=1) > 0), :]
            
            # X軸（列）のカテゴリ絞り込み処理
            single_cat_settings = X_AXIS_CATEGORIES_TO_USE.get('単一カテゴリ', {})
            if variable in single_cat_settings:
                categories_to_keep = single_cat_settings[variable]
                actual_cols = [col for col in categories_to_keep if col in crosstab_detail.columns]
                
                if actual_cols:
                    print(f"   -> Filtering X-axis columns: {actual_cols}")
                    crosstab_detail = crosstab_detail[actual_cols]
                else:
                    print(f"   -> Warning: Specified categories for {variable} not found.")

            # 列がすべて0のものを除外（フィルタリング後）
            crosstab_detail = crosstab_detail.loc[:, (crosstab_detail.sum(axis=0) > 0)]
            
            if crosstab_detail.empty:
                print("   -> Result is empty, skipping.")
                continue

            # Plotting
            width = max(12, crosstab_detail.shape[1] * 0.8)
            height = max(8, crosstab_detail.shape[0] * 0.6)
            plt.figure(figsize=(width, height))
            
            sns.heatmap(
                crosstab_detail, annot=True, cmap='YlGnBu', fmt='d', annot_kws={"size": 16}
            )
            
            graph_title = f"'{target_col}' vs '{variable}' のクロス集計"
            plt.title(graph_title, fontsize=18)
            plt.ylabel(target_col, fontsize=16)
            plt.xlabel(variable, fontsize=16)
            plt.tick_params(axis='both', labelsize=16)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=45, ha='right')
            plt.tight_layout()
            
            filename = sanitize_filename(f"V{v_score:.3f}__{graph_title}") + ".png"
            plt.savefig(os.path.join(output_dir, filename), dpi=600)
            
        except Exception as e:
            print(f"   -> Error creating heatmap for {variable}: {e}")
        finally:
            plt.close()
            
    print(f"\nDone. Saved {len(high_corr_vars)} heatmaps.")

def save_multiselect_heatmaps(df, target_col, groups, output_dir):
    """
    複数選択（ダミー変数）グループとターゲット変数のクロス集計ヒートマップを作成する
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "="*50)
    print(f"(Multi-Select) Generating heatmaps for dummy variable groups...")
    print("="*50 + "\n")
    
    if not groups:
        print("No multi-select groups defined.")
        return
        
    all_columns = set(df.columns)
    
    for group in groups:
        group_name = group['group_name']
        prefix = group['prefix']
        filter_col = group['filter_col']
        exclude_suffixes = set(group['exclude_suffixes'])
        
        print(f"--- Group: '{group_name}' ---")
        
        # 必要な列の存在チェック
        if filter_col not in all_columns:
            print(f"   Warning: Filter column '{filter_col}' not found. Skipping.")
            continue
            
        # 集計対象列の特定
        target_dummy_cols = [
            col for col in all_columns 
            if col.startswith(prefix) and col[len(prefix):] not in exclude_suffixes
        ]
        
        if not target_dummy_cols:
            print(f"   Warning: No valid columns found for prefix '{prefix}'. Skipping.")
            continue
            
        # データフィルタリング（"あり"フラグが1のもの）
        filtered_df = df[df[filter_col] == 1].copy()
        if filtered_df.empty:
            print(f"   Info: No data where '{filter_col}' is 1. Skipping.")
            continue
            
        try:
            # GroupBy集計
            crosstab_data = filtered_df.groupby(target_col)[target_dummy_cols].sum()
            
            # データがない行・列を削除
            crosstab_data = crosstab_data.loc[:, (crosstab_data.sum() > 0)]
            crosstab_data = crosstab_data.loc[(crosstab_data.sum(axis=1) > 0), :]

            # X軸（列）のカテゴリ絞り込み処理
            multi_cat_settings = X_AXIS_CATEGORIES_TO_USE.get('複数選択', {})
            if group_name in multi_cat_settings:
                cats_short = multi_cat_settings[group_name]
                cats_full = [prefix + cat for cat in cats_short]
                actual_cols = [col for col in cats_full if col in crosstab_data.columns]
                
                if actual_cols:
                    print(f"   -> Filtering columns: {cats_short}")
                    crosstab_data = crosstab_data[actual_cols]
                else:
                    print(f"   -> Warning: Specified categories for {group_name} not found.")

            if crosstab_data.empty:
                print("   Info: Result is empty after filtering. Skipping.")
                continue

            # 表示用に列名を短縮
            crosstab_data.columns = [col[len(prefix):] for col in crosstab_data.columns]
            
            # Plotting
            width = max(15, crosstab_data.shape[1] * 0.6)
            height = max(10, crosstab_data.shape[0] * 0.5)
            plt.figure(figsize=(width, height))
            
            sns.heatmap(
                crosstab_data, annot=True, cmap='YlGnBu', fmt='d', 
                linewidths=.5, annot_kws={"size": 16}
            )
            
            graph_title = f"'{target_col}' vs '{group_name}' (複数選択)"
            plt.title(graph_title, fontsize=18)
            plt.ylabel(target_col, fontsize=16)
            plt.xlabel(f"{group_name} の項目", fontsize=16)
            plt.tick_params(axis='both', labelsize=16)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=45, ha='right')
            plt.tight_layout()
            
            filename = sanitize_filename(f"MultiSelect__{graph_title}") + ".png"
            plt.savefig(os.path.join(output_dir, filename), dpi=600)
            print(f"   -> Saved heatmap to '{filename}'")
            plt.close()

        except Exception as e:
            print(f"   -> Error: {e}")
        finally:
            plt.close()

# ==========================================================================
# 3. メイン実行処理 (MAIN EXECUTION)
# ==========================================================================

def main():
    # 1. データ読み込み
    df = load_data(INPUT_FILE)
    if df is None:
        return

    # 1.5. 分析対象のフィルタリング（Y軸）
    if TARGET_CATEGORIES_TO_USE:
        print("\n" + "="*50)
        print(f"Filtering Data based on Y-axis categories: {TARGET_VARIABLE}")
        print(TARGET_CATEGORIES_TO_USE)
        
        before_count = len(df)
        df = df[df[TARGET_VARIABLE].isin(TARGET_CATEGORIES_TO_USE)].copy()
        after_count = len(df)
        
        print(f"Rows: {before_count} -> {after_count}")
        if after_count == 0:
            print("Error: No data remaining after filtering. Check your category names.")
            return
        print("="*50 + "\n")
    else:
        print("\nUsing all categories (No filtering).")

    # -----------------------------------------
    # 分析 1: カテゴリ vs カテゴリ (Cramer's V)
    # -----------------------------------------
    ranking_df = calculate_cramer_ranking(df, TARGET_VARIABLE, COLUMNS_TO_ANALYZE)
    
    if not ranking_df.empty:
        print(f"\n--- Variable Correlation Ranking (Target: {TARGET_VARIABLE}) ---")
        print(ranking_df.head(10).round(3))  # Top 10を表示

        save_ranking_plot(ranking_df, TARGET_VARIABLE, OUTPUT_RANKING_IMAGE)
        
        save_crosstab_heatmaps(
            df, TARGET_VARIABLE, ranking_df, CRAMER_V_THRESHOLD, OUTPUT_HEATMAP_FOLDER
        )
    else:
        print("No valid columns found for analysis.")

    # -----------------------------------------
    # 分析 2: カテゴリ vs 複数選択 (Dummy Groups)
    # -----------------------------------------
    save_multiselect_heatmaps(
        df, TARGET_VARIABLE, MULTI_SELECT_GROUPS, OUTPUT_MULTISELECT_HEATMAP_FOLDER
    )

if __name__ == "__main__":
    main()