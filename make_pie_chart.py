import pandas as pd
import matplotlib.pyplot as plt
import os
import japanize_matplotlib  # 日本語フォント対応
import re
import sys

# ==========================================================================
# 1. 設定セクション (CONFIGURATION)
# ==========================================================================

# 入力ファイル
CSV_FILE_PATH = 'motodata_1_pre_processed.csv'
LOCATION_COLUMN = '場所'

# 出力フォルダ名（比較結果用）
COMPARATIVE_OUTPUT_FOLDER = "_Comparative_Analysis_Results"

# ---------------------------------------------------------
# 分析対象グループの定義
# ---------------------------------------------------------

# 基本分析 (Type A) 対象グループ
BASIC_TARGET_GROUPS = {
    '口唇部_断面形': [
        '口唇部_断面形_平坦', '口唇部_断面形_丸み_内面', '口唇部_断面形_丸み_外面', '口唇部_断面形_丸み_直立',
        '口唇部_断面形_傾き_内面', '口唇部_断面形_傾き_外面', '口唇部_断面形_尖り_内面', '口唇部_断面形_尖り_外面',
        '口唇部_断面形_尖り_直立', '口唇部_断面形_突出_内面', '口唇部_断面形_突出_外面', '口唇部_断面形_突出_直立',
        '口唇部_断面形_面取り_内部', '口唇部_断面形_面取り_外部'
    ],
    '口唇部_装飾': [
        '口唇部_装飾_なし', '口唇部_装飾_縄文', '口唇部_装飾_沈線', '口唇部_装飾_刻目', '口唇部_装飾_刺突'
    ],
    '口唇部_器面調整': ['口唇部_器面調整_なし', '口唇部_器面調整_ナデ', '口唇部_器面調整_ミガキ'],
    '口縁部_形状': [
        '口縁部_形状_水平', '口縁部_形状_緩やかな波状', '口縁部_形状_突起状山形_平板',
        '口縁部_形状_突起状山形_横断面がコ', '口縁部_形状_不明'
    ],
    '口縁部_方向': ['口縁部_方向_内彎', '口縁部_方向_内折', '口縁部_方向_直立', '口縁部_方向_外反', '口縁部_方向_不明'],
    '口縁部_器面調整': ['口縁部_器面調整_なし', '口縁部_器面調整_ナデ', '口縁部_器面調整_ミガキ'],
    '口縁部直下': ['口縁部直下_無文', '口縁部直下_縄文', '口縁部直下_条痕', '口縁部直下_その他'],
    '頸部の傾向': ['頸部の傾向_分離', '頸部の傾向_一体化', '頸部の傾向_同化'],
    '頸部の状態': [
        '頸部の状態_段・肥厚_内面', '頸部の状態_段・肥厚_外面', '頸部の状態_隆帯', '頸部の状態_屈曲',
        '頸部の状態_文様_縄文', '頸部の状態_文様_沈線', '頸部の状態_文様_その他'
    ],
    '胴部方向': ['胴部方向_内彎', '胴部方向_外反', '胴部方向_直立', '胴部方向_くびれ', '胴部方向_不明'],
    '胴部_器面調整': ['胴部_器面調整_なし', '胴部_器面調整_ナデ', '胴部_器面調整_ミガキ', '胴部_器面調整_その他']
}

# 技法分析 (Type B) 固定カテゴリ
TECHNIQUE_FIXED_CATEGORIES = [
    '磨消縄文', '幅広の沈線', '細い沈線', '沈線系(幅広の沈線+細い沈線)', '縄文', '無文', '刺突',
    '刺突＋沈線系', '刺突＋磨消縄文', '刺突＋縄文', '沈線系+縄文', '磨消縄文+沈線系', 'その他'
]

# モチーフ分析 (Type C) 対象グループ
MOTIF_TARGET_GROUPS = {
    '口縁部_主モチーフ': [
        '口縁部_主モチーフ_無文', '口縁部_主モチーフ_区画文_方形・窓枠状', '口縁部_主モチーフ_区画文_楕円形', '口縁部_主モチーフ_区画文_波状',
        '口縁部_主モチーフ_曲線文_スペード文', '口縁部_主モチーフ_曲線文_円文', '口縁部_主モチーフ_曲線文_横に長いJ字文',
        '口縁部_主モチーフ_曲線文_波状文', '口縁部_主モチーフ_曲線文_渦巻文', '口縁部_主モチーフ_曲線文_連弧文',
        '口縁部_主モチーフ_直線文_三角文', '口縁部_主モチーフ_直線文_横位線', '口縁部_主モチーフ_直線文_縦位線',
        '口縁部_主モチーフ_特殊文_刺突文', '口縁部_主モチーフ_特殊文_羽状文', '口縁部_主モチーフ_モチーフ不明:沈線',
        '口縁部_主モチーフ_モチーフ不明:磨消縄文', '口縁部_主モチーフ_モチーフ不明:その他',
    ],
    '胴部_主モチーフ': [
        '胴部_主モチーフ_無文', '胴部_主モチーフ_区画文_方形・窓枠状', '胴部_主モチーフ_区画文_楕円形', '胴部_主モチーフ_区画文_波状',
        '胴部_主モチーフ_曲線文_スペード文', '胴部_主モチーフ_曲線文_横に長いJ字文', '胴部_主モチーフ_曲線文_波状文',
        '胴部_主モチーフ_曲線文_渦巻文', '胴部_主モチーフ_曲線文_縦に長いJ字文', '胴部_主モチーフ_曲線文_連弧文',
        '胴部_主モチーフ_直線文_斜文', '胴部_主モチーフ_直線文_縦位線', '胴部_主モチーフ_特殊文_紡錘文', '胴部_主モチーフ_特殊文_その他',
        '胴部_主モチーフ_モチーフ不明:沈線', '胴部_主モチーフ_モチーフ不明:磨消縄文', '胴部_主モチーフ_モチーフ不明:その他'
    ]
}

# ---------------------------------------------------------
# 分析実行ターゲットの設定
# ---------------------------------------------------------
ANALYSIS_TARGETS = [
    {'mode': 'prefix', 'value': ['Daikan'], 'heritage_name': '大官大寺下層遺跡', 'run_basic_analysis': True, 'run_technique_analysis': True, 'run_motif_analysis': True},
    {'mode': 'prefix', 'value': ['Shimocha'], 'heritage_name': '下茶屋地蔵谷遺跡', 'run_basic_analysis': True, 'run_technique_analysis': True, 'run_motif_analysis': True},
    {'mode': 'prefix', 'value': ['Yano'], 'heritage_name': '矢野遺跡', 'run_basic_analysis': True, 'run_technique_analysis': True, 'run_motif_analysis': True},
    {'mode': 'prefix', 'value': ['Kosaka'], 'heritage_name': '小阪遺跡', 'run_basic_analysis': True, 'run_technique_analysis': True, 'run_motif_analysis': True},
    {'mode': 'prefix', 'value': ['Higashi'], 'heritage_name': '東庄内A遺跡', 'run_basic_analysis': True, 'run_technique_analysis': True, 'run_motif_analysis': True},
    {'mode': 'prefix', 'value': ['Kaku'], 'heritage_name': '覚正垣内遺跡', 'run_basic_analysis': True, 'run_technique_analysis': True, 'run_motif_analysis': True},
    {'mode': 'prefix', 'value': ['Hutsunami'], 'heritage_name': '仏並遺跡', 'run_basic_analysis': True, 'run_technique_analysis': True, 'run_motif_analysis': True},
    {'mode': 'prefix', 'value': ['Ina'], 'heritage_name': '稲ヶ原A遺跡', 'run_basic_analysis': True, 'run_technique_analysis': True, 'run_motif_analysis': True},
    {'mode': 'prefix', 'value': ['Matsu'], 'heritage_name': '松風台遺跡', 'run_basic_analysis': True, 'run_technique_analysis': True, 'run_motif_analysis': True},
    {'mode': 'prefix', 'value': ['Koigakubo'], 'heritage_name': '恋ヶ窪東遺跡', 'run_basic_analysis': True, 'run_technique_analysis': True, 'run_motif_analysis': True},
    {'mode': 'prefix', 'value': ['Matsu', 'Koigakubo', 'Ina'], 'heritage_name': '関東の遺跡', 'run_basic_analysis': True, 'run_technique_analysis': True, 'run_motif_analysis': True},
    {'mode': 'prefix', 'value': ['Matsu', 'Koigakubo', 'Ina', 'Daikan', 'Kosaka', 'Kaku', 'Yano', 'Higashi', 'Hutsunami', 'Shimocha'], 'heritage_name': '全遺跡', 'run_basic_analysis': True, 'run_technique_analysis': True, 'run_motif_analysis': True},
]


# ==========================================================================
# 2. ヘルパー関数 & カラーマップ (HELPER FUNCTIONS)
# ==========================================================================

def categorize_techniques(row, prefix):
    """
    技法フラグの組み合わせから、主要な技法カテゴリを判定する関数
    """
    # 各主要技法の有無を取得 (該当する列の値が1かチェック)
    has_masho = row.get(f'{prefix}_磨消縄文', 0) == 1
    has_futo_sen = row.get(f'{prefix}_幅広の沈線', 0) == 1
    has_hoso_sen = row.get(f'{prefix}_細い沈線', 0) == 1
    has_jomon = row.get(f'{prefix}_縄文', 0) == 1
    has_mumon = row.get(f'{prefix}_無文', 0) == 1
    
    # 刺突は複数の列が存在する可能性があるため部分一致で検索
    has_shitotsu = any(row.get(col, 0) == 1 for col in row.index if f'{prefix}_刺突' in col)
    
    # 沈線系 (幅広 or 細い)
    has_senkei = has_futo_sen or has_hoso_sen

    # --- 複合技法の判定 ---
    if has_shitotsu and has_senkei: return '刺突＋沈線系'
    if has_shitotsu and has_masho: return '刺突＋磨消縄文'
    if has_shitotsu and has_jomon: return '刺突＋縄文'
    if has_senkei and has_jomon: return '沈線系+縄文'
    if has_masho and has_senkei: return '磨消縄文+沈線系'
    if has_futo_sen and has_hoso_sen: return '沈線系(幅広の沈線+細い沈線)'
    
    # --- 単独技法の判定 ---
    if has_masho: return '磨消縄文'
    if has_futo_sen: return '幅広の沈線'
    if has_hoso_sen: return '細い沈線'
    if has_shitotsu: return '刺突'
    if has_jomon: return '縄文'
    if has_mumon: return '無文'
    
    # --- 「その他」の詳細分析 ---
    # どの固定カテゴリにも該当しない場合、立っているフラグを収集する
    active_flags = []
    
    # 基本技法として処理済みのものは除外リストに入れる
    base_tech_prefixes = ['磨消縄文', '幅広の沈線', '細い沈線', '縄文', '無文', '沈線']
    
    for col in row.index:
        if col.startswith(prefix) and row.get(col, 0) == 1:
            flag_name = col.replace(prefix + '_', '')
            
            # 無視するフラグ
            if flag_name.lower() == 'nan':
                continue
            
            # 基本技法またはその派生形は「その他」の詳細には含めない
            is_base_technique = False
            for base in base_tech_prefixes:
                if flag_name == base or flag_name.startswith(base + '_'):
                    is_base_technique = True
                    break
            
            if is_base_technique:
                continue

            active_flags.append(flag_name)
    
    if not active_flags:
        return 'その他: [フラグなし]'
    else:
        active_flags.sort()
        return f"その他: [{', '.join(active_flags)}]"

def get_special_condition_mask(df):
    """特定の条件下（口縁部が無文で同化し、胴部がある等）の行を判定するマスクを返す"""
    try:
        mask = (
            (df['口縁部_あり'] == 1) &
            (df['口縁部_主モチーフ_無文'] == 1) &
            (df['頸部_あり'] == 1) &
            (df['頸部の傾向_同化'] == 1) &
            (df['胴部_あり'] == 1) &
            (df['口縁部_技法_無文'] == 1)
        )
        return mask
    except KeyError as e:
        print(f"Warning: Columns for special condition check not found: {e}")
        return pd.Series([False] * len(df), index=df.index)

# --- カラーマップ生成関数 ---
def _generate_extended_colors():
    colors1 = plt.get_cmap('tab20').colors
    colors2 = plt.get_cmap('tab20b').colors
    return list(colors1) + list(colors2)

def get_technique_color_map():
    colors = _generate_extended_colors()
    return {cat: colors[i % len(colors)] for i, cat in enumerate(TECHNIQUE_FIXED_CATEGORIES)}

def get_motif_master_color_map():
    all_motifs = []
    for group, cols in MOTIF_TARGET_GROUPS.items():
        all_motifs.extend([c.replace(group + '_', '') for c in cols])
    unique_motifs = sorted(list(set(all_motifs)))
    colors = _generate_extended_colors()
    return {motif: colors[i % len(colors)] for i, motif in enumerate(unique_motifs)}

def get_basic_master_color_map():
    all_items = []
    for group, cols in BASIC_TARGET_GROUPS.items():
        all_items.extend([c.replace(group + '_', '') for c in cols])
    unique_items = sorted(list(set(all_items)))
    colors = _generate_extended_colors()
    return {item: colors[i % len(colors)] for i, item in enumerate(unique_items)}


# ==========================================================================
# 3. 描画関数 (VISUALIZATION FUNCTIONS)
# ==========================================================================

def create_pie_chart(counts, colors, title, output_path, legend_labels):
    """円グラフを作成して保存する"""
    plt.figure(figsize=(7, 5))
    wedges, texts, autotexts = plt.pie(
        counts, colors=colors, autopct='%1.1f%%', startangle=90, pctdistance=0.8
    )
    plt.title(title, fontsize=10)
    plt.setp(autotexts, size=6, weight="bold", color="white")
    plt.legend(wedges, legend_labels, title="凡例", loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, title_fontsize=9)
    plt.ylabel('')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def create_comparative_stacked_bar_chart(df_plot, group_name, color_map, output_folder, site_totals):
    """比較用の100%積み上げ棒グラフを作成して保存する"""
    output_path = os.path.join(output_folder, f'_Comparative_{group_name}_StackedBar.png')
    
    # 項目に応じた色設定
    colors = [color_map.get(item, '#CCCCCC') for item in df_plot.columns]
    
    # グラフサイズ調整
    num_sites = len(df_plot.index)
    width = max(5, num_sites * 0.7 + 3)
    
    fig, ax = plt.subplots(figsize=(width, 6))
    df_plot.plot(kind='bar', stacked=True, color=colors, ax=ax, width=0.4)
    
    ax.set_title(f'「{group_name}」の遺跡別 構成割合', fontsize=12)
    ax.set_ylabel('割合 (%)', fontsize=9)
    ax.set_xlabel('遺跡・遺跡群', fontsize=9)
    ax.set_ylim(0, 100)
    
    # X軸ラベルにN数を追加
    new_x_labels = [f"{site}\n(n={site_totals.get(site, 0)})" for site in df_plot.index]
    ax.set_xticklabels(new_x_labels, rotation=30, ha='right', fontsize=9, fontweight='bold')
    
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), title="凡例", fontsize=9, title_fontsize=9)
    plt.yticks(fontsize=8)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def save_comparative_summary_table(df_counts, group_name, output_folder):
    """比較用の集計表(CSV)を保存する"""
    output_path = os.path.join(output_folder, f'_Comparative_{group_name}_Summary.csv')
    
    # 割合(%)の算出
    df_percents = 100 * df_counts.div(df_counts.sum(axis=0), axis=1)
    df_percents = df_percents.fillna(0)
    
    # 出力用DataFrame構築
    df_summary = pd.DataFrame(index=df_counts.index)
    df_summary.index.name = '項目名'
    
    # ANALYSIS_TARGETS順に列を並べる
    site_names_in_order = [t['heritage_name'] for t in ANALYSIS_TARGETS]
    sorted_columns = []
    
    for site in site_names_in_order:
        if site in df_counts.columns:
            df_summary[f'{site} (件数)'] = df_counts[site].astype(int)
            df_summary[f'{site} (割合%)'] = df_percents[site]
            sorted_columns.append(f'{site} (件数)')
            sorted_columns.append(f'{site} (割合%)')
            
    df_summary = df_summary[sorted_columns]
    
    # 合計行の追加
    total_counts = df_summary.filter(like='(件数)').sum(axis=0)
    total_counts.name = '--- 合計 ---'
    df_total_row = pd.DataFrame(total_counts).T
    df_summary = pd.concat([df_summary, df_total_row])
    
    df_summary.to_csv(output_path, encoding='utf-8-sig', index=True, float_format='%.2f')


# ==========================================================================
# 4. 分析ロジック (ANALYSIS LOGIC)
# ==========================================================================

def run_basic_analysis(df, output_folder, heritage_name, filter_str, color_map):
    """Type A: 基本分析実行"""
    print(f"\n--- {heritage_name} ({filter_str}): 基本分析 (Type A) ---")
    collected_counts = {}

    for group_name, cols in BASIC_TARGET_GROUPS.items():
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols: continue

        counts = df[valid_cols].sum()
        counts = counts[counts > 0].sort_values(ascending=False)
        if counts.empty: continue

        # ラベルの整形
        cleaned_labels = [l.replace(group_name + '_', '') for l in counts.index]
        colors = [color_map.get(l, '#CCCCCC') for l in cleaned_labels]
        
        # 円グラフ保存
        title = f'「{heritage_name}」における「{group_name}」の割合 (n={counts.sum()})'
        legend_labels = [f'{l} ({p:.1f}%)' for l, p in zip(cleaned_labels, 100 * counts / counts.sum())]
        output_path = os.path.join(output_folder, f'{filter_str}_{group_name}_pie_chart.png')
        create_pie_chart(counts, colors, title, output_path, legend_labels)
        
        # 比較用データ格納
        counts.index = cleaned_labels
        collected_counts[group_name] = counts

    return collected_counts

def run_technique_analysis(df, output_folder, heritage_name, filter_str, all_cols, color_map):
    """Type B: 技法分析実行"""
    print(f"\n--- {heritage_name} ({filter_str}): 技法分析 (Type B) ---")
    
    technique_groups = {
        '口縁部_技法': [c for c in all_cols if '口縁部_技法' in c],
        '胴部_技法': [c for c in all_cols if '胴部_技法' in c],
    }
    condition_mask = get_special_condition_mask(df)
    collected_counts = {}

    for group_name, cols in technique_groups.items():
        if not cols: continue
        
        # カテゴリ判定
        if group_name == '口縁部_技法':
            labels_normal = df.loc[~condition_mask].apply(categorize_techniques, axis=1, prefix='口縁部_技法')
            labels_special = df.loc[condition_mask].apply(categorize_techniques, axis=1, prefix='胴部_技法')
            all_labels = pd.concat([labels_normal, labels_special])
        else:
            all_labels = df.apply(categorize_techniques, axis=1, prefix='胴部_技法')
        
        # 「その他」の詳細出力処理
        other_details = all_labels[all_labels.str.startswith('その他: ')]
        other_details_valid = other_details[other_details != 'その他: [フラグなし]']
        
        if not other_details_valid.empty:
            other_counts = other_details_valid.value_counts().sort_values(ascending=False)
            csv_path = os.path.join(output_folder, f'{filter_str}_{group_name}_Others_Details.csv')
            pd.DataFrame({'その他分類': other_counts.index, '件数': other_counts.values}).to_csv(csv_path, encoding='utf-8-sig', index=False)
            print(f"   -> Saved detailed 'Others' report to {csv_path}")

        # 集計処理（[フラグなし]を除外し、詳細は「その他」にまとめる）
        labels_clean = all_labels[all_labels != 'その他: [フラグなし]'].replace(r'^その他: .*', 'その他', regex=True)
        counts = labels_clean.value_counts().reindex(TECHNIQUE_FIXED_CATEGORIES, fill_value=0)
        
        # 円グラフ保存 (0件の項目は除外)
        counts_pie = counts[counts > 0].sort_values(ascending=False)
        if not counts_pie.empty:
            colors = [color_map.get(l, '#CCCCCC') for l in counts_pie.index]
            title = f'「{heritage_name}」における「{group_name}」の割合 (n={counts_pie.sum()})'
            legend_labels = [f'{l} ({p:.1f}%)' for l, p in zip(counts_pie.index, 100 * counts_pie / counts_pie.sum())]
            output_path = os.path.join(output_folder, f'{filter_str}_{group_name}_categorized_pie_chart.png')
            create_pie_chart(counts_pie, colors, title, output_path, legend_labels)
        
        collected_counts[group_name] = counts

    return collected_counts

def run_motif_analysis(df, output_folder, heritage_name, filter_str, color_map):
    """Type C: モチーフ分析実行"""
    print(f"\n--- {heritage_name} ({filter_str}): 主モチーフ分析 (Type C) ---")
    condition_mask = get_special_condition_mask(df)
    collected_counts = {}

    for group_name, cols in MOTIF_TARGET_GROUPS.items():
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols: continue

        if group_name == '口縁部_主モチーフ':
            # 特殊条件: 口縁部主モチーフとして胴部の情報をカウントする場合
            counts_normal = df.loc[~condition_mask, valid_cols].sum()
            
            doubu_cols = [c.replace('口縁部', '胴部') for c in valid_cols]
            doubu_cols = [c for c in doubu_cols if c in df.columns]
            counts_special = df.loc[condition_mask, doubu_cols].sum()
            counts_special.index = counts_special.index.str.replace('胴部_主モチーフ', '口縁部_主モチーフ')
            
            counts = counts_normal.add(counts_special, fill_value=0)
        else:
            counts = df[valid_cols].sum()

        counts_pie = counts[counts > 0].sort_values(ascending=False)
        
        # 比較用データ (0件も含む全項目を保存するため、ラベル短縮処理)
        counts.index = counts.index.str.replace(group_name + '_', '')
        collected_counts[group_name] = counts

        if counts_pie.empty: continue

        # 円グラフ保存
        cleaned_labels = [l.replace(group_name + '_', '') for l in counts_pie.index]
        colors = [color_map.get(l, '#CCCCCC') for l in cleaned_labels]
        title = f'「{heritage_name}」における「{group_name}」の割合 (n={counts_pie.sum()})'
        legend_labels = [f'{l} ({p:.1f}%)' for l, p in zip(cleaned_labels, 100 * counts_pie / counts_pie.sum())]
        output_path = os.path.join(output_folder, f'{filter_str}_{group_name}_pie_chart.png')
        create_pie_chart(counts_pie, colors, title, output_path, legend_labels)

    return collected_counts


def process_comparative_outputs(collected_data, color_maps, output_folder):
    """全分析結果を統合して比較グラフ・表を作成"""
    print(f"\n=== 全遺跡比較分析開始 (Output: {output_folder}) ===")
    os.makedirs(output_folder, exist_ok=True)
    
    site_order = [t['heritage_name'] for t in ANALYSIS_TARGETS]

    for analysis_type, data in collected_data.items():
        color_map = color_maps[analysis_type]
        
        for group_name, group_results in data.items():
            # 結合DataFrame作成 (列順序指定)
            df_combined = pd.DataFrame(group_results, columns=site_order).fillna(0).astype(int)
            df_combined = df_combined.sort_index()
            
            # 各遺跡の総数（N数）計算
            site_totals = df_combined.sum(axis=0)
            
            # CSV保存
            save_comparative_summary_table(df_combined, group_name, output_folder)
            
            # グラフ作成 (データがあるもののみ)
            items_to_plot = df_combined.sum(axis=1) > 0
            df_plot = df_combined[items_to_plot]
            
            if not df_plot.empty:
                # 割合に変換して転置
                df_percent = 100 * df_plot.div(df_plot.sum(axis=0), axis=1).fillna(0)
                create_comparative_stacked_bar_chart(
                    df_percent.T, group_name, color_map, output_folder, site_totals
                )


# ==========================================================================
# 5. メイン実行部 (MAIN EXECUTION)
# ==========================================================================

def main():
    try:
        if not os.path.exists(CSV_FILE_PATH):
            raise FileNotFoundError(f"File not found: {CSV_FILE_PATH}")
        
        df_all = pd.read_csv(CSV_FILE_PATH)
        all_columns = df_all.columns
        print(f"Loaded data: {len(df_all)} rows")

        if LOCATION_COLUMN not in df_all.columns:
            print(f"Error: Column '{LOCATION_COLUMN}' not found.")
            return

        # カラーマップ初期化
        color_maps = {
            'basic': get_basic_master_color_map(),
            'technique': get_technique_color_map(),
            'motif': get_motif_master_color_map()
        }

        collected_data = {'basic': {}, 'technique': {}, 'motif': {}}

        # --- 個別分析ループ ---
        for target in ANALYSIS_TARGETS:
            heritage_name = target['heritage_name']
            value = target['value']
            mode = target['mode']
            
            # フィルタリング文字列作成
            filter_str = '_'.join(value) if isinstance(value, list) else str(value)
            
            # データの絞り込み
            df_loc = df_all[LOCATION_COLUMN].astype(str)
            if mode == 'prefix':
                if isinstance(value, list):
                    pattern = '|'.join(f'^{re.escape(str(v))}' for v in value)
                    df_filtered = df_all[df_loc.str.contains(pattern, na=False)].copy()
                else:
                    df_filtered = df_all[df_loc.str.startswith(str(value))].copy()
            else: # exact match (implied)
                # ... (必要なら実装) ...
                continue

            if df_filtered.empty:
                print(f"Skipping {heritage_name}: No data found.")
                continue

            # 出力先フォルダ作成
            output_folder = filter_str
            os.makedirs(output_folder, exist_ok=True)

            # 各分析実行
            if target.get('run_basic_analysis'):
                res = run_basic_analysis(df_filtered, output_folder, heritage_name, filter_str, color_maps['basic'])
                for k, v in res.items():
                    if k not in collected_data['basic']: collected_data['basic'][k] = {}
                    collected_data['basic'][k][heritage_name] = v
            
            if target.get('run_technique_analysis'):
                res = run_technique_analysis(df_filtered, output_folder, heritage_name, filter_str, all_columns, color_maps['technique'])
                for k, v in res.items():
                    if k not in collected_data['technique']: collected_data['technique'][k] = {}
                    collected_data['technique'][k][heritage_name] = v

            if target.get('run_motif_analysis'):
                res = run_motif_analysis(df_filtered, output_folder, heritage_name, filter_str, color_maps['motif'])
                for k, v in res.items():
                    if k not in collected_data['motif']: collected_data['motif'][k] = {}
                    collected_data['motif'][k][heritage_name] = v

        # --- 全体比較分析実行 ---
        process_comparative_outputs(collected_data, color_maps, COMPARATIVE_OUTPUT_FOLDER)
        print("\nAll analysis completed successfully.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()