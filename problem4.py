import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.special import expit
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set font for international compatibility
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

class SmoothTimeDrivenScoringSystem:
    """
    Smooth Time-Driven Scoring System with Technical Hard Threshold Protection
    Proposed New System for Dancing with the Stars - DYNAMIC WEEK VERSION
    """
    
    def __init__(self, data_path=None, threshold_type='median', threshold_multiplier=1.0):
        self.threshold_type = threshold_type
        self.threshold_multiplier = threshold_multiplier
        
        if data_path:
            self.data = self._load_data(data_path)
        else:
            print("⚠️ No data file provided, generating mock data...")
            self.data = self._generate_mock_data()
        
        self.results_df = None
        self.intervention_log = []
        
        print(f"✓ Data loaded: {len(self.data)} records across {self.data['season'].nunique()} seasons")
        
    def _load_data(self, path):
        """Load CSV and deduplicate by (season, week, celebrity)"""
        df = pd.read_csv(path)
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        
        col_map = {
            'total_score': ['total_score', 'score', 'judge_score', 'judge'],
            'vote_pct': ['vote_pct', 'fan_vote', 'vote', 'fan_pct', 'votes'],
            'week': ['week', 'week_number'],
            'season': ['season', 'season_number'],
            'celebrity': ['celebrity', 'celebrity_name', 'name', 'star']
        }
        for std, alts in col_map.items():
            for alt in alts:
                if alt in df.columns and std not in df.columns:
                    df.rename(columns={alt: std}, inplace=True)
        
        for col in ['total_score', 'vote_pct', 'week', 'season']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['season', 'week', 'celebrity'])
        before = len(df)
        df = df.drop_duplicates(subset=['season', 'week', 'celebrity'], keep='first')
        after = len(df)
        if before != after:
            print(f"⚠️ Deduplication: Removed {before-after} duplicate records (season+week+celebrity)")
        
        df = df[df['vote_pct'] > 0]
        
        # ==================== 关键修改1：动态周数检测 ====================
        self.max_week_global = int(df['week'].max())
        print(f"✓ Detected maximum week: {self.max_week_global}")
        
        return df.reset_index(drop=True)
    
    def _generate_mock_data(self):
        """Generate simulated data including controversial contestants - DYNAMIC VERSION"""
        np.random.seed(42)
        data = []
        
        # ==================== 关键修改2：动态生成周数 ====================
        # 每季随机5-20周，模拟真实动态情况
        for season in range(1, 21):  # 减少季数以便测试
            max_w = np.random.randint(8, 25)  # 随机周数：8-24周
            n_c = np.random.randint(8, 14)    # 随机选手数
            
            print(f"Generating Season {season}: {max_w} weeks, {n_c} contestants")
            
            # 特定选手配置（如果周数不足则截断）
            profiles = {
                'Jerry Rice': {'s': 2, 'judge': [15,16,14,15,16,15,16,15,16,17], 
                              'fan': [0.35,0.38,0.32,0.40,0.45,0.42,0.48,0.50,0.52,0.55]},
                'Bristol Palin': {'s': 11, 'judge': [12,14,13,15,14,16,15,17,16,18], 
                                 'fan': [0.30,0.32,0.35,0.38,0.40,0.45,0.48,0.50,0.52,0.48]},
                'Bobby Bones': {'s': 27, 'judge': [14,13,15,14,16,15,17,16,18,19], 
                               'fan': [0.45,0.50,0.47,0.53,0.55,0.57,0.60,0.63,0.65,0.70]},
            }
            
            # 生成普通选手
            for i in range(n_c - 2):
                for w in range(1, max_w+1):
                    data.append({
                        'season': season, 'week': w,
                        'celebrity': f'Normal_{i}_S{season}',
                        'total_score': np.clip(np.random.normal(25-w*0.4, 2.5), 15, 30),
                        'vote_pct': np.random.uniform(0.05, 0.20)
                    })
            
            # 生成特定选手（如果周数足够）
            for name, p in profiles.items():
                if p['s'] == season and max_w >= len(p['judge']):
                    for w in range(1, len(p['judge'])+1):
                        data.append({
                            'season': season, 'week': w,
                            'celebrity': name,
                            'total_score': p['judge'][w-1],
                            'vote_pct': p['fan'][w-1]/n_c
                        })
        
        return pd.DataFrame(data)
    
    # ==================== 关键修改3：自适应权重函数 ====================
    def double_sigmoid_weight(self, t, T):
        """ADAPTIVE Double-sigmoid weight function - works with ANY week count"""
        if T <= 0: 
            T = 1
        
        # 核心改进：将任意周数映射到标准0-10虚拟轴
        # 第一阶段：0-40% 为初赛期
        # 第二阶段：40%-80% 为过渡期  
        # 第三阶段：80%-100% 为决赛期
        virtual_t = (t / T) * 10  # 归一化到0-10标准轴
        
        # 保持原有的双S型拐点位置（4和8）
        S1 = expit(2.0 * (virtual_t - 4.0))
        S2 = expit(2.0 * (virtual_t - 8.0))
        w_j = 0.55 + 0.15 * (S1 - S2)
        
        # 确保权重在合理范围内
        w_j = np.clip(w_j, 0.4, 0.9)
        
        return w_j, 1 - w_j
    
    def calculate_ranks(self, scores, ascending=True):
        return stats.rankdata(scores, method='average') if ascending else stats.rankdata(-scores, method='average')
    
    def apply_hard_threshold(self, df_week, week, max_week):
        """Core logic for Technical Hard Threshold protection mechanism"""
        is_final = (week == max_week)
        if not is_final or len(df_week) <= 1:
            df_week = df_week.copy()
            df_week['threshold_intervened'] = False
            df_week['threshold_beneficiary'] = False
            df_week['final_rank'] = df_week['initial_rank']
            return df_week
        
        if self.threshold_type == 'median':
            tech_line = df_week['total_score'].median() * self.threshold_multiplier
        else:
            tech_line = df_week['total_score'].mean() * self.threshold_multiplier
        
        df = df_week.copy()
        df['tech_qualified'] = df['total_score'] >= tech_line
        df['threshold_intervened'] = False
        df['threshold_beneficiary'] = False
        
        champion_mask = df['initial_rank'] == 1
        
        if champion_mask.any():
            champ_idx = champion_mask.idxmax()
            champ_score = df.loc[champ_idx, 'total_score']
            champ_name = df.loc[champ_idx, 'celebrity']
            
            if champ_score < tech_line:
                qualified = df[df['tech_qualified'] == True]
                
                if len(qualified) > 0:
                    new_champ_idx = qualified['initial_rank'].idxmin()
                    new_champ_name = df.loc[new_champ_idx, 'celebrity']
                    
                    df.loc[champ_idx, 'threshold_intervened'] = True
                    df.loc[new_champ_idx, 'threshold_beneficiary'] = True
                    
                    df['final_rank'] = df['initial_rank'].copy()
                    df.loc[champ_idx, 'final_rank'] = min(3, len(df))
                    df.loc[new_champ_idx, 'final_rank'] = 1
                    
                    self.intervention_log.append({
                        'season': int(df['season'].iloc[0]),
                        'week': int(week),
                        'blocked': str(champ_name),
                        'promoted': str(new_champ_name),
                        'champ_score': float(champ_score),
                        'threshold': float(tech_line)
                    })
                    
                    print(f"  ⚠️ Hard Threshold Intervention: {champ_name}({champ_score:.1f}) blocked from winning, champion transferred to {new_champ_name}")
        
        if 'final_rank' not in df.columns:
            df['final_rank'] = df['initial_rank']
            
        return df
    
    def process_season(self, season):
        """Process individual season with DYNAMIC weeks"""
        s_data = self.data[self.data['season'] == season]
        if len(s_data) == 0: 
            return None
        
        max_w = int(s_data['week'].max())
        results = []
        
        for w in range(1, max_w + 1):
            w_data = s_data[s_data['week'] == w].copy()
            if len(w_data) == 0: 
                continue
            
            n = len(w_data)
            # ==================== 关键修改4：传入实际最大周数 ====================
            w_j, w_f = self.double_sigmoid_weight(w, max_w)
            is_final = (w == max_w)
            
            # 动态安全区计算
            early_cutoff = max(3, int(max_w * 0.3))  # 前30%为早期
            late_cutoff = min(max_w - 1, int(max_w * 0.8))  # 后20%为晚期
            
            if w <= early_cutoff or n >= 10:
                method = 'Rank-based'
                score = 0.5 * self.calculate_ranks(w_data['total_score']) + \
                       0.5 * self.calculate_ranks(w_data['vote_pct'])
            elif is_final and n <= 4:
                method = 'Final-HardThreshold'
                j_min, j_max = w_data['total_score'].min(), w_data['total_score'].max()
                j_pct = (w_data['total_score'] - j_min) / (j_max - j_min + 1e-6)
                f_pct = w_data['vote_pct'] / (w_data['vote_pct'].sum() + 1e-6)
                score = w_j * j_pct + w_f * f_pct
            else:
                method = 'Hybrid'
                j_r = self.calculate_ranks(w_data['total_score']) / n
                f_r = self.calculate_ranks(w_data['vote_pct']) / n
                score = w_j * (1-j_r) + w_f * (1-f_r)
            
            w_data['initial_rank'] = stats.rankdata(score)
            
            j_z = (w_data['total_score'] - w_data['total_score'].mean()) / (w_data['total_score'].std() + 1e-6)
            f_z = (w_data['vote_pct'] - w_data['vote_pct'].mean()) / (w_data['vote_pct'].std() + 1e-6)
            div = np.abs(j_z - f_z)
            w_data['controversy_level'] = pd.cut(div, bins=[0, 0.5, 1.0, np.inf], 
                                                  labels=['Normal', 'Notice', 'High']).astype(str)
            w_data['divergence'] = div.values
            
            if self.threshold_type == 'median':
                tech_line = w_data['total_score'].median() * self.threshold_multiplier
            else:
                tech_line = w_data['total_score'].mean() * self.threshold_multiplier
            w_data['tech_threshold'] = tech_line
            
            w_data = self.apply_hard_threshold(w_data, w, max_w)
            
            w_data['season'] = season
            w_data['week'] = w
            w_data['method'] = method
            w_data['judge_weight'] = w_j
            w_data['fan_weight'] = w_f
            w_data['safe_zone'] = 2 if w <= early_cutoff else (5 if w >= late_cutoff else 3)
            
            results.append(w_data)
        
        if results:
            return pd.concat(results, ignore_index=True)
        return None
    
    def analyze_all(self):
        """Analyze all seasons with DYNAMIC weeks"""
        all_res = []
        print("Processing all seasons...")
        
        for season in sorted(self.data['season'].unique()):
            r = self.process_season(season)
            if r is not None:
                all_res.append(r)
        
        if all_res:
            self.results_df = pd.concat(all_res, ignore_index=True)
            print(f"✓ Analysis complete: {len(self.results_df)} total records")
        return self.results_df
    
    # ==================== 关键修改5：动态可视化 ====================
    def plot_intervention_analysis(self, save_path='fig1_intervention_analysis.png'):
        """Figure 1: Analysis of Hard Threshold Intervention Effects"""
        if self.results_df is None:
            return
        
        interventions = pd.DataFrame(self.intervention_log) if self.intervention_log else pd.DataFrame()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        colors = {
            'blocked': '#C75B39',
            'passed': '#5A8F7B',
            'judge': '#4A6FA5',
            'neutral': '#7D848C'
        }
        
        # Fig 1a: Blocked Contestants
        if len(interventions) > 0:
            y_pos = np.arange(len(interventions))
            bars = ax1.barh(y_pos, interventions['champ_score'], 
                           color=colors['blocked'], edgecolor='black', alpha=0.8)
            ax1.axvline(interventions['threshold'].mean(), color=colors['judge'], 
                       linestyle='--', linewidth=2, label=f'Technical Threshold ({self.threshold_type})')
            
            for i, row in interventions.iterrows():
                ax1.text(row['champ_score'] + 0.2, i, f"{row['champ_score']:.1f}", 
                        va='center', fontsize=9)
            
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(interventions['blocked'])
            ax1.set_xlabel('Judge Score (Technical Merit)')
            ax1.set_title('Blocked Contestants (Below Technical Threshold)', fontweight='bold')
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'No Interventions Triggered', ha='center', va='center', transform=ax1.transAxes)
        
        # Fig 1b: Ranking Changes
        targets = ['Bobby Bones', 'Jerry Rice', 'Bristol Palin']
        x_pos = np.arange(len(targets))
        width = 0.35
        
        status_colors = {
            'intervened': colors['blocked'],
            'passed': colors['passed'],
            'neutral': colors['neutral']
        }
        
        initial_ranks = []
        final_ranks = []
        final_colors = []
        
        for name in targets:
            data = self.results_df[(self.results_df['celebrity']==name) & 
                                  (self.results_df['method']=='Final-HardThreshold')]
            if len(data) > 0:
                row = data.iloc[0]
                initial_ranks.append(row['initial_rank'])
                final_ranks.append(row['final_rank'])
                if row['threshold_intervened']:
                    final_colors.append(status_colors['intervened'])
                else:
                    final_colors.append(status_colors['passed'])
            else:
                initial_ranks.append(0)
                final_ranks.append(0)
                final_colors.append(status_colors['neutral'])
        
        bars1 = ax2.bar(x_pos - width/2, initial_ranks, width, color=colors['neutral'], alpha=0.6, label='Before Intervention')
        
        legend_patches = []
        for i, (rank, color) in enumerate(zip(final_ranks, final_colors)):
            bar = ax2.bar([x_pos[i] + width/2], [rank], width, color=color, alpha=0.8)
            if i == 0:
                if color == status_colors['intervened']:
                    legend_patches.append(mpatches.Patch(color=status_colors['intervened'], label='After Intervention (Blocked)'))
                elif color == status_colors['passed']:
                    legend_patches.append(mpatches.Patch(color=status_colors['passed'], label='After Intervention (Passed)'))
                else:
                    legend_patches.append(mpatches.Patch(color=status_colors['neutral'], label='After Intervention (No Data)'))
        
        ax2.axhline(1, color='gold', linestyle='-', alpha=0.5, label='Champion Position')
        ax2.set_ylabel('Ranking (1 = Champion)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(targets, rotation=15)
        ax2.set_title('Controversial Contestants: Ranking Changes', fontweight='bold')
        
        all_legend_items = [bars1[0], mpatches.Patch(color='gold', alpha=0.5)] + legend_patches
        all_labels = ['Before Intervention', 'Champion Position'] + [patch.get_label() for patch in legend_patches]
        ax2.legend(all_legend_items, all_labels)
        
        # Fig 1c: Technical Score Distribution
        finals = self.results_df[self.results_df['method']=='Final-HardThreshold']
        if len(finals) > 0:
            qualified = finals[finals['total_score'] >= finals['tech_threshold']]
            disqualified = finals[finals['total_score'] < finals['tech_threshold']]
            
            ax3.hist(qualified['total_score'], bins=8, alpha=0.7, color=colors['passed'], 
                    edgecolor='black', label=f'Qualified (n={len(qualified)})')
            ax3.hist(disqualified['total_score'], bins=8, alpha=0.7, color=colors['blocked'], 
                    edgecolor='black', label=f'Disqualified (n={len(disqualified)})')
            ax3.axvline(finals['tech_threshold'].mean(), color=colors['judge'], 
                       linestyle='--', linewidth=2, label=f'Threshold Line')
            ax3.set_xlabel('Judge Technical Score')
            ax3.set_ylabel('Number of Contestants')
            ax3.set_title('Final technical Score Distribution', fontweight='bold')
            ax3.legend()
        
        # Fig 1d: Intervention Timeline
        if len(interventions) > 0:
            ax4.scatter(interventions['week'], interventions['season'], 
                       s=200, c=colors['blocked'], marker='X', label='Intervention Event')
            for _, row in interventions.iterrows():
                ax4.annotate(row['blocked'], (row['week'], row['season']), 
                            xytext=(5, 5), textcoords='offset points', fontsize=8)
            ax4.set_xlabel('Week')
            ax4.set_ylabel('Season')
            ax4.set_title('Temporal Distribution of Interventions', fontweight='bold')
            ax4.legend()
        
        plt.suptitle('Technical Hard Threshold Protection Mechanism Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")
        plt.show()
    
    def plot_system_mechanism(self, save_path='fig2_system_mechanism.png'):
        """Figure 2: Visualization of Proposed System Mechanism - DYNAMIC"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # ==================== 关键修改6：动态生成周数范围 ====================
        max_w = getattr(self, 'max_week_global', 10)
        weeks = np.linspace(1, max_w, 200)  # 高分辨率曲线
        
        # Left: Weight Trajectory
        weights = [self.double_sigmoid_weight(t, max_w)[0] for t in weeks]
        
        ax1.plot(weeks, weights, linewidth=3, color='#4A6FA5', label=f'Judge Weight (max_w={max_w})')
        ax1.fill_between(weeks, weights, alpha=0.2, color='#4A6FA5')
        
        # 动态标记决赛保护区域（最后20%周次）
        final_start = max_w * 0.8
        ax1.axhline(0.85, color='#C75B39', linestyle='--', label='Hard Threshold Mode (0.85)')
        ax1.fill_between([final_start, max_w], 0.4, 0.9, alpha=0.1, color='#C75B39', 
                        label='Finals Protection Zone')
        
        # 标记关键拐点（4周和8周对应的标准化位置）
        for w in [max_w * 0.4, max_w * 0.8]:  # 对应虚拟周的4和8
            wval, _ = self.double_sigmoid_weight(w, max_w)
            ax1.scatter([w], [wval], s=80, c='#4A6FA5', zorder=5)
        
        ax1.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Equal Weight (0.5)')
        ax1.set_xlabel(f'Competition Week (1-{max_w})')
        ax1.set_ylabel('Judge Weight Coefficient')
        ax1.set_title('Dynamic Double-Sigmoid Weight System', fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.4, 0.9)
        
        # Right: Contestant Trajectory
        targets = ['Bobby Bones', 'Jerry Rice']
        legend_handles = []
        legend_labels = []
        
        for idx, name in enumerate(targets):
            data = self.results_df[self.results_df['celebrity']==name].sort_values('week')
            if len(data) == 0:
                continue
            
            j_norm = (data['total_score'] - data['total_score'].min()) / (data['total_score'].max() - data['total_score'].min() + 1e-6)
            f_norm = (data['vote_pct'] - data['vote_pct'].min()) / (data['vote_pct'].max() - data['vote_pct'].min() + 1e-6)
            
            offset = idx * 0.05
            
            # 使用线条样式区分参赛者
            line_style = '-' if idx == 0 else '--'
            marker_style = 'o' if idx == 0 else 's'
            
            line1 = ax2.plot(data['week'], j_norm + offset, linestyle=line_style, marker=marker_style,
                           linewidth=2, markersize=6, color='#4A6FA5')[0]
            line2 = ax2.plot(data['week'], f_norm + offset, linestyle=line_style, marker=marker_style,
                           linewidth=2, markersize=6, color='#D4A373')[0]
            
            if idx == 0:
                legend_handles.extend([line1, line2])
                legend_labels.extend(['Technical Score', 'Fan Vote %'])
            
            final = data[data['method']=='Final-HardThreshold']
            if len(final) > 0 and final.iloc[0]['threshold_intervened']:
                ax2.axvline(final.iloc[0]['week'], color='#C75B39', linestyle='-.', alpha=0.5, 
                           label='Intervention Point' if idx == 0 else "")
        
        ax2.set_xlabel(f'Competition Week (1-{max_w})')
        ax2.set_ylabel('Normalized Score (0-1)')
        ax2.set_title('Technical Merit vs. Fan Popularity Trajectories', fontweight='bold')
        ax2.legend(legend_handles, legend_labels, fontsize=8, loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")
        plt.show()
    
    def plot_finals_analysis(self, save_path='fig3_finals_analysis.png'):
        """Figure 3: Comprehensive Finals Analysis - DYNAMIC"""
        if self.results_df is None:
            return
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        finals = self.results_df[self.results_df['method']=='Final-HardThreshold']
        colors = {'blocked': '#C75B39', 'passed': '#5A8F7B', 'judge': '#4A6FA5', 'fan': '#D4A373'}
        
        # Fig 3a: Scatter Plot
        if len(finals) > 0:
            qualified = finals[finals['total_score'] >= finals['tech_threshold']]
            disqualified = finals[finals['total_score'] < finals['tech_threshold']]
            
            ax1.scatter(qualified['total_score'], qualified['vote_pct'], 
                       s=100, c=colors['passed'], alpha=0.8, label=f'Qualified (n={len(qualified)})', edgecolors='black')
            ax1.scatter(disqualified['total_score'], disqualified['vote_pct'], 
                       s=150, c=colors['blocked'], marker='X', alpha=0.9, label=f'Blocked (n={len(disqualified)})', edgecolors='black')
            ax1.axvline(finals['tech_threshold'].mean(), color=colors['judge'], 
                       linestyle='--', linewidth=2, label='Technical Threshold')
            ax1.set_xlabel('Judge Technical Score')
            ax1.set_ylabel('Fan Vote Percentage')
            ax1.set_title('Finals Contestant Distribution', fontweight='bold')
            ax1.legend()
        
        # Fig 3b: Dual Axis
        max_w = getattr(self, 'max_week_global', 10)
        weekly = self.results_df[self.results_df['week'] <= max_w].groupby('week').agg({
            'judge_weight': 'mean',
            'controversy_level': lambda x: (x=='High').mean()*100
        }).reset_index()
        
        line1 = ax2.plot(weekly['week'], weekly['judge_weight'], 'o-', color=colors['judge'], 
                        linewidth=2, markersize=6, label='Judge Weight')[0]
        ax2.set_xlabel(f'Competition Week (1-{max_w})')
        ax2.set_ylabel('Judge Weight Coefficient', color=colors['judge'])
        
        ax2_twin = ax2.twinx()
        line2 = ax2_twin.plot(weekly['week'], weekly['controversy_level'], 's--', color=colors['fan'], 
                             linewidth=2, markersize=6, label='High Controversy Rate %')[0]
        ax2_twin.set_ylabel('High Controversy Rate (%)', color=colors['fan'])
        
        ax2.set_title('System Dynamics Over Time', fontweight='bold')
        
        lines = [line1, line2]
        labels = ['Judge Weight', 'High Controversy Rate %']
        ax2.legend(lines, labels, loc='center right')
        
        # Fig 3c: Method Usage Pie Chart
        method_counts = self.results_df['method'].value_counts()
        ax3.pie(method_counts.values, labels=method_counts.index, autopct='%1.1f%%',
               wedgeprops={'edgecolor': 'white', 'linewidth': 2},
               colors=['#4A6FA5', '#D4A373', '#C75B39'])
        ax3.set_title('Scoring Method Distribution', fontweight='bold')
        
        plt.suptitle(f'Comprehensive Analysis of Finals Stage (Max Week: {max_w})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")
        plt.show()
    
    def generate_report(self):
        """Generate comprehensive analysis report for DYNAMIC week system"""
        if self.results_df is None: 
            return "Please run analyze_all() first"
        
        total = len(self.results_df)
        finals = self.results_df[self.results_df['method']=='Final-HardThreshold']
        n_interventions = len(self.intervention_log)
        intervention_rate = n_interventions / max(1, len(finals)) * 100
        
        max_w = getattr(self, 'max_week_global', 'Unknown')
        
        report = f"""
{'='*70}
PROPOSED NEW SYSTEM: Smooth Time-Driven Scoring with Technical Hard Threshold
DYNAMIC WEEK VERSION - Analysis Report for Dancing with the Stars
{'='*70}

【SYSTEM MECHANISM】
• Dynamic Week Support: 1-{max_w} weeks per season
• Regular Stage: Double-Sigmoid Smooth Weighting (Adaptive to season length)
• Finals Stage: Technical Hard Threshold Protection (Median×{self.threshold_multiplier:.1f})
• Color Coding:
  - Blue: Judge Scores, Technical Threshold, Weight Trajectory
  - Orange: Fan Votes, Popularity Metrics
  - Green: Technically Qualified Contestants
  - Red: Blocked Controversial Contestants

【OPERATIONAL STATISTICS】
• Total Records Processed: {total}
• Final Episodes: {len(finals)}
• Hard Threshold Interventions: {n_interventions} ({intervention_rate:.1f}%)

【INTERVENTION DETAILS】
"""
        if n_interventions > 0:
            for log in self.intervention_log:
                report += f"• Season {log['season']}, Week {log['week']}: {log['blocked']} (Score: {log['champ_score']:.1f}) blocked from winning\n"
        else:
            report += "• No hard threshold interventions triggered\n"
        
        # Case studies...
        # ... (保持原有案例研究代码)
        
        report += f"\n{'='*70}"
        return report

# ==================== MAIN EXECUTION ====================

def main():
    print("="*70)
    print("TECHNICAL HARD THRESHOLD PROTECTION SYSTEM - DYNAMIC WEEK VERSION")
    print("Proposed New Scoring Model for Dancing with the Stars")
    print("="*70)
    
    system = SmoothTimeDrivenScoringSystem(
        data_path='processed_weekly_data.csv',  # 使用动态生成的模拟数据
        threshold_type='median',
        threshold_multiplier=1.0
    )
    
    results = system.analyze_all()
    
    print("\nGenerating visualization figures...")
    system.plot_intervention_analysis()
    system.plot_system_mechanism()
    system.plot_finals_analysis()
    
    print(system.generate_report())
    
    results.to_csv('scoring_results_dynamic_system.csv', index=False)
    print("\n✓ Detailed results saved to: scoring_results_dynamic_system.csv")
    print("="*70)

if __name__ == "__main__":
    main()