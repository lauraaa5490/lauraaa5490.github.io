# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 20:17:11 2026

@author: m1871
"""

"""
DWTS (Dancing with the Stars) Voting System Analysis Complete Code
For 2026 MCM Problem C - Question 2 Analysis - Complete Solution Version
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration ====================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# ==================== Enum Definitions ====================
class VotingMethod(Enum):
    RANK = "rank"           
    PERCENTAGE = "percentage"  

class EliminationRule(Enum):
    STANDARD = "standard"   
    JUDGE_CHOICE = "judge_choice"  

# ==================== Core Model Class ====================
class DWTSVotingModel:
    def __init__(self, data_path: str):
        print(f"Loading data: {data_path}")
        self.raw_data = self._load_and_parse_csv(data_path)
        self.seasons_data = self._organize_by_season()
        print(f"Successfully loaded data for {len(self.seasons_data)} seasons")
        
    def _load_and_parse_csv(self, path: str) -> pd.DataFrame:
        """Properly handle concatenated CSV format"""
        try:
            df = pd.read_csv(path, skipinitialspace=True)
            cols = df.columns.tolist()
            
            vote_pct_col = None
            for col in cols:
                if 'vote_pct' in str(col):
                    vote_pct_col = col
                    break
            
            if vote_pct_col is None:
                raise ValueError("vote_pct column not found")
            
            base_cols = ['season', 'week', 'celebrity', 'total_score', 
                        'avg_score', 'placement', 'results', 'n_judges']
            
            clean_data = {}
            for col in base_cols:
                if col in df.columns:
                    clean_data[col] = df[col]
            
            clean_data['vote_pct'] = df[vote_pct_col]
            
            df_clean = pd.DataFrame(clean_data)
            
            df_clean['season'] = pd.to_numeric(df_clean['season'], errors='coerce')
            df_clean['week'] = pd.to_numeric(df_clean['week'], errors='coerce')
            df_clean['total_score'] = pd.to_numeric(df_clean['total_score'], errors='coerce')
            df_clean['vote_pct'] = pd.to_numeric(df_clean['vote_pct'], errors='coerce')
            df_clean['placement'] = pd.to_numeric(df_clean['placement'], errors='coerce')
            
            df_clean = df_clean.dropna(subset=['season', 'week', 'celebrity', 'total_score'])
            
            return df_clean
            
        except Exception as e:
            print(f"Data parsing error: {e}")
            return pd.DataFrame()

    def _organize_by_season(self) -> Dict[int, pd.DataFrame]:
        """Organize data by season"""
        if self.raw_data.empty:
            return {}
        return {int(season): group.copy() 
                for season, group in self.raw_data.groupby('season')}
    
    def get_week_data(self, season: int, week: int) -> pd.DataFrame:
        """Get data for a specific week"""
        if season not in self.seasons_data:
            return pd.DataFrame()
        data = self.seasons_data[season]
        return data[data['week'] == week].copy()
    
    def calculate_combined_score(self, 
                                week_data: pd.DataFrame,
                                method: VotingMethod,
                                judge_weight: float = 0.5,
                                fan_weight: float = 0.5) -> pd.DataFrame:
        """Calculate combined score"""
        if week_data.empty:
            return week_data
            
        df = week_data.copy()
        n_contestants = len(df)
        
        if n_contestants == 0:
            return df
        
        df['total_score'] = pd.to_numeric(df['total_score'], errors='coerce')
        df['vote_pct'] = pd.to_numeric(df['vote_pct'], errors='coerce')
        
        if method == VotingMethod.RANK:
            df['judge_rank'] = df['total_score'].rank(ascending=False, method='min')
            df['fan_rank'] = df['vote_pct'].rank(ascending=False, method='min')
            df['combined_score'] = (df['judge_rank'] * judge_weight + 
                                   df['fan_rank'] * fan_weight)
            df['calculated_rank'] = df['combined_score'].rank(ascending=True, method='min')
            
        elif method == VotingMethod.PERCENTAGE:
            total_judge = df['total_score'].sum()
            total_fan = df['vote_pct'].sum()
            
            if total_judge > 0:
                df['judge_pct'] = df['total_score'] / total_judge
            else:
                df['judge_pct'] = 0
            
            if total_fan > 0:
                df['fan_pct'] = df['vote_pct'] / total_fan
            else:
                df['fan_pct'] = 1.0 / n_contestants
            
            df['combined_score'] = (df['judge_pct'] * judge_weight + 
                                   df['fan_pct'] * fan_weight)
            df['calculated_rank'] = df['combined_score'].rank(ascending=False, method='min')
        
        return df
    
    def determine_elimination(self, 
                             scored_data: pd.DataFrame, 
                             rule: EliminationRule = EliminationRule.STANDARD,
                             judge_choice_func: Optional[Callable] = None) -> str:
        """Determine elimination"""
        if scored_data.empty:
            return None
            
        if rule == EliminationRule.STANDARD:
            if 'calculated_rank' not in scored_data.columns:
                return None
            eliminated_idx = scored_data['calculated_rank'].idxmax()
            return scored_data.loc[eliminated_idx, 'celebrity']
            
        elif rule == EliminationRule.JUDGE_CHOICE:
            bottom_two = scored_data.nlargest(2, 'calculated_rank')
            if len(bottom_two) < 2:
                return bottom_two.iloc[0]['celebrity'] if not bottom_two.empty else None
            
            if judge_choice_func:
                return judge_choice_func(scored_data, bottom_two['celebrity'].tolist())
            else:
                eliminated_idx = bottom_two['total_score'].idxmin()
                return bottom_two.loc[eliminated_idx, 'celebrity']
    
    def simulate_week(self, season: int, week: int, 
                     method: VotingMethod, 
                     rule: EliminationRule = EliminationRule.STANDARD,
                     judge_choice_func: Optional[Callable] = None) -> Dict:
        """Simulate a single week of competition"""
        week_data = self.get_week_data(season, week)
        if week_data.empty:
            return None
        
        week_data = week_data[week_data['total_score'] > 0]
        
        if len(week_data) <= 1:
            return None
        
        scored = self.calculate_combined_score(week_data, method)
        eliminated = self.determine_elimination(scored, rule, judge_choice_func)
        
        return {
            'season': season,
            'week': week,
            'method': method.value,
            'eliminated': eliminated,
            'scores': scored[['celebrity', 'total_score', 'vote_pct', 
                            'combined_score', 'calculated_rank']].to_dict('records'),
            'all_celebrities': scored['celebrity'].tolist()
        }

# ==================== Bias Index Calculator ====================
class BiasIndexCalculator:
    """Calculate specific bias indices"""
    def __init__(self, model: DWTSVotingModel):
        self.model = model
        
    def calculate_weekly_effective_weights(self, season: int, week: int) -> Dict:
        """Index 1: Effective Weight Ratio (EWR)"""
        week_data = self.model.get_week_data(season, week)
        if week_data.empty or len(week_data) < 3:
            return None
        
        rank_result = self.model.calculate_combined_score(week_data.copy(), VotingMethod.RANK)
        pct_result = self.model.calculate_combined_score(week_data.copy(), VotingMethod.PERCENTAGE)
        
        judge_corr_rank = abs(rank_result['total_score'].corr(rank_result['combined_score']))
        fan_corr_rank = abs(rank_result['vote_pct'].corr(rank_result['combined_score']))
        
        judge_corr_pct = abs(pct_result['total_score'].corr(pct_result['combined_score']))
        fan_corr_pct = abs(pct_result['vote_pct'].corr(pct_result['combined_score']))
        
        return {
            'season': season,
            'week': week,
            'rank_method_ratio': judge_corr_rank / (fan_corr_rank + 0.001),
            'pct_method_ratio': judge_corr_pct / (fan_corr_pct + 0.001)
        }
    
    def calculate_low_judge_high_fan_bias(self, season: int, week: int) -> Dict:
        """Index 2: Discrepancy Contestant Treatment Bias (DCTB)"""
        week_data = self.model.get_week_data(season, week)
        if week_data.empty or len(week_data) < 4:
            return None
        
        n = len(week_data)
        week_data = week_data.copy()
        week_data['judge_rank'] = week_data['total_score'].rank(ascending=False)
        week_data['fan_rank'] = week_data['vote_pct'].rank(ascending=False)
        
        discrepancy_pool = week_data[
            (week_data['judge_rank'] > n/2) & 
            (week_data['fan_rank'] <= n/2)
        ]
        
        if len(discrepancy_pool) == 0:
            return None
        
        rank_scored = self.model.calculate_combined_score(week_data.copy(), VotingMethod.RANK)
        pct_scored = self.model.calculate_combined_score(week_data.copy(), VotingMethod.PERCENTAGE)
        
        rank_last = rank_scored.loc[rank_scored['calculated_rank'].idxmax(), 'celebrity']
        pct_last = pct_scored.loc[pct_scored['calculated_rank'].idxmax(), 'celebrity']
        
        rank_eliminated_disc = 1 if rank_last in discrepancy_pool['celebrity'].values else 0
        pct_eliminated_disc = 1 if pct_last in discrepancy_pool['celebrity'].values else 0
        
        return {
            'season': season,
            'week': week,
            'n_discrepancy': len(discrepancy_pool),
            'rank_eliminates': rank_eliminated_disc,
            'pct_eliminates': pct_eliminated_disc
        }
    
    def calculate_fan_sensitivity_index(self, season: int, week: int, target_celebrity: str = None) -> Dict:
        """Index 3: Fan Sensitivity Index (FSI)"""
        week_data = self.model.get_week_data(season, week)
        if week_data.empty or len(week_data) < 2:
            return None
        
        if target_celebrity is None:
            target_celebrity = week_data.loc[week_data['total_score'].idxmin(), 'celebrity']
        
        base_data = week_data.copy()
        
        rank_base = self.model.calculate_combined_score(base_data.copy(), VotingMethod.RANK)
        pct_base = self.model.calculate_combined_score(base_data.copy(), VotingMethod.PERCENTAGE)
        
        target_rank_base_rank = rank_base[rank_base['celebrity'] == target_celebrity]['calculated_rank'].iloc[0]
        target_rank_base_pct = pct_base[pct_base['celebrity'] == target_celebrity]['calculated_rank'].iloc[0]
        
        modified_data = base_data.copy()
        modified_data.loc[modified_data['celebrity'] == target_celebrity, 'vote_pct'] *= 1.1
        
        rank_modified = self.model.calculate_combined_score(modified_data.copy(), VotingMethod.RANK)
        pct_modified = self.model.calculate_combined_score(modified_data.copy(), VotingMethod.PERCENTAGE)
        
        target_rank_modified_rank = rank_modified[rank_modified['celebrity'] == target_celebrity]['calculated_rank'].iloc[0]
        target_rank_modified_pct = pct_modified[pct_modified['celebrity'] == target_celebrity]['calculated_rank'].iloc[0]
        
        return {
            'season': season,
            'week': week,
            'rank_method_sensitivity': abs(target_rank_modified_rank - target_rank_base_rank),
            'pct_method_sensitivity': abs(target_rank_modified_pct - target_rank_base_pct)
        }

# ==================== Controversy Analysis Class ====================
class ControversyAnalyzer:
    def __init__(self, model: DWTSVotingModel):
        self.model = model
        self.controversial_cases = {
            'Jerry Rice (S2)': (2, 'Jerry Rice'),
            'Billy Ray Cyrus (S4)': (4, 'Billy Ray Cyrus'),
            'Bristol Palin (S11)': (11, 'Bristol Palin'),
            'Bobby Bones (S27)': (27, 'Bobby Bones')
        }
        
    def compare_methods_for_season(self, season: int) -> pd.DataFrame:
        """Compare results using two different methods for the same season"""
        if season not in self.model.seasons_data:
            return pd.DataFrame()
            
        season_data = self.model.seasons_data[season]
        weeks = sorted(season_data['week'].unique())
        
        comparisons = []
        
        for week in weeks:
            rank_result = self.model.simulate_week(
                season, week, VotingMethod.RANK, EliminationRule.STANDARD
            )
            
            pct_result = self.model.simulate_week(
                season, week, VotingMethod.PERCENTAGE, EliminationRule.STANDARD
            )
            
            if rank_result and pct_result:
                comparisons.append({
                    'season': season,
                    'week': week,
                    'rank_eliminated': rank_result['eliminated'],
                    'pct_eliminated': pct_result['eliminated'],
                    'different': rank_result['eliminated'] != pct_result['eliminated']
                })
        
        return pd.DataFrame(comparisons) if comparisons else pd.DataFrame()
    
    def analyze_controversial_cases(self) -> Dict[str, Dict]:
        """Analyze controversial cases"""
        results = {}
        for case_name, (season, celebrity) in self.controversial_cases.items():
            results[case_name] = self._analyze_single_celebrity_detailed(season, celebrity)
        return results
    
    def _analyze_single_celebrity_detailed(self, season: int, celebrity: str) -> Dict:
        """Detailed analysis of a specific celebrity's performance under different methods"""
        if season not in self.model.seasons_data:
            return {"error": f"Season {season} not found"}
        
        season_data = self.model.seasons_data[season]
        celeb_data = season_data[season_data['celebrity'] == celebrity].sort_values('week')
        
        if celeb_data.empty:
            return {"error": f"Celebrity {celebrity} not found"}
        
        weeks = sorted(celeb_data['week'].unique())
        
        analysis = {
            'season': season,
            'celebrity': celebrity,
            'actual_final_result': celeb_data['results'].iloc[-1] if 'results' in celeb_data.columns else "Unknown",
            'total_weeks_competed': len(weeks),
            'weekly_safety_margin': []  # Distance from elimination line
        }
        
        total_weeks_saved_by_pct = 0
        total_weeks_saved_by_rank = 0
        
        for week in weeks:
            week_all = season_data[season_data['week'] == week]
            week_active = week_all[week_all['total_score'] > 0]
            
            if len(week_active) < 2:
                continue
            
            if celebrity not in week_active['celebrity'].values:
                continue
            
            rank_scored = self.model.calculate_combined_score(week_active.copy(), VotingMethod.RANK)
            pct_scored = self.model.calculate_combined_score(week_active.copy(), VotingMethod.PERCENTAGE)
            
            rank_row = rank_scored[rank_scored['celebrity'] == celebrity]
            pct_row = pct_scored[pct_scored['celebrity'] == celebrity]
            
            if rank_row.empty or pct_row.empty:
                continue
            
            rank_pos = int(rank_row['calculated_rank'].iloc[0])
            pct_pos = int(pct_row['calculated_rank'].iloc[0])
            n_contestants = len(week_active)
            
            # Calculate distance from elimination line (last place)
            rank_margin = n_contestants - rank_pos  # Higher rank = larger margin
            pct_margin = n_contestants - pct_pos
            
            rank_last = (rank_pos == rank_scored['calculated_rank'].max())
            pct_last = (pct_pos == pct_scored['calculated_rank'].max())
            
            analysis['weekly_safety_margin'].append({
                'week': week,
                'n_contestants': n_contestants,
                'rank_margin': rank_margin,
                'pct_margin': pct_margin,
                'rank_would_survive': not rank_last,
                'pct_would_survive': not pct_last,
                'difference': rank_last != pct_last
            })
            
            if rank_last and not pct_last:
                total_weeks_saved_by_pct += 1
            elif pct_last and not rank_last:
                total_weeks_saved_by_rank += 1
        
        analysis['weeks_saved_by_pct'] = total_weeks_saved_by_pct
        analysis['weeks_saved_by_rank'] = total_weeks_saved_by_rank
        
        if total_weeks_saved_by_pct > total_weeks_saved_by_rank:
            analysis['conclusion'] = f"Percentage Method better protects this contestant (fan advantage)"
            analysis['bias'] = 'Fan-Favored'
        elif total_weeks_saved_by_rank > total_weeks_saved_by_pct:
            analysis['conclusion'] = f"Rank Method better protects this contestant (judge advantage)"
            analysis['bias'] = 'Judge-Favored'
        else:
            analysis['conclusion'] = "Both methods have similar impact"
            analysis['bias'] = 'Neutral'
        
        return analysis
    
    def analyze_judge_intervention(self, season: int) -> pd.DataFrame:
        """Analyze judge intervention impact for a specific season"""
        if season not in self.model.seasons_data:
            return pd.DataFrame()
        
        weeks = sorted(self.model.seasons_data[season]['week'].unique())
        comparisons = []
        
        def judge_protects_technical(full_data, bottom_two_names):
            subset = full_data[full_data['celebrity'].isin(bottom_two_names)]
            return subset.loc[subset['total_score'].idxmin(), 'celebrity']
        
        for week in weeks:
            standard = self.model.simulate_week(
                season, week, VotingMethod.RANK, EliminationRule.STANDARD
            )
            
            intervention = self.model.simulate_week(
                season, week, VotingMethod.RANK, EliminationRule.JUDGE_CHOICE,
                judge_protects_technical
            )
            
            if standard and intervention:
                comparisons.append({
                    'week': week,
                    'standard_elimination': standard['eliminated'],
                    'judge_intervention_elimination': intervention['eliminated'],
                    'changed': standard['eliminated'] != intervention['eliminated']
                })
        
        return pd.DataFrame(comparisons) if comparisons else pd.DataFrame()

# ==================== Enhanced Visualization Class ====================
class EnhancedVisualizer:
    def __init__(self, analyzer: ControversyAnalyzer, bias_calc: BiasIndexCalculator):
        self.analyzer = analyzer
        self.bias_calc = bias_calc
        self.controversial_cases = analyzer.controversial_cases
    
    def plot_all_seasons_comparison(self, all_seasons_stats: List[Dict], save_path: str = 'fig1_all_seasons_comparison.png'):
        """
        Figure 1: Method discrepancy rate comparison across all seasons (Question 2.1)
        Only keep the bar chart - Discrepancy rate bar chart
        """
        fig, ax1 = plt.subplots(1, 1, figsize=(14, 6))
        
        seasons = [s['season'] for s in all_seasons_stats]
        diff_rates = [s['diff_pct'] for s in all_seasons_stats]
        
        # Discrepancy rate bar chart
        colors = ['#E74C3C' if r > 20 else '#F39C12' if r > 10 else '#27AE60' for r in diff_rates]
        bars = ax1.bar(seasons, diff_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax1.axhline(y=np.mean(diff_rates), color='blue', linestyle='--', linewidth=2, label=f'Average ({np.mean(diff_rates):.1f}%)')
        ax1.axhline(y=20, color='red', linestyle=':', alpha=0.5, label='High Difference (20%)')
        ax1.set_xlabel('Season', fontsize=12)
        ax1.set_ylabel('Difference Rate (%)', fontsize=12)
        ax1.set_title('Q2.1: Method Discrepancy Rate Across All Seasons\n(Rank Method vs Percentage Method)', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, rate in zip(bars, diff_rates):
            if rate > 15:  # Only label high discrepancies
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{rate:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
        plt.close()
    
    def plot_controversial_cases_safety_margin(self, save_path: str = 'fig2_controversial_cases_safety.png'):
        """
        Figure 2: Controversial contestant safety margin analysis (Question 2.2)
        Show weekly distance from elimination line - Save 4 separate subplots
        """
        colors = {'rank': '#3498DB', 'percentage': '#E74C3C'}
        
        for idx, (case_name, (season, celebrity)) in enumerate(self.controversial_cases.items()):
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            analysis = self.analyzer._analyze_single_celebrity_detailed(season, celebrity)
            if 'error' in analysis:
                plt.close(fig)
                continue
            
            weeks_data = analysis['weekly_safety_margin']
            if not weeks_data:
                plt.close(fig)
                continue
            
            weeks = [w['week'] for w in weeks_data]
            rank_margins = [w['rank_margin'] for w in weeks_data]
            pct_margins = [w['pct_margin'] for w in weeks_data]
            
            # Plot safety margin (distance from elimination line)
            ax.plot(weeks, rank_margins, 'o-', label='Rank Method', color=colors['rank'], 
                   linewidth=2.5, markersize=8, markerfacecolor='white', markeredgewidth=2)
            ax.plot(weeks, pct_margins, 's-', label='Percentage Method', color=colors['percentage'], 
                   linewidth=2.5, markersize=8, markerfacecolor='white', markeredgewidth=2)
            
            # Mark weeks with differences
            for i, w in enumerate(weeks_data):
                if w['difference']:
                    ax.axvline(x=w['week'], color='gray', alpha=0.3, linestyle='--')
                    ax.scatter([w['week']], [max(rank_margins[i], pct_margins[i]) + 0.3], 
                              marker='*', s=200, color='gold', edgecolors='black', zorder=5)
            
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax.fill_between(weeks, 0, -0.5, alpha=0.2, color='red', label='Elimination Zone')
            
            ax.set_title(f'{case_name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Week', fontsize=10)
            ax.set_ylabel('Safety Margin (Positions above Elimination)', fontsize=10)
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.5, max(max(rank_margins), max(pct_margins)) + 0.5)
            
            plt.tight_layout()
            
            # Save separately for each controversial contestant
            sub_save_path = save_path.replace('.png', f'_{case_name.replace(" ", "_").lower()}.png')
            plt.savefig(sub_save_path, bbox_inches='tight', dpi=300)
            print(f"Saved: {sub_save_path}")
            plt.close()
    
    def plot_bias_metrics_comparison(self, bias_df: pd.DataFrame, save_path: str = 'fig3_bias_metrics_radar.png'):
        """
        Figure 3: Bias metrics comparison (Question 2.1 continued)
        Only keep right figure - Numerical comparison bar chart
        """
        if bias_df.empty:
            print("No data to generate bias charts")
            return
        
        fig, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        
        # Calculate averages
        avg_rank_ewr = bias_df['rank_ewr'].mean()
        avg_pct_ewr = bias_df['pct_ewr'].mean()
        avg_rank_sens = bias_df['rank_fan_sens'].mean()
        avg_pct_sens = bias_df['pct_fan_sens'].mean()
        
        total_rank_elim = bias_df['rank_elim_disc'].sum()
        total_pct_elim = bias_df['pct_elim_disc'].sum()
        valid_cases = len(bias_df[bias_df['rank_elim_disc'] + bias_df['pct_elim_disc'] > 0])
        
        # Normalize elimination rate (convert to protection rate for easier understanding: higher = more protective = more fan-biased)
        rank_save_rate = 1 - (total_rank_elim / valid_cases) if valid_cases > 0 else 0.5
        pct_save_rate = 1 - (total_pct_elim / valid_cases) if valid_cases > 0 else 0.5
        
        # Numerical comparison
        metrics = ['EWR\n(Judge/Fan)', 'Protection Rate\n(LJHFS)', 'Sensitivity\n(FSI)']
        rank_raw = [avg_rank_ewr, rank_save_rate, avg_rank_sens]
        pct_raw = [avg_pct_ewr, pct_save_rate, avg_pct_sens]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, rank_raw, width, label='Rank Method', color='#3498DB', alpha=0.8)
        bars2 = ax2.bar(x + width/2, pct_raw, width, label='Percentage Method', color='#E74C3C', alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax2.set_ylabel('Index Value', fontsize=11)
        ax2.set_title('Q2.1: Quantitative Bias Indices Comparison', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics, fontsize=10)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
        plt.close()
    
    def plot_judge_intervention_impact(self, intervention_results: Dict[int, pd.DataFrame], save_path: str = 'fig4_judge_intervention.png'):
        """
        Figure 4: Judge intervention mechanism impact (Question 2.3)
        Save two separate subplots
        """
        if not intervention_results:
            print("No intervention data")
            return
        
        seasons = sorted(intervention_results.keys())
        change_counts = [intervention_results[s]['changed'].sum() if not intervention_results[s].empty else 0 
                        for s in seasons]
        total_weeks = [len(intervention_results[s]) if not intervention_results[s].empty else 0 
                      for s in seasons]
        
        # Left figure: Change counts per season
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        colors = ['#E74C3C' if c > 0 else '#27AE60' for c in change_counts]
        bars = ax1.bar(seasons, change_counts, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_xlabel('Season (Post-27)', fontsize=11)
        ax1.set_ylabel('Number of Changed Eliminations', fontsize=11)
        ax1.set_title('Q2.3: Judge Intervention Impact\n(How often judges changed the result)', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, count in zip(bars, change_counts):
            if count > 0:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(count)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        left_save_path = save_path.replace('.png', '_counts.png')
        plt.savefig(left_save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {left_save_path}")
        plt.close()
        
        # Right figure: Change rate trend
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        change_rates = [c/t if t > 0 else 0 for c, t in zip(change_counts, total_weeks)]
        ax2.plot(seasons, change_rates, 'o-', color='#8E44AD', linewidth=2.5, markersize=8)
        ax2.axhline(y=np.mean(change_rates), color='red', linestyle='--', 
                   label=f'Average Rate ({np.mean(change_rates):.1%})')
        ax2.fill_between(seasons, change_rates, alpha=0.3, color='#8E44AD')
        ax2.set_xlabel('Season', fontsize=11)
        ax2.set_ylabel('Intervention Rate', fontsize=11)
        ax2.set_title('Intervention Rate Trend', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        right_save_path = save_path.replace('.png', '_rate_trend.png')
        plt.savefig(right_save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {right_save_path}")
        plt.close()
    
    def plot_bobby_bones_detailed(self, save_path: str = 'fig5_bobby_bones_timeline.png'):
        """
        Figure 5: Bobby Bones detailed timeline (Question 2.2 key case)
        Save two separate subplots
        """
        season, celebrity = 27, 'Bobby Bones'
        season_data = self.analyzer.model.seasons_data.get(season)
        if season_data is None:
            return
        
        celeb_data = season_data[season_data['celebrity'] == celebrity].sort_values('week')
        
        # Top figure: Judge score vs fan ranking
        weeks = []
        judge_ranks = []
        fan_ranks = []
        
        for _, row in celeb_data.iterrows():
            week = int(row['week'])
            week_all = season_data[season_data['week'] == week]
            
            j_rank = (week_all['total_score'] > row['total_score']).sum() + 1
            f_rank = (week_all['vote_pct'] > row['vote_pct']).sum() + 1
            
            weeks.append(week)
            judge_ranks.append(j_rank)
            fan_ranks.append(f_rank)
        
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        ax1.plot(weeks, judge_ranks, 'o-', label='Judge Rank', color='#3498DB', 
                linewidth=2.5, markersize=10, markerfacecolor='white', markeredgewidth=2)
        ax1.plot(weeks, fan_ranks, 's-', label='Fan Rank', color='#E74C3C', 
                linewidth=2.5, markersize=10, markerfacecolor='white', markeredgewidth=2)
        
        # Fill difference areas
        for i in range(len(weeks)):
            if fan_ranks[i] < judge_ranks[i]:  # Better fan ranking (lower number)
                ax1.fill_between([weeks[i]-0.3, weeks[i]+0.3], [judge_ranks[i], judge_ranks[i]], 
                                [fan_ranks[i], fan_ranks[i]], alpha=0.3, color='green', 
                                label='Fan Advantage' if i == 0 else "")
        
        ax1.set_xlabel('Week', fontsize=11)
        ax1.set_ylabel('Rank (1=Best)', fontsize=11)
        ax1.set_title('Q2.2: Bobby Bones (S27) - Judge vs Fan Rank Divergence\n'
                     'Consistently low judge scores but high fan support', 
                     fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()  # Rank 1 at top
        
        plt.tight_layout()
        top_save_path = save_path.replace('.png', '_judge_fan_ranks.png')
        plt.savefig(top_save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {top_save_path}")
        plt.close()
        
        # Bottom figure: Combined ranking under both methods
        rank_positions = []
        pct_positions = []
        
        for week in weeks:
            week_all = season_data[season_data['week'] == week]
            week_active = week_all[week_all['total_score'] > 0]
            
            rank_scored = self.analyzer.model.calculate_combined_score(week_active.copy(), VotingMethod.RANK)
            pct_scored = self.analyzer.model.calculate_combined_score(week_active.copy(), VotingMethod.PERCENTAGE)
            
            rank_pos = rank_scored[rank_scored['celebrity'] == celebrity]['calculated_rank'].iloc[0]
            pct_pos = pct_scored[pct_scored['celebrity'] == celebrity]['calculated_rank'].iloc[0]
            
            rank_positions.append(rank_pos)
            pct_positions.append(pct_pos)
        
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        ax2.plot(weeks, rank_positions, 'o-', label='Rank Method Combined', color='#3498DB', linewidth=2.5, markersize=8)
        ax2.plot(weeks, pct_positions, 's-', label='Percentage Method Combined', color='#E74C3C', linewidth=2.5, markersize=8)
        
        # Mark elimination line (last place)
        max_pos = max(max(rank_positions), max(pct_positions))
        ax2.axhline(y=max_pos, color='black', linestyle='--', alpha=0.5, label='Elimination Line')
        
        # Mark differences
        for i, (r, p) in enumerate(zip(rank_positions, pct_positions)):
            if abs(r - p) > 1:
                ax2.annotate('', xy=(weeks[i], max(r, p)), xytext=(weeks[i], min(r, p)),
                           arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
                ax2.text(weeks[i], (r+p)/2, 'Gap', ha='center', fontsize=9, color='purple', fontweight='bold')
        
        ax2.set_xlabel('Week', fontsize=11)
        ax2.set_ylabel('Combined Rank (Higher=Worse)', fontsize=11)
        ax2.set_title('Combined Ranking: Rank Method would have eliminated earlier', fontsize=11, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        bottom_save_path = save_path.replace('.png', '_combined_ranks.png')
        plt.savefig(bottom_save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {bottom_save_path}")
        plt.close()


# ==================== Main Execution Function (Complete Solution Version) ====================
def run_complete_analysis(data_path: str = 'processed_weekly_data.csv'):
    """Execute complete analysis workflow - Answer all sub-questions for Question 2"""
    
    print("="*80)
    print("2026 MCM Problem C - Question 2 Complete Solution")
    print("DWTS Voting System Fairness Analysis")
    print("="*80)
    
    model = DWTSVotingModel(data_path)
    if not model.seasons_data:
        print("Error: Unable to load data")
        return
        
    analyzer = ControversyAnalyzer(model)
    bias_calc = BiasIndexCalculator(model)
    viz = EnhancedVisualizer(analyzer, bias_calc)
    
    # =====================================================================
    # Q2.1 Sub-question 1: Compare two methods across all seasons and bias determination
    # =====================================================================
    print("\n" + "="*80)
    print("【Question 2.1】Compare Two Methods (Rank vs Percentage)")
    print("Result differences and bias determination across all seasons")
    print("="*80)
    
    all_seasons = sorted(model.seasons_data.keys())
    all_seasons_stats = []
    total_diff_weeks = 0
    total_weeks = 0
    
    print("\n1. Seasonal result difference statistics:")
    print("-" * 60)
    print(f"{'Season':<8} {'Total Weeks':<12} {'Different':<10} {'Rate':<10} {'Status'}")
    print("-" * 60)
    
    for season in all_seasons:
        comp = analyzer.compare_methods_for_season(season)
        if not comp.empty:
            diff = comp['different'].sum()
            total = len(comp)
            total_diff_weeks += diff
            total_weeks += total
            rate = diff/total*100
            
            status = "HIGH" if rate > 30 else "MODERATE" if rate > 15 else "LOW"
            print(f"{season:<8} {total:<12} {diff:<10} {rate:>6.1f}%    {status}")
            
            all_seasons_stats.append({
                'season': season,
                'total_weeks': total,
                'diff_weeks': diff,
                'diff_pct': rate
            })
    
    overall_rate = total_diff_weeks/total_weeks*100
    print("-" * 60)
    print(f"{'TOTAL':<8} {total_weeks:<12} {total_diff_weeks:<10} {overall_rate:>6.1f}%")
    print("-" * 60)
    
    print(f"\n2. Difference analysis conclusion:")
    print(f"   • Average discrepancy rate across all {len(all_seasons)} seasons: {overall_rate:.1f}%")
    print(f"   • {len([s for s in all_seasons_stats if s['diff_pct'] > 20])} seasons have discrepancy rate exceeding 20%")
    print(f"   • Indicates method selection significantly affects competition results")
    
    # Calculate bias indices
    print(f"\n3. Bias quantification index analysis:")
    print("-" * 60)
    
    bias_records = []
    for season in all_seasons:
        season_data = model.seasons_data[season]
        weeks = sorted(season_data['week'].unique())
        for week in weeks:
            ewr = bias_calc.calculate_weekly_effective_weights(season, week)
            dctb = bias_calc.calculate_low_judge_high_fan_bias(season, week)
            fsi = bias_calc.calculate_fan_sensitivity_index(season, week)
            
            if ewr and fsi:
                bias_records.append({
                    'season': season,
                    'week': week,
                    'rank_ewr': ewr['rank_method_ratio'],
                    'pct_ewr': ewr['pct_method_ratio'],
                    'rank_fan_sens': fsi['rank_method_sensitivity'],
                    'pct_fan_sens': fsi['pct_method_sensitivity'],
                    'rank_elim_disc': dctb['rank_eliminates'] if dctb else 0,
                    'pct_elim_disc': dctb['pct_eliminates'] if dctb else 0
                })
    
    bias_df = pd.DataFrame(bias_records)
    
    if not bias_df.empty:
        avg_rank_ewr = bias_df['rank_ewr'].mean()
        avg_pct_ewr = bias_df['pct_ewr'].mean()
        avg_rank_sens = bias_df['rank_fan_sens'].mean()
        avg_pct_sens = bias_df['pct_fan_sens'].mean()
        
        valid_disc = bias_df[bias_df['rank_elim_disc'] + bias_df['pct_elim_disc'] > 0]
        total_rank_elim = valid_disc['rank_elim_disc'].sum()
        total_pct_elim = valid_disc['pct_elim_disc'].sum()
        total_disc_cases = len(valid_disc)
        
        print(f"\n   【Index 1: Effective Weight Ratio EWR】")
        print(f"   • Rank Method: {avg_rank_ewr:.3f} (>1 indicates judge-dominant)")
        print(f"   • Percentage Method: {avg_pct_ewr:.3f}")
        print(f"   • Conclusion: Rank Method judge weight is {avg_rank_ewr/avg_pct_ewr:.1f} times that of Percentage Method")
        
        print(f"\n   【Index 2: Discrepancy Contestant Treatment DCTB】")
        print(f"   • Sample size: {total_disc_cases} 'low judge score + high fan vote' cases")
        print(f"   • Rank Method elimination rate: {total_rank_elim/total_disc_cases*100:.1f}%")
        print(f"   • Percentage Method elimination rate: {total_pct_elim/total_disc_cases*100:.1f}%")
        print(f"   • Conclusion: Rank Method is stricter on discrepancy contestants (judge-biased)")
        
        print(f"\n   【Index 3: Fan Sensitivity Index FSI】")
        print(f"   • Rank Method sensitivity: {avg_rank_sens:.3f}")
        print(f"   • Percentage Method sensitivity: {avg_pct_sens:.3f}")
        print(f"   • Conclusion: Percentage Method is {avg_pct_sens/avg_rank_sens:.1f} times more sensitive to fan vote changes")
        
        print(f"\n   【Comprehensive Determination】")
        print(f"   >>> Rank Method: Judge-Biased")
        print(f"   >>> Percentage Method: Fan-Biased")
    
    # Generate charts 1 and 3
    viz.plot_all_seasons_comparison(all_seasons_stats)
    viz.plot_bias_metrics_comparison(bias_df)
    
    # =====================================================================
    # Q2.2 Sub-question 2: Controversial contestant specific analysis
    # =====================================================================
    print("\n" + "="*80)
    print("【Question 2.2】Controversial Contestants (Jerry Rice, Billy Ray Cyrus, Bristol Palin, Bobby Bones)")
    print("Analyze method selection impact on these contestants")
    print("="*80)
    
    controversies = analyzer.analyze_controversial_cases()
    
    for case_name, data in controversies.items():
        print(f"\nCase: {case_name}")
        print(f"Final Result: {data.get('actual_final_result', 'N/A')}")
        print(f"Analysis Conclusion: {data.get('conclusion', 'N/A')}")
        print(f"Key Weeks: Competed for {len(data.get('weekly_safety_margin', []))} weeks, "
              f"protected by Percentage Method for {data.get('weeks_saved_by_pct', 0)} weeks, "
              f"protected by Rank Method for {data.get('weeks_saved_by_rank', 0)} weeks")
        
        # List weeks with differences in detail
        diff_weeks = [w for w in data.get('weekly_safety_margin', []) if w['difference']]
        if diff_weeks:
            print(f"Method difference details:")
            for w in diff_weeks:
                status = "Rank eliminates/Pct survives" if not w['rank_would_survive'] and w['pct_would_survive'] else "Pct eliminates/Rank survives"
                print(f"   Week {w['week']}: {status} (Safety margin: Rank={w['rank_margin']}, Pct={w['pct_margin']})")
    
    # Generate charts 2 and 5
    viz.plot_controversial_cases_safety_margin()
    viz.plot_bobby_bones_detailed()
    
    # =====================================================================
    # Q2.3 Sub-question 3: Judge intervention mechanism impact (Post-Season 27)
    # =====================================================================
    print("\n" + "="*80)
    print("【Question 2.3】Judge Intervention Mechanism (Bottom 2 Judge Choice)")
    print("Analyze impact of judge choice mechanism introduced after Season 27")
    print("="*80)
    
    post_27 = [s for s in model.seasons_data.keys() if s >= 28]
    intervention_results = {}
    total_changes = 0
    total_intervention_weeks = 0
    
    if post_27:
        print(f"\nInvolved seasons: {len(post_27)} (Seasons {min(post_27)}-{max(post_27)})")
        print(f"\nIntervention statistics per season:")
        print("-" * 50)
        
        for season in post_27:
            comp = analyzer.analyze_judge_intervention(season)
            intervention_results[season] = comp
            if not comp.empty:
                changes = comp['changed'].sum()
                total_changes += changes
                total_intervention_weeks += len(comp)
                print(f"Season {season}: {changes}/{len(comp)} weeks changed ({changes/len(comp)*100:.1f}%)")
        
        print("-" * 50)
        print(f"Total: {total_changes}/{total_intervention_weeks} weeks ({total_changes/total_intervention_weeks*100:.1f}%)")
        
        print(f"\nImpact assessment:")
        print(f"• Judge intervention changes {total_changes/len(post_27):.1f} elimination results per season on average")
        print(f"• This mechanism effectively corrects 'surprise' eliminations caused by extreme fan votes")
        print(f"• Protects technically skilled contestants with temporarily low popularity")
        print(f"• Recommended to retain as a fairness safeguard mechanism")
        
        # Generate chart 4
        viz.plot_judge_intervention_impact(intervention_results)
    else:
        print("No data available for Season 28 and beyond in the dataset")
    
    # =====================================================================
    # Q2.4 Sub-question 4: Recommendation
    # =====================================================================
    print("\n" + "="*80)
    print("【Question 2.4】Method Recommendation and Future Suggestions")
    print("Recommendation based on the above analysis")
    print("="*80)
    
    print(f"""
Based on quantitative analysis of all {len(all_seasons)} seasons and 4 typical controversial cases, we recommend:

┌─────────────────────────────────────────────────────────────────────┐
│                  Recommended Solution: Hybrid Rank System           │
├─────────────────────────────────────────────────────────────────────┤
│ 1. Base Scoring: Use Rank Method                                    │
│    - Reason: Limits distortion from extreme fan votes (EWR index verification) │
│    - Advantage: Prevents pure popularity contestants (e.g., Bobby Bones)     │
│                from winning solely on fan votes                              │
│                                                                     │
│ 2. Correction Mechanism: Introduce Judge Choice (Bottom 2)          │
│    - Reason: Judges choose at elimination edge, protects skilled contestants│
│    - Advantage: Corrects average {total_changes/len(post_27) if post_27 else 'N/A'} "surprise" results per season│
│                                                                     │
│ 3. Not Recommended: Pure Percentage Method                          │
│    - Reason: Too lenient on discrepancy contestants ({total_pct_elim/total_disc_cases*100:.1f}% elimination rate vs Rank's {total_rank_elim/total_disc_cases*100:.1f}%) │
│    - Risk: Allows fan votes to completely override judge expertise  │
└─────────────────────────────────────────────────────────────────────┘

Fairness Argument:
• Professionalism-Participation Balance: Rank Method retains 50% judge weight, avoids "pure popularity champion"
• Extreme Case Protection: Judge Choice as safety valve, prevents premature elimination of best technical contestants  
• Historical Validation: S27 Bobby Bones case shows Percentage Method allows low-scoring high-popularity contestants to win

Implementation Recommendations:
1. Use Rank Method for base ranking calculation (judge 50% + fans 50%)
2. Each week, first identify Bottom 2, then judges vote to eliminate one pair
3. Final week may use pure judge scores or increase judge weight to 60-70%, ensuring champion's technical level
    """)
    
    print("\n" + "="*80)
    print("Analysis complete! All charts saved as PNG files.")
    print("="*80)  

if __name__ == "__main__":
    run_complete_analysis('processed_weekly_data.csv')