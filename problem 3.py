# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 20:16:04 2026

@author: m1871
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy import stats
from scipy.stats import chi2

class DanceCompetitionModel:
    def __init__(self, data_path):
        """
        Initialize the dance competition model
        
        Parameters:
        data_path: Path to data file
        """
        self.data = pd.read_csv(data_path)
        self.models = {}
        self.results = {}
        
    def preprocess_data(self):
        """
        Data preprocessing function
        """
        df = self.data.copy()
        
        # Create survival variable (1=advanced, 0=eliminated)
        df['survival'] = np.where(df['placement'] == 'Eliminated', 0, 1)
        
        # Handle categorical variables - Key modification: Ensure three core variables are correctly encoded
        categorical_vars = ['celebrity_industry', 'celebrity_homestate', 
                          'celebrity_homecountry/region', 'ballroom_partner']
        
        for var in categorical_vars:
            le = LabelEncoder()
            df[var + '_encoded'] = le.fit_transform(df[var].astype(str))
        
        # Log transform followers (if available)
        if 'celebrity_followers' in df.columns:
            df['log_followers'] = np.log(df['celebrity_followers'] + 1)
        
        # Create season dummy variables
        season_dummies = pd.get_dummies(df['season'], prefix='season')
        df = pd.concat([df, season_dummies], axis=1)
        
        # Create week dummy variables
        week_dummies = pd.get_dummies(df['week'], prefix='week')
        df = pd.concat([df, week_dummies], axis=1)
        
        self.processed_data = df
        return df
    
    def model_m1_judge_scores(self):
        """
        Model M1: Judge score influencing factors model (linear panel model)
        Core variables: ballroom_partner (professional dancer), celebrity_age_during_season (age), celebrity_industry (industry)
        """
        df = self.processed_data
        
        # Select valid judge score data
        judge_data = df[df['week_total_score'] > 0].copy()
        
        # Model formula - Fixed: Explicitly include ballroom_partner (professional dancer)
        formula = """
        week_total_score ~ 
        C(ballroom_partner_encoded) +
        celebrity_age_during_season + 
        C(celebrity_industry_encoded) + 
        C(season) + 
        C(week)
        """
        
        # Use mixed effects model (panel data)
        try:
            model = smf.mixedlm(formula, judge_data, groups=judge_data['celebrity_name'])
            result = model.fit()
            self.models['m1'] = result
            return result
        except:
            # If mixed model fails, use OLS
            model = smf.ols(formula, judge_data)
            result = model.fit()
            self.models['m1'] = result
            return result
    
    def model_m2_audience_votes(self):
        """
        Model M2: Audience vote influencing factors model (linear panel model)
        Core variables: ballroom_partner (professional dancer), celebrity_age_during_season (age), celebrity_industry (industry)
        """
        df = self.processed_data
        
        # Select valid audience vote data
        vote_data = df[df['vote_pct'] > 0].copy()
        
        # Model formula - Fixed: Explicitly include ballroom_partner (professional dancer)
        formula = """
        vote_pct ~ 
        C(ballroom_partner_encoded) +
        celebrity_age_during_season + 
        C(celebrity_industry_encoded) + 
        C(season) + 
        C(week)
        """
        
        try:
            model = smf.mixedlm(formula, vote_data, groups=vote_data['celebrity_name'])
            result = model.fit()
            self.models['m2'] = result
            return result
        except:
            model = smf.ols(formula, vote_data)
            result = model.fit()
            self.models['m2'] = result
            return result
    
    def model_m3_survival_probability(self):
        """
        Model M3: Weekly advancement probability model (panel Logit model)
        Core variables: ballroom_partner (professional dancer), celebrity_age_during_season (age), celebrity_industry (industry)
        """
        df = self.processed_data
        
        # Prepare panel data - Need to create time series for each contestant
        panel_data = []
        
        for name in df['celebrity_name'].unique():
            celeb_data = df[df['celebrity_name'] == name].sort_values('week')
            for i in range(1, len(celeb_data)):
                current_week = celeb_data.iloc[i]
                prev_week = celeb_data.iloc[i-1]
                
                panel_data.append({
                    'celebrity_name': name,
                    'week': current_week['week'],
                    'survival': current_week['survival'],
                    'prev_judge_score': prev_week['week_total_score'],
                    'prev_vote_pct': prev_week['vote_pct'],
                    'age': current_week['celebrity_age_during_season'],
                    'industry': current_week['celebrity_industry_encoded'],
                    'ballroom_partner': current_week['ballroom_partner_encoded'],  # Add professional dancer variable
                    'season': current_week['season']
                })
        
        panel_df = pd.DataFrame(panel_data)
        panel_df = panel_df.dropna()
        
        # Logit model formula - Fixed: Include professional dancer variable
        formula = """
        survival ~ 
        prev_judge_score + 
        prev_vote_pct + 
        C(ballroom_partner) +
        age + 
        C(industry) + 
        C(season) + 
        C(week)
        """
        
        try:
            model = smf.logit(formula, panel_df)
            result = model.fit()
            self.models['m3'] = result
            return result, panel_df
        except:
            # If Logit doesn't converge, use Probit
            model = smf.probit(formula, panel_df)
            result = model.fit()
            self.models['m3'] = result
            return result, panel_df
    
    def hypothesis_testing(self):
        """
        Hypothesis testing framework
        Test significance of three core variables (professional dancer, age, industry)
        """
        results = {}
        
        # 1. Main effect significance test
        if 'm1' in self.models:
            results['m1_main_effects'] = self.models['m1'].pvalues
        if 'm2' in self.models:
            results['m2_main_effects'] = self.models['m2'].pvalues
        if 'm3' in self.models:
            results['m3_main_effects'] = self.models['m3'].pvalues
        
        # 2. Judge vs audience influence difference test (for three core variables)
        comparison_results = self.compare_judge_audience_influence()
        results['judge_vs_audience'] = comparison_results
        
        return results
    
    def compare_judge_audience_influence(self):
        """
        Compare judge and audience influence differences
        For three core variables: ballroom_partner, celebrity_age_during_season, celebrity_industry
        """
        comparison_results = {}
        
        if 'm1' not in self.models or 'm2' not in self.models:
            return comparison_results
        
        m1_params = self.models['m1'].params
        m2_params = self.models['m2'].params
        m1_cov = self.models['m1'].cov_params()
        m2_cov = self.models['m2'].cov_params()
        
        # Define variables to compare (Note: categorical variables have multiple coefficients, compare overall effect)
        variables_to_compare = [
            'celebrity_age_during_season',  # Age
        ]
        
        # For categorical variables, compare differences in each set of coefficients
        # Extract coefficients for ballroom_partner and industry
        partner_vars_m1 = [var for var in m1_params.index if 'ballroom_partner' in var]
        partner_vars_m2 = [var for var in m2_params.index if 'ballroom_partner' in var]
        industry_vars_m1 = [var for var in m1_params.index if 'celebrity_industry' in var]
        industry_vars_m2 = [var for var in m2_params.index if 'celebrity_industry' in var]
        
        # Test age coefficient difference
        if 'celebrity_age_during_season' in m1_params.index and 'celebrity_age_during_season' in m2_params.index:
            beta_diff = m1_params['celebrity_age_during_season'] - m2_params['celebrity_age_during_season']
            se_diff = np.sqrt(m1_cov.loc['celebrity_age_during_season', 'celebrity_age_during_season'] + 
                             m2_cov.loc['celebrity_age_during_season', 'celebrity_age_during_season'])
            z_stat = beta_diff / se_diff
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            comparison_results['age'] = {
                'coefficient_diff': beta_diff,
                'z_statistic': z_stat,
                'p_value': p_value,
                'conclusion': 'Significantly different' if p_value < 0.05 else 'No significant difference'
            }
        
        # For professional dancer and industry variables, perform joint significance test (F-test or chi-square test)
        if len(partner_vars_m1) > 0 and len(partner_vars_m2) > 0:
            comparison_results['ballroom_partner'] = {
                'note': 'Categorical variable, recommend likelihood ratio test for nested models',
                'm1_significant': any(self.models['m1'].pvalues[var] < 0.05 for var in partner_vars_m1 if var in self.models['m1'].pvalues.index),
                'm2_significant': any(self.models['m2'].pvalues[var] < 0.05 for var in partner_vars_m2 if var in self.models['m2'].pvalues.index)
            }
        
        if len(industry_vars_m1) > 0 and len(industry_vars_m2) > 0:
            comparison_results['celebrity_industry'] = {
                'note': 'Categorical variable, recommend likelihood ratio test for nested models',
                'm1_significant': any(self.models['m1'].pvalues[var] < 0.05 for var in industry_vars_m1 if var in self.models['m1'].pvalues.index),
                'm2_significant': any(self.models['m2'].pvalues[var] < 0.05 for var in industry_vars_m2 if var in self.models['m2'].pvalues.index)
            }
        
        return comparison_results
    
    def likelihood_ratio_test(self, restricted_formula, unrestricted_formula, data, model_type='ols'):
        """
        Likelihood ratio test (for comparing coefficient constraints of specific variables in judge and audience models)
        """
        if model_type == 'ols':
            restricted = smf.ols(restricted_formula, data).fit()
            unrestricted = smf.ols(unrestricted_formula, data).fit()
        else:
            restricted = smf.logit(restricted_formula, data).fit(disp=0)
            unrestricted = smf.logit(unrestricted_formula, data).fit(disp=0)
        
        lr_stat = 2 * (unrestricted.llf - restricted.llf)
        df_diff = unrestricted.df_model - restricted.df_model
        p_value = 1 - chi2.cdf(lr_stat, df_diff)
        
        return {
            'lr_statistic': lr_stat,
            'df': df_diff,
            'p_value': p_value,
            'restricted_ll': restricted.llf,
            'unrestricted_ll': unrestricted.llf
        }
    
    def create_results_tables(self):
        """
        Create results tables
        Show effects of three core variables (ballroom_partner, celebrity_age_during_season, celebrity_industry)
        """
        tables = {}
        
        # Table 1: Analysis of three core influencing factors (judge vs audience)
        if 'm1' in self.models and 'm2' in self.models:
            table1_data = []
            m1_params = self.models['m1'].params
            m2_params = self.models['m2'].params
            m1_pvalues = self.models['m1'].pvalues
            m2_pvalues = self.models['m2'].pvalues
            m1_bse = self.models['m1'].bse
            m2_bse = self.models['m2'].bse
            
            # 1. Age (continuous variable)
            if 'celebrity_age_during_season' in m1_params.index:
                table1_data.append({
                    'Variable': 'Age',
                    'Variable Type': 'Continuous',
                    'Judge Score Effect': f"{m1_params['celebrity_age_during_season']:.4f} (±{m1_bse['celebrity_age_during_season']:.4f}) [p={m1_pvalues['celebrity_age_during_season']:.4f}]",
                    'Audience Vote Effect': f"{m2_params['celebrity_age_during_season']:.4f} (±{m2_bse['celebrity_age_during_season']:.4f}) [p={m2_pvalues['celebrity_age_during_season']:.4f}]"
                })
            
            # 2. Professional Dancer (categorical variable) - Show significance overview
            partner_vars_m1 = [var for var in m1_params.index if 'ballroom_partner' in var]
            partner_vars_m2 = [var for var in m2_params.index if 'ballroom_partner' in var]
            
            if len(partner_vars_m1) > 0:
                sig_partners_m1 = sum(1 for var in partner_vars_m1 if m1_pvalues[var] < 0.05)
                sig_partners_m2 = sum(1 for var in partner_vars_m2 if m2_pvalues[var] < 0.05)
                
                table1_data.append({
                    'Variable': 'Professional Dancer (Ballroom Partner)',
                    'Variable Type': 'Categorical',
                    'Judge Score Effect': f"{sig_partners_m1}/{len(partner_vars_m1)} groups significant (p<0.05)",
                    'Audience Vote Effect': f"{sig_partners_m2}/{len(partner_vars_m2)} groups significant (p<0.05)"
                })
            
            # 3. Industry (categorical variable)
            industry_vars_m1 = [var for var in m1_params.index if 'celebrity_industry' in var]
            industry_vars_m2 = [var for var in m2_params.index if 'celebrity_industry' in var]
            
            if len(industry_vars_m1) > 0:
                sig_industry_m1 = sum(1 for var in industry_vars_m1 if m1_pvalues[var] < 0.05)
                sig_industry_m2 = sum(1 for var in industry_vars_m2 if m2_pvalues[var] < 0.05)
                
                table1_data.append({
                    'Variable': 'Industry',
                    'Variable Type': 'Categorical',
                    'Judge Score Effect': f"{sig_industry_m1}/{len(industry_vars_m1)} groups significant (p<0.05)",
                    'Audience Vote Effect': f"{sig_industry_m2}/{len(industry_vars_m2)} groups significant (p<0.05)"
                })
            
            tables['table1_core_factors'] = pd.DataFrame(table1_data)
        
        # Table 2: Advancement model driving force test (includes three core variables)
        if 'm3' in self.models:
            table2_data = []
            m3_params = self.models['m3'].params
            m3_pvalues = self.models['m3'].pvalues
            m3_bse = self.models['m3'].bse
            
            # Lagged variables
            lag_vars = ['prev_judge_score', 'prev_vote_pct']
            for var in lag_vars:
                if var in m3_params.index:
                    or_value = np.exp(m3_params[var])
                    table2_data.append({
                        'Variable': var,
                        'Category': 'Lagged Performance',
                        'Coefficient Estimate': f"{m3_params[var]:.4f} (±{m3_bse[var]:.4f})",
                        'Odds Ratio (OR)': f"{or_value:.4f}",
                        'P-value': f"{m3_pvalues[var]:.4f}"
                    })
            
            # Three core variables
            core_vars = {
                'C(ballroom_partner)': 'Professional Dancer',
                'age': 'Age',
                'C(industry)': 'Industry'
            }
            
            for var_key, var_name in core_vars.items():
                matching_vars = [var for var in m3_params.index if var_key in var]
                if len(matching_vars) > 0:
                    # If categorical variable, calculate joint significance or show first coefficient
                    if var_key in ['C(ballroom_partner)', 'C(industry)']:
                        sig_count = sum(1 for var in matching_vars if m3_pvalues[var] < 0.05)
                        table2_data.append({
                            'Variable': var_name,
                            'Category': 'Celebrity Characteristics',
                            'Coefficient Estimate': f"{len(matching_vars)} categories, {sig_count} significant",
                            'Odds Ratio (OR)': '-',
                            'P-value': f'Significance rate: {sig_count/len(matching_vars):.1%}'
                        })
                    else:  # Age
                        var = matching_vars[0]
                        or_value = np.exp(m3_params[var])
                        table2_data.append({
                            'Variable': var_name,
                            'Category': 'Celebrity Characteristics',
                            'Coefficient Estimate': f"{m3_params[var]:.4f} (±{m3_bse[var]:.4f})",
                            'Odds Ratio (OR)': f"{or_value:.4f}",
                            'P-value': f"{m3_pvalues[var]:.4f}"
                        })
            
            tables['table2_survival_drivers'] = pd.DataFrame(table2_data)
        
        # Table 3: Judge vs audience influence difference test results
        comparison_results = self.compare_judge_audience_influence()
        if comparison_results:
            table3_data = []
            for var, result in comparison_results.items():
                if 'coefficient_diff' in result:  # Continuous variable (age)
                    table3_data.append({
                        'Variable': var,
                        'Test Method': 'Wald Test (Coefficient Difference)',
                        'Statistic': f"Z = {result['z_statistic']:.4f}",
                        'P-value': f"{result['p_value']:.4f}",
                        'Conclusion': result['conclusion']
                    })
                else:  # Categorical variable
                    table3_data.append({
                        'Variable': var,
                        'Test Method': 'Individual Significance Comparison',
                        'Statistic': '-',
                        'P-value': f"Judge significant: {result.get('m1_significant', 'N/A')}, Audience significant: {result.get('m2_significant', 'N/A')}",
                        'Conclusion': result.get('note', 'Requires likelihood ratio test')
                    })
            
            tables['table3_comparison'] = pd.DataFrame(table3_data)
        
        return tables
    
    def visualize_results(self):
        """
        Results visualization - three independent charts
        Highlight three core influencing factors
        """
        # Set global font sizes
        title_fontsize = 18
        label_fontsize = 14
        tick_fontsize = 12
        legend_fontsize = 11
        
        plt.rcParams['font.size'] = label_fontsize
        plt.rcParams['axes.titlesize'] = title_fontsize
        plt.rcParams['axes.labelsize'] = label_fontsize
        plt.rcParams['xtick.labelsize'] = tick_fontsize
        plt.rcParams['ytick.labelsize'] = tick_fontsize
        plt.rcParams['legend.fontsize'] = legend_fontsize
        
        # ==================== Figure 1: 3D Scatter Plot (Age vs Score vs Vote) ====================
        print("Generating Figure 1: 3D Scatter Plot...")
        fig1 = plt.figure(figsize=(12, 9))
        ax1 = fig1.add_subplot(111, projection='3d')
        
        # Prepare data
        scatter_data = self.processed_data[
            (self.processed_data['week_total_score'] > 0) & 
            (self.processed_data['vote_pct'] > 0) & 
            (self.processed_data['celebrity_age_during_season'].notna())
        ].copy()
        
        if len(scatter_data) > 0:
            # Limit number of data points for better visualization
            if len(scatter_data) > 2000:
                scatter_data = scatter_data.sample(n=2000, random_state=42)
            
            x = scatter_data['celebrity_age_during_season']
            y = scatter_data['week_total_score']
            z = scatter_data['vote_pct']
            
            # Color by age groups
            age_bins = pd.cut(x, bins=5, labels=['≤20', '21-25', '26-30', '31-35', '>35'])
            
            # Plot scatter
            scatter = ax1.scatter(x, y, z, c=pd.factorize(age_bins)[0], 
                                cmap='viridis', alpha=0.6, s=30, edgecolors='w', linewidth=0.3)
            
            # Set axis labels
            ax1.set_xlabel('Age', fontsize=label_fontsize, fontweight='bold')
            ax1.set_ylabel('Judge Score', fontsize=label_fontsize, fontweight='bold')
            ax1.set_zlabel('Audience Vote Percentage (%)', fontsize=label_fontsize, fontweight='bold')
            ax1.set_title('3D Relationship: Age - Judge Score - Audience Vote', fontsize=title_fontsize, 
                         fontweight='bold', pad=20)
            
            # Add legend
            colors = plt.cm.viridis(np.linspace(0, 1, len(age_bins.unique())))
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=colors[i], markersize=8,
                                        label=f'{label}') 
                             for i, label in enumerate(age_bins.unique())]
            ax1.legend(handles=legend_elements, loc='upper left', 
                      bbox_to_anchor=(0.02, 0.98), ncol=5)
            
            # Add grid
            ax1.grid(True, alpha=0.3)
            
            # Set viewing angle
            ax1.view_init(elev=20, azim=45)
        else:
            ax1.text2D(0.5, 0.5, 'No data available', transform=ax1.transAxes, 
                      ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        plt.savefig('dance_competition_3d_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Figure 1 saved to: dance_competition_3d_scatter.png")
        
        # ==================== Figure 2: Professional Dancer Score Distribution (Violin Plot) ====================
        print("Generating Figure 2: Professional Dancer Influence Analysis...")
        fig2, ax2 = plt.subplots(figsize=(14, 7))
        
        violin_data = self.processed_data[
            self.processed_data['week_total_score'] > 0
        ].copy()
        
        if len(violin_data) > 0:
            # Calculate score statistics for each professional dancer, select top performers
            partner_stats = violin_data.groupby('ballroom_partner')['week_total_score'].agg(['mean', 'count'])
            partner_stats = partner_stats[partner_stats['count'] >= 3]  # At least 3 ratings
            top_partners = partner_stats.nlargest(10, 'mean').index
            
            plot_data = violin_data[violin_data['ballroom_partner'].isin(top_partners)]
            
            # Plot violin plot
            parts = ax2.violinplot(
                [plot_data[plot_data['ballroom_partner'] == partner]['week_total_score'].values 
                 for partner in top_partners],
                positions=range(len(top_partners)),
                showmeans=True,
                showmedians=True,
                widths=0.8,
                bw_method='silverman'
            )
            
            # Set violin plot style - Assign different colors to each violin
            colors = plt.cm.tab20(np.linspace(0, 1, len(top_partners)))
            
            for idx, pc in enumerate(parts['bodies']):
                # Use tab20 color palette sequentially
                pc.set_facecolor(colors[idx])
                pc.set_alpha(0.7)
                # Set border color to darker shade of same color for better visual distinction
                pc.set_edgecolor(plt.cm.tab20(np.linspace(0, 1, len(top_partners))[idx] * 0.6))
                pc.set_linewidth(1.5)
            
            parts['cmeans'].set_color('red')
            parts['cmeans'].set_linewidth(2)
            parts['cmedians'].set_color('green')
            parts['cmedians'].set_linewidth(2)
            parts['cmaxes'].set_color('navy')
            parts['cmins'].set_color('navy')
            parts['cbars'].set_color('navy')
            
            # Set axes
            ax2.set_xticks(range(len(top_partners)))
            ax2.set_xticklabels([partner[:12] + '...' if len(partner) > 12 else partner 
                               for partner in top_partners], rotation=30, ha='right')
            ax2.set_xlabel('Professional Dancer Name', fontsize=label_fontsize, fontweight='bold')
            ax2.set_ylabel('Score', fontsize=label_fontsize, fontweight='bold')
            ax2.set_title('Influence Distribution of Professional Dancers on Scores', fontsize=title_fontsize, 
                         fontweight='bold', pad=15)
            
            # Add grid
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.set_axisbelow(True)
            
            # Add legend description
            legend_elements = [
                plt.Line2D([0], [0], color='red', lw=2, label='Mean'),
                plt.Line2D([0], [0], color='green', lw=2, label='Median')
            ]
            ax2.legend(handles=legend_elements, loc='upper right')
        else:
            ax2.text(0.5, 0.5, 'No data available', transform=ax2.transAxes, 
                    ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        plt.savefig('dance_competition_violin_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Figure 2 saved to: dance_competition_violin_plot.png")
        
        # ==================== Figure 3: Industry Characteristics Radar Chart ====================
        print("Generating Figure 3: Industry Characteristics Analysis...")
        fig3 = plt.figure(figsize=(10, 10))
        ax3 = fig3.add_subplot(111, projection='polar')
        
        radar_data = self.processed_data.copy()
        
        # Calculate three metrics for each industry: average score, average vote percentage, advancement rate
        industry_stats = radar_data.groupby('celebrity_industry').agg({
            'week_total_score': 'mean',
            'vote_pct': 'mean',
            'survival': 'mean'  # Advancement rate
        }).dropna()
        
        if len(industry_stats) > 0:
            # Select major industries for display
            industry_counts = radar_data['celebrity_industry'].value_counts()
            top_industries = industry_counts.nlargest(8).index
            industry_stats = industry_stats[industry_stats.index.isin(top_industries)]
            
            # Normalize data to 0-1 range
            stats_normalized = industry_stats.copy()
            for col in stats_normalized.columns:
                min_val = stats_normalized[col].min()
                max_val = stats_normalized[col].max()
                if max_val > min_val:
                    stats_normalized[col] = (stats_normalized[col] - min_val) / (max_val - min_val)
                else:
                    stats_normalized[col] = 0.5
            
            # Prepare radar chart data
            categories = ['Average Score', 'Average Vote Percentage', 'Advancement Rate']
            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the shape
            
            # Plot radar lines for each industry
            colors = plt.cm.Set2(np.linspace(0, 1, len(industry_stats)))
            
            for idx, (industry, row) in enumerate(stats_normalized.iterrows()):
                values = row.values.tolist()
                values += values[:1]  # Close the shape
                
                ax3.plot(angles, values, 'o-', linewidth=2, 
                        color=colors[idx], label=industry)
                ax3.fill(angles, values, alpha=0.15, color=colors[idx])
            
            # Set axis labels
            ax3.set_xticks(angles[:-1])
            ax3.set_xticklabels(categories, fontsize=13, fontweight='bold')
            ax3.set_ylim(0, 1)
            ax3.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax3.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=11)
            ax3.set_title('Industry Impact on Comprehensive Performance', fontsize=title_fontsize, 
                         fontweight='bold', pad=20)
            
            # Add grid
            ax3.grid(True, alpha=0.3)
            
            # Add legend (place outside)
            ax3.legend(loc='upper left', bbox_to_anchor=(-0.35, 1.0), 
                      fontsize=legend_fontsize, framealpha=0.95)
        else:
            ax3.text(0, 0, 'No data available', transform=ax3.transAxes, 
                    ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        plt.savefig('dance_competition_radar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Figure 3 saved to: dance_competition_radar_chart.png")

    
    def comprehensive_analysis(self):
        """
        Comprehensive analysis: Run all models and analyses
        """
        print("Starting data preprocessing...")
        self.preprocess_data()
        
        print("Training judge score model (M1)...")
        m1_result = self.model_m1_judge_scores()
        
        print("Training audience vote model (M2)...")
        m2_result = self.model_m2_audience_votes()
        
        print("Training advancement probability model (M3)...")
        m3_result, panel_data = self.model_m3_survival_probability()
        
        print("Performing hypothesis testing...")
        hypothesis_results = self.hypothesis_testing()
        
        print("Generating results tables...")
        tables = self.create_results_tables()
        
        print("Generating visualization results...")
        self.visualize_results()
        
        # Save results
        self.save_results()
        
        return {
            'models': self.models,
            'hypothesis_results': hypothesis_results,
            'tables': tables
        }
    
    def save_results(self):
        """Save results to files"""
        # Save model summaries
        with open('model_results_summary.txt', 'w', encoding='utf-8') as f:
            for name, model in self.models.items():
                f.write(f"=== {name.upper()} MODEL RESULTS ===\n")
                f.write(str(model.summary()))
                f.write("\n\n")
        
        # Save processed data
        self.processed_data.to_csv('processed_dance_data.csv', index=False, encoding='utf-8')
        
        print("Results saved to files")

# Usage example
def main():
    # Initialize model
    model = DanceCompetitionModel('dance_data_complete.csv')
    
    # Perform comprehensive analysis
    results = model.comprehensive_analysis()
    
    # Print key results
    print("\n" + "="*50)
    print("KEY ANALYSIS RESULTS")
    print("="*50)
    
    # Display table results
    if 'table1_core_factors' in results['tables']:
        print("\nTable 1: Core Influencing Factors Analysis (Professional Dancer, Age, Industry)")
        print(results['tables']['table1_core_factors'].to_string(index=False))
    
    if 'table2_survival_drivers' in results['tables']:
        print("\nTable 2: Advancement Decision Factors Analysis")
        print(results['tables']['table2_survival_drivers'].to_string(index=False))
    
    if 'table3_comparison' in results['tables']:
        print("\nTable 3: Judge vs Audience Influence Difference Tests")
        print(results['tables']['table3_comparison'].to_string(index=False))
    
    # Model goodness-of-fit
    print("\nModel Goodness-of-Fit:")
    for name, model in results['models'].items():
        if hasattr(model, 'rsquared'):
            print(f"{name}: R² = {model.rsquared:.4f}")
        if hasattr(model, 'prsquared'):
            print(f"{name}: Pseudo R² = {model.prsquared:.4f}")
    
    # Key findings summary
    print("\n" + "="*50)
    print("CORE FINDINGS SUMMARY")
    print("="*50)
    
    # Check influence differences of three core variables
    hyp_results = results['hypothesis_results']
    
    if 'judge_vs_audience' in hyp_results:
        comparison = hyp_results['judge_vs_audience']
        print("\n【JUDGE VS AUDIENCE INFLUENCE DIFFERENCE TESTS】")
        for var, res in comparison.items():
            if 'conclusion' in res:
                print(f"• {var}: {res['conclusion']} (Z={res.get('z_statistic', 'N/A'):.4f}, p={res.get('p_value', 'N/A'):.4f})")
            else:
                print(f"• {var}: Judge model significant={res.get('m1_significant', 'N/A')}, Audience model significant={res.get('m2_significant', 'N/A')}")
    
    print("\n【VARIABLE IMPORTANCE】")
    print("• Professional Dancer (ballroom_partner): Examines differential impact of different professional dancers on scores/votes")
    print("• Age (celebrity_age_during_season): Examines linear/non-linear impact of age on performance")
    print("• Industry (celebrity_industry): Examines whether industry background provides competitive advantage")

if __name__ == "__main__":
    main()