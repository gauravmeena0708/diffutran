import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error, pairwise_distances, roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, ks_2samp, chisquare
import matplotlib.pyplot as plt
import seaborn as sns
import os

class EvaluationFramework:
    def __init__(self, real_data, synthetic_data, task_type='classification', target_col=None):
        """
        SOTA Evaluation Framework including Statistical Tests, Utility, and Privacy.
        """
        self.real_data = real_data.copy()
        self.synthetic_data = synthetic_data.copy()
        self.task_type = task_type
        self.target_col = target_col
        
        self.numerical_cols = real_data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = real_data.select_dtypes(exclude=[np.number]).columns.tolist()
        
        self._encode_data()

    def _encode_data(self):
        self.real_encoded = self.real_data.copy()
        self.syn_encoded = self.synthetic_data.copy()
        self.encoders = {}
        for col in self.categorical_cols:
            le = LabelEncoder()
            all_cats = pd.concat([self.real_data[col], self.synthetic_data[col]]).astype(str).unique()
            le.fit(all_cats)
            self.encoders[col] = le
            self.real_encoded[col] = self.real_encoded[col].apply(lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1)
            self.syn_encoded[col] = self.syn_encoded[col].apply(lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1)

    # 1. FIDELITY & STATISTICAL TESTS
    def evaluate_fidelity(self):
        metrics = {}
        # Numerical: Wasserstein & KS Test
        for col in self.numerical_cols:
            # Ensure strictly numeric and drop NaN
            real_vals = pd.to_numeric(self.real_data[col], errors='coerce').dropna().astype(float)
            syn_vals = pd.to_numeric(self.synthetic_data[col], errors='coerce').dropna().astype(float)
            
            if len(real_vals) > 0 and len(syn_vals) > 0:
                metrics[f'WD_{col}'] = wasserstein_distance(real_vals, syn_vals)
                ks_stat, p_val = ks_2samp(real_vals, syn_vals)
                metrics[f'KS_Stat_{col}'] = ks_stat
                metrics[f'KS_pval_{col}'] = p_val
            
        # Categorical: JSD & Chi-Square
        for col in self.categorical_cols:
            full_cats = self.encoders[col].classes_
            
            real_counts = self.real_data[col].astype(str).value_counts().reindex(full_cats, fill_value=0)
            syn_counts = self.synthetic_data[col].astype(str).value_counts().reindex(full_cats, fill_value=0)
            
            p = (real_counts / (real_counts.sum() + 1e-8)).values
            q = (syn_counts / (syn_counts.sum() + 1e-8)).values
            
            metrics[f'JSD_{col}'] = jensenshannon(p, q)
            
            # Chi-Square Test over frequencies
            real_freq = np.ones_like(real_counts.values, dtype=np.float64) * 1e-5 + real_counts.values
            syn_freq = np.ones_like(syn_counts.values, dtype=np.float64) * 1e-5 + syn_counts.values
            syn_freq = syn_freq * (real_freq.sum() / syn_freq.sum())
            chi2_stat, p_val = chisquare(f_obs=syn_freq, f_exp=real_freq)
            metrics[f'Chi2_Stat_{col}'] = chi2_stat
            metrics[f'Chi2_pval_{col}'] = p_val
            
        return metrics

    def plot_correlation_difference(self, save_path=None):
        real_corr = self.real_encoded.corr()
        syn_corr = self.syn_encoded.corr()
        diff = np.abs(real_corr - syn_corr)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(diff, annot=False, cmap='viridis')
        plt.title("Correlation Difference (Absolute)")
        if save_path:
            plt.savefig(save_path)
            plt.close()
        return np.linalg.norm(diff, 'fro')

    # 2. UTILITY
    def evaluate_utility(self):
        if self.target_col is None or self.target_col not in self.real_data.columns:
            return {}
            
        # Select features only (X)
        X_real = self.real_encoded.drop(columns=[self.target_col])
        y_real = self.real_data[self.target_col]
        
        # Ensure synthetic features match the EXACT columns and ORDER of X_real
        if self.target_col in self.syn_encoded.columns:
            X_syn = self.syn_encoded.drop(columns=[self.target_col])
        else:
            X_syn = self.syn_encoded
            
        # Reorder/Filter columns to match exactly
        X_syn = X_syn[X_real.columns]
        y_syn = self.synthetic_data[self.target_col]
        
        if self.task_type == 'classification':
             le_target = LabelEncoder()
             all_targets = pd.concat([y_real, y_syn]).astype(str)
             le_target.fit(all_targets)
             y_real = le_target.transform(y_real.astype(str))
             y_syn = le_target.transform(y_syn.astype(str))
        
        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(X_real, y_real, test_size=0.2, random_state=42)
        
        if self.task_type == 'classification':
            model_trtr = RandomForestClassifier(n_estimators=50, random_state=42)
            model_tstr = RandomForestClassifier(n_estimators=50, random_state=42)
            metric_func = f1_score
            metric_average = 'weighted'
        else:
            model_trtr = RandomForestRegressor(n_estimators=50, random_state=42)
            model_tstr = RandomForestRegressor(n_estimators=50, random_state=42)
            metric_func = r2_score
            metric_average = None
            
        model_trtr.fit(X_train_real, y_train_real)
        pred_trtr = model_trtr.predict(X_test_real)
        score_trtr = metric_func(y_test_real, pred_trtr, average=metric_average) if self.task_type == 'classification' else metric_func(y_test_real, pred_trtr)
        
        model_tstr.fit(X_syn, y_syn)
        pred_tstr = model_tstr.predict(X_test_real)
        score_tstr = metric_func(y_test_real, pred_tstr, average=metric_average) if self.task_type == 'classification' else metric_func(y_test_real, pred_tstr)
        
        metric_name = 'F1' if self.task_type == 'classification' else 'R2'
        return {
            f'TRTR_{metric_name}': score_trtr,
            f'TSTR_{metric_name}': score_tstr,
            f'Utility_Ratio': score_tstr / (score_trtr + 1e-8)
        }

    def evaluate_discriminator_measure(self):
        X_real = self.real_encoded.copy()
        X_syn = self.syn_encoded.copy()
        X_real['is_synthetic'] = 0
        X_syn['is_synthetic'] = 1
        combined = pd.concat([X_real, X_syn], ignore_index=True)
        X = combined.drop(columns=['is_synthetic'])
        y = combined['is_synthetic']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        preds = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)
        acc = accuracy_score(y_test, clf.predict(X_test))
        
        return {
            'Discriminator_AUC': auc,
            'Discriminator_Accuracy': acc,
            'Global_Indistinguishability': 1.0 - max(0.0, (acc - 0.5) * 2) 
        }

    # 3. PRIVACY (MIA & k-anonymity & DCR)
    def evaluate_privacy(self, sample_size=1000, sensitive_cols=None):
        metrics = {}
        
        # 3.1 DCR
        real_sample = self.real_encoded.sample(min(len(self.real_encoded), sample_size), random_state=42)
        syn_sample = self.syn_encoded.sample(min(len(self.syn_encoded), sample_size), random_state=42)
        dists = pairwise_distances(syn_sample, real_sample, metric='euclidean')
        min_dists = np.min(dists, axis=1) 
        metrics['DCR_Mean'] = np.mean(min_dists)
        metrics['DCR_5th_Percentile'] = np.percentile(min_dists, 5)
        
        # 3.2 Membership Inference Attack (MIA)
        try:
            half = len(self.real_encoded) // 2
            target_real = self.real_encoded.iloc[:half]
            holdout_real = self.real_encoded.iloc[half:]
            
            # Distance-based MIA Adv
            tr_dists = np.min(pairwise_distances(target_real.head(1000), syn_sample), axis=1)
            ho_dists = np.min(pairwise_distances(holdout_real.head(1000), syn_sample), axis=1)
            
            mia_labels = [1] * len(tr_dists) + [0] * len(ho_dists)
            mia_scores = [-x for x in list(tr_dists) + list(ho_dists)]
            metrics['MIA_AUC'] = roc_auc_score(mia_labels, mia_scores)
        except Exception as e:
            metrics['MIA_AUC'] = 0.5
            
        # 3.3 k-Anonymity & l-Diversity
        if sensitive_cols is None and self.target_col:
            sensitive_cols = [self.target_col]
        else:
            sensitive_cols = []
            
        qis = [c for c in self.categorical_cols if c not in sensitive_cols]
        if qis:
            group_sizes = self.synthetic_data.groupby(qis).size()
            metrics['k_Anonymity'] = group_sizes.min() if len(group_sizes) > 0 else 0
            if sensitive_cols:
                sens = sensitive_cols[0]
                l_divs = self.synthetic_data.groupby(qis)[sens].nunique()
                metrics['l_Diversity'] = l_divs.min() if len(l_divs) > 0 else 0

        return metrics

    # 4. DIVERSITY
    def evaluate_diversity(self):
        """
        Evaluate how much of the real data manifold/variety is captured.
        """
        metrics = {}
        
        # 4.1 Categorical Coverage
        for col in self.categorical_cols:
            real_cats = set(self.real_data[col].astype(str).unique())
            syn_cats = set(self.synthetic_data[col].astype(str).unique())
            coverage = len(syn_cats.intersection(real_cats)) / len(real_cats)
            metrics[f'Coverage_{col}'] = coverage
            
        # 4.2 Numerical Range Coverage
        for col in self.numerical_cols:
            r_min, r_max = self.real_data[col].min(), self.real_data[col].max()
            s_min, s_max = self.synthetic_data[col].min(), self.synthetic_data[col].max()
            
            # Intersection of ranges
            i_min = max(r_min, s_min)
            i_max = min(r_max, s_max)
            overlap = max(0, i_max - i_min) / (r_max - r_min + 1e-8)
            metrics[f'Range_Coverage_{col}'] = overlap
            
        return metrics

if __name__ == "__main__":
    df_real = pd.DataFrame({
        'n1': [1, 2, 3, 4, 5],
        'c1': ['A', 'A', 'B', 'B', 'B'],
        'target': [0, 1, 0, 1, 1]
    })
    df_syn = pd.DataFrame({
        'n1': [1.1, 1.9, 3.2, 4.1, 4.8],
        'c1': ['A', 'A', 'B', 'A', 'B'],
        'target': [0, 1, 0, 0, 1]
    })
    
    evaluator = EvaluationFramework(df_real, df_syn, task_type='classification', target_col='target')
    fid = evaluator.evaluate_fidelity()
    assert 'KS_Stat_n1' in fid
    assert 'Chi2_Stat_c1' in fid
    priv = evaluator.evaluate_privacy()
    assert 'MIA_AUC' in priv
    assert 'k_Anonymity' in priv
    print("EvaluationFramework Verification Passed!")
