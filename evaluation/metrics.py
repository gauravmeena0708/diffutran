import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns
import os

class EvaluationFramework:
    def __init__(self, real_data, synthetic_data, task_type='classification', target_col=None):
        """
        Args:
            real_data: pd.DataFrame
            synthetic_data: pd.DataFrame
            task_type: 'classification' or 'regression'
            target_col: Target column name for Utility metrics
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.task_type = task_type
        self.target_col = target_col
        
        # Identify columns
        self.numerical_cols = real_data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = real_data.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Precompute encodings for utility/privacy
        self._encode_data()

    def _encode_data(self):
        """Label encode categorical columns for metric calculation"""
        self.real_encoded = self.real_data.copy()
        self.syn_encoded = self.synthetic_data.copy()
        
        self.encoders = {}
        for col in self.categorical_cols:
            le = LabelEncoder()
            # Fit on both to handle all categories
            all_cats = pd.concat([self.real_data[col], self.synthetic_data[col]]).astype(str).unique()
            le.fit(all_cats)
            self.encoders[col] = le
            
            self.real_encoded[col] = self.real_encoded[col].apply(lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1)
            self.syn_encoded[col] = self.syn_encoded[col].apply(lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1)

    # =================
    # 1. FIDELITY
    # =================
    def evaluate_fidelity(self):
        """
        Compute Statistical Similarity (JSD for categorical, WD for numerical)
        """
        metrics = {}
        
        # Numerical: Wasserstein Distance
        for col in self.numerical_cols:
            wd = wasserstein_distance(self.real_data[col], self.synthetic_data[col])
            metrics[f'WD_{col}'] = wd
            
        # Categorical: Jensen-Shannon Divergence
        for col in self.categorical_cols:
            # Compute probabilities
            full_cats = self.encoders[col].classes_
            p = self.real_data[col].value_counts(normalize=True).reindex(full_cats, fill_value=0).values
            q = self.synthetic_data[col].value_counts(normalize=True).reindex(full_cats, fill_value=0).values
            
            jsd = jensenshannon(p, q)
            metrics[f'JSD_{col}'] = jsd
            
        return metrics

    def plot_correlation_difference(self, save_path=None):
        """Plot difference in correlation matrices"""
        real_corr = self.real_encoded.corr()
        syn_corr = self.syn_encoded.corr()
        diff = np.abs(real_corr - syn_corr)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(diff, annot=False, cmap='viridis')
        plt.title("Correlation Difference (Absolute)")
        if save_path:
            plt.savefig(save_path)
            plt.close()
        return np.linalg.norm(diff, 'fro') # Frobenius norm of difference

    # =================
    # 2. UTILITY
    # =================
    def evaluate_utility(self):
        """
        Compute TSTR (Train on Synthetic, Test on Real) and TRTR (Baseline)
        """
        if self.target_col is None or self.target_col not in self.real_data.columns:
            return {}

        X_real = self.real_encoded.drop(columns=[self.target_col])
        y_real = self.real_data[self.target_col] # Use original real data
        
        X_syn = self.syn_encoded.drop(columns=[self.target_col])
        y_syn = self.synthetic_data[self.target_col] # Use original synthetic data
        
        # Align Target Types (Encoding)
        if self.task_type == 'classification':
             le_target = LabelEncoder()
             # Fit on combined to ensure coverage
             all_targets = pd.concat([y_real, y_syn]).astype(str)
             le_target.fit(all_targets)
             
             y_real = le_target.transform(y_real.astype(str))
             y_syn = le_target.transform(y_syn.astype(str))
        
        # Split Real for Test set
        from sklearn.model_selection import train_test_split
        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
            X_real, y_real, test_size=0.2, random_state=42
        )
        
        # Initialize Models
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
            
        # TRTR
        model_trtr.fit(X_train_real, y_train_real)
        pred_trtr = model_trtr.predict(X_test_real)
        if self.task_type == 'classification':
             score_trtr = metric_func(y_test_real, pred_trtr, average=metric_average)
        else:
             score_trtr = metric_func(y_test_real, pred_trtr)
        
        # TSTR
        model_tstr.fit(X_syn, y_syn)
        pred_tstr = model_tstr.predict(X_test_real)
        if self.task_type == 'classification':
             score_tstr = metric_func(y_test_real, pred_tstr, average=metric_average)
        else:
             score_tstr = metric_func(y_test_real, pred_tstr)
        
        metric_name = 'F1' if self.task_type == 'classification' else 'R2'
        
        return {
            f'TRTR_{metric_name}': score_trtr,
            f'TSTR_{metric_name}': score_tstr,
            f'Utility_Ratio': score_tstr / (score_trtr + 1e-8)
        }

    # =================
    # 3. GLOBAL UTILITY / FIDELITY (Discriminator)
    # =================
    def evaluate_discriminator_measure(self):
        """
        Train a classifier to distinguish Real (0) vs Synthetic (1).
        Accuracy ~ 0.5 implies high global utility/fidelity (indistinguishable).
        Accuracy ~ 1.0 implies low global utility (easy to distinguish).
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
        
        # Prepare Mix
        X_real = self.real_encoded.copy()
        X_syn = self.syn_encoded.copy()
        
        X_real['is_synthetic'] = 0
        X_syn['is_synthetic'] = 1
        
        combined = pd.concat([X_real, X_syn], ignore_index=True)
        X = combined.drop(columns=['is_synthetic'])
        y = combined['is_synthetic']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train Discriminator
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        # Evaluate
        preds = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)
        
        # Accuracy
        acc = accuracy_score(y_test, clf.predict(X_test))
        
        return {
            'Discriminator_AUC': auc,
            'Discriminator_Accuracy': acc,
            'Global_Indistinguishability': 1.0 - max(0.0, (acc - 0.5) * 2) # Normalized 0-1 (1 is best)
        }

    # =================
    # 4. PRIVACY
    # =================
    def evaluate_privacy(self, sample_size=1000):
        """
        Compute Distance to Closest Record (DCR)
        Warning: Computationally expensive for large datasets
        """
        # Downsample for speed if needed
        real_sample = self.real_encoded.sample(min(len(self.real_encoded), sample_size), random_state=42)
        syn_sample = self.syn_encoded.sample(min(len(self.syn_encoded), sample_size), random_state=42)
            
        # Compute distances from Synthetic to nearest Real
        # Using euclidean distance on encoded data (standard for DCR)
        # Better: scaled to [0,1] first? stick to raw for now as simplest baseline
        
        dists = pairwise_distances(syn_sample, real_sample, metric='euclidean')
        min_dists = np.min(dists, axis=1) # For each synthetic, closest real
        
        dcr_mean = np.mean(min_dists)
        dcr_5th = np.percentile(min_dists, 5) # 5th percentile (risk of overfitting)
        
        return {
            'DCR_Mean': dcr_mean,
            'DCR_5th_Percentile': dcr_5th
        }

    # =================
    # 5. RELATIONSHIP REPORT
    # =================
    def generate_relationship_report(self, data_type='synthetic'):
        """
        Generate a natural language description of relationships in the data.
        Args:
            data_type: 'synthetic' or 'real'
        """
        if data_type == 'synthetic':
            df = self.synthetic_data
        else:
            df = self.real_data

        report = [f"Relationship Analysis Report ({data_type.capitalize()} Data):"]
        report.append("-" * 40)

        # 1. Numerical Correlations
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr()
            # Iterate over upper triangle
            for i in range(len(numerical_cols)):
                for j in range(i+1, len(numerical_cols)):
                    col1 = numerical_cols[i]
                    col2 = numerical_cols[j]
                    corr = corr_matrix.iloc[i, j]

                    if abs(corr) > 0.7:
                        strength = "strongly" if abs(corr) > 0.85 else "moderately"
                        direction = "increases" if corr > 0 else "decreases"
                        report.append(f"- {col1} and {col2} are {strength} {'positively' if corr > 0 else 'negatively'} correlated ({corr:.2f}). As {col1} increases, {col2} tends to {direction}.")

        # 2. Categorical-Categorical Associations
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        for i in range(len(categorical_cols)):
            for j in range(len(categorical_cols)):
                if i == j: continue
                col1 = categorical_cols[i]
                col2 = categorical_cols[j]

                # Contingency Table
                try:
                    contingency = pd.crosstab(df[col1], df[col2], normalize='index')

                    # Check for strong conditional probabilities
                    for val1 in contingency.index:
                        for val2 in contingency.columns:
                            prob = contingency.loc[val1, val2]
                            if prob > 0.8:
                                report.append(f"- When {col1} is '{val1}', {col2} is likely '{val2}' (probability: {prob:.2f}).")
                except Exception as e:
                    pass # Skip if error in crosstab

        # 3. Categorical-Numerical Differences
        for cat_col in categorical_cols:
            for num_col in numerical_cols:
                try:
                    groups = df.groupby(cat_col)[num_col]
                    means = groups.mean()

                    if len(means) > 0 and means.std() > 0 and (means.max() - means.min() > means.std() * 2):
                         highest_cat = means.idxmax()
                         lowest_cat = means.idxmin()
                         report.append(f"- {num_col} varies by {cat_col}. It is highest when {cat_col} is '{highest_cat}' ({means[highest_cat]:.2f}) and lowest when {cat_col} is '{lowest_cat}' ({means[lowest_cat]:.2f}).")
                except Exception as e:
                    pass

        if len(report) == 2:
            report.append("No strong relationships found.")

        return "\n".join(report)
