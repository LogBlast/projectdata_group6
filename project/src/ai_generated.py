from scipy import stats


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

class RandomForestAnalyzer:
    def __init__(self, df, target_column, test_size=0.2, random_state=42):
        """
        Initialize the Random Forest Analyzer
        
        Parameters:
        df (pd.DataFrame): Your cleaned and preprocessed dataset
        target_column (str): Name of the target column
        test_size (float): Proportion of data for testing (default: 0.2)
        random_state (int): Random state for reproducibility (default: 42)
        """
        self.df = df.copy()
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        
        # Initialize attributes
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.y_pred = None
        self.feature_importances = None
        
    def prepare_data(self):
        """Separate features and target, then split the data"""
        print("Preparing data...")
        
        # Separate features and target
        self.X = self.df.drop(columns=[self.target_column])
        self.y = self.df[self.target_column]
        
        print(f"Features shape: {self.X.shape}")
        print(f"Target shape: {self.y.shape}")
        print(f"Target distribution:\n{self.y.value_counts()}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=self.y
        )
        
        print(f"\nTraining set: {self.X_train.shape}")
        print(f"Testing set: {self.X_test.shape}")
        
    def train_model(self, n_estimators=100, **kwargs):
        """
        Train the Random Forest model
        
        Parameters:
        n_estimators (int): Number of trees in the forest
        **kwargs: Additional parameters for RandomForestClassifier
        """
        print("\nTraining Random Forest model...")
        
        # Default parameters
        default_params = {
            'n_estimators': n_estimators,
            'random_state': self.random_state,
            'class_weight': 'balanced',
            'n_jobs': -1
        }
        
        # Update with any custom parameters
        default_params.update(kwargs)
        
        # Create and train the model
        self.model = RandomForestClassifier(**default_params)
        self.model.fit(self.X_train, self.y_train)
        
        print("Model training completed!")
        
        # Make predictions
        self.y_pred = self.model.predict(self.X_test)
        
        # Cross-validation
        print("\nPerforming 5-fold cross-validation...")
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, 
                                   cv=5, scoring='accuracy')
        print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
    def evaluate_model(self):
        """Evaluate the model performance"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Basic metrics
        accuracy = accuracy_score(self.y_test, self.y_pred)
        print(f"Test Accuracy: {accuracy:.3f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.y_pred))
        
        # ROC-AUC for binary classification
        if len(np.unique(self.y)) == 2:
            try:
                auc = roc_auc_score(self.y_test, self.y_pred)
                print(f"ROC-AUC Score: {auc:.3f}")
            except:
                pass
    
    def plot_confusion_matrix(self, figsize=(8, 6)):
        """Plot confusion matrix"""
        plt.figure(figsize=figsize)
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=np.unique(self.y), 
                   yticklabels=np.unique(self.y))
        plt.title('Random Forest Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
    
    def analyze_feature_importance(self, top_n=15, figsize=(12, 8)):
        """Analyze and plot feature importances"""
        print(f"\nAnalyzing feature importance (top {top_n})...")
        
        # Get feature importances
        importances = self.model.feature_importances_
        feature_names = self.X.columns
        
        # Create DataFrame
        self.feature_importances = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        print(f"Top {top_n} most important features:")
        print(self.feature_importances.head(top_n))
        
        # Plot feature importances
        plt.figure(figsize=figsize)
        top_features = self.feature_importances.head(top_n)
        sns.barplot(data=top_features, x='Importance', y='Feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importances - Random Forest')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()
        
        return self.feature_importances
    
    def run_complete_analysis(self, n_estimators=100, top_features=15, **kwargs):
        """
        Run the complete analysis pipeline
        
        Parameters:
        n_estimators (int): Number of trees
        top_features (int): Number of top features to display
        **kwargs: Additional model parameters
        """
        print("="*60)
        print("RANDOM FOREST COMPLETE ANALYSIS")
        print("="*60)
        
        # Step 1: Prepare data
        self.prepare_data()
        
        # Step 2: Train model
        self.train_model(n_estimators=n_estimators, **kwargs)
        
        # Step 3: Evaluate model
        self.evaluate_model()
        
        # Step 4: Plot confusion matrix
        self.plot_confusion_matrix()
        
        # Step 5: Analyze feature importance
        feature_importance_df = self.analyze_feature_importance(top_n=top_features)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED!")
        print("="*60)
        
        return {
            'model': self.model,
            'predictions': self.y_pred,
            'feature_importances': feature_importance_df,
            'test_accuracy': accuracy_score(self.y_test, self.y_pred)
        }

class AI_Gen :
    def en_cours(data: pd.DataFrame):
        """
        Complete EDA analysis for gaming engagement dataset
        """
        
        # Basic data exploration
        print("="*80)
        print("GAMING ENGAGEMENT DATASET - COMPLETE ANALYSIS")
        print("="*80)
        
        print("First 4 rows:\n", data.head(4))
        print("\nDataset shape:", data.shape)
        print("\nData types:\n", data.dtypes)
        print("\nMissing values:\n", data.isnull().sum())
        print("\nDuplicate rows:", data.duplicated().sum())
        print("\nTop 5 players by session duration:\n", data.sort_values(by='AvgSessionDurationMinutes', ascending=False).head())
        
        # Target variable analysis
        print("\n" + "="*50)
        print("TARGET VARIABLE ANALYSIS")
        print("="*50)
        engagement_counts = data['EngagementLevel'].value_counts()
        engagement_props = data['EngagementLevel'].value_counts(normalize=True)
        print("Engagement Level Distribution:")
        for level in ['High', 'Medium', 'Low']:
            if level in engagement_counts.index:
                print(f"{level}: {engagement_counts[level]} ({engagement_props[level]:.2%})")
        
        # Check for class imbalance
        max_prop = engagement_props.max()
        min_prop = engagement_props.min()
        imbalance_ratio = max_prop / min_prop
        print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}")
        if imbalance_ratio > 2:
            print("⚠️  Significant class imbalance detected - consider stratified sampling")
        else:
            print("✓ Classes are relatively balanced")

        # Features to analyze
        features = [
            "Age", "Gender", "Location", "GameGenre", "PlayTimeHours",
            "InGamePurchases", "GameDifficulty", "SessionsPerWeek",
            "AvgSessionDurationMinutes", "PlayerLevel", "AchievementsUnlocked"
        ]
        
        # Define feature types for better analysis
        numerical_features = ["Age", "PlayTimeHours", "SessionsPerWeek", "AvgSessionDurationMinutes", "PlayerLevel", "AchievementsUnlocked"]
        categorical_features = ["Gender", "Location", "GameGenre", "GameDifficulty"]
        binary_features = ["InGamePurchases"]
        
        # Set up colors
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue
        engagement_colors = {'High': colors[0], 'Medium': colors[1], 'Low': colors[2]}

        # Set up the main plot grid (your original visualization)
        print("\n" + "="*50)
        print("FEATURE OVERVIEW GRID")
        print("="*50)
        
        fig, axs = plt.subplots(6, 2, figsize=(20, 30))
        axs = axs.flatten()

        for i, feature in enumerate(features):
            ax = axs[i]
            if data[feature].dtype == 'object' or data[feature].dtype.name == 'category':
                sns.countplot(data=data, x=feature, hue='EngagementLevel', ax=ax, 
                            palette=engagement_colors, hue_order=['High', 'Medium', 'Low'])
            else:
                sns.boxplot(data=data, x='EngagementLevel', y=feature, ax=ax,
                        palette=engagement_colors, order=['High', 'Medium', 'Low'])
            ax.set_title(f'{feature} vs EngagementLevel', fontsize=12, weight='bold')
            ax.set_xlabel(feature, fontsize=10)
            ax.set_ylabel('Value', fontsize=10)

            # Improve readability for categorical axes
            if data[feature].dtype == 'object' and data[feature].nunique() > 4:
                ax.tick_params(axis='x', rotation=45)

        # Hide any unused subplots
        for j in range(len(features), len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.show()
        
        # Now detailed analysis for each feature type
        print("\n" + "="*50)
        print("DETAILED NUMERICAL FEATURES ANALYSIS")
        print("="*50)
        
        for feature in numerical_features:
            print(f"\n{'-'*40}")
            print(f"ANALYZING: {feature}")
            print(f"{'-'*40}")
            
            # Create comprehensive plots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
            
            # Box plot
            sns.boxplot(data=data, x='EngagementLevel', y=feature, ax=ax1,
                    palette=engagement_colors, order=['High', 'Medium', 'Low'])
            ax1.set_title(f'{feature} Box Plot by Engagement', fontsize=12, weight='bold')
            
            # Violin plot
            sns.violinplot(data=data, x='EngagementLevel', y=feature, ax=ax2,
                        palette=engagement_colors, order=['High', 'Medium', 'Low'])
            ax2.set_title(f'{feature} Distribution Shape', fontsize=12, weight='bold')
            
            # Histogram overlay
            for level in ['High', 'Medium', 'Low']:
                subset = data[data['EngagementLevel'] == level][feature]
                ax3.hist(subset, alpha=0.6, label=level, bins=20, 
                        color=engagement_colors[level], density=True)
            ax3.set_title(f'{feature} Density Distribution', fontsize=12, weight='bold')
            ax3.set_xlabel(feature)
            ax3.set_ylabel('Density')
            ax3.legend()
            
            # Statistical summary table
            stats_summary = data.groupby('EngagementLevel')[feature].agg(['count', 'mean', 'median', 'std', 'min', 'max']).round(2)
            ax4.axis('tight')
            ax4.axis('off')
            table = ax4.table(cellText=stats_summary.values, 
                            rowLabels=stats_summary.index,
                            colLabels=stats_summary.columns,
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            ax4.set_title(f'{feature} Statistics', fontsize=12, weight='bold')
            
            plt.tight_layout()
            plt.show()
            
            # Statistical tests
            high_vals = data[data['EngagementLevel'] == 'High'][feature]
            medium_vals = data[data['EngagementLevel'] == 'Medium'][feature]
            low_vals = data[data['EngagementLevel'] == 'Low'][feature]
            
            # ANOVA
            f_stat, p_value = stats.f_oneway(high_vals, medium_vals, low_vals)
            print(f"ANOVA Results:")
            print(f"  F-statistic: {f_stat:.4f}")
            print(f"  p-value: {p_value:.6f}")
            print(f"  Result: {'Significant difference' if p_value < 0.05 else 'No significant difference'}")
            
            # Pairwise t-tests
            print(f"\nPairwise t-tests:")
            pairs = [('High', 'Medium'), ('High', 'Low'), ('Medium', 'Low')]
            for pair in pairs:
                val1 = data[data['EngagementLevel'] == pair[0]][feature]
                val2 = data[data['EngagementLevel'] == pair[1]][feature]
                t_stat, p_val = stats.ttest_ind(val1, val2)
                print(f"  {pair[0]} vs {pair[1]}: t={t_stat:.3f}, p={p_val:.6f}")
        
        print("\n" + "="*50)
        print("DETAILED CATEGORICAL FEATURES ANALYSIS")
        print("="*50)
        
        for feature in categorical_features:
            print(f"\n{'-'*40}")
            print(f"ANALYZING: {feature}")
            print(f"{'-'*40}")
            
            unique_vals = data[feature].nunique()
            print(f"Unique values: {unique_vals}")
            print(f"Values: {data[feature].unique()}")
            
            # Create plots based on number of categories
            if unique_vals <= 5:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
            else:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
            
            # Count plot
            sns.countplot(data=data, x=feature, hue='EngagementLevel', ax=ax1,
                        palette=engagement_colors, hue_order=['High', 'Medium', 'Low'])
            ax1.set_title(f'{feature} Count by Engagement', fontsize=12, weight='bold')
            ax1.legend(title='Engagement Level')
            if unique_vals > 5:
                ax1.tick_params(axis='x', rotation=45)
            
            # Proportion plot
            prop_data = pd.crosstab(data[feature], data['EngagementLevel'], normalize='index')
            prop_data = prop_data.reindex(columns=['High', 'Medium', 'Low'], fill_value=0)
            prop_data.plot(kind='bar', ax=ax2, color=[engagement_colors[col] for col in prop_data.columns])
            ax2.set_title(f'{feature} Engagement Proportions', fontsize=12, weight='bold')
            ax2.set_ylabel('Proportion')
            ax2.legend(title='Engagement Level')
            ax2.tick_params(axis='x', rotation=45)
            
            # Heatmap
            crosstab = pd.crosstab(data[feature], data['EngagementLevel'])
            sns.heatmap(crosstab, annot=True, fmt='d', ax=ax3, cmap='YlOrRd')
            ax3.set_title(f'{feature} vs Engagement Heatmap', fontsize=12, weight='bold')
            
            # Pie chart for feature distribution
            feature_counts = data[feature].value_counts()
            ax4.pie(feature_counts.values, labels=feature_counts.index, autopct='%1.1f%%', startangle=90)
            ax4.set_title(f'{feature} Overall Distribution', fontsize=12, weight='bold')
            
            plt.tight_layout()
            plt.show()
            
            # Chi-square test
            contingency_table = pd.crosstab(data[feature], data['EngagementLevel'])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            
            print(f"Chi-square Test Results:")
            print(f"  Chi-square statistic: {chi2:.4f}")
            print(f"  p-value: {p_value:.6f}")
            print(f"  Degrees of freedom: {dof}")
            print(f"  Result: {'Significant association' if p_value < 0.05 else 'No significant association'}")
            
            # Cramer's V (effect size)
            n = contingency_table.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
            print(f"  Cramer's V (effect size): {cramers_v:.4f}")
            
            print(f"\nCross-tabulation:")
            print(contingency_table)
            print(f"\nProportions within each {feature}:")
            print(prop_data.round(3))
        
        print("\n" + "="*50)
        print("BINARY FEATURES ANALYSIS")
        print("="*50)
        
        for feature in binary_features:
            print(f"\n{'-'*40}")
            print(f"ANALYZING: {feature}")
            print(f"{'-'*40}")
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
            
            # Count plot
            sns.countplot(data=data, x=feature, hue='EngagementLevel', ax=ax1,
                        palette=engagement_colors, hue_order=['High', 'Medium', 'Low'])
            ax1.set_title(f'{feature} Count by Engagement', fontsize=12, weight='bold')
            ax1.legend(title='Engagement Level')
            
            # Proportion within engagement levels
            prop_engagement = pd.crosstab(data['EngagementLevel'], data[feature], normalize='index')
            prop_engagement.plot(kind='bar', ax=ax2, color=['lightcoral', 'lightgreen'])
            ax2.set_title(f'{feature} Proportion within Engagement Levels', fontsize=12, weight='bold')
            ax2.set_ylabel('Proportion')
            ax2.tick_params(axis='x', rotation=0)
            
            # Proportion within binary categories
            prop_binary = pd.crosstab(data[feature], data['EngagementLevel'], normalize='index')
            prop_binary = prop_binary.reindex(columns=['High', 'Medium', 'Low'], fill_value=0)
            prop_binary.plot(kind='bar', ax=ax3, color=[engagement_colors[col] for col in prop_binary.columns])
            ax3.set_title(f'Engagement Distribution within {feature} Groups', fontsize=12, weight='bold')
            ax3.set_ylabel('Proportion')
            ax3.tick_params(axis='x', rotation=0)
            
            # Summary statistics
            summary_stats = []
            for level in ['High', 'Medium', 'Low']:
                subset = data[data['EngagementLevel'] == level]
                purchase_rate = subset[feature].mean()
                count = len(subset)
                summary_stats.append([level, count, purchase_rate])
            
            ax4.axis('tight')
            ax4.axis('off')
            table = ax4.table(cellText=summary_stats,
                            colLabels=['Engagement', 'Count', f'{feature} Rate'],
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1.2, 1.5)
            ax4.set_title(f'{feature} Summary Statistics', fontsize=12, weight='bold')
            
            plt.tight_layout()
            plt.show()
            
            # Statistical analysis
            contingency_table = pd.crosstab(data[feature], data['EngagementLevel'])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            
            print(f"Statistical Analysis:")
            print(f"  Chi-square: {chi2:.4f}, p-value: {p_value:.6f}")
            print(f"  Result: {'Significant association' if p_value < 0.05 else 'No significant association'}")
            
            print(f"\nBinary Feature Rates by Engagement:")
            for level in ['High', 'Medium', 'Low']:
                rate = data[data['EngagementLevel'] == level][feature].mean()
                print(f"  {level}: {rate:.3f}")
        
        print("\n" + "="*50)
        print("CORRELATION ANALYSIS")
        print("="*50)
        
        # Encode categorical variables for correlation
        data_encoded = data.copy()
        from sklearn.preprocessing import LabelEncoder
        
        categorical_cols = ['Gender', 'Location', 'GameGenre', 'GameDifficulty', 'EngagementLevel']
        label_encoders = {}
        
        for col in categorical_cols:
            if col in data_encoded.columns:
                le = LabelEncoder()
                data_encoded[col] = le.fit_transform(data_encoded[col])
                label_encoders[col] = le
        
        # Create correlation matrix
        correlation_matrix = data_encoded.drop(['PlayerID'], axis=1, errors='ignore').corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, mask=mask, fmt='.2f', linewidths=0.5)
        plt.title('Feature Correlation Matrix', fontsize=16, weight='bold')
        plt.tight_layout()
        plt.show()
        
        # Target correlations
        if 'EngagementLevel' in correlation_matrix.columns:
            target_correlations = correlation_matrix['EngagementLevel'].abs().sort_values(ascending=False)
            print("\nFeatures ranked by absolute correlation with EngagementLevel:")
            for feature, corr in target_correlations.items():
                if feature != 'EngagementLevel':
                    print(f"  {feature}: {corr:.4f}")
        
        # Feature importance insights
        print("\n" + "="*50)
        print("KEY INSIGHTS AND RECOMMENDATIONS")
        print("="*50)
        
        print("1. DATA QUALITY:")
        print(f"   - Dataset size: {data.shape[0]} rows, {data.shape[1]} columns")
        print(f"   - Missing values: {data.isnull().sum().sum()}")
        print(f"   - Duplicates: {data.duplicated().sum()}")
        print(f"   - Class balance: {imbalance_ratio:.2f} ratio")
        
        print("\n2. FEATURE ENGINEERING OPPORTUNITIES:")
        print("   - Create interaction features (Age × GameGenre)")
        print("   - Calculate total weekly playtime (SessionsPerWeek × AvgSessionDurationMinutes)")
        print("   - Achievement rate (AchievementsUnlocked / PlayerLevel)")
        print("   - Player intensity score (PlayTimeHours × SessionsPerWeek)")
        
        print("\n3. PREPROCESSING RECOMMENDATIONS:")
        if imbalance_ratio > 2:
            print("   - Use stratified sampling for train/test splits")
            print("   - Consider SMOTE or class weights for model training")
        print("   - Scale numerical features (Age, PlayTimeHours, etc.)")
        print("   - One-hot encode categorical features with low cardinality")
        print("   - Consider target encoding for high-cardinality features")
        
        print("\n4. MODEL SELECTION GUIDANCE:")
        print("   - Multi-class classification problem (High/Medium/Low)")
        print("   - Consider: Random Forest, XGBoost, Neural Networks")
        print("   - Use stratified cross-validation")
        print("   - Focus on balanced accuracy and F1-scores")
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE")
        print("="*50)
        
        return data_encoded, correlation_matrix, label_encoders