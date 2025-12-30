import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from wordcloud import WordCloud
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.inspection import permutation_importance
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

class ResultVisualizer:
    def __init__(self, figsize=(10, 6), style='seaborn'):
        self.figsize = figsize
        self.style = style
        # Ensure output directory exists to avoid errors when saving images
        try:
            os.makedirs('results', exist_ok=True)
        except Exception:
            pass

        self.set_style()
    
    """set plotting style"""
    def set_style(self):
        # Try to use seaborn themes first; fall back to matplotlib defaults if unavailable
        try:
            if isinstance(self.style, str) and self.style.lower().startswith('seaborn'):
                import seaborn as sns
                try:
                    sns.set_theme()
                except Exception:
                    sns.set()   # for older seaborn versions
            else:
                plt.style.use(self.style)
        except Exception:   # if specified style fails, try seaborn then matplotlib default
            try:
                import seaborn as sns
                try:
                    sns.set_theme()
                except Exception:
                    sns.set()
            except Exception:
                try:
                    plt.style.use('default')
                except Exception:
                    pass

        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        # Set seaborn color palette (if seaborn is available)
        try:
            import seaborn as sns
            sns.set_palette("husl")
        except Exception:
            pass
        
    """Plot data distributions"""
    def plot_data_distribution(self, data):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Sentiment label distribution
        label_counts = data['label'].value_counts()
        colors = ['#ff6b6b', '#51cf66']  # red - negative, green - positive

        axes[0, 0].pie(label_counts.values, labels=['Negative', 'Positive'], 
                  colors=colors, autopct='%1.1f%%', startangle=90)  
        axes[0, 0].set_title('Sentiment Label Distribution')
        
        # Text length distribution
        if 'text_length' in data.columns:
            sns.histplot(data['text_length'], bins=50, ax=axes[0, 1], kde=True)
            axes[0, 1].set_title('Comment Length Distribution')
            axes[0, 1].set_xlabel('Characters')
            axes[0, 1].set_ylabel('Frequency')
        
        # Compare lengths of positive vs negative samples
        if 'text_length' in data.columns:
            sns.boxplot(x='label', y='text_length', data=data, ax=axes[1, 0])
            axes[1, 0].set_title('Positive vs Negative Comment Lengths')
            axes[1, 0].set_xlabel('Sentiment Label')
            axes[1, 0].set_ylabel('Characters')
            axes[1, 0].set_xticks([0, 1])
            axes[1, 0].set_xticklabels(['Negative', 'Positive'])
        
        # Word count distribution
        if 'word_count' in data.columns:
            sns.histplot(data['word_count'], bins=30, ax=axes[1, 1], kde=True)
            axes[1, 1].set_title('Word Count Distribution')
            axes[1, 1].set_xlabel('Word Count')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('results/data_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    """Plot model performance comparisons"""
    def plot_model_comparison(self, results_df):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        results_df['Val Acc'] = results_df['Val Acc'].astype(float)
        results_df['Val F1'] = results_df['Val F1'].astype(float)
        
        # Validation accuracy comparison
        models = results_df['Model']
        val_acc = results_df['Val Acc']
        
        axes[0, 0].barh(models, val_acc, color='skyblue')
        axes[0, 0].set_xlabel('Validation Accuracy')
        axes[0, 0].set_title('Model Validation Accuracy Comparison')
        axes[0, 0].invert_yaxis()
        
        for i, v in enumerate(val_acc): # corrected line
            axes[0, 0].text(v + 0.001, i, f'{v:.3f}', va='center')
        
        # Validation F1 comparison
        val_f1 = results_df['Val F1']
        
        axes[0, 1].barh(models, val_f1, color='lightgreen')
        axes[0, 1].set_xlabel('Validation F1')
        axes[0, 1].set_title('Model Validation F1 Comparison')
        axes[0, 1].invert_yaxis()
        
        for i, v in enumerate(val_f1):
            axes[0, 1].text(v + 0.001, i, f'{v:.3f}', va='center')
        
        # Train vs validation accuracy comparison
        if 'Train Acc' in results_df.columns:
            train_acc = results_df['Train Acc'].astype(float)
            
            x = np.arange(len(models))
            width = 0.35
            
            axes[1, 0].bar(x - width/2, train_acc, width, label='Train Accuracy', color='lightblue')
            axes[1, 0].bar(x + width/2, val_acc, width, label='Validation Accuracy', color='orange')
            axes[1, 0].set_xlabel('Model')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].set_title('Train vs Validation Accuracy Comparison')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
            axes[1, 0].legend()
            axes[1, 0].set_ylim([0, 1])
        
        # Model performance heatmap
        performance_metrics = results_df[['Val Acc', 'Val F1']].astype(float)
        
        sns.heatmap(performance_metrics.T, annot=True, fmt='.3f', 
                   cmap='YlOrRd', ax=axes[1, 1], cbar_kws={'label': '分数'})
        axes[1, 1].set_title('Model Performance Heatmap')
        axes[1, 1].set_xlabel('Model')
        
        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    """Plot confusion matrix"""
    def plot_confusion_matrix(self, cm, model_name):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Predicted Negative', 'Predicted Positive'],
               yticklabels=['Actual Negative', 'Actual Positive'],
               ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_title(f'{model_name} - confusion matrix')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        # Compute metrics
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / np.sum(cm)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Add metrics text
        metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1 Score: {f1:.3f}'
        ax.text(0.5, -0.15, metrics_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               horizontalalignment='center')
        
        plt.tight_layout()
        plt.savefig(f'results/confusion_matrix_{model_name}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
    """Plot ROC curve"""
    def plot_roc_curve(self, y_true, y_prob, model_name):
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC Curve (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{model_name} - ROC Curve')
        ax.legend(loc="lower right")
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results/roc_curve_{model_name}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return roc_auc
    
    """Plot precision-recall curve"""
    def plot_precision_recall_curve(self, y_true, y_prob, model_name):
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot PR curve
        ax.plot(recall, precision, color='darkgreen', lw=2)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'{model_name} - Precision-Recall Curve')
        
        ax.grid(True, alpha=0.3)    # add grid
        
        # Compute AP score
        ap_score = auc(recall, precision)
        ax.text(0.05, 0.95, f'AP Score: {ap_score:.3f}', 
               transform=ax.transAxes, fontsize=12,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'results/pr_curve_{model_name}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return ap_score
        
    """Generate a word cloud"""
    def generate_wordcloud(self, texts, title='Word Cloud', max_words=100):
        if isinstance(texts, list):
            text = ' '.join([' '.join(words) for words in texts])
        else:
            text = ' '.join(texts)
        
        if not text.strip():
            print("Text content is empty; cannot generate word cloud")
            return
        
        wordcloud = WordCloud(
            font_path='simhei.ttf',
            width=800,
            height=400,
            background_color='white',
            max_words=max_words,
            colormap='viridis',
            contour_width=1,
            contour_color='steelblue'
        ).generate(text)
        
        # Render word cloud
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16)
        
        plt.tight_layout()
        filename = title.replace(' ', '_').replace(':', '')
        plt.savefig(f'results/{filename}.png', dpi=300, bbox_inches='tight')
        plt.show()

    """Plot feature importance"""
    def plot_feature_importance(self, model, feature_names, top_n=20, X=None, y=None, n_repeats=10, random_state=42):
        """Plot feature importance.

        Fallback to permutation importance if model has no `feature_importances_` or `coef_`.
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            # Try permutation importance as a model-agnostic fallback
            if X is not None and y is not None:
                try:
                    print('Computing permutation importance as fallback...')
                    r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
                    importances = r.importances_mean
                except Exception as e:
                    print(f'Permutation importance failed: {e}')
                    print('Model does not support feature importance analysis')
                    return
            else:
                print('Model does not support feature importance analysis')
                print('Provide validation data (X, y) to compute permutation importance as fallback')
                return
        
        # Ensure number of features match
        if len(importances) != len(feature_names):
            print(f"Feature count mismatch: importances={len(importances)}, feature_names={len(feature_names)}")
            return
        
        indices = np.argsort(importances)[-top_n:]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw horizontal bar chart
        y_pos = np.arange(len(indices))
        ax.barh(y_pos, importances[indices])
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.invert_yaxis()  # 最重要的显示在顶部
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top-{top_n} Important Features')
        
        plt.tight_layout()
        plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    """Plot error analysis"""
    def plot_error_analysis(self, y_true, y_pred, texts):
        # Find misclassified samples
        errors = np.where(y_true != y_pred)[0]
        
        if len(errors) == 0:
            print("No misclassified samples")
            return
        
        # Analyze error types
        false_positives = []  # actual negative but predicted positive
        false_negatives = []  # actual positive but predicted negative
        
        for idx in errors:
            if y_true[idx] == 0 and y_pred[idx] == 1:
                false_positives.append(idx)
            elif y_true[idx] == 1 and y_pred[idx] == 0:
                false_negatives.append(idx)
        
        # Count error types
        error_counts = {
            'False Positive': len(false_positives),
            'False Negative': len(false_negatives)
        }
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Error type pie chart
        axes[0].pie(error_counts.values(), labels=error_counts.keys(),
               autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
        axes[0].set_title('Error Type Distribution')
        
        # Example misclassified samples
        error_examples = []
        if false_positives:
            fp_example = texts[false_positives[0]][:50] + '...' if len(texts[false_positives[0]]) > 50 else texts[false_positives[0]]
            error_examples.append(f"False Positive: {fp_example}")
        
        if false_negatives:
            fn_example = texts[false_negatives[0]][:50] + '...' if len(texts[false_negatives[0]]) > 50 else texts[false_negatives[0]]
            error_examples.append(f"False Negative: {fn_example}")
        
        axes[1].text(0.1, 0.5, '\n'.join(error_examples), 
                    transform=axes[1].transAxes, fontsize=10,
                    verticalalignment='center')
        axes[1].set_title('Error Sample Examples')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/error_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    """Create an integrated dashboard"""
    def create_dashboard(self, data, results, best_model_info):
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Sentiment Distribution', 'Model Performance Comparison', 'Confusion Matrix',
                          'ROC Curve', 'Feature Importance', 'Word Cloud'),
            specs=[[{'type': 'pie'}, {'type': 'bar'}, {'type': 'heatmap'}],
                  [{'type': 'scatter'}, {'type': 'bar'}, {'type': 'image'}]]
        )
        
        # Sentiment distribution pie chart
        label_counts = data['label'].value_counts()
        fig.add_trace(
            go.Pie(labels=['Negative', 'Positive'], values=label_counts.values),
            row=1, col=1
        )
        
        # model performance comparison bar chart
        model_names = list(results.keys())
        accuracies = [results[name]['val_accuracy'] for name in model_names]
        
        fig.add_trace(
            go.Bar(x=model_names, y=accuracies),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Takeaway Sentiment Analysis Dashboard"
        )
        
        # Save as HTML
        fig.write_html("results/dashboard.html")
        print("Dashboard saved to results/dashboard.html")

# test the visualizer with dummy data
if __name__ == "__main__":
    np.random.seed(42)
    
    data = pd.DataFrame({
        'label': np.random.randint(0, 2, 100),
        'text_length': np.random.randint(10, 100, 100),
        'word_count': np.random.randint(5, 50, 100)
    })
    
    results_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest', 'SVM', 'Naive Bayes'],
        'Val Acc': [0.85, 0.87, 0.83, 0.80],
        'Val F1': [0.84, 0.86, 0.82, 0.79],
        'Train Acc': [0.88, 0.90, 0.85, 0.82]
    })
    
    cm = np.array([[45, 5], [8, 42]])
    
    y_true = np.random.randint(0, 2, 100)
    y_prob = np.random.rand(100)
    
    visualizer = ResultVisualizer()
    
    visualizer.plot_data_distribution(data)
    visualizer.plot_model_comparison(results_df)
    visualizer.plot_confusion_matrix(cm, "Random Forest")
    visualizer.plot_roc_curve(y_true, y_prob, "Random Forest")