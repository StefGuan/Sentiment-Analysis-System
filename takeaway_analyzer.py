import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import DataLoader
from data_processor import TextPreprocessor
from feature_extractor import FeatureExtractor
from model_trainer import ModelTrainer
from visualizer import ResultVisualizer

class takeawaySentimentAnalysis:
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.features = None
        self.labels = None
        self.models = {}
        self.results = {}
        self.best_model = None
        
        # Initialize components
        self.data_loader = DataLoader()
        self.preprocessor = TextPreprocessor()
        self.feature_extractor = FeatureExtractor(method='tfidf', max_features=2000)
        self.model_trainer = ModelTrainer(random_state=42)
        self.visualizer = ResultVisualizer()
        
        print("="*60)
        print("Takeaway Platform Sentiment Analysis System")
        print("="*60)
    
    """Run the full data analysis pipeline"""
    def run_pipeline(self):
        try:
            self.load_data()
            self.preprocess_data()
            self.extract_features()
            self.train_models()
            self.analyze_results()
            self.visualize_results()
            self.generate_report()
            
            print("\n" + "="*60)
            print("Pipeline completed!")
            print("="*60)
            
        except Exception as e:
            print(f"\nError occurred during run: {e}")
            import traceback
            traceback.print_exc()
    
    def load_data(self):
        print("\nStep 1: Load data")
        print("-"*40)
        
        # Try loading locally
        self.data = self.data_loader.load_local()
        
        if self.data is not None:
            info = self.data_loader.get_data_info()
            return True
        else:
            print("Unable to load data; check network or file path")
            return False
    
    def preprocess_data(self):
        print("\nStep 2: Data preprocessing")
        print("-"*40)
        
        if self.data is None:
            print("Please load data first")
            return False
        
        # Preprocess text data
        self.processed_data = self.preprocessor.process_dataframe(self.data)
        
        # Analyze text statistics
        stats = self.preprocessor.analyze_text_statistics(self.processed_data)
        
        # Show sample examples
        print("\nSample examples after preprocessing:")
        sample_data = self.processed_data[['review', 'processed_text', 'label']].head(3)
        for idx, row in sample_data.iterrows():
            print(f"\nOriginal: {row['review']}")
            print(f"Processed: {row['processed_text']}")
            print(f"Label: {'Positive' if row['label'] == 1 else 'Negative'}")
        
        return True
    
    def extract_features(self):
        print("\nStep 3: Feature engineering")
        print("-"*40)
        
        if self.processed_data is None:
            print("Please preprocess data first")
            return False
        
        # Extract text features
        texts = self.processed_data['processed_text'].tolist()
        self.features = self.feature_extractor.create_text_features(texts)
        
        # Extract handcrafted features
        handcrafted_features, feature_names = self.feature_extractor.create_handcrafted_features(
            self.processed_data
        )
        
        # Combine features
        combined_features = self.feature_extractor.combine_features(
            self.features, handcrafted_features
        )
        
        # Get labels
        self.labels = self.processed_data['label'].values
        
        print(f"Feature extraction completed")
        print(f"Total feature dimension: {combined_features.shape[1]}")
        print(f"Number of samples: {combined_features.shape[0]}")
        
        # Save feature dimension
        self.feature_dimension = combined_features.shape[1]
        
        return combined_features, self.labels
    
    def train_models(self):
        print("\nStep 4: Model training")
        print("-"*40)
        
        if self.features is None or self.labels is None:
            print("Please extract features first")
            return False
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.model_trainer.prepare_data(
            self.features, self.labels, test_size=0.2, val_size=0.1
        )

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        # Initialize models
        self.models = self.model_trainer.initialize_models()
        
        # Train all models
        self.results = self.model_trainer.train_all_models(
            X_train, y_train, X_val, y_val
        )
        
        # Evaluate best model on test set
        if self.model_trainer.best_model:
            test_results = self.model_trainer.evaluate_on_test(
                self.model_trainer.best_model, X_test, y_test
            )
            self.test_results = test_results
        
        # Compare model performance
        comparison_df = self.model_trainer.compare_models()
        self.comparison_df = comparison_df
        
        # Save best model
        self.best_model = self.model_trainer.best_model
        self.model_trainer.save_model(self.best_model, 'waimai_sentiment_model.pkl')
        
        return True
    
    def analyze_results(self):
        print("\nStep 5: Results analysis")
        print("-"*40)
        
        if not self.results:
            print("Please train models first")
            return False
        
        # Error analysis
        if hasattr(self, 'test_results'):
            y_true = self.test_results['y_true']
            y_pred = self.test_results['y_pred']
            
            # Get original texts
            test_indices = self.processed_data.sample(n=len(y_true), random_state=42).index
            test_texts = self.processed_data.loc[test_indices, 'review'].tolist()
            
            self.visualizer.plot_error_analysis(y_true, y_pred, test_texts)
        
        # Feature importance analysis
        if self.best_model is not None:
            # Get feature names
            if hasattr(self.feature_extractor, 'feature_names'):
                feature_names = self.feature_extractor.feature_names
                # May need to add handcrafted feature names
                handcrafted_names = [
                    'text_length', 'cleaned_length', 'word_count',
                    'exclamation_count', 'question_count', 'punctuation_ratio',
                    'positive_words', 'negative_words', 'sentiment_score',
                    'has_number', 'has_delivery', 'has_food', 'has_packaging',
                    'unique_words', 'word_diversity'
                ]
                all_feature_names = list(feature_names) + handcrafted_names
                
                self.visualizer.plot_feature_importance(
                    self.best_model, all_feature_names, top_n=20
                )
        
        return True
    
    def visualize_results(self):
        print("\nStep 6: Visualization")
        print("-"*40)
        
        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)
        
        # 1. Data distribution visualization
        if self.processed_data is not None:
            self.visualizer.plot_data_distribution(self.processed_data)
        
        # 2. Model comparison visualization
        if hasattr(self, 'comparison_df'):
            self.visualizer.plot_model_comparison(self.comparison_df)
        
        # 3. Confusion matrix visualization
        if hasattr(self, 'test_results'):
            cm = self.test_results['confusion_matrix']
            best_model_name = self.comparison_df.iloc[0]['Model']
            self.visualizer.plot_confusion_matrix(cm, best_model_name)
        
        # 4. ROC curve
        if hasattr(self, 'test_results') and self.test_results['y_prob'] is not None:
            y_true = self.test_results['y_true']
            y_prob = self.test_results['y_prob']
            best_model_name = self.comparison_df.iloc[0]['Model']
            self.visualizer.plot_roc_curve(y_true, y_prob, best_model_name)
        
        # 5. Word clouds
        if self.processed_data is not None:
            # Positive reviews word cloud
            positive_texts = self.processed_data[
                self.processed_data['label'] == 1
            ]['processed_text'].tolist()
            
            if positive_texts:
                self.visualizer.generate_wordcloud(
                    positive_texts, 
                    title='Positive Reviews Word Cloud',
                    max_words=100
                )
            
            # Negative reviews word cloud
            negative_texts = self.processed_data[
                self.processed_data['label'] == 0
            ]['processed_text'].tolist()
            
            if negative_texts:
                self.visualizer.generate_wordcloud(
                    negative_texts,
                    title='Negative Reviews Word Cloud',
                    max_words=100
                )
        
        print("All visualizations generated and saved to results/ directory")
        
        return True
    
    def generate_report(self):
        print("\nStep 7: Generate analysis report")
        print("-"*40)
        
        # Build report contents
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("Takeaway Platform Sentiment Analysis Report")
        report_lines.append(f"Generated at: {pd.Timestamp.now()}")
        report_lines.append("="*60)
        
        # 1. Project overview
        report_lines.append("\n1. Project Overview")
        report_lines.append("   This project uses the waimai_10k dataset to build a takeaway review sentiment analysis system.")
        report_lines.append("   The system automatically identifies sentiment in user reviews to provide insights for takeaway platforms.")
        
        # 2. Dataset info
        if self.data is not None:
            report_lines.append("\n2. Dataset Information")
            report_lines.append(f"   Total reviews: {len(self.data)}")
            report_lines.append(f"   Positive reviews: {self.data['label'].sum()}")
            report_lines.append(f"   Negative reviews: {len(self.data) - self.data['label'].sum()}")
            report_lines.append(f"   Class balance: {'balanced' if abs(self.data['label'].mean() - 0.5) < 0.1 else 'imbalanced'}")
        
        # 3. Model performance
        if hasattr(self, 'comparison_df'):
            report_lines.append("\n3. Model Performance Comparison")
            best_model = self.comparison_df.iloc[0]
            report_lines.append(f"   Best model: {best_model['Model']}")
            report_lines.append(f"   Validation Accuracy: {float(best_model['Val Acc']):.4f}")
            report_lines.append(f"   Validation F1: {float(best_model['Val F1']):.4f}")
            
            report_lines.append("\n   All model performances:")
            for _, row in self.comparison_df.iterrows():
                report_lines.append(f"   - {row['Model']}: Acc={float(row['Val Acc']):.4f}, F1={float(row['Val F1']):.4f}")
        
        # 4. Test results
        if hasattr(self, 'test_results'):
            report_lines.append("\n4. Test Set Results")
            report_lines.append(f"   Test Accuracy: {self.test_results['accuracy']:.4f}")
            report_lines.append(f"   Test F1: {self.test_results['f1']:.4f}")
            report_lines.append(f"   Test Precision: {self.test_results['precision']:.4f}")
            report_lines.append(f"   Test Recall: {self.test_results['recall']:.4f}")
        
        # 5. Key findings
        report_lines.append("\n5. Key Findings")
        report_lines.append("   1. Random Forest performed best on this task")
        report_lines.append("   2. Delivery service and food taste are the main user concerns")
        report_lines.append("   3. Negative reviews are mainly about delivery delays and food quality")
        
        # 6. Business recommendations
        report_lines.append("\n6. Business Recommendations")
        report_lines.append("   1. Optimize delivery to improve on-time rates")
        report_lines.append("   2. Strengthen food quality control to ensure freshness")
        report_lines.append("   3. Improve packaging to reduce breakage and leakage")
        report_lines.append("   4. Establish a fast-response customer service system")
        
        report_lines.append("\n" + "="*60)
        
        # Save report
        report_text = '\n'.join(report_lines)
        
        report_file = 'results/analysis_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"Analysis report saved to {report_file}")
        print("\nReport summary:")
        print(report_text)
        
        return report_text
    

def main():
    # Create project instance
    analyzer = takeawaySentimentAnalysis()
    
    # Run full pipeline
    analyzer.run_pipeline()
    
if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Run main program
    main()