# Sentiment-Analysis-System
  (for takeaway reviews temporarily)

## 📋 Introduction
基于waimai_10k数据集的外卖评价情感分析系统，使用机器学习算法自动识别用户评论的情感倾向。  
A complete machine learning pipeline for sentiment analysis of takeaway platform reviews, supporting data loading, text preprocessing, feature extraction, multi-model training, visualization, and report generation.

## 🗂️ Project Structure
Takeaway_sentiment_analysis/  
├── takeaway_analyzer.py            
├── data_loader.py                  
├── data_processor.py               
├── feature_extractor.py            
├── model_trainer.py                
├── visualizer.py                   
├── data/                           
├── results/                        
├── models/                         
└── README.md                       

## 🚀 Quick Start
Prerequisites: requirement.txt  

Installation:  
-Clone the repository  
-Install dependencies  
-Prepare dataset  

  
Running the System: python takeaway_analyzer.py  
The program will automatically execute the following 7 steps:  
-Data Loading  
-Data Preprocessing  
-Feature Engineering  
-Model Training  
-Result Analysis  
-Visualization Generation  
-Report Generation  

## 📊 Output Results
After running, the following will be generated in the results/ directory:  
·analysis_report.txt - Detailed analysis report  
·Multiple visualization charts (PNG format)  
·Model performance comparison table  

## 🔧 Extension & Customization
Adding new models:  
Add new models in the initialize_models() method in model_trainer.py  

  
Adding new features:  
Add custom features in the create_handcrafted_features() method in feature_extractor.py

  
Adjusting Visualizations:  
Modify corresponding plotting methods in visualizer.py to adjust chart styles or add new chart types
