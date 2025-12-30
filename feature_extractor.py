# generate features extraction module
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD  #for dimensionality reduction
import numpy as np
from collections import Counter # for counting word frequencies

class FeatureExtractor:
    def __init__(self, method='tfidf', max_features=2000):
        self.method = method
        self.max_features = max_features
        self.vectorizer = None 
        self.svd = None 

    """create text feature which supports 'tfidf' and 'count' methods"""
    def create_text_features(self, texts):
        print(f"Creating text features using {self.method} method...")
        # If a vectorizer was already fitted (e.g., during training), reuse it and
        # only call `transform` to avoid refitting on a single input which can
        # cause max_df/min_df validation errors.
        if self.vectorizer is not None:
            X = self.vectorizer.transform(texts)
        else:
            # When creating a new vectorizer, adjust `max_df` when there's only
            # a single document to prevent the sklearn error about
            # "max_df corresponds to < documents than min_df".
            use_max_df = 1 if len(texts) <= 1 else 0.95

            if self.method == 'tfidf':  # TF-IDF vectorization
                self.vectorizer = TfidfVectorizer(
                    max_features=self.max_features, # limit to top N features
                    ngram_range=(1, 2), # unigrams and bigrams
                    min_df=1,
                    max_df=use_max_df,
                    sublinear_tf=True   # using 1+log(tf) instead of tf
                )
            elif self.method == 'count':    # Count vectorization
                self.vectorizer = CountVectorizer(
                    max_features=self.max_features, # limit to top N features
                    ngram_range=(1, 2), # unigrams and bigrams
                    min_df=1,
                    max_df=use_max_df,
                )
            elif self.method == 'binary':   # Binary occurrence vectorization
                self.vectorizer = CountVectorizer(
                    max_features=self.max_features,
                    binary=True,  # binary occurrence
                    ngram_range=(1, 2),  # unigrams and bigrams
                    min_df=1,
                    max_df=use_max_df,
                )

            X = self.vectorizer.fit_transform(texts)    # fit and transform the texts
        self.feature_names = self.vectorizer.get_feature_names_out() # get feature names
        print(f"Generated feature matrix with shape: {X.shape}")
        print(f"the length of feature names: {len(self.feature_names)}")

        return X
    
    """create hand-crafted statistical features"""
    def create_handcrafted_features(self, df):
        features = []   
        feature_names = [
            'text_length',
            'cleaned_length',
            'word_count',
            'exclamation_count',
            'question_count',
            'punctuation_ratio',
            'sentiment_score',
            'has_delivery',
            'has_food',
            'has_packaging',
            'unique_words',
            'word_diversity'
        ]

        positive_words = {'好', '喜欢', '满意', '棒', '赞', '推荐', '不错', '愉快', '美味', '超赞','很好', '实惠', '新鲜', '划算', '舒服', '贴心', '惊喜', '必点', '正宗', '丰富'}
        negative_words = {'差', '失望', '不好', '糟糕', '讨厌', '差评', '难吃', '投诉', '垃圾', '坑爹', '雷', '贵', '凉'}

        for idx, row in df.iterrows():
            text = row.get('review', '')
            cleaned_text = row.get('cleaned_text', '')
            words = row.get('segmented_text', [])

            text_length = len(text)
            cleaned_length = len(cleaned_text)
            word_count = len(words)

            exclamation_count = text.count('！') + text.count('!')
            question_count = text.count('？') + text.count('?')
            punctuation_ratio = (exclamation_count + question_count) / max(text_length, 1)

            pos_count = sum(1 for word in words if word in positive_words)
            neg_count = sum(1 for word in words if word in negative_words)
            sentiment_score = (pos_count - neg_count) / max(word_count, 1)

            has_delivery = 1 if any(word in text for word in ['配送', '送达', '快递', '骑手']) else 0
            has_food = 1 if any(word in text for word in ['味道', '口感', '好吃', '难吃']) else 0
            has_packaging = 1 if any(word in text for word in ['包装', '盒子', '袋子']) else 0

            unique_words = len(set(words))
            word_diversity = unique_words / max(word_count, 1)

            handcrafted_feats = [
                text_length,
                cleaned_length,
                word_count,
                exclamation_count,
                question_count,
                punctuation_ratio,
                sentiment_score,
                has_delivery,
                has_food,
                has_packaging,
                unique_words,
                word_diversity
            ]
            features.append(handcrafted_feats)

        print(f"handcrafted feature names: {feature_names}")
        return np.array(features), feature_names
        
    """combine text features and handcrafted features"""   
    def combine_features(self, text_features, handcrafted_features):
        if handcrafted_features is not None:
            combined = np.hstack((text_features.toarray(), handcrafted_features))
            print(f"Combined feature matrix shape: {combined.shape}")
            return combined
        else:
            return text_features

    """reduce dimensionality using SVD"""
    def reduce_dimensionality(self, features, n_components=100):
        print(f"Reducing dimensionality to {n_components} components using SVD...")
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced_features = svd.fit_transform(features)
        explained_variance = svd.explained_variance_ratio_.sum()
        print(f"Reduced feature matrix shape: {reduced_features.shape}")
        print(f"keep explained variance ratio: {explained_variance:.2%}")
        return reduced_features

    """get top N important features based on IDF scores (for TF-IDF)"""
    def get_top_features(self, n=20):
        if self.vectorizer is not None and hasattr(self.vectorizer, 'idf_'):
            feature_importance = self.vectorizer.idf_
            top_indices = np.argsort(feature_importance)[-n:][::-1]
            top_features = [(self.feature_names[i], feature_importance[i]) for i in top_indices]

            print(f"\nTop {n} important features based on IDF scores:")
            for feature, importance in top_features:
                print(f"{feature}: {importance:.4f}")

            return top_features

        return None

# test feature extraction module
if __name__ == "__main__":
    texts = [
        '味道不错配送很快',
        '菜都凉了配送太慢',
        '包装很好味道正宗',
        '价格太贵分量不足'
    ]
    
    extractor = FeatureExtractor(method='tfidf', max_features=50)
    features = extractor.create_text_features(texts)
    
    print(f"\nexample of feature matrix:")
    print(features.toarray()[:2, :10])  # display first 2 samples and first 10 features