# text preprocessing module
import re
import jieba
import jieba.posseg as pseg
from collections import defaultdict
import pandas as pd

class TextPreprocessor:
    def __init__(self):
        self.stopwords = set()
        self.custom_dict = []
        self.initailze_resources()

    """Initialize stopwords and custom dictionary"""
    def initailze_resources(self):
        jieba.initialize()
        self.add_food_dictionary()
        self.load_stopwords()

    """Add takeaway food-related terms"""
    def add_food_dictionary(self):
        food_words =  [
            ('麻辣烫', 'n'), ('黄焖鸡', 'n'), ('披萨', 'n'), ('汉堡', 'n'), ('奶茶', 'n'),
            ('炸鸡', 'n'), ('寿司', 'n'), ('拉面', 'n'), ('盖饭', 'n'), ('烧烤', 'n'),
            ('米线', 'n'), ('饺子', 'n'), ('炒饭', 'n'), ('面条', 'n'), ('快餐', 'n'),
            ('便当', 'n'), ('甜品', 'n'), ('咖啡', 'n'), ('果汁', 'n'), ('蛋糕', 'n')
        ]
        takeaway_words = [
            ('外卖', 'n'), ('配送', 'vn'), ('骑手', 'n'), ('打包', 'v'), ('送餐', 'v'),
            ('送达', 'v'), ('准时达', 'n'), ('超时', 'v'), ('保温', 'v'), ('包装袋', 'n'),
            ('外卖盒', 'n'), ('配送费', 'n'), ('满减', 'n'), ('优惠券', 'n')
        ]
        sentiment_words = [
            ('超好吃', 'a'), ('巨难吃', 'a'), ('五星好评', 'n'), ('差评', 'n'),
            ('绝绝子', 'a'), ('yyds', 'a'), ('踩雷', 'v'), ('拔草', 'v'),
            ('种草', 'v'), ('回购', 'v'), ('推荐', 'v'), ('避坑', 'v')
        ]

        for word, flag in food_words + takeaway_words + sentiment_words:    # add to jieba dictionary
            jieba.add_word(word, freq=1000, tag=flag)
            self.custom_dict.append((word, flag))

    """Load stopwords from file"""
    def load_stopwords(self):
        basic_stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
            '你', '会', '着', '没有', '看', '好', '自己', '这', '这个',
            '那', '那个', '在', '还', '我们', '他们', '你们', '她', '他',
            '它', '啊', '哦', '嗯', '呢', '吧', '吗', '啦', '呀', '哇'
        }
        takeaway_stopwords = {
            '外卖', '订单', '配送', '商家', '用户', '手机', '平台', 'app',
            '美团', '饿了么', '点餐', '订购', '购买'
        }
        punctuation = {
            '，', '。', '！', '？', '、', '；', '：', '「', '」', '『', '』',
            '（', '）', '《', '》', '【', '】', '｛', '｝', '—', '～', '·',
            '．', '﹐', '﹒', '﹔', '﹕', '﹖', '﹗', '＂', '＃', '＄', '％',
            '＆', '＇', '（', '）', '＊', '＋', '，', '－', '．', '／', '：',
            '；', '＜', '＝', '＞', '？', '＠', '［', '＼', '］', '＾', '＿',
            '｀', '｛', '｜', '｝', '～'
        }
        self.stopwords = basic_stopwords.union(takeaway_stopwords).union(punctuation)   # combine all stopwords
    
    """Clean and preprocess text"""
    def clean_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).strip().lower()   # convert to string, strip whitespace, lowercase

        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)    # remove URLs
        text = re.sub(r'@\w+\s?', '', text)
        text = re.sub(r'#\w+#', '', text)
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？、；："'']', ' ', text)   # remove special characters
        text = re.sub(r'\s+', ' ', text)    # normalize whitespace
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # limit repeated characters to two
        return text.strip()
    
    """Segment text and remove stopwords"""
    def segment_text(self, text):
        if not text:
            return []
        words_with_pos = pseg.cut(text)

        filtered_words = []
        for word, flag in words_with_pos:
            if (
                word not in self.stopwords 
                and len(word) > 1   # filter single characters
                and re.search('[\u4e00-\u9fa5]', word)  # filter Chinese words
                and flag not in ['x', 'w', 'm']):  # filter non-informative POS tags
                filtered_words.append(word)

        return filtered_words

    """process text in DataFrame column"""
    def process_dataframe(self, df, text_column='review'):
        process_df = df.copy()  # avoid modifying original DataFrame
        process_df['cleaned_text'] = process_df[text_column].apply(self.clean_text) # clean text
        process_df['segmented_text'] = process_df['cleaned_text'].apply(self.segment_text)  # segment text
        process_df['processed_text'] = process_df['segmented_text'].apply(lambda x: ' '.join(x))  # join back to string
        process_df['text_length'] = process_df['processed_text'].apply(len)  # calculate text length
        process_df['word_count'] = process_df['segmented_text'].apply(len)  # calculate word count

        print("Text preprocessing completed.")
        print(f" original column: {text_column} ")
        print(" new columns added: cleaned_text, segmented_text, processed_text, text_length, word_count ")
        return process_df
    
    """analyze text statistics"""
    def analyze_text_statistics(self, df):
        stats = {
            'avg_text_length': df['text_length'].mean(),  
            'avg_word_count': df['word_count'].mean(),
            'min_text_length': df['text_length'].min(),
            'max_text_length': df['text_length'].max(),
            'total_words': df['word_count'].sum(),
            'unique_words': len(set([word for words in df['segmented_text'] for word in words]))
        }

        print("="*50)   # separator line
        print("Text Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
        return stats
    
# test the TextPreprocessor class
if __name__ == "__main__":
    test_data = pd.DataFrame({
        'review': [
            "这个外卖真的超好吃！强烈推荐！http://example.com",
            "服务态度差，送餐超时，差评！",
            "包装还可以，就是味道一般般。",
            None,
            "我觉得还不错，下次还会再点的！😊"
        ],
        'label': [1, 0, 1, 0, 1]
    })

    preprocessor = TextPreprocessor()
    processed_data = preprocessor.process_dataframe(test_data)

    print(processed_data[['review', 'processed_text', 'word_count']].head())
    

