import time
from datetime import datetime
import pandas as pd
from keybert import KeyBERT
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from FileReader import *
import json


class Extractor:
    def __init__(self, model_name="monologg/kobert", stopwords_path='stopword.txt'):

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)

        self.range_parameter_tuple = (2, 4)     # Number of extracted keyword's word

        # Stop words
        self.stopwords_path = stopwords_path
        with open(self.stopwords_path, 'r', encoding='utf-8') as file:
            self.stopwords = [line.strip() for line in file.readlines()]

        # csvpath = 'data/Reviews.csv'
        self.filereader = Filereader()
        self.review = []
        #self.review = self.filereader.readReviews(self.csvpath, 'cp949')

        # Extractor
        # self.extractor = KeyBERT()
        self.extractor = KeyBERT(self.model)

    def _read_review(self, review_path: str, encoding='cp949'):
        self.review = self.filereader.readReviews(review_path, encoding=encoding)

    def extract_keyword_string(self, review: str, show_similarity=True) -> list:

        temp = []
        keywords = self.extractor.extract_keywords(
            review,
            keyphrase_ngram_range=self.range_parameter_tuple,
            use_maxsum=True,
            # use_mmr=True,
            # stop_words=stopwords,
            top_n=5,
        )
        # [ (title, similarity) ... ]

        if not show_similarity:
            for keyword in keywords:
                temp.append(keyword[0])
            keywords = temp

        return keywords

    def extract_keywords(self, review_path='data/Review_good.csv', encoding='cp949', show_similarity=True):
        self._read_review(review_path=review_path, encoding=encoding)
        #print(self.review)
        keys = {}   # To handle the case that a same book has multiple reviews
        start_time = time.time()

        for item in self.review:
            # Skip Review-less book
            if len(item) < 2:
                continue

            title = item[0]
            text = item[1]

            keywords = self.extract_keyword_string(text, show_similarity=show_similarity)

            if title not in keys:
                keys[title] = [keywords]
            else:
                keys[title].extend([keywords])

        end_time = time.time()

        execution_time = end_time - start_time
        print(f"실행 시간: {execution_time:.6f} 초")

        print(f"Keywords\n{keys}")
        return keys

        #with open('results/data.json', 'w', encoding='utf-8') as json_file:
        #    json.dump(keys, json_file, ensure_ascii=False, indent=4)

    def save_keywords_json(self, review_path='data/Review_good.csv', encoding='cp949'):
        keys = self.extract_keywords(review_path, encoding)

        current_time = datetime.now().strftime("%y%m%d%H%M")
        file_name = 'results/review_keyword_' + current_time + ".json"
        with open(file_name, 'w', encoding='utf-8') as file:
            json.dump(keys, file, ensure_ascii=False, indent=4)

    def save_keywords_csv(self, review_path='data/Review_book.csv', encoding='cp949', show_similarity=False):
        keys = self.extract_keywords(review_path=review_path, encoding=encoding, show_similarity=show_similarity)

        current_time = datetime.now().strftime("%y%m%d%H%M")
        file_name = 'results/review_keyword_' + current_time + ".csv"

        rows = []
        for title, keywords in keys.items():
            row = [title] + keywords[0]     # TODO ad-hoc design. If 1 book has multiple review-> should be changed
            rows.append(row)

        columns = ['title', 'keyword1', 'keyword2', 'keyword3', 'keyword4', 'keyword5']
        dataframe = pd.DataFrame(rows, columns=columns)
        dataframe.to_csv(file_name, index=False, encoding='utf-8-sig')


