import time
from textEX import textList
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

    def extract_keyword_string(self, review: str) -> list:

        keywords = self.extractor.extract_keywords(
            review,
            keyphrase_ngram_range=self.range_parameter_tuple,
            use_maxsum=True,
            # use_mmr=True,
            # stop_words=stopwords,
            top_n=5,
        )

        return keywords

    def extract_keywords_json(self, review_path='data/Review_good.csv', encoding='cp949'):
        self._read_review(review_path=review_path, encoding=encoding)
        #print(self.review)
        keys = {}
        start_time = time.time()

        for item in self.review:
            if len(item) < 2:
                continue

            title = item[0]
            text = item[1]

            keywords = self.extract_keyword_string(text)

            if title not in keys:
                keys[title] = [keywords]
            else:
                keys[title].extend([keywords])

        end_time = time.time()

        execution_time = end_time - start_time
        print(f"실행 시간: {execution_time:.6f} 초")

        with open('results/data.json', 'w', encoding='utf-8') as json_file:
            json.dump(keys, json_file, ensure_ascii=False, indent=4)





