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

        # Stop words
        self.stopwords_path = stopwords_path
        with open(self.stopwords_path, 'r', encoding='utf-8') as file:
            self.stopwords = [line.strip() for line in file.readlines()]

        # File reader
        self.csvpath = 'data/Review_good.csv'
        # csvpath = 'data/Reviews.csv'
        self.filereader = Filereader()
        self.review = self.filereader.readReviews(self.csvpath, 'cp949')

        # Extractor
        # self.extractor = KeyBERT()
        self.extractor = KeyBERT(self.model)

    def extract_keywords_json(self):
        keys = {}
        start_time = time.time()
        for item in self.review:
            title = item[0]
            text = item[1]

            keywords = self.extractor.extract_keywords(
                text,
                keyphrase_ngram_range=(2, 4),
                use_maxsum=True,
                # use_mmr=True,
                # stop_words=stopwords,
                top_n=5,
            )

            if title not in keys:
                keys[title] = [keywords]
            else:
                keys[title].extend([keywords])

        end_time = time.time()

        execution_time = end_time - start_time
        print(f"실행 시간: {execution_time:.6f} 초")

        with open('data.json', 'w', encoding='utf-8') as json_file:
            json.dump(keys, json_file, ensure_ascii=False, indent=4)





