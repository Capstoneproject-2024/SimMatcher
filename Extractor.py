import time
from datetime import datetime
import pandas as pd
from keybert import KeyBERT
from sympy import false
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from FileReader import *
import json
from konlpy.tag import Okt
from collections import Counter
import re
#from models.test import extract_pos


class Extractor:
    def __init__(self, model_name="monologg/kobert", stopwords_path='stopword.txt'):

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)

        self.range_parameter_tuple = (2, 4)     # Number of extracted keyword's word
        self.time_format = "%y%m%d-%H%M%S"

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

        print(f'Extractor.py: extractor ready')

    def _read_review(self, review_path: str, encoding='cp949'):
        self.review = self.filereader.readReviews(review_path, encoding=encoding)

    def extract_pos(self, text):
        okt = Okt()

        # 텍스트에서 특수문자 제거
        text = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9\s]', '', text)

        # Stopwords
        stopwords = ['되다', '하다']

        # 형태소 분석하여 명사, 형용사, 동사만 추출
        words = okt.pos(text, stem=True)
        keywords = [word for word, pos in words if pos in ['Noun', 'Verb', 'Adjective']]

        keywords = [word for word in keywords if len(word) > 1]  # 길이가 1인 단어 제거 (불필요한 조사일 가능성 높음)

        # 정해진 품사로만 이루어진 문장 생성
        refined_sentence = ' '.join(keywords)
        return refined_sentence

        # word_counts = Counter(keywords)                # 키워드 빈도 계산
        # top_keywords = word_counts.most_common(10)     # 빈도 높은 키워드 상위 10개 추출
        # return top_keywords


    def extract_keyword_string(self, review: str, show_similarity=True, pos=False) -> list:
        if pos:
            review = self.extract_pos(review)

        stopwords = ['하다', '되다', '있다']
        temp = []
        keywords = self.extractor.extract_keywords(
            review,
            keyphrase_ngram_range=self.range_parameter_tuple,
            use_maxsum=True,
            # use_mmr=True,
            stop_words=stopwords,
            top_n=5,
        )
        # [ (title, similarity) ... ]

        if not show_similarity:
            for keyword in keywords:
                temp.append(keyword[0])
            keywords = temp

        return keywords

    def extract_keywords(self, review_path='data/Review_good.csv', encoding='cp949', show_similarity=True, pos=False):
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

            keywords = self.extract_keyword_string(text, show_similarity=show_similarity, pos=pos)

            if title not in keys:
                keys[title] = [keywords]
            else:
                keys[title].extend([keywords])

        end_time = time.time()

        execution_time = end_time - start_time
        print(f"실행 시간: {execution_time:.6f} 초")

        #print(f"Keywords\n{keys}")
        return keys

        #with open('results/data.json', 'w', encoding='utf-8') as json_file:
        #    json.dump(keys, json_file, ensure_ascii=False, indent=4)

    def save_keywords_json(self, review_path='data/Review_good.csv', encoding='cp949'):
        keys = self.extract_keywords(review_path, encoding)

        current_time = datetime.now().strftime(self.time_format)
        file_name = 'results/review_keyword_' + current_time + ".json"
        with open(file_name, 'w', encoding='utf-8') as file:
            json.dump(keys, file, ensure_ascii=False, indent=4)

    def save_keywords_csv(self, review_path='data/Review_book.csv', encoding='cp949', show_similarity=False, pos=False):
        keys = self.extract_keywords(review_path=review_path, encoding=encoding, show_similarity=show_similarity, pos=pos)

        current_time = datetime.now().strftime(self.time_format)
        file_name = 'results/review_keyword_' + current_time + ".csv"

        rows = []
        for title, keywords in keys.items():
            #print(keywords)
            row = [title] + keywords[0]     # TODO ad-hoc design. If 1 book has multiple review-> should be changed
            rows.append(row)

        columns = ['title', 'keyword1', 'keyword2', 'keyword3', 'keyword4', 'keyword5']
        dataframe = pd.DataFrame(rows, columns=columns)
        dataframe.to_csv(file_name, index=False, encoding='utf-8-sig')

    def save_keywords_pos_csv(self, review_path='data/Review_book.csv', encoding='cp949', show_similarity=False):
        """
        Extract keyword, then apply POS
        :param review_path:
        :param encoding:
        :param show_similarity:
        :return:
        """
        keys = self.extract_keywords(review_path=review_path, encoding=encoding, show_similarity=show_similarity)

        current_time = datetime.now().strftime(self.time_format)
        file_name = 'results/review_keyword_' + current_time + ".csv"

        rows = []
        for title, keywords in keys.items():
            # separate pos
            # print(keywords[0])
            for i, keyword in enumerate(keywords[0]):
                keywords[0][i] = self.extract_pos(keyword)      # POS 처리를 Extraction 이후에 적용

            # make row for csv
            row = [title] + keywords[0]     # TODO ad-hoc design. If 1 book has multiple review-> should be changed
            rows.append(row)

        columns = ['title', 'keyword1', 'keyword2', 'keyword3', 'keyword4', 'keyword5']
        dataframe = pd.DataFrame(rows, columns=columns)
        dataframe.to_csv(file_name, index=False, encoding='utf-8-sig')

    def save_status_to_exit(self):
        #TODO
        print("Extractor saved")


    ###################################################################################################################

