from gensim.models import fasttext
import numpy as np
from FileReader import *
from enum import IntEnum

class Matcher:
    def __init__(self, modelpath='models/cc.ko.300.bin.gz', use_model=True):
        self.books = []
        self.reviews = []
        if use_model:
            self.model = fasttext.load_facebook_vectors('models/cc.ko.300.bin.gz')
        self.reader = Filereader()
        self.review_proportion = 0.5    # Proportion of Review

        self.keywords = {} # {book_title: {info: [], review: []}}

        self.set_keywords()
        print("Made model of sim_matcher")

    def print_all_keywords(self):
        print("------------------------------------------------------------")
        for title, keywords in self.keywords.items():
            print(f'Title: "{title}"\n'
                  f'Info Keywords  : {keywords[Keytype.INFO.name]}\n'
                  f'Review Keywords: {keywords[Keytype.REVIEW.name]}\n')
        print("------------------------------------------------------------")

    def set_keywords(self, book_keyword_path='BookInfo.txt', review_keyword_path='data.json'):
        self.getBooks(book_path=book_keyword_path)
        self.getReviews(review_path=review_keyword_path)

        for book in self.books:
            self._add_keyword(book[0].lower(), book[1], Keytype.INFO)

        for review in self.reviews:
            self._add_keyword(review[0].lower(), review[1], Keytype.REVIEW)

        print("Keywords set")
        #self.print_all_keywords()

    def _add_keyword(self, title: str, keywords: list, key_type: int):
        if title not in self.keywords:
            self.keywords[title] = {Keytype.INFO.name: [], Keytype.REVIEW.name: []}

        if key_type == Keytype.REVIEW:
            self.keywords[title][Keytype.REVIEW.name] = keywords

        elif key_type == Keytype.INFO:
            self.keywords[title][Keytype.INFO.name] = keywords

    def _s2v_mean(self, sentence: str, voo='similar'):
        """
        Calculate vector of a single sentence using arithmetic mean
        :param sentences:
        :return:
        """
        words = sentence.split()
        word_vec = []

        for word in words:
            if word in self.model:
                word_vec.append(self.model[word])
            else:
                if voo == 'similar':
                    # if word doesn't exist in model, find the most similar word in model
                    similar_word = self.model.most_similar(word, topn=1)[0][0]
                    word_vec.append(self.model[similar_word])
                    print(f'{word} not in model -> changed into {similar_word}')
        return np.mean(word_vec, axis=0)

    def _s2v_single(self, word: str):
        if word in self.model:
            return self.model[word]
        else:
            sim_word = self.model.most_similar(word, topn=1)[0][0]
            return self.model[sim_word]

    def _cosine_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        return dot_product/(norm_a * norm_b)

    def sentence_similarity(self, sen1: str, sen2: str):
        vec1 = self._s2v_mean(sen1)
        vec2 = self._s2v_mean(sen2)
        return self._cosine_similarity(vec1, vec2)

    def test_similarity(self, word1: str, word2: str):
        word1_vec = self._s2v_single(word1)
        word2_vec = self._s2v_single(word2)
        print(f"Word1: '{word1}', Word2: '{word2}', similarity: '{self._cosine_similarity(word1_vec, word2_vec)}'")


    def getBooks(self, book_path='BookInfo.txt'):
        """
        Read book information from publishers
        [ ['title', ['key', 'words', .....] ]...... ]
        :param book_path:
        :return:
        """
        self.books = self.reader.readBooks(book_path)

    def getReviews(self, review_path='data.json'):
        """
        Read reviews from json file
        [ ['title', ['key', 'words', .....] ]...... ]
        :param review_path:
        :return:
        """
        self.reviews = self.reader.readReviewFromJson(review_path)

    def set_proportion(self, review_proportion: int):
        if 0 <= review_proportion <= 100:
            self.review_proportion = review_proportion/100
        else:
            print("WARNING: Proportion should be in 0~100")

    def match_both(self):
        r_proportion = self.review_proportion
        i_proportion = 1 - self.review_proportion

        print(f"Match test - review[{r_proportion}] + info[{i_proportion}]")

        for i, review in enumerate(self.reviews):
            print(f'{i}: {review}')
        review_num = int(input('\nEnter review number: '))

        review_sample = self.reviews[0]

        while 0 <= review_num <= len(self.reviews):
            book_similarity = []
            review_sample = self.reviews[review_num]
            print(f'Sample review: {review_sample}')

            for title, keywords in self.keywords.items():
                info_keywords = keywords[Keytype.INFO.name]
                review_keywords = keywords[Keytype.REVIEW.name]
                sims_info = []
                sims_review = []

                for keyword in info_keywords:
                    for review in review_sample[1]:
                        sims_info.append(self.sentence_similarity(review, keyword))

                for keyword in review_keywords:
                    for review in review_sample[1]:
                        sims_review.append(self.sentence_similarity(review, keyword))

                similarity = 0

                if len(sims_info) != 0 and len(sims_review) != 0:
                    info_sim_avg = sum(sims_info) / len(sims_info)
                    review_sim_avg = sum(sims_review) / len(sims_review)
                    similarity = (r_proportion * review_sim_avg) + (i_proportion * info_sim_avg)

                else:
                    if len(sims_info) != 0:
                        similarity = sum(sims_info) / len(sims_info)
                    elif len(sims_review) != 0:
                        similarity = sum(sims_review) / len(sims_review)

                book_similarity.append([title, similarity])

            book_similarity.sort(key=lambda x: x[1], reverse=True)
            print(book_similarity[:5])
            review_num = int(input('\nEnter review number: '))

    def match_book2review(self, reviews, books):
        print("Match test")

        for i, line in enumerate(reviews):
            print(f'{i}: {line}')

        review_num = int(input('\nEnter review number: '))
        review_sample = reviews[0]
        while 0 <= review_num <= len(reviews):
            book_similarity = []
            review_sample = reviews[review_num]
            print(f'Sample review: {review_sample}')
            for book in books:
                title = book[0]
                keywords = book[1]
                sims = []
                #print(f'title: {title}, keywords: {keywords}')
                for keyword in keywords:
                    for review in review_sample[1]:
                        sims.append(self.sentence_similarity(review, keyword))

                book_similarity.append([title, sum(sims)/len(sims)])

            book_similarity.sort(key=lambda x: x[1], reverse=True)
            print(book_similarity)
            review_num = int(input('\nEnter review number: '))

    def match_review2review(self):
        pass


class Keytype(IntEnum):
    INFO = 0
    REVIEW = 1

