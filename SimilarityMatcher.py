from gensim.models import Word2Vec, KeyedVectors, fasttext
from sklearn.metrics.pairwise import cosine_similarity
from FileReader import *
import numpy as np
from FileReader import *

class Matcher:
    def __init__(self, modelpath='models/cc.ko.300.bin.gz'):
        self.books = []
        self.reviews = []
        self.model = fasttext.load_facebook_vectors('models/cc.ko.300.bin.gz')
        self.reader = Filereader()
        print("Made model of sim_matcher")

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

    def _cosine_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        return dot_product/(norm_a * norm_b)

    def sentence_similarity(self, sen1, sen2):
        vec1 = self._s2v_mean(sen1)
        vec2 = self._s2v_mean(sen2)
        return self._cosine_similarity(vec1, vec2)

    def getBooks(self, bookPath='BookInfo.txt'):
        """
        Read book information from publishers
        :param bookPath:
        :return:
        """
        path = bookPath
        reader = Filereader()
        self.books = reader.readBooks(bookPath)

    def match(self, reviews, books):
        print("Match test")
            # [ [title, similarity_mean] ...... ]

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

