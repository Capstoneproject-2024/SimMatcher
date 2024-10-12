import csv
import re
import json

class Filereader:
    def __init__(self):
        self.status = 0

    def readReviews(self, csvpath: str, encoding='utf-8', skip=1) -> list:
        reviews = []
        with open(csvpath, mode='r', encoding=encoding) as file:
            reader = csv.reader(file)

            for _ in range(skip):
                next(reader)

            for row in reader:
                reviews.append(row)

        return reviews

    def readBooks(self, path: str, encoding='utf-8') -> list:
        books = []
        with open(path, mode='r', encoding=encoding) as file:
            lines = file.readlines()
            for i in range(0, len(lines), 6):
                title = lines[i].strip()
                reviews = [lines[i + j].strip() for j in range(1, 6)]

                for n, review in enumerate(reviews):
                    filtered_review = re.sub(r'^\d+\.\s*', '', review)
                    reviews[n] = filtered_review

                books.append([title, reviews])
        return books

    def readReviewFromJson(self, jsonpath: str, encoding='utf-8') -> list:
        reviews_processed = []
        with open(jsonpath, 'r', encoding=encoding) as file:
            data = json.load(file)
            for book, reviews in data.items():
                #print(f'book = {book} : {reviews}')
                for keywords in reviews:
                    keyword_list = []
                    for keyword_set in keywords:
                        #print(keyword_set)
                        keyword_list.append(keyword_set[0])
                    reviews_processed.append([book, keyword_list])

        return reviews_processed


