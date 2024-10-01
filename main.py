from Extractor import *
from FileReader import *
from SimilarityMatcher import *
import json

#extractor = Extractor()
#extractor.extract_keywords_json()


review_path = 'data.json'
book_path = 'BookInfo.txt'

reader = Filereader()
reviews = reader.readReviewFromJson(review_path)
#reviews = reader.readReviews(review_path, encoding='cp949')
books = reader.readBooks(book_path)

#print("Check readers")

# Reviews : [ ['title', ['key', 'words', .....] ] ...... ] -> keywords are in sentence form, not a word
# Books   : [ ['title', ['key', 'words', .....] ...... ] ] -> keywords are in sentence form, not a word

#print(f'Review\n{reviews}\n')
#print(f'Book\n{books}\n')



matcher = Matcher()
matcher.getBooks()
matcher.match(reviews, books)


