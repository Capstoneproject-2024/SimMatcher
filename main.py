from Extractor import *
from FileReader import *
from SimilarityMatcher import *
import traceback
import json

#extractor = Extractor()
#extractor.extract_keywords_json(review_path='data/Review_book.csv')

"""

review_path = 'data.json'
book_path = 'BookInfo.txt'

reader = Filereader()
reviews = reader.readReviewFromJson(review_path)
books = reader.readBooks(book_path)

#print("Check readers")

# Reviews : [ ['title', ['key', 'words', .....] ]...... ] -> keywords are in sentence form, not a word
# Books   : [ ['title', ['key', 'words', .....] ]...... ] -> keywords are in sentence form, not a word

#print(f'Review\n{reviews}\n')
#print(f'Book\n{books}\n')
"""

print("Program Start")
matcher = Matcher()
print("Matcher ready")

while True:
    print("0: Exit\n"
          "1: Match\n"
          "2: Set Review Proportion (0~100)\n"
          "3: Change file\n"
          "4: Extractor\n"
          "5: Print all keywords")
    user_input = input("choose>>")

    try:
        if user_input == '0':
            exit(0)

        elif user_input == '1':
            matcher.match_both()

        elif user_input == '2':
            proportion = input("Type proportion of review (Review keyword : book keyword, 0~100)")
            matcher.set_proportion(int(proportion))

        elif user_input == '5':
            matcher.print_all_keywords()

    except Exception as e:
        traceback.print_exc()
