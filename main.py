from numpy.lib.function_base import extract

from Extractor import *
from FileReader import *
from SimilarityMatcher import *
import traceback
import json
from pathlib import Path

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

using_matcher = input("Will you use matcher and extractor? (Y to use) >>")
if using_matcher in ['y', 'Y']:
    matcher = Matcher()
    print("main.py: Matcher ready - with W2V model")
else:
    matcher = Matcher(use_model=False)
    print("main.py: Matcher ready - NO W2V model")
extractor = Extractor()

while True:
    print("0: Exit\n"
          "1: Match\n"
          "2: Set Review Proportion (0~100)\n"
          "3: Change file (x)\n"
          "4: Extractor (x)\n"
          "5: Print all keywords\n"
          "6: Extract and save as csv\n"
          "7: Extract, then apply POS, and save as csv\n"
          "8: Test Matcher, then save as csv"
          )
    user_input = input("choose>>")

    try:
        if user_input == '0':
            extractor.save_status_to_exit()
            exit(0)

        elif user_input == '1' and matcher is not None:
            matcher.match_both_test()

        elif user_input == '2':
            proportion = input("Type proportion of review (Review keyword : book keyword, 0~100)")
            matcher.set_proportion(int(proportion))

        elif user_input == '4':
            print("Testing novel brand new keyword extracting LMFOOOOOO (but not yet)")

        elif user_input == '5':
            matcher.print_all_keywords()
            #matcher.print_all_keywords_json()

        elif user_input == '6':
            extractor.save_keywords_csv(pos=False)

        elif user_input == '7':
            extractor.save_keywords_pos_csv()

        elif user_input == '8':
            path = input('Type a path of keywords csv file [title, keyword{1~5}] format needed\n>>')
            file = Path(path)   # 존재 확인
            if file.is_file():
                matcher.test_and_save_as_csv(path)
            else:
                print(f'Wrong file path: "{path}"')

    except Exception as e:
        traceback.print_exc()
