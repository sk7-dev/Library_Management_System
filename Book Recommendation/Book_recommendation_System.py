import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Load book data
books = pd.read_csv("Datasets/Books.csv", delimiter=';', encoding='ISO-8859-1')
ratings = pd.read_csv("Datasets/Book-Ratings.csv", delimiter=';', encoding='ISO-8859-1')

# Capture user input for recommendation
bookName = input("Enter a book name: ")
number = int(input("Enter number of books to recommend: "))

# Preprocess and Vectorize
listOfDictionaries = []
indexMap = {}
reverseIndexMap = {}
ptr = 0
for groupKey in ratings.groupby('ISBN').groups.keys():
    tempDict = dict(ratings[ratings['ISBN'] == groupKey][['User-ID', 'Book-Rating']].values)
    indexMap[ptr] = groupKey
    reverseIndexMap[groupKey] = ptr
    listOfDictionaries.append(tempDict)
    ptr += 1

dictVectorizer = DictVectorizer(sparse=True)
vector = dictVectorizer.fit_transform(listOfDictionaries)
pairwiseSimilarity = cosine_similarity(vector)

# Recommendation function
def getTopRecommendations(bookID):
    print("Input Book:", bookID)
    row = reverseIndexMap[bookID]
    similar = []
    for i in np.argsort(pairwiseSimilarity[row])[:-2][::-1]:
        if len(similar) >= number:
            break
        similar.append(indexMap[i])
    return similar

# Fetch book ISBN for input book and get recommendations
isbn = books.loc[books['Book-Title'] == bookName, 'ISBN'].values[0]
recommended_books = getTopRecommendations(isbn)

# Print recommendations
print("\nRecommended Books:\n")
for rec_isbn in recommended_books:
    print(books.loc[books['ISBN'] == rec_isbn, 'Book-Title'].values[0])
