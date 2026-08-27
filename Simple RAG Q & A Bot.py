# Simple RAG Mini Q&A Bot
# RAG = Retrieve relevant information and give it as an answer


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# STEP 1: Read the documents.txt file

with open("chapter 1/.vscode/documents.txt", "r", encoding="utf-8") as file:
    text = file.read()


# STEP 2: Split document into passages

passages = text.split("\n\n")

passages = [p for p in passages if p.strip()]

# STEP 3: Ask the user a question

question = input("Ask your question: ")

# STEP 4: Convert text into numbers

# TF-IDF calculate the importance of each word in the passages and the question. 
# It converts the text into numerical vectors that can be compared for similarity.
vectorizer = TfidfVectorizer()

# Adding question and passage in a single list for vectorization
all_text = [question] + passages

#Converting the text into numerical values using TF-IDF vectorization
vectors = vectorizer.fit_transform(all_text)

# STEP 5: Calculate similarity

# First vector is the question vector.
question_vector = vectors[0]

# Rest vectors are the passage vectors.
passage_vectors = vectors[1:]

# Calculating the similaritybetween the question and each passage using cosine similiarity.
scores = cosine_similarity(question_vector, passage_vectors)[0]


# STEP 6: Find the best passage

# Finding maximum score
best_index = scores.argmax()

# Finding the passage with the highest similarity score
answer = passages[best_index]

# Finding the highest similarity score
best_score = scores[best_index]


# STEP 7: Print the answer

print("\nAnswer:")
print(answer)

print("\nSimilarity Score:")
print(round(best_score, 2))