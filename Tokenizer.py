import pandas as pd
import re
import os
import pickle
from collections import Counter

#Task 1: Data Preprocessing & Tokenization 


file_path = "C:/Users/Sanika Salunke/Desktop/Applied_ML_Assigmnt/movie_reviews.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

stopwords = set([
    "the", "a", "an", "is", "it", "to", "in", "of", "and", "for", "on", "with",
    "this", "that", "was", "as", "at", "by", "but", "from", "or", "not", "be",
    "are", "we", "you", "your", "they", "their", "if", "so", "too", "very", "just"
])


def preprocess_text(text, vocab=None):
    text = text.lower()
    text = re.sub(r"\b(not)\s+(\w+)", r"\1_\2", text)  
    tokens = re.findall(r"\b\w+\b", text)  
    tokens = [t for t in tokens if t not in stopwords]  

    bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
    trigrams = [f"{tokens[i]}_{tokens[i+1]}_{tokens[i+2]}" for i in range(len(tokens)-2)]
    all_tokens = tokens + bigrams + trigrams

    # Replacing Out of vocabulary words
    if vocab:
        all_tokens = [t if t in vocab else "<OOV>" for t in all_tokens]

    return all_tokens


def preprocess_text_bigrams(text, vocab=None):
    text = text.lower()
    text = re.sub(r"\b(not)\s+(\w+)", r"\1_\2", text)  
    tokens = re.findall(r"\b\w+\b", text)  
    tokens = [t for t in tokens if t not in stopwords]  

    
    bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]

    # Replacing out of vocab for bigrams since 2 models are being trained one with only 
    # bigrams and other with bigrams trigrams and all tokens

    if vocab:
        bigrams = [b if b in vocab else "<OOV>" for b in bigrams]

    return bigrams


df["Processed_Review"] = df["Review"].apply(preprocess_text)
df["Processed_Bigrams"] = df["Review"].apply(preprocess_text_bigrams)


train_data = df[df["Split"] == "train"]["Processed_Review"].tolist()
test_data = df[df["Split"] == "test"]["Processed_Review"].tolist()
train_labels = df[df["Split"] == "train"]["Sentiment"].values
test_labels = df[df["Split"] == "test"]["Sentiment"].values

# train test split for bigrams only model
train_data_bigrams = df[df["Split"] == "train"]["Processed_Bigrams"].tolist()
test_data_bigrams = df[df["Split"] == "test"]["Processed_Bigrams"].tolist()


vocab = set(token for review in train_data for token in review)
vocab_bigrams = set(bigram for review in train_data_bigrams for bigram in review)

df["Processed_Review"] = df["Review"].apply(lambda x: preprocess_text(x, vocab=vocab))
df["Processed_Bigrams"] = df["Review"].apply(lambda x: preprocess_text_bigrams(x, vocab=vocab_bigrams))

# Save everything
output_dir = "C:/Users/Sanika Salunke/Desktop/Applied_ML_Assigmnt/Tokenizer_Result"
os.makedirs(output_dir, exist_ok=True)

pickle.dump(train_data, open(os.path.join(output_dir, "train_data.pkl"), "wb"))
pickle.dump(test_data, open(os.path.join(output_dir, "test_data.pkl"), "wb"))
pickle.dump(train_labels, open(os.path.join(output_dir, "train_labels.pkl"), "wb"))
pickle.dump(test_labels, open(os.path.join(output_dir, "test_labels.pkl"), "wb"))
pickle.dump(vocab, open(os.path.join(output_dir, "vocab.pkl"), "wb"))

# Saving bigram data
pickle.dump(train_data_bigrams, open(os.path.join(output_dir, "train_data_bigrams.pkl"), "wb"))
pickle.dump(test_data_bigrams, open(os.path.join(output_dir, "test_data_bigrams.pkl"), "wb"))
pickle.dump(vocab_bigrams, open(os.path.join(output_dir, "vocab_bigrams.pkl"), "wb"))

print("All done! Processed data saved in Tokenizer_Result.")
print("Bigram data saved too. You're good to go!")
