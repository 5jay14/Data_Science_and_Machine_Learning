import pandas as pd

with open(r"C:\Users\vijay\Desktop\DS ML\18-Naive-Bayes-and-NLP\One.txt") as text1:
    a = text1.read().lower().split()  # this reads in entire text file and creates a vocabulary
    a_uni_words = set(a)
    # a = text.readlines() # this gives a list of each sentences

with open(r"C:\Users\vijay\Desktop\DS ML\18-Naive-Bayes-and-NLP\Two.txt") as text2:
    b = text2.read().lower().split()  # this reads in entire text file and creates a vocabulary
    b_uni_words = set(b)

all_uni_words = set()
all_uni_words.update(a_uni_words)
all_uni_words.update(b_uni_words)

# assigning a number/index for each word
full_vocab = dict()
i = 0

for word in all_uni_words:
    full_vocab[word] = i
    i = i + 1
print(full_vocab)  # notice the words are not sorted when iterated, its the nature of how set works. python works in the
# efficient way or arranging the items

one_frequency = [0] * len(full_vocab)
two_frequency = [0] * len(full_vocab)
all_words = [''] * len(full_vocab)
print(one_frequency,two_frequency)
with open(r"C:\Users\vijay\Desktop\DS ML\18-Naive-Bayes-and-NLP\One.txt") as text:
    a1 = text.read().lower().split()

with open(r"C:\Users\vijay\Desktop\DS ML\18-Naive-Bayes-and-NLP\Two.txt") as text:
    b2 = text.read().lower().split()

for word in a1:
    word_index = full_vocab[word]
    one_frequency[word_index] +=1

for word in b2:
    word_index = full_vocab[word]
    two_frequency[word_index] +=1

for word in full_vocab:
    word_index= full_vocab[word]
    all_words[word_index] = word

bow = pd.DataFrame(data=[one_frequency,two_frequency], columns=all_words)
print(bow) # bow = bag of words