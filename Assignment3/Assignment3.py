import numpy as np 
import nltk 
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer
import operator
from itertools import islice
from nltk.util import ngrams
import collections
from decimal import *

#Check if you have this package for tokenizer to work - downloads if it doesnt exist
nltk.download('punkt')

######################### Part 1(a) ##############################################################

#Read the corpus given
corpus = open("Plato_Republic.txt", 'r').read()

#Asked to convert it to lower case
corpus = corpus.lower()

#Tokenizer which only tokenizes alpha numeric elements and ignores punctuations and stuff
tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(corpus)

#Counts of words and unique words in corpus
word_count = len(tokens)
unique_word_count = len(set(tokens))


######################### Part 1(b) ###############################################################

min_len = 8
top_words = 5

#Pick top 5 highest frequency words with length of at least 8  
print(list(islice(sorted({k: v for i in range(min_len,len(nltk.ConditionalFreqDist((len(word), word) for word in tokens))+1) for k, v in nltk.ConditionalFreqDist((len(word), word) for word in tokens)[i].items()}.items(), key=operator.itemgetter(1),reverse=True),top_words)))

######################## Part 1(c) ################################################################

#Create your unigrams,bigrams,unigram frequencies and bigram frequencies
bigrams = ngrams(tokens, 2)
bigram_freq = dict(collections.Counter(bigrams))
unigrams = ngrams(tokens,1)
unigram_freq = dict(collections.Counter(unigrams))


def probability(x1,x2):
	numerator = None
	denominator = None

	#check if bigrams and unigrams exist in corpus
	try:
		numerator = bigram_freq[(x1,x2)]

	except:

		try:
			numerator = bigram_freq[(x2,x1)]

		except:
			print("Bigram does not exist")
			return None

	try:
		denominator = unigram_freq[(x1,)]

	except:
		print("Unigram does not exist")
		return None

	return numerator/denominator	

######################## Part 1(D) ###################################################################

total = Decimal(1)

print(bigram_freq[('and','said')])
print(unigram_freq[('and',)])


for i in range(0,len(tokens)-2): 
	total = total*Decimal(probability(tokens[i],tokens[i+1]))
	#print(total)
	


perplexity = total**Decimal(-1*(1/(len(tokens)-1)))
print(perplexity)

######################## Part 2(A) ###################################################################

#Prepare Variables
letters = ['h','e','l','l','o']
index_to_letter = {
						0 : "h",
						1 : "e",
						2 : "l",
						3 : "o"		
					}

embedding = OrderedDict()
embedding["h"] = np.array([
								[1],
								[0],
								[0],
								[0]
						])

embedding["e"] = np.array([
								[0],
								[1],
								[0],
								[0]
						])

embedding["l"] = np.array([
								[0],
								[0],
								[1],
								[0]
						])

embedding["o"] = np.array([
								[0],
								[0],
								[0],
								[1]
						])

A = np.array([
				[1,-1,-0.5,0.5],
				[1,1,-0.5,-1]
			])

B = np.array([
				[1,1],
				[0.5,1],
				[-1,0],
				[0,-0.5]
			])

R = np.array([
				[1,0],
				[0,1]
			])

H = np.array([
				[0],
				[0]
			])

#Deduce
for letter in letters:
	#print(embedding[v])
	print("H: ",H)
	H = np.tanh( np.matmul(R,H) + np.matmul(A,embedding[letter]) )
	Y = np.matmul(B,H)
	#print("Y: ",Y)
	print("Deduction ",index_to_letter[np.argmax(Y, axis = 0)[0]])
	
####################### Part2(B) #############################################