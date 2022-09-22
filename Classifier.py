from selectors import EpollSelector
import sys
from collections import defaultdict
import math
import random
import os
import os.path


def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    
    ngram = []   
    
    for i in range(n-1):
        sequence.insert(0,"START")
    if n == 1:
        sequence.insert(0,"START")

    sequence.append("STOP")

    for i in range(len(sequence)+2):
        if i + n > len(sequence):
            break
        temp = []
        for j in range(i,i+n):
            temp.append(sequence[j])
        ngram.append(tuple(temp))
    
    for i in range(n-1):
        sequence.pop(0)
    if n == 1:
        sequence.pop(0)

    sequence.remove("STOP")
    return ngram


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)
        self.bigramcountsdenom = 0
        self.unigramcountsdenom = 0
        self.trigramcountsdenom = 0
        for i in self.trigramcounts:
            self.trigramcountsdenom += self.trigramcounts[i]
        for i in self.bigramcounts:
            self.bigramcountsdenom += self.bigramcounts[i]
        for i in self.unigramcounts:
            self.unigramcountsdenom += self.unigramcounts[i]


    def count_ngrams(self, corpus):
        

        self.unigramcounts = {} # might want to use defaultdict or Counter instead
        self.bigramcounts = {}
        self.trigramcounts = {}

        for sentence in corpus:
            for i, word in enumerate(sentence):
                if i == 0:
                    if ("START",) in self.unigramcounts:
                        self.unigramcounts[("START",)] += 1
                    else:
                        self.unigramcounts[("START",)] = 1
                    if (word,) in self.unigramcounts:
                        self.unigramcounts[(word,)] += 1
                    else:
                        self.unigramcounts[(word,)] = 1

                    if ('START',word) in self.bigramcounts:
                        self.bigramcounts[('START',word)] += 1
                    else:
                        self.bigramcounts[('START',word)] = 1
                    if ('START','START',word) in self.trigramcounts:
                        self.trigramcounts[('START','START',word)] += 1
                    else:
                        self.trigramcounts[('START','START',word)] = 1
                if i == 1: 
                    if ('START',sentence[i-1],word) in self.trigramcounts:
                        self.trigramcounts[('START',sentence[i-1],word)] += 1
                    else:
                        self.trigramcounts[('START',sentence[i-1],word)] = 1
                else:
                    if (word,) in self.unigramcounts:
                        self.unigramcounts[(word,)] += 1
                    else:
                        self.unigramcounts[(word,)] = 1
                    
                    if (sentence[i-1],word) in self.bigramcounts:
                        self.bigramcounts[(sentence[i-1],word)] += 1
                    else:
                        self.bigramcounts[(sentence[i-1],word)] = 1
                    
                    if (sentence[i-2],sentence[i-1],word) in self.trigramcounts:
                        self.trigramcounts[(sentence[i-2],sentence[i-1],word)] += 1
                    else:
                        self.trigramcounts[(sentence[i-2],sentence[i-1],word)] = 1

                    
                if i == len(sentence) - 1:
                    if ("STOP",) in self.unigramcounts:
                        self.unigramcounts[('STOP',)] += 1
                    else:
                        self.unigramcounts[('STOP',)] = 1
                    if (sentence[i],'STOP') in self.bigramcounts:

                        self.bigramcounts[(sentence[i],'STOP')] += 1
                    else:
                        self.bigramcounts[(sentence[i],'STOP')] = 1
                    if (sentence[i-1],sentence[i],'STOP') in self.trigramcounts:

                        self.trigramcounts[(sentence[i-1],sentence[i],'STOP')] += 1
                    else:
                        self.trigramcounts[(sentence[i-1],sentence[i],'STOP')] = 1

              
       
        return

    def raw_trigram_probability(self,trigram):
        
        num = 0
        if trigram in self.trigramcounts:
            num = self.trigramcounts[trigram]       
        return num/self.trigramcountsdenom
        

    def raw_bigram_probability(self, bigram):
         
        num = 0
        if bigram in self.bigramcounts:
            num = self.bigramcounts[bigram]
        
        return num/self.bigramcountsdenom
        
    
    def raw_unigram_probability(self, unigram):
        
        num = 0
        if unigram in self.unigramcounts:
            num = self.unigramcounts[unigram]
        
        return num/self.unigramcountsdenom
                     

    def smoothed_trigram_probability(self, trigram):
        
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0     
        
        term1 = lambda1 * (self.raw_trigram_probability(trigram[0:3])) 
        term2 = lambda2 * (self.raw_bigram_probability(trigram[1:])\
            /(self.raw_unigram_probability(trigram[1:2])+10**-10) * self.raw_bigram_probability(trigram[0:2]))

        term3 = lambda3 * (self.raw_unigram_probability(trigram[2::])\
             * self.raw_bigram_probability(trigram[0:2]))


        return term1 + term2 + term3
        
    def sentence_logprob(self, sentence):
        
        sum = 0.0
        ngrams = get_ngrams(sentence,3)
        for i in ngrams:
            temp = self.smoothed_trigram_probability(i)
            if temp != 0:                
                sum += math.log2(temp)


        return float(sum)

    def perplexity(self, corpus):
        
        sum = 0.0
        for i in corpus:
            sum += self.sentence_logprob(i)
        sum = sum/self.unigramcountsdenom
        
        return 2**(-sum)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            if pp >= pp2:                
                correct+=1
            total+= 1
        for f in os.listdir(testdir2):
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp2 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            if pp >= pp2:                
                correct+=1
            total+= 1
        
        return correct/total

if __name__ == "__main__":

    
    acc = essay_scoring_experiment('train_high.txt', 'train_low.txt", "test_high", "test_low")
    print(acc)

