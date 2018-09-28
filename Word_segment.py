import glob
from nltk.tokenize import RegexpTokenizer  ### for nltk word tokenization
tokenizer = RegexpTokenizer(r'\w+')
stop_words=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']

class word_segment:
    def __init__(self, line):
        self.words=[]
        self.line=line

    def seg_word(self):
        self.words=tokenizer.tokenize(self.line)
        self.words=[word1 for word1 in self.words if word1 not in stop_words]
        return self.words

class parse_documents(word_segment):
    def __init__(self, Corpus, directory):
       self.Corpus=Corpus
       self.directory1=str(directory)
       self.term_freq={}
       self.posting_list={}
       self.num_doc=0

    def parse_doc(self):

       for doc1 in glob.glob(self.directory1+"*.txt"):
           text1=open(doc1,"r", encoding="UTF-8").readlines()
           # lines1=line_segment.get_lines(text1)
           for line_num, line in enumerate(text1):
               dummy_word=word_segment(line).seg_word()
               self.num_doc+=1
               for word1 in dummy_word:
                   if word1 in self.term_freq.keys():
                      self.term_freq[str(word1)]+=1
                      self.posting_list[str(word1)].append(line_num)
                   else:
                      self.term_freq.update({str(word1):1})
                      self.posting_list.update({str(word1):[]})
       return self.posting_list, self.term_freq, self.num_doc