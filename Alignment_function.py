
import ast
import numpy as np
from nltk.tokenize import RegexpTokenizer  ### for nltk word tokenization
import json
tokenizer = RegexpTokenizer(r'\w+')

stop_words=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()

stop_words=[lmtzr.lemmatize(w1) for w1 in stop_words]
stop_words=list(set(stop_words))


def Ques_Emb(ques1, word_embs, IDF):

    emb_size = len(word_embs[ques1[0]])
    # print("emb size is: ",emb_size)
    Ques_Matrix = np.empty((0, emb_size), float)
    IDF_Mat = np.empty((0, 1), float)  ##### IDF is of size = 1 coz its a value
    query_term = []
    for q_term in ques1:
        if q_term not in word_embs.keys():
           print("This should not happen, recheck whats going on ", ques1) ## print the index to trace better
        else:
           Ques_Matrix = np.append(Ques_Matrix, np.array([word_embs[q_term]]), axis=0)
           if q_term in IDF.keys():
              IDF_Mat = np.append(IDF_Mat, np.array([[IDF[q_term]]]), axis=0)
           else:
              IDF_Mat = np.append(IDF_Mat, np.array([[3]]), axis=0)
           query_term.append(q_term)

    return Ques_Matrix, IDF_Mat, query_term




def Word2Vec_score(Question, IDF_Mat, Corpus, IDF, Justification_threshold, Q_terms_list, emb_size):

    Document_score=[[0] for i in range(len(Question))]
    Justification_ind = [[0] for i in range(len(Question))]
    Jind = 0
    with open(Corpus) as f:
        for line in f:
            just_data = json.loads(line)

            for ind1 in range(Justification_threshold):

                Doc_Matrix = np.empty((0, emb_size), float)  ####################### DIMENSION of EMBEDDING
                for key2 in just_data[str(ind1)]["word_emb"]:

                    Doc_Matrix = np.append(Doc_Matrix, np.array([just_data[str(ind1)]["word_emb"][key2]]), axis=0)
                if Doc_Matrix.size == 0:
                    print("this situation should never come, check your retrieved justifications: ")
                    pass
                else:
                    Doc_Matrix = Doc_Matrix.transpose()
                    ques1 = Question[Jind]
                    Score = np.matmul(ques1, Doc_Matrix)

                    Score = np.sort(Score, axis=1)

                    ## max score
                    max_score1 = Score[:, -1:]  ## taking just one positive alignment
                    max_score1 = np.multiply(IDF_Mat[Jind], max_score1)
                    max_score = 0

                    for qind1, qword1 in enumerate(max_score1):
                        for i1, s1 in enumerate(qword1):
                            max_score += (s1 / float(i1 + 1))

                    ## min score
                    min_score = Score[:, 0:1]
                    min_score1 = np.multiply(IDF_Mat[Jind], min_score)
                    # min_score_d = np.multiply(np.transpose(Doc_IDF_Mat_min), min_score)  ### Becky suggestion which is not working

                    min_score = 0
                    for qind1, qword1 in enumerate(min_score1):

                        for i1, s1 in enumerate(qword1):
                            min_score += (s1 / float(i1 + 1))  ## i1 +

                    total_score = max_score + 0.2 * (min_score)  ## + max_score_d + min_score_d
                    total_score = total_score / float(ques1.shape[0])
                    Document_score[Jind].append(total_score)
            if Jind%50==0:
               print("we have calculated alignment upto this justification: ",Jind)
               print("and the alignment scores are: ", Document_score[Jind])
            Jind+=1

    return Document_score

