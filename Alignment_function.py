
import ast
import numpy as np
from nltk.tokenize import RegexpTokenizer  ### for nltk word tokenization
tokenizer = RegexpTokenizer(r'\w+')

stop_words=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()

stop_words=[lmtzr.lemmatize(w1) for w1 in stop_words]
stop_words=list(set(stop_words))


def Ques_Emb(ques1, IDF, embeddings_index, emb_size):
    query_term=[]
    Ques_Matrix = np.empty((0, emb_size), float)
    IDF_Mat = np.empty((0, 1), float)  ##### IDF is of size = 1 coz its a value
    for q_term in ques1:
        if q_term in embeddings_index.keys():
           Ques_Matrix = np.append(Ques_Matrix, np.array([embeddings_index[q_term]]), axis=0)
           if q_term in IDF.keys():
              IDF_Mat = np.append(IDF_Mat, np.array([[IDF[q_term]]]), axis=0)
           else:
              IDF_Mat = np.append(IDF_Mat, np.array([[3]]), axis=0)
           query_term.append(q_term)

    return Ques_Matrix, IDF_Mat, query_term




def Word2Vec_score(Question, IDF_Mat, Corpus, IDF, Justification_threshold, embeddings_index, emb_size, Q_terms_list, PMI_vals):

    Document_score=[[0] for i in range(len(Question))]
    Justification_ind = [[0] for i in range(len(Question))]
    PMI_scores = [[] for i in range(len(Question))]

    Corpus=open(Corpus,"r")
    for Jind, Justifications in enumerate(Corpus):

        if Jind%1000==0:
           print (Jind)
           # print(threshold_vals)
        Justification_set = []
        Justifications = Justifications.strip()
        cols = Justifications.split("\t")  ## cols[0] has the question number, cols[1]  has the candidate option number for that specific question.
        Feature_col = cols
        # print (len(Feature_col))
        if len(Feature_col) >= Justification_threshold:
            """
            for ind1 in range(Justification_threshold):  #### we take only top 10 justifications.
                just_sent = " ".join(Feature_col[ind1].strip().split()[1:])
                Justification_set.append(just_sent.lower())
            """
            for justifications1 in Feature_col:
                if len(Justification_set) < Justification_threshold:
                    just_sent = " ".join(justifications1.strip().split()[1:])
                    if len(just_sent) >= 0:
                        Justification_set.append(just_sent.lower())

        max_PMI = [[] for i in range(len(Q_terms_list[Jind]))]

        for just_ind, just1 in enumerate(Justification_set):
            Doc_set = tokenizer.tokenize(just1)
            # Doc_set=list(set(Doc_set))
            Doc_set = [lmtzr.lemmatize(w1) for w1 in Doc_set]
            Doc_set = [w for w in Doc_set if not w in stop_words]

            ## Calculating PMI scores here, we have justification terms from above and we have query terms from Q_term_list[Jind]

            for tind1, term in enumerate(Q_terms_list[Jind]):
                term_PMI = []
                for tj1 in Doc_set:
                    try:
                        term_PMI.append(float(PMI_vals[term + " " + tj1]))
                    except KeyError:
                        pass

                if len(term_PMI) > 0:
                    max_PMI[tind1] += sorted(term_PMI)[-1:]
                    max_PMI[tind1] += [min(term_PMI)]

            #####################

            Doc_Matrix = np.empty((0, emb_size), float)  ####################### DIMENSION of EMBEDDING
            Doc_len=0
            for key in Doc_set:
                if key in embeddings_index.keys():
                   Doc_Matrix=np.append(Doc_Matrix, np.array([embeddings_index[key]]), axis=0)
                   Doc_len+=1
            if Doc_Matrix.size==0:
               pass
            else:

                Doc_IDF_Mat = np.empty((0, 1), float)
                Doc_IDF_Mat_min = np.empty((0, 1), float)

                Q_term_Mat = np.empty((0, 1), float)

                Doc_Matrix=Doc_Matrix.transpose()
                #print(Doc_Matrix.shape)
                ques1=Question[Jind]
                #threshold_vals = math.ceil(0.75 * float(ques1.shape[0])) ## math.ceil
                threshold_vals = ques1.shape[0]

                Score=np.matmul(ques1,Doc_Matrix)
                max_indices = np.argmax(Score, axis=1)
                min_indices = np.argmin(Score, axis=1)
                max_list=[]
                for mind1 in max_indices:
                    if Doc_set[mind1] in IDF.keys():
                        Doc_IDF_Mat = np.append(Doc_IDF_Mat, np.array([[IDF[Doc_set[mind1]]]]), axis=0)
                        max_list.append(Doc_set[mind1])
                    else:
                        Doc_IDF_Mat = np.append(Doc_IDF_Mat, np.array([[5.379046132954042]]), axis=0)


                #if Jind<8:
                   #print (max_list)
                   #print (ques1.shape)
                counter2=0
                for mind1 in min_indices:
                    if Doc_set[mind1] in IDF.keys():
                        Doc_IDF_Mat_min = np.append(Doc_IDF_Mat_min, np.array([[IDF[Doc_set[mind1]]]]), axis=0)
                    else:

                        counter2+=1
                        Doc_IDF_Mat_min= np.append(Doc_IDF_Mat_min, np.array([[5.379046132954042]]), axis=0)

                Score = np.sort(Score, axis=1)
                max_score1 = Score[:, -1:]
                # max_score=np.multiply(np.transpose(IDF_Mat[Jind]),max_score)
                max_score1 = np.multiply(IDF_Mat[Jind], max_score1)

                # max_score=(sum(max_score1))#.item(0) ## this is the original without any threshold on the values.
                max_score = 0

                for qind1, qword1 in enumerate(max_score1):
                    # max_val=0
                    # qword1=qword1[::-1]
                    for i1, s1 in enumerate(qword1):
                        max_score += (s1 / float(i1 + 1))
                #max_score_d= (sum(max_score_d))

                #print (max_score)
                min_score = Score[:, 0:1]
                min_score1 = np.multiply(IDF_Mat[Jind], min_score)
                #min_score_d = np.multiply(np.transpose(Doc_IDF_Mat_min), min_score)  ### Becky suggestion which is not working

                min_score = 0
                for qind1, qword1 in enumerate(min_score1):
                    # qword1 = qword1[::-1]

                    for i1, s1 in enumerate(qword1):
                        min_score += (s1 / float(i1 + 1))  ## i1 +


                total_score=max_score + 0.2*(min_score)  ## + max_score_d + min_score_d
                total_score=total_score/float(ques1.shape[0])
                Document_score[Jind].append(total_score)
                Justification_ind[Jind].append(just_ind)

        for qterm_array in max_PMI:
            PMI_scores[Jind]+=qterm_array # [1:]

    return Document_score, Justification_ind, PMI_scores

