
import ast, os
import numpy as np
import collections
import json


embeddings_index = {}
"""
# glove_emb = open('glove.6B.100d.txt','r', encoding='utf-8')
# f = open(os.path.join("/extra/vikasy/","glove.840B.300d.txt"),'r', encoding='utf-8')
f = open(os.path.join("/Users/vikasy/Glove_vectors/","glove.840B.300d.txt"),'r', encoding='utf-8')
# f = open('all_emb_FB.txt','r', encoding='utf-8')

#f = open('ss_qz_04.dim50vecs.txt')
for line in f:
    values = line.split()
    word = values[0]
    # word=lmtzr.lemmatize(word)
    try:
       coefs = np.asarray(values[1:], dtype='float32')
       b = np.linalg.norm(coefs, ord=2)
       coefs=coefs/b
       emb_size=coefs.shape[0]
    except ValueError:
       print (values[0])
       continue
    embeddings_index[word] = coefs

"""





from Word_segment import parse_documents
from IDF import cal_IDF, Query_IDF
from Preprocess_ARC import Preprocess_Arc, Preprocess_KB_sentences, Write_ARC_KB, get_IDF_weights, Query_boosting_sent
import math
from Alignment_function import Word2Vec_score, Ques_Emb
# from BM_25_Clark import Word2Vec_score, Ques_Emb
from Evaluation_ranking import evals, SIGIR_just
import collections


from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()


Question_json_file = "/xdisk/vikasy/ARC-Easy-Test_FLAIR_Word_emb.jsonl"

cols_sizes, questions, candidates, algebra, All_words, correct_ans, negative_ques=Preprocess_Arc("ARC","/extra/vikasy/ARC_Challenge_FLAIR/ARC_corpus/ARC-Easy/ARC-Easy-Test.csv").preprocess()

print (len(questions), len(candidates))
print (collections.Counter(correct_ans),"correct ans len")

# Query_IDF(All_words,"IDF_doc_Test.txt") ### If you want to build IDF of query terms, call this function.

file2=open("IDF_doc_Test.txt","r")
for line2 in file2:
    IDF=ast.literal_eval(line2)

## read PMI file
"""
PMI_values = open("PMI_ARC_8th_grade_ARISTO_window_10.txt","r")
for line3 in PMI_values:
    PMI_vals=ast.literal_eval(line3)
"""

Correct_ans = []#[]
All_questions = []
IDF_Mat=[]

cands=[]
cands_idf=[]
All_q_terms = []

counter_1 = 0
with open(Question_json_file) as f:
    for line in f:
        if counter_1%50==0:
           print(counter_1)
        counter_1 += 1

        ques_data = json.loads(line)
        preprocessed_sent = Preprocess_KB_sentences(ques_data["sent_text"], 1)
        # Query_file_lucene.write(preprocessed_sent) #### (str(ind) + " " + preprocessed_sent)
        Q1_matrix, IDF_Mat1, q_term1 = Ques_Emb(preprocessed_sent.split(), ques_data["word_emb"], IDF)

        cands.append(Q1_matrix)
        cands_idf.append(IDF_Mat1)
        All_q_terms.append(q_term1)

    All_questions += cands
    IDF_Mat += cands_idf
    cands=[]
    cands_idf=[]


performance_file = open("ARC_easy_test_performance_BiDAF_0.2_neg.txt","w")
# J_Threshold_set = [i+1 for i in range(9)]
J_Threshold_set = [2]

Ensemble_file = "EASY_test_goodJ_scores.txt"
out_file = open("EASY_test_good_J_scores.txt","w")
file1="/xdisk/vikasy/ARC-Easy-Test_JUSTIFICATION_FLAIR_Word_emb_10.jsonl"
for J_Threshold in J_Threshold_set:
    # file1="Justifications_Easy_TEST_cand_BOOST_3_40_BM25.txt"

    Score_matrix = Word2Vec_score(All_questions, IDF_Mat,  file1, IDF, J_Threshold, All_q_terms, 4196)

    Accuracy, Ranked_justifications_index=evals(Score_matrix, candidates, correct_ans, Ensemble_file)

    print ("We are getting this much ",Accuracy)

    print ("the threshold value is: ", J_Threshold)
    newline = str(J_Threshold) + "\t" + str(Accuracy)
    performance_file.write(newline + "\n")

    SIGIR_ranked_file = "SIGIR_easy_test_justification_4.txt"
    SIGIR_just(file1, SIGIR_ranked_file, Ranked_justifications_index)

    print("ranked justifications look like: ", collections.Counter(Ranked_justifications_index))
