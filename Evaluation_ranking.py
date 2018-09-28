
## Calculating accuracy here in the same file.
import ast
import numpy as np
from statistics import mean

def SIGIR_just(overlap_just_name, SIGIR_just_name, Ranked_indexes):
    overlap_just_file = open(overlap_just_name,"r")
    SIGIR_just = open(SIGIR_just_name,"w")
    count = 0
    for line in overlap_just_file:
        justifications = line.strip().split("\t")
        best_just = justifications[Ranked_indexes[count]]
        SIGIR_just.write(best_just+"\n")
        count+=1

def evals(scores, candidates, Correct_ans, outfile1, write1 = 0):
    if write1 == 1:
        outfile = open(outfile1,"w")
    ind_score=[]
    All_score=[]
    Predicted_ans=[]
    Accuracy = 0
    score_index=0
    Ranked_justifications_index = []
    print (len(candidates))
    new_line = ""
    for cindex,cand1 in enumerate(candidates):
        num_options=len(cand1)
        upper_limit=score_index + num_options
        while score_index < upper_limit:
              # print (cand1, score_index, scores[score_index])

              final_score=0

              Ranked_justifications_index.append(scores[score_index].index(max(scores[score_index])))

              for i1, s1 in enumerate(scores[score_index]):
                  final_score+= ((s1) /float(i1+1))
              # ind_score.append(sum(scores[score_index]))
              ind_score.append(final_score)
              new_line+= " " + str(final_score)
              # All_score.append(sum(scores[score_index]))
              score_index+=1
        ind_score=np.asarray(ind_score)

        # Predicted_ans.append(np.argmax(ind_score))
        new_line+= "\t" + str(Correct_ans[cindex]) + "\n"
        if write1==1:
           outfile.write(new_line)
        new_line=""
        if Correct_ans[cindex]==np.argmax(ind_score):
           Accuracy+=1
        ind_score = []

    return (Accuracy/float(cindex)), Ranked_justifications_index