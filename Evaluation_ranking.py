
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

def evals(scores, candidates, Correct_ans, outfile1, J_threshold, write1 = 0):
    All_accuracies = []
    for J_thresh in range(J_threshold):
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

                  final_score=0

                  Ranked_justifications_index.append(scores[score_index][0:J_thresh+1].index(max(scores[score_index][0:J_thresh+1])))

                  for i1, s1 in enumerate(scores[score_index][0:J_thresh+1]):
                      final_score+= ((s1) /float(i1+1))

                  ind_score.append(final_score)
                  new_line+= " " + str(final_score)

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
            """
            if upper_limit>1820:
               Accuracy_1820 =  Accuracy/float(cindex)
               print("the accuracy till this point is: ", Accuracy_1820)
               break
            """

        Accuracy_1820 = Accuracy / float(cindex)
        print("Accuracy with ", J_thresh+1, " is ", Accuracy_1820)
        All_accuracies.append(Accuracy_1820)

    return (All_accuracies, Ranked_justifications_index)