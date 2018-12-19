# Dataset Conversion
import os
import json
import collections
from Overlap_analysis import calculate_overlap


Space=' '
new_line=[]

######## Change configs from here
write_label_files = 0
overlap_analysis = 0

language_dataset = "English"

if language_dataset=="English":

    json_file_name = "English_pred_labels_perEpoch_dev.jsonl"
    pred_directory = "stability_test_2/"
    NER_test_file =  'English_NER_data/valid.txt'
    smooth_file = open("English_dev_20_C.txt", "w")

elif language_dataset == "Dutch":
    json_file_name = "Dutch_pred_labels_perEpoch_withOthers.jsonl"  # "English_pred_labels_perEpoch.jsonl"
    pred_directory = "Res_full3/"  # "stability_test/"
    NER_test_file = "Dutch_NER_data/ned.testb.txt"  # 'English_NER_data/valid.txt'
    smooth_file = open("Dutch_testb_20_C.txt", "w")

## if you want to write the merge files, then do this, else just read the scores from json file.
if write_label_files == 1:
    linenum=0
    list1=[48]
    all_pred_files = os.listdir(pred_directory) ## list of all predicted files

    All_predicted_labels_json = open(json_file_name,"w")
    # name1="AWS_GPU_BEST_"
    epoch_num = []
    predictions_each_epoch = []

    for pred_file_name in all_pred_files:
        linenum = 0
        ind = 0
        pred_labels_dict = {}

        pred_file=open(pred_directory+pred_file_name,'r')
        epoch_num.append(pred_file_name.split("_")[-1]) ## storing all the epoch nums
        Word_array = []
        for line in pred_file:
            # word=line.split()
            word=line.strip()
            Word_array.append(word)

        file3="Res_dummy"
        #empty_file=open(file3,'a')
        file5=open(NER_test_file,'r', encoding="ISO-8859-1")
        print (pred_file_name)
        line_counter = 0
        for line in file5:                ## Spanish

            line_counter+=1
            if linenum>len(Word_array):
               pass
            else:
                #print (linenum)
                linenum+=1
                words = list(line.split())
                # print words
                if len(words)>1:
                   new_line=words[0]+Space+words[-1]+Space+str(Word_array[ind])

                   # if words[-1] == Word_array[ind]:
                   if Word_array[ind] in pred_labels_dict.keys():
                      pred_labels_dict[Word_array[ind]].append(line_counter)
                   else:
                      pred_labels_dict.update({Word_array[ind]:[line_counter]})

                   ind=ind+1
                   new_line=[]
                else:
                    if len(Word_array[ind])!=0:
                       print ("Mismatch in line numbers- cross veryify")
                       break

                    ind=ind+1

        predictions_each_epoch.append(pred_labels_dict)
        json_line = {str(epoch_num[-1]):pred_labels_dict}
        json.dump(json_line, All_predicted_labels_json)
        All_predicted_labels_json.write("\n")

    print (len(predictions_each_epoch), len(epoch_num))

else: # read from the json file
   json_file = open(json_file_name,"r")
   predictions_each_epoch = []
   epoch_num = []

   for line in json_file:
       json_data = json.loads(line)
       for key2 in json_data.keys(): ## there will be only 1 key, i.e. current epoch number
           predictions_each_epoch.append(json_data[key2])
           epoch_num.append(key2)

## now calculating overlap between consecutive epochs
if overlap_analysis == 1:
    for index1, epoch in enumerate(epoch_num[:-6]):

        overlap_vals = calculate_overlap(predictions_each_epoch[index1], predictions_each_epoch[index1+1])
        print (epoch_num[index1], epoch_num[index1+4], overlap_vals)


## calculating ensemble:
start_epoch = 160
end_epoch = 200

All_scores_counted_over_epochs = {}

for ind1, epoch_nums1 in enumerate(epoch_num):
    if int(epoch_nums1)>=start_epoch and int(epoch_nums1)<end_epoch:
       for label1 in predictions_each_epoch[ind1]:
           if label1 in All_scores_counted_over_epochs.keys():
              All_scores_counted_over_epochs[label1]+=(predictions_each_epoch[ind1][label1]) ## we want to maintain a single list for the collection counter to work in a single line
           else:
              All_scores_counted_over_epochs.update({label1:predictions_each_epoch[ind1][label1]})

# All_scores_with_counts = {}

for key1 in All_scores_counted_over_epochs.keys():
    All_scores_counted_over_epochs[key1] = collections.Counter(All_scores_counted_over_epochs[key1])

print (All_scores_counted_over_epochs)

##### Writing the smoothed scores over epochs

file5=open(NER_test_file,'r', encoding="ISO-8859-1")

Low_precision_threshold = 30
High_precision_threshold = 38

threshold_dict = {"B-MISC":Low_precision_threshold, "I-MISC":Low_precision_threshold, "B-ORG":Low_precision_threshold, "I-ORG":Low_precision_threshold, "B-PER":High_precision_threshold, "I-PER":High_precision_threshold, "B-LOC":High_precision_threshold, "I-LOC":High_precision_threshold, "O":High_precision_threshold }
precision_threshold_vals = {}
recall_threshold_vals = {}

line_counter = 0
for line in file5:                ## Spanish

    line_counter+=1

    words = list(line.split())
    # print words
    if len(words)>1:

      new_line=words[0]+Space+words[-1]+Space
      candidate_label={}
      for labkey in All_scores_counted_over_epochs.keys():
          if line_counter in All_scores_counted_over_epochs[labkey].keys():  ## check if it is present
             if  All_scores_counted_over_epochs[labkey][line_counter] > threshold_dict[labkey]: ## threshold   ## higher threshold will improve precision and lower will improve recall - tune to get best Fscore

                 candidate_label.update({labkey:All_scores_counted_over_epochs[labkey][line_counter]})

      if len(candidate_label)>0:
         max_count = 0
         for labkey1 in candidate_label.keys():
             if candidate_label[labkey1]>max_count:
                predicted_label = labkey1

         new_line+=predicted_label + "\n"
      else:
         print ("so, we do come here which is nice... ")
         new_line+= "O" +"\n"


      if len(new_line.split())!=3:
         print ("check this line number ",line_counter)
      smooth_file.write(new_line)
    else:
       smooth_file.write("\n")
