from Preprocess_ARC import get_IDF_weights
import math

class cal_IDF:
    def __init__(self, dict, num_doc): ## We are taking 4 corpuses - WikiDump, AristoMini, Science_textbook and Flashcards
        self.dict1=dict
        self.num_doc=num_doc
    def get_IDF(self):
        for key1 in self.dict1.keys():
            self.dict1[key1]=self.dict1[key1]/float(self.num_doc)
def Cal_IDF(filename="Lemmatized_Arc_Coprus_stop_rem.txt"):
    IDF = {}
    for each_word in All_words:
        IDF[str(each_word)] = 0

    print ("vocab len should be same", len(IDF))
    Doc_lengths, All_Documents, AW1, IDF2 = get_IDF_weights(filename, IDF)

    print (len(IDF2))
    for terms_TF in All_Documents:
        for tf_key in terms_TF:
            terms_TF[tf_key] = 1 + math.log(terms_TF[tf_key])

    Total_doc = len(All_Documents)
    Avg_Doc_len = sum(Doc_lengths) / float(len(Doc_lengths))

    for each_word in All_words:
        doc_count = IDF2[str(each_word)]

        IDF2[str(each_word)] = math.log10((Total_doc - doc_count + 0.5) / float(doc_count + 0.5))

    IDF_file = open("IDF_doc_dev.txt", "w")
    IDF_file.write(str(IDF2))


def Query_IDF(All_words, file_name):
    IDF = {}
    for each_word in All_words:
        IDF[str(each_word)] = 0

    print ("vocab len should be same", len(IDF))
    Doc_lengths, All_Documents, AW1, IDF2 = get_IDF_weights("Lemmatized_Arc_Coprus_stop_rem.txt", IDF)

    print (len(IDF2))
    for terms_TF in All_Documents:
        for tf_key in terms_TF:
            terms_TF[tf_key] = 1 + math.log(terms_TF[tf_key])

    Total_doc = len(All_Documents)
    Avg_Doc_len = sum(Doc_lengths) / float(len(Doc_lengths))

    for each_word in All_words:
        doc_count = IDF2[str(each_word)]

        IDF[str(each_word)] = math.log10((Total_doc - doc_count + 0.5) / float(doc_count + 0.5))

    IDF_file = open(file_name, "w")
    IDF_file.write(str(IDF))

