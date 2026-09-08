import time
import os


# ================================================
#         set variable
# ================================================

# ------ output_version 

output_version = 1


# ------ software_name 

# software_name = "Camel"
# software_name = "CloudStack"
# software_name = "Geode"
software_name = "Hbase"



# ================================================
#         somehow static variable
# ================================================

dataset_file_names = {
    "Camel":      "Camel_DE - v.02",
    "CloudStack": "CloudStack_DE - v.01",
    "Geode":      "Geode_DE - v.01",
    "Hbase":      "Hbase_DE - v.01"
}

Methods_Folder_names = {
    "random":           "01_RandomLabelingOfBugs",
    "MultiNB":          "10_NaiveBayesMultinomialNB",
    "genTFIDF":         "11_TFIDF",
    "MultiNB-w2v-NKB":  "12_NaiveBayesMultinomialNB_w2v",
    "MultiNB-w2v-KB":   "12_NaiveBayesMultinomialNB_w2v",
    "DT":               "20_DecisionTree",
    "DT-w2v-NKB":       "21_DecisionTree_w2v",
    "DT-w2v-KB":        "21_DecisionTree_w2v",
    "knnCS":            "30_KNN_CosineSimilarity",
    "CNN-KB":           "50_OnePhaseMethod_CNN",
    "CNN-NKB":          "50_OnePhaseMethod_CNN"
}

output_file_names = {
    "random":           "_random-performance_",
    "MultiNB":          "_MultiNB-performance_",
    "genTFIDF":         "_FromOnePhaseData_tfidf-word-weights",
    "MultiNB-w2v-NKB":  "_MultiNB-w2v-NKB-performance_",
    "MultiNB-w2v-KB":   "_MultiNB-w2v-KB-performance_",
    "DT":               "_DT-tfidf-performance_",
    "DT-w2v-NKB":       "_DT-w2v-NKB-performance_",
    "DT-w2v-KB":        "_DT-w2v-KB-performance_",
    "knnCS":            "_knnCS-performance_",
    "CNN-KB":           "_FromOnePhaseData_CNN-w2v-KB-performance_",
    "CNN-NKB":          "_FromOnePhaseData_CNN-w2v-NKB-performance_"
}



# ================================================
#         mypaths
# ================================================

mypaths = {

    "data": {
        "dataset":              "",
        "tfidf_word_weights":   "",
        "w2v_word_vectors":     ""
    },

    "output": {
        "performance": ""
    }
}


# ================================================
#         funcions
# ================================================

# ------- createDatasetFilePath 

def createDatasetFilePath():
    
    return (
        # --- dataset folder path:
        "../../../10- Data/one-phase method/" + 
        software_name + "/" + 
        # --- dataset file name:
        dataset_file_names[software_name] + ".csv"
        )


mypaths["data"]["dataset"] = createDatasetFilePath()
mypaths["data"]["w2v_word_vectors"] = "../../../10- Data/w2vGoogle/GoogleNews-vectors-negative300.bin"


# ------- createOutputFilePath 

def createOutputFilePath(
    processingDataMethod, 
    isForTFIDF=False,
    isCNNBalance=False
    ):

    temp_outputFileVersion = ""
    temp_output_subfolder = ""
    temp_output_subFilename = ""


    if(not isForTFIDF):
        temp_outputFileVersion = (
            time.strftime("%Y-%m-%d_%H-%M") +
            "_v.{}".format(output_version)
            )


    if ("-NKB" in processingDataMethod):
        temp_output_subfolder = "NKB/"
    elif ("-KB" in processingDataMethod):
        temp_output_subfolder = "KB/"


    if("CNN" in processingDataMethod):
        if(isCNNBalance):
            temp_output_subFilename = "_balanced_"
        else:
            temp_output_subFilename = "_imbalance_"

    return (
        # --- output folder path:
        Methods_Folder_names[processingDataMethod] +
        "/00- Output/" + 
        software_name + "/" + 
        #dataset_file_names[software_name] + "/" +
        temp_output_subfolder +
        # --- output file name:
        #software_name +
        dataset_file_names[software_name] +
        output_file_names[processingDataMethod] +
        temp_output_subFilename +
        # output file version
        temp_outputFileVersion + 
        # file extension
        ".json"
        )


mypaths["data"]["tfidf_word_weights"] = createOutputFilePath("genTFIDF", True)


# ------- getOutputFilePath 

def getOutputFilePath(processingDataMethod):
    # we choose the last file in the output directory
    # we just need the directory
    # so a file with the name may not really exist
    tempOutputFileDirectory = createOutputFilePath(processingDataMethod)
    myOutputDirectory = os.path.dirname(tempOutputFileDirectory)

    myOutputFileName = os.listdir(myOutputDirectory)[-1]
    #myOutputFileName = "Hbase_random-performance_2025-12-08_22-27_v.1.json"
    
    return (myOutputDirectory + "/" + myOutputFileName), myOutputDirectory





