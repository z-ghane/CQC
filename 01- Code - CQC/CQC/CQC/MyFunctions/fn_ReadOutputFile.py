import numpy as np
import json

from Config import Config_Paths as cfgp

from MyFunctions.fn_ShowCfMatrixDetails import fn_ShowCfMatrixDetails
from MyFunctions.fn_PlotConfusionMatrics import fn_PlotConfusionMatrics




# ********************* fn_ShowDetailsAndPlot *********************

def fn_ReadOutputFile():

    # ----- Read Output File

    with open(cfgp.mypaths["output"]["performance"]) as filehandle:
        whole_data = json.load(filehandle)


    # bug_classes
    class_labels = whole_data["preprocessing_params"]["data"]["dataset"]["bug_classes"]

    # confusion_matrix
    cf_matrix = whole_data["model_results"]["confusion_matrix"]
    cf_matrix = np.array(cf_matrix)



    return class_labels, cf_matrix


