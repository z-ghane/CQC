import json

from Config import Config_Other as cfg
from Config import Config_Paths as cfgp

from MyFunctions.fn_ReadOutputFile import fn_ReadOutputFile
from MyFunctions.fn_ShowCfMatrixDetails import fn_ShowCfMatrixDetails
from MyFunctions.fn_PlotConfusionMatrics import fn_PlotConfusionMatrics


def fn_ShowOutputFileResults(processingMethod, outputFileName=""):

    # ************** set paths **************

    print()
    print("*** set paths ***")
    print()

    outputFilePath, outputFolderPath = cfgp.getOutputFilePath(processingMethod)
    
    # Empty strings are "falsy"
    if not outputFileName:
        cfgp.mypaths["output"]["performance"] = (
            outputFilePath
            )
    else:
        cfgp.mypaths["output"]["performance"] = (
            outputFolderPath + "/" + outputFileName
            )
        


    print("-" * 15)
    print("mypaths:")
    print(json.dumps(cfgp.mypaths, indent=4, sort_keys=True))

    print()
    print("-" * 15)
    print("preprocessing_params:")
    print(json.dumps(cfg.preprocessing_params, indent=4, sort_keys=True))

    print()



    # ************** Read Output File **************
    
    print()
    print("*** Read Output File ***")
    print()

    class_labels, cf_matrix = fn_ReadOutputFile()



    # ************** Show Result **************
    
    print()
    print("*** Show Result ***")
    print()

    # ----- ShowCfMatrixDetails
    fn_ShowCfMatrixDetails(cf_matrix)
    print()

    # ----- PlotConfusionMatrics
    fn_PlotConfusionMatrics(cf_matrix, class_labels)


    return

