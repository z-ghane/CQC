from Config import Config_Paths as cfgp
from Config import Config_Other as cfg




def fn_GetPerformanceResultInJsonFormat(whichMethod, confusion_matrix):
    tempStructure = {
        "dataset": cfgp.mypaths["data"]["dataset"],
        "preprocessing_params": cfg.preprocessing_params,
        #"class_probability": cfg.class_probability,
        "model_results": {
            "confusion_matrix": confusion_matrix.get_cf()
        }
    }
    if whichMethod == "random":
        items = list(tempStructure.items())
        items.insert(2, ('class_probability', cfg.class_probability))
        tempStructure = dict(items)

    return tempStructure

