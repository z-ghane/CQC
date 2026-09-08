import numpy as np
import json

from Config import Config_Other as cfg
from Config import Config_Paths as cfgp

from MyClasses.cls_MyConfusionMatrix import cls_MyConfusionMatrix

from MyFunctions.fn_ReadAndCompose import fn_ReadAndCompose
from MyFunctions.fn_GetPerformanceResultInJsonFormat import fn_GetPerformanceResultInJsonFormat
from MyFunctions.fn_SaveDataToFile import fn_SaveDataToFile
from MyFunctions.fn_ShowCfMatrixDetails import fn_ShowCfMatrixDetails




# ================================================
#         initial
# ================================================

print()
print("*** initial ***")
print()

processingMethod = "random"



# ================================================
#         set paths
# ================================================

print()
print("*** set paths ***")
print()


cfgp.mypaths["output"]["performance"] = (
    cfgp.createOutputFilePath(processingMethod)
    )


print("-" * 15)
print("mypaths:")
print(json.dumps(cfgp.mypaths, indent=4, sort_keys=True))

print()
print("-" * 15)
print("preprocessing_params:")
print(json.dumps(cfg.preprocessing_params, indent=4, sort_keys=True))



# ================================================
#         Read Files
# ================================================

print()
print("*** Read Files ***")
print()

texts, labels, _ = fn_ReadAndCompose()



# ================================================
#         Predict
# ================================================

print()
print("*** Predict ***")
print()

predicted = np.random.choice(
    cfg.class_probability["classes"], 
    len(labels),
    p = cfg.class_probability["probabilities"]
    )



# ================================================
#         Calculate confusion_matrix
# ================================================

print()
print("*** Calculate confusion_matrix ***")
print()

confusion_matrix = cls_MyConfusionMatrix(
    cfg.preprocessing_params["data"]["dataset"]["num_bug_classes"]
    )

confusion_matrix.update(labels, predicted)



# ================================================
#         Save Result
# ================================================

print()
print("*** Save Result ***")
print()

temp_PerformanceResultInJsonFormat = fn_GetPerformanceResultInJsonFormat(
    processingMethod,
    confusion_matrix
    )

fn_SaveDataToFile(
    temp_PerformanceResultInJsonFormat)



# ================================================
#         Show Result
# ================================================

print()
print("*** Show Result ***")
print()

cf_matrix = confusion_matrix.get_cf()
cf_matrix = np.array(cf_matrix)


# ----- ShowCfMatrixDetails
fn_ShowCfMatrixDetails(cf_matrix)
print()


# ----- should have equal values
print("-" * 15)
print("len(texts)     : ", len(texts))
print("len(labels)    : ", len(labels))
print("len(predicted) : ", len(predicted))
print()


