import json

import numpy as np

from Config import Config_Other as cfg
from Config import Config_Paths as cfgp

from MyClasses.cls_MyConfusionMatrix import cls_MyConfusionMatrix

from MyFunctions.fn_ReadAndCompose import fn_ReadAndCompose
from MyFunctions.fn_GetPerformanceResultInJsonFormat import fn_GetPerformanceResultInJsonFormat
from MyFunctions.fn_SaveDataToFile import fn_SaveDataToFile
from MyFunctions.fn_ShowCfMatrixDetails import fn_ShowCfMatrixDetails
from MyFunctions.fn_Predict_MNB_DT_w2v import fn_Predict_MNB_DT_w2v



# seems that always have the same output
# that's why the save section could be commented

# ================================================
#         set variable
# ================================================

print()
print("*** set variable ***")
print()

# -------- token_threshold
token_threshold = 20000

# -------- is_keyword_Base
is_keyword_Base = True
# is_keyword_Base = False

# -------- my_docMaxLen
my_docMaxLen = 100 if is_keyword_Base else None



# ================================================
#         initial
# ================================================

# -------- processingMethod
processingMethod = "MultiNB-w2v-KB" if is_keyword_Base else "MultiNB-w2v-NKB"

# -------- cfg.preprocessing_params
cfg.preprocessing_params["is_keyword_Base"] = is_keyword_Base
cfg.preprocessing_params["docMaxLen"] = my_docMaxLen



# ================================================
#         set paths
# ================================================

print()
print("*** set paths ***")
print()


cfgp.mypaths["output"]["performance"] = (
    cfgp.createOutputFilePath(processingMethod)
    )


# --- print
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
#         predict
# ================================================

print()
print("*** predict ***")
print()

y_test, predicted = fn_Predict_MNB_DT_w2v(
    texts, labels, token_threshold, processingMethod)

print()
print(type(predicted))
print("len(texts)             : ", len(texts))
print("len(predicted)         : ", len(predicted))
print("len(labels)            : ", len(labels))



# ================================================
#         Calculate confusion_matrix
# ================================================

print()
print("*** Calculate confusion_matrix ***")
print()

confusion_matrix = cls_MyConfusionMatrix(
    cfg.preprocessing_params["data"]["dataset"]["num_bug_classes"])

confusion_matrix.update(y_test, predicted)



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



np.random.choice(
  ['pooh', 'rabbit', 'piglet', 'Christopher'], 
  5,
  p=[0.5, 0.1, 0.1, 0.3]
)


