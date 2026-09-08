import json

import cls_Preprocessing_TFIDF

from Config import Config_Other as cfg
from Config import Config_Paths as cfgp

from MyFunctions.fn_ReadAndCompose import fn_ReadAndCompose
from MyFunctions.fn_SaveDataToFile import fn_SaveDataToFile


# for the same input, 
#       the output of this .py file  will be the same
#       so it needs to be run just ONE time.




# ================================================
#         initial
# ================================================

print()
print("*** initial ***")
print()

processingMethod = "genTFIDF"



# ================================================
#         set paths
# ================================================

print()
print("*** set paths ***")
print()


cfgp.mypaths["output"]["performance"] = (
    cfgp.createOutputFilePath(processingMethod, isForTFIDF=True)
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
#         Vectorize
# ================================================

print()
print("*** Vectorize ***")
print()

ds = cls_Preprocessing_TFIDF.cls_Preprocessing_TFIDF()

ds.tokenize(texts)
ds.vectorize_tfidf()


# See properties

# --- should have same values
# print("len df_main before compose :", len_df_main_before_compse)
# print("len df_main after  compose :", len_df_main_after_compse)
# print("df_main length             :", len(df_main))
print("vector_tfidf               : ", len(ds.vector_tfidf))
print("-" * 40)
# should have same values | after applying token_threshold
print("bugRepTokens               : ", len(ds.bugRepTokens))
# --- end

print("docMaxTokenNo_org          : ", ds.docMaxTokenNo_org) # orginal
print("w2vDic           : ", len(ds.w2vDic)) # vocabulary



# ================================================
#         Save Result
# ================================================

print()
print("*** Save Result ***")
print()

fn_SaveDataToFile(ds.vector_tfidf)



# ================================================
#         Show Result
# ================================================

print()
print("*** Show Result ***")
print()








