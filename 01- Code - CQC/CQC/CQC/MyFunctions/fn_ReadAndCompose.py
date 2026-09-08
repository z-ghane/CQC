from torchvision import transforms
from MyClasses.cls_Rows import cls_Rows
import pandas as pd
from Config import Config_Other as cfg
from Config import Config_Paths as cfgp

def fn_ReadAndCompose():
    # Read File

    df_main = pd.read_csv(
        cfgp.mypaths["data"]["dataset"], 
        names = cfg.preprocessing_params["data"]["dataset"]["columns_name"], 
        dtype = cfg.preprocessing_params["data"]["dataset"]["columns_dtype"],
        header = None, 
        skip_blank_lines = True
    )

    #print("df_main length            : ", len(df_main))

    len_df_main_before_compse = len(df_main)
    


    ## obj
    composed_pre = transforms.Compose([
        cls_Rows(
            cfg.preprocessing_params["data"]["dataset"]["columns_name"], 
            cfg.preprocessing_params["data"]["dataset"]["bug_classes"]
        )
    ])

    df_main = composed_pre(df_main)

    len_df_main_after_compse = len(df_main)
    
    
    texts = df_main["text"].tolist()
    labels = df_main["bug_class_2"].tolist()


    print("len df_main before compose: ", len_df_main_before_compse)
    print("len df_main after  compose: ", len_df_main_after_compse)
    

    return texts, labels, df_main

