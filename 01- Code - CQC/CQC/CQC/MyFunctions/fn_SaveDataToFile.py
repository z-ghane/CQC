from Config import Config_Paths as cfgp

import json
import os


def fn_SaveDataToFile(myData):
    mypath = os.path.dirname(cfgp.mypaths["output"]["performance"])

    isExist = os.path.exists(mypath)
    if not isExist:
        os.makedirs(mypath)


    with open(cfgp.mypaths["output"]["performance"], "w") as fout:
        json.dump(myData, fout)



