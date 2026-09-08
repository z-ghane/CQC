from Config import Config_Paths as cfgp


temp = cfgp.mypaths["data"]["dataset"]
#temp = cfgp.w2vpath
print(temp)

# see if it could identify my relative path

from pathlib import Path
path_location = Path(temp)

print(path_location.exists())






