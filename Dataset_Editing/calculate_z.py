#! /usr/bin/python

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "arwawali"
__date__ = "$Jun 17, 2015 6:01:25 PM$"
import sys
from common_functions import readdataFile
import math
if __name__ == "__main__":
    data_path=sys.argv[1]
    dataset_name=sys.argv[2]
    z_value=sys.argv[3]
    data=readdataFile(data_path+dataset_name+".dvf")
    cols=len(data[0])
    del data
    spars_cols=int(math.ceil(float(cols)*float(z_value)))
    print spars_cols
