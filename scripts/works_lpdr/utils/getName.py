import pandas as pd
import os
import sys
import numpy as np

# 添加项目根目录到 sys.path
project_root_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(project_root_dir)

from src.constantClass.getBaseInfo import getBaseInfo
from src.constantClass import productName

shopinfo = getBaseInfo()["shopInfo"]
sale_Product = productName.WEILONG
big_Product = productName.BIG_PRODUCT
