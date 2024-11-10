import os

import shutil
import random
import time
import math as m
import numpy as np
import sys
import json

import choreo 

import js
import pyodide

js.postMessage(
    funname = "Python_Imports_Done",
    args    = pyodide.ffi.to_js(
        {
        },
        dict_converter=js.Object.fromEntries
    )
)
