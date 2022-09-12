import os
import js
import json

def DLSavedFiledJSNoPrompt(filename,readtype='rt'):

    with open(filename, readtype) as fh:
        thefile = fh.read()
        
    blob = js.Blob.new([thefile], {type : 'application/text'})
    url = js.window.URL.createObjectURL(blob) 

    downloadLink = js.document.createElement("a")
    downloadLink.href = url
    downloadLink.download = filename
    js.document.body.appendChild(downloadLink)
    downloadLink.click()
    downloadLink.remove()


ToPython = js.ToPython.to_py()

filename = "data.json"

with open(filename, "w") as jsonFile:
    jsonString = json.dumps(ToPython, indent=4, sort_keys=True)
    jsonFile.write(jsonString)
    

# DLSavedFiledJSNoPrompt(filename)