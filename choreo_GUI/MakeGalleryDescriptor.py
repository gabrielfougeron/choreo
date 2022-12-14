import sys
import os
import json

gallery_root = './choreo-gallery'
out_filename = "gallery_descriptor.json"

Gallery_dict = {}

for (dirpath, dirnames, filenames) in os.walk(gallery_root):
    parts = dirpath.split(os.sep)
    parts.pop(0)

    npy_json_pairs = {}
    for the_file in filenames:
        base,ext = os.path.splitext(the_file)
        if ext in ['.npy','.json']:
            if not(base in npy_json_pairs):
                npy_json_pairs[base] = {}
            npy_json_pairs[base][ext] = os.path.join(*parts,the_file)

    folder_dict = {}
    folder_dict.setdefault('name',parts.pop())
    folder_dict.setdefault('dirs', [])
    folder_dict.setdefault('files',npy_json_pairs)

    if len(parts) > 0 :
        # Move to current dir
        curr = Gallery_dict
        while (len(parts) > 0 ):
            seek = parts.pop(0)
            for sub_folder in curr['dirs']:
                if (sub_folder["name"] == seek):
                    curr = sub_folder
        # Add the folder dict to the list of dirs
        curr['dirs'].append(folder_dict)

    else:
        Gallery_dict = folder_dict


def sort_dirs_list_inplace(the_dict):

    the_dict["dirs"].sort(key=lambda x: x["name"])

    for sub_dir in the_dict["dirs"]:
        sort_dirs_list_inplace(sub_dir)
        
sort_dirs_list_inplace(Gallery_dict)


jsonString = json.dumps(Gallery_dict, indent=4, sort_keys=True)


with open(out_filename, "w") as jsonFile:
    jsonFile.write(jsonString)


