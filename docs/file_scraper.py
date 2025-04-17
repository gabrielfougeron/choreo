from glob import glob
import shutil
import os
import functools
from sphinx_gallery.scrapers import figure_rst

def ext_scraper(block, block_vars, gallery_conf, extlist=['.mp4']):
    # Find all files of a given ext in the example directory.
    path_current_example = os.path.dirname(block_vars['src_file'])
    path_target_example = os.path.dirname(block_vars['target_file'])
    
    all_files = []
    for root, dirs, files in os.walk(path_current_example):
        for filename in files:
            for ext in extlist:
                if filename.endswith(ext):
                    all_files.append(os.path.join(root, filename))
                    break
                    
    # Iterate through files, copy them to the Sphinx-Gallery output directory
    image_names = list()
    seen = set()
    for media_file in all_files:
        if media_file not in seen:
            seen |= set(media_file)

            relpath = os.path.relpath(media_file, path_current_example)
            new_path = os.path.join(path_target_example, relpath)
            
            image_names.append(new_path)
            if not os.path.isfile(new_path):
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                shutil.copy(media_file, new_path)

    # Use the `figure_rst` helper function to generate reST for image files
    return figure_rst(image_names, gallery_conf['src_dir']) 
