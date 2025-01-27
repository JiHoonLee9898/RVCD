import os, sys
sys.path.append("../MAIN_CODES/eval")
import chair
from chair import CHAIR  

def count_files_in_folder(folder_path):
    try:
        entries = os.listdir(folder_path)
        file_count = sum(1 for entry in entries if os.path.isfile(os.path.join(folder_path, entry)))
        print(f"The folder '{folder_path}' contains {file_count} files.")
        return file_count
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage
if __name__ == "__main__":

    DB_path = '/home/donut2024/JIHOON/RVCD/DB_single_concept_images_flux_generated/generated_images'
    
    count_files_in_folder(DB_path)
    coco_all_word_include_synonyms_list = [word.strip() for line in chair.synonyms_txt.splitlines() for word in line.split(',') if word.strip()]
    already_made_img_list = []
    not_yet_list = []
    for root, dirs, files in os.walk(DB_path):
        for file in files:
            entity = os.path.join(root, file).split('.png')[0].split('/')[-1]
            already_made_img_list.append(entity)
    for word in coco_all_word_include_synonyms_list:
        if word not in already_made_img_list:
            not_yet_list.append(word)
    print(f'not_yet_list: {not_yet_list}')