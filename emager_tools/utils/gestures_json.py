import os
import json

### GESTURES JSON UTILS ###

def get_images_list(images_folder:str):
    images = [images_folder + f for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f)) and (f.endswith('.png') or f.endswith('.jpg'))]
    return images

def get_images_folder(images_list:list):
    if len(images_list) == 0:
        raise ValueError("No images provided")
    if not os.path.exists(images_list[0]):
        raise FileNotFoundError(f"Image file not found: {images_list[0]}")
    images_folder = os.path.dirname(os.path.abspath(images_list[0]))
    return images_folder

def get_gestures_dict(images):
    '''
    Get the gestures dictionary from the images.
    images must be a list of images or a string of a folder containing images.
    '''
    if isinstance(images, list):
        images = get_images_folder(images)
    images_folder = images
    if isinstance(images_folder, str):
        if not os.path.exists(images_folder):
            raise FileNotFoundError(f"Folder not found: {images_folder}")
        list_file = list(filter(lambda f: f.endswith("json"), os.listdir(images_folder)))[0]
    if not list_file:
        print(f"No JSON file found in {images_folder}")
        return None
    with open(images_folder + "/" + list_file, "r") as f:
        gestures_dict = json.load(f)
    return gestures_dict

def get_index_from_label(label:int, images, gestures_dict=None):
    """
    Get the index from the label.
    images must be a list of images or a string of a folder containing images.
    """
    if gestures_dict is None:
        gestures_dict = get_gestures_dict(images)
    if isinstance(images, str):
        images = get_images_list(images)
    
    # Get images path and index
    images_name = gestures_dict[str(label)]
    for index, img in enumerate(images):
        if images_name in img:
            return index
    print(f"Image not found: {images_name}")
    return None

def get_label_from_index(index:int, images, gestures_dict=None):
    """
    Get the label from the index.
    images must be a list of images or a string of a folder containing images.
    """
    if gestures_dict is None:
        gestures_dict = get_gestures_dict(images)
    if isinstance(images, str):
        images = get_images_list(images)
    
    # Get images path and index
    image_path = images[index]
    images_name = os.path.splitext(os.path.basename(image_path))[0]
    label = None
    for key, value in gestures_dict.items():
        if value == images_name:
            label = int(key)
            break
    return label