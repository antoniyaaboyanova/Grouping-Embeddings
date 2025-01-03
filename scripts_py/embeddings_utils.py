import numpy as np
import os
import torch
import pickle 
import logging 
from tqdm import tqdm

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor, LlavaForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel


##### Helpers ####
def load_data(file):
    logging.info('Loading file: %s', file)
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data

def dump_data(data, filename):
    logging.info('Writing file: %s', filename)
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


#### Captions generation #####
def run_blip_model(project_dir, image_dir):
    cache_dir = os.path.join(project_dir, "models")
    fmri_stim = np.load(os.path.join(project_dir, "files", "fmri_train_stim.npy"), allow_pickle=True)
    imagePaths = []

    for im in tqdm(fmri_stim):
        im_cat = im.split(".")[0][0:-4]
        imagePaths.append(os.path.join(image_dir, im_cat, im))

    # Set up the BLIP model and processor
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large",
        cache_dir=cache_dir)
    
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-large",
        cache_dir=cache_dir)

    model = model.to("cuda")
    simulated_captions = []

    for path in tqdm(imagePaths):
        raw_image = Image.open(path).convert('RGB')

        # Conditional image captioning
        category = path.split("/")[-2]
        if "_" in category:
            category = category.replace("_", "")
        text = "a picture of {}".format(category)
        inputs = processor(raw_image, text, return_tensors="pt").to("cuda")
        out1 = model.generate(**inputs)

        # Unconditional image captioning
        inputs = processor(raw_image, return_tensors="pt").to("cuda")
        out2 = model.generate(**inputs)
        simulated_captions.append([processor.decode(out1[0], skip_special_tokens=True), 
                                processor.decode(out2[0], skip_special_tokens=True)])
        
        # Clear GPU memory before processing the next batch
        torch.cuda.empty_cache()
    
    # Save    
    file_name = "fmri_train_caps_blip.npy"
    save_dir = os.path.join(project_dir, "files", file_name)
    np.save(save_dir, np.array(simulated_captions))

def run_llava_model(project_dir, image_dir):
    cache_dir = os.path.join(project_dir, "models")
    fmri_stim = np.load(os.path.join(project_dir, "files", "fmri_train_stim.npy"), allow_pickle=True)
    imagePaths = []

    for im in tqdm(fmri_stim):
        im_cat = im.split(".")[0][0:-4]
        imagePaths.append(os.path.join(image_dir, im_cat, im))

    # Set up the LLAVA model and processor
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        revision='a272c74',
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=cache_dir)
    
    processor = AutoProcessor.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        revision='a272c74',
        cache_dir=cache_dir,
        force_download=True)

    model = model.to("cuda")
    processor.patch_size = model.config.vision_config.patch_size
    processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy

    # Get prompts for LLAVA
    prompts = [
        "USER: <image>\nPlease provide a brief description of the image above. Include key visible objects, their attributes (colors, sizes, positions), actions occurring, emotions conveyed, and any relevant background or context.\nASSISTANT:",
    ]
    batch_size = 7
    simulated_captions = []

    for i in tqdm(range(0, len(imagePaths), batch_size)):

        # Process the current batch
        batch_images = [Image.open(imagePaths[j]).convert('RGB') for j in range(i, min(i+batch_size, len(imagePaths)))]
        inputs = processor(text=prompts * len(batch_images), images=batch_images, padding=True, return_tensors="pt").to("cuda")
        
        # Generate the output
        output = model.generate(**inputs, max_new_tokens=40)
        generated_text = processor.batch_decode(output, skip_special_tokens=True)
        generated_text = [text.split("ASSISTANT:")[-1] for text in generated_text]

        # Store caption
        for text in generated_text:
            simulated_captions.append(text)

        # Clear GPU memory before processing the next batch
        torch.cuda.empty_cache()

    file_name = "fmri_train_caps_llava.npy"
    save_dir = os.path.join(project_dir, "files", file_name)
    np.save(save_dir, np.array(simulated_captions))

#### CLIP extractors #####
def clip_vis(project_dir, image_dir):

    cache_dir = os.path.join(project_dir, "models")
    fmri_stim = np.load(os.path.join(project_dir, "files", "fmri_train_stim.npy"), allow_pickle=True)
    imagePaths = []

    for im in tqdm(fmri_stim):
        im_cat = im.split(".")[0][0:-4]
        imagePaths.append(os.path.join(image_dir, im_cat, im))

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14",         
                                    torch_dtype=torch.float16,
                                    device_map="auto",
                                    cache_dir=cache_dir)
    
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14",                                                                             
                                              cache_dir=cache_dir)
    
    model = model.to("cuda")
    batch_size = 8 
    all_image_embeddings = []

    # Process images in batches
    for i in tqdm(range(0, len(imagePaths), batch_size)):
        batch_images = []
        for path in imagePaths[i:i + batch_size]:
            image = Image.open(path).convert("RGB")
            batch_images.append(image)

        # Use the CLIP processor to prepare the batch of images
        inputs = processor(images=batch_images, return_tensors="pt", padding=True).to("cuda")

        # Obtain the image embeddings
        with torch.no_grad():
            image_embeddings = model.get_image_features(**inputs)

        all_image_embeddings.append(image_embeddings.detach().cpu().numpy())

    # Convert the list of embeddings to a single NumPy array
    all_image_embeddings = np.concatenate(all_image_embeddings, axis=0)
    print("Total Image Embeddings Shape:", all_image_embeddings.shape)

    CLIP_vis = {"stimuli": fmri_stim,
            "stimuli_paths": imagePaths,
            "embeddings": all_image_embeddings}
    
    file_name = "CLIP_vis_fmri.pickle"
    save_dir = os.path.join(project_dir, "files", file_name)
    dump_data(CLIP_vis, save_dir)


def clip_txt(project_dir, language_model="blip"): 

    cap_name = f"fmri_train_caps_{language_model}.npy"
    captions = np.load(os.path.join(project_dir,"files", cap_name), allow_pickle=True)
    cache_dir = os.path.join(project_dir, "models")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14",         
                                    torch_dtype=torch.float16,
                                    device_map="auto",
                                    cache_dir=cache_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14",                                                                             
                                              cache_dir=cache_dir)
    
    
    model = model.to(device)

    if language_model == "blip":
        embeddings_1 = []  
        for cap1, cap2 in tqdm(captions):
            inputs_1 = processor(text=cap1, return_tensors="pt", padding=True).to(device)
            
            with torch.no_grad():  
                embed_1 = model.get_text_features(**inputs_1)
            embeddings_1.append(embed_1.detach().cpu().numpy())

        embeddings_1 = np.array(embeddings_1)
        CLIP_txt = {"captions": captions[:,0],
           "embeddings": embeddings_1.reshape(8640, -1)}

    if language_model == "llava":
        embeddings_1 = []  
        for cap1 in tqdm(captions):
            inputs_1 = processor(text=cap1, return_tensors="pt", padding=True).to(device)
            
            with torch.no_grad():  
                embed_1 = model.get_text_features(**inputs_1)
            embeddings_1.append(embed_1.detach().cpu().numpy())

        embeddings_1 = np.array(embeddings_1)
        CLIP_txt = {"captions": captions,
                    "embeddings": embeddings_1.reshape(8640, -1)}
    
    file_name = f"CLIP_txt_fmri_{language_model}.pickle"
    save_dir = os.path.join(project_dir, "files", file_name)
    dump_data(CLIP_txt, save_dir)
    


        



