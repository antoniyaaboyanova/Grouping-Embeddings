import numpy as np
import os
import torch
import pickle 
import logging 
from tqdm import tqdm

from PIL import Image
from scipy.spatial import distance
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


def clip_txt(project_dir, image_dir,  language_model="blip"):
    
    fmri_stim = np.load(os.path.join(project_dir, "files", "fmri_train_stim.npy"), allow_pickle=True)
    imagePaths = []

    for im in tqdm(fmri_stim):
        im_cat = im.split(".")[0][0:-4]
        imagePaths.append(os.path.join(image_dir, im_cat, im))

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
        CLIP_txt = {"stimuli": fmri_stim,
                    "stimuli_paths": imagePaths,
                    "captions": captions[:,0],
                    "embeddings": embeddings_1.reshape(8640, -1)}

    if language_model == "llava":
        embeddings_1 = []  
        for cap1 in tqdm(captions):
            inputs_1 = processor(text=cap1, return_tensors="pt", padding=True).to(device)
            
            with torch.no_grad():  
                embed_1 = model.get_text_features(**inputs_1)
            embeddings_1.append(embed_1.detach().cpu().numpy())

        embeddings_1 = np.array(embeddings_1)
        CLIP_txt = {"stimuli": fmri_stim,
                    "stimuli_paths": imagePaths,
                    "captions": captions,
                    "embeddings": embeddings_1.reshape(8640, -1)}
    
    file_name = f"CLIP_txt_fmri_{language_model}.pickle"
    save_dir = os.path.join(project_dir, "files", file_name)
    dump_data(CLIP_txt, save_dir)
    
def extract_centroids(project_dir, clip_file_name, class_type):
    
    import hdbscan
    import umap 
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    
    if class_type not in ["inanimate", "animate"]:
        raise ValueError("class_type must be either 'inanimate' (default) or 'animate'.")
    
    clip_file = load_data(os.path.join(project_dir, "files", clip_file_name))
    animal_mask = np.load(os.path.join(project_dir, "files", "animate_mask.npy"))
    
    if class_type =="inanimate":
        animal_mask = ~animal_mask

     ###### update dictionary  ######
    clip_file["category"] = np.array([x[0:-8] for x in clip_file["stimuli"]])

    ###### get clusters ######
    reducer = umap.UMAP(n_neighbors=30, n_components=5, random_state=42)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=50, metric='euclidean')

    embeddings = clip_file["embeddings"][animal_mask, :]

    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)
    reduced_embeddings = reducer.fit_transform(embeddings)
    labels = clusterer.fit_predict(reduced_embeddings)
    

    for key in clip_file.keys():
        clip_file[key] =  np.array(clip_file[key])[animal_mask]

    clip_file['embeddings'] = embeddings # scaled! 
    clip_file["cluster"] = labels 

    ### Compute centroid
    embeddings_df = pd.DataFrame(clip_file["embeddings"].tolist())
    embeddings_df['cluster'] = clip_file["cluster"]
    centroids = embeddings_df.groupby('cluster').mean().to_numpy()

    clip_file["cluster_centroids"] = centroids
    print(f"New embedding shape: {clip_file['embeddings'].shape}")

    new_name = clip_file_name.split(".")[0]
    file_path = os.path.join(project_dir, "files", class_type)
    file_name = f"{new_name}_{class_type}.pickle"

    # Ensure the directory exists
    os.makedirs(file_path, exist_ok=True)

    # Save the file
    dump_data(clip_file, os.path.join(file_path, file_name))

    return clip_file




def sort_data(c_vis, c_txt, mode="vis"):
    """
    Sorts data based on visual or textual clusters.

    Args:
        c_vis (dict): Dictionary containing visual data, including embeddings, clusters, and metadata.
        c_txt (dict): Dictionary containing textual data, including embeddings, clusters, and metadata.
        mode (str): Sorting mode, either "vis" (default) or "txt".

    Returns:
        dict: Sorted data including embeddings, stimuli, and associated metadata.
    """
    if mode not in ["vis", "txt"]:
        raise ValueError("Invalid mode. Must be 'vis' or 'txt'.")

    if mode == "vis":
        source = c_vis
        target = c_txt
        sorted_keys = ["image_embeddings", "text_embeddings", "stimuli", "stimuli_paths", "category", "clusters"]
    
    elif mode == "txt":
        
        source = c_txt
        target = c_vis
        sorted_keys = ["text_embeddings", "image_embeddings", "stimuli", "stimuli_paths", "category", "clusters"]

    sorted_data = {key: [] for key in sorted_keys}
    top_25_embeddings = {key: [] for key in sorted_keys}

    for cl_id, cl in enumerate(np.unique(source["cluster"])):
        cluster_mask = source["cluster"] == cl
        cluster_embeddings = source["embeddings"][cluster_mask]
        cluster_target_embeddings = target["embeddings"][cluster_mask]
        cluster_centroid = source["cluster_centroids"][cl_id, :].reshape(1, -1)
        distances = distance.cdist(cluster_embeddings, cluster_centroid, "euclidean").flatten()
        order = np.argsort(distances)

        sorted_data[sorted_keys[0]].append(cluster_embeddings[order])  # First key: primary embeddings
        sorted_data[sorted_keys[1]].append(cluster_target_embeddings[order])  # Second key: target embeddings
        sorted_data["stimuli"].append(source["stimuli"][cluster_mask][order])
        sorted_data["stimuli_paths"].append(target["stimuli_paths"][cluster_mask][order])
        sorted_data["category"].append(source["category"][cluster_mask][order])
        sorted_data["clusters"].append(np.full(order.shape, cl))

        
        top_25_embeddings[sorted_keys[0]].append(cluster_embeddings[order[:25]])  
        top_25_embeddings[sorted_keys[1]].append(cluster_target_embeddings[order[:25]])  
        top_25_embeddings["stimuli"].append(source["stimuli"][cluster_mask][order[:25]])
        top_25_embeddings["stimuli_paths"].append(target["stimuli_paths"][cluster_mask][order[:25]])
        top_25_embeddings["category"].append(source["category"][cluster_mask][order[:25]])
        top_25_embeddings["clusters"].append(np.full(order[:25].shape, cl))


    # Concatenate lists into arrays
    for key in sorted_data:
        sorted_data[key] = np.concatenate(sorted_data[key])
    # Concatenate the top 25 embeddings for each cluster
    for key in top_25_embeddings:
        top_25_embeddings[key] = np.concatenate(top_25_embeddings[key])

    
    return sorted_data, top_25_embeddings






    
