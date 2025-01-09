import os
import warnings
# Suppress specific TensorFlow backend logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

warnings.filterwarnings("ignore")

import argparse
from embeddings_utils import extract_centroids, sort_data, dump_data


def main():
    
    # Argument parser
    parser = argparse.ArgumentParser(description="CLIP Embeddings - centroid computation")
    parser.add_argument('--project_dir', type=str, default='/projects/crunchie/boyanova/EEG_Things/Grouping-Embeddings', help="Path to the project directory")
    parser.add_argument('--interactive', action='store_true', help="Enable interactive mode to choose embedding steps")

    args = parser.parse_args()

    # Steps to execute
    steps = {
        "CLIP image embeddings": False,
        "CLIP text embeddings (BLIP model)": False,
        "CLIP text embeddings (LLAVA model)": False,
        "inanimate or animate": "inanimate"

    }
    if args.interactive:
            print("\nInteractive mode enabled. Select steps to execute:\n")
            
            # Ask user for each step
            for step_name, default_value in steps.items():
                
                if isinstance(default_value, bool):
                    user_input = input(f"{step_name}? (yes/no) [default: {'yes' if default_value else 'no'}]: ").strip().lower()
                    steps[step_name] = user_input in ["yes", "y", ""] if default_value else user_input in ["yes", "y"]
                
                elif isinstance(default_value, str):
                    user_input = input(f"{step_name} [default: {default_value}]: ").strip()
                    steps[step_name] = user_input if user_input else default_value

            print("\nSelected Steps:")
            for step, value in steps.items():
                print(f"  - {step}: {value}")

    else:
            print("\nInteractive mode is disabled. Use the --interactive flag to choose specific steps.")

    # Execute the chosen steps
    if steps["CLIP image embeddings"]:
        print("<<< Getting CLIP image centroids >>>")
        clip_f_name = "CLIP_vis_fmri.pickle"
        clip_vis = extract_centroids(args.project_dir, clip_f_name, steps["inanimate or animate"])
    
    if steps["CLIP text embeddings (BLIP model)"]:
        print("<<< Getting CLIP text (BLIP) centroids >>>")
        clip_f_name = "CLIP_txt_fmri_blip.pickle"
        clip_blip = extract_centroids(args.project_dir, clip_f_name, steps["inanimate or animate"])

        if steps["CLIP image embeddings"]:
             # sort the data 
             print("Sorting data...")
             for mode in ['vis', 'txt']:
                sorted_dict, top_dict = sort_data(clip_vis, clip_blip, mode = mode)
                dump_data(sorted_dict, os.path.join(args.project_dir, "files", 
                                                    steps['inanimate or animate'], 
                                                "sorted_CLIP_{mode}_blip.pickle"))
                
                dump_data(top_dict, os.path.join(args.project_dir, "files", 
                                                steps['inanimate or animate'], 
                                                "top25_CLIP_{mode}_blip.pickle"))

    if steps["CLIP text embeddings (LLAVA model)"]:
        print("<<< Getting CLIP text (LLAVA) centroids >>>")
        clip_f_name = "CLIP_txt_fmri_llava.pickle"
        clip_llava = extract_centroids(args.project_dir, clip_f_name, steps["inanimate or animate"])
        if steps["CLIP image embeddings"]:
             # sort the data 
             print("Sorting data...")
             for mode in ['vis', 'txt']:
                sorted_dict, top_dict = sort_data(clip_vis, clip_llava, mode = mode)
                dump_data(sorted_dict, os.path.join(args.project_dir, "files", 
                                                    steps['inanimate or animate'], 
                                                "sorted_CLIP_{mode}_llava.pickle"))
                
                dump_data(top_dict, os.path.join(args.project_dir, "files", 
                                                steps['inanimate or animate'], 
                                                "top25_CLIP_{mode}_llava.pickle"))        

if __name__ == "__main__":
    main()


