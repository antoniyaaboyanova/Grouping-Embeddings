import argparse
from embeddings_utils import clip_vis, clip_txt
import warnings

warnings.filterwarnings("ignore")

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="CLIP Embedding Generation Pipeline")
    parser.add_argument('--project_dir', type=str, default='/projects/crunchie/boyanova/EEG_Things/Grouping-Embeddings', help="Path to the project directory")
    parser.add_argument('--image_dir', type=str, default='/projects/crunchie/boyanova/EEG_Things/data_set/Images', help="Path to the image directory")
    parser.add_argument('--interactive', action='store_true', help="Enable interactive mode to choose embedding steps")
    
    args = parser.parse_args()

    # Steps to execute
    steps = {
        "CLIP image embeddings": False,
        "CLIP text embeddings (BLIP model)": False,
        "CLIP text embeddings (LLAVA model)": False,
    }

    if args.interactive:
        print("Interactive mode enabled.")
        
        # Ask user for each step
        for step_name in steps:
            user_input = input(f"Do you want to get {step_name}? (yes/no): ").strip().lower()
            steps[step_name] = user_input in ["yes", "y"]
    else:
        print("Enable interactive mode by using the --interactive flag to choose specific steps.")

    # Execute the chosen steps
    if steps["CLIP image embeddings"]:
        print("<<< Getting CLIP image embeddings >>>")
        clip_vis(args.project_dir, args.image_dir)
    
    if steps["CLIP text embeddings (BLIP model)"]:
        print("<<< Getting CLIP text embeddings with BLIP model >>>")
        clip_txt(args.project_dir, "blip")
    
    if steps["CLIP text embeddings (LLAVA model)"]:
        print("<<< Getting CLIP text embeddings with LLAVA model >>>")
        clip_txt(args.project_dir, "llava")

if __name__ == "__main__":
    main()



