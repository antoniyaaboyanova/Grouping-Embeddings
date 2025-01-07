import argparse
from embeddings_utils import extract_centroids
import warnings

warnings.filterwarnings("ignore")

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
        extract_centroids(args.project_dir, clip_f_name, steps["inanimate or animate"])
    
    if steps["CLIP text embeddings (BLIP model)"]:
        print("<<< Getting CLIP text (BLIP) centroids >>>")
        clip_f_name = "CLIP_txt_fmri_blip.pickle"
        extract_centroids(args.project_dir, clip_f_name, steps["inanimate or animate"])
        
    
    if steps["CLIP text embeddings (LLAVA model)"]:
        print("<<< Getting CLIP text (LLAVA) centroids >>>")
        clip_f_name = "CLIP_txt_fmri_llava.pickle"
        extract_centroids(args.project_dir, clip_f_name, steps["inanimate or animate"])
        

if __name__ == "__main__":
    main()


