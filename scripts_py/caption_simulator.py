import argparse
import os
from embeddings_utils import run_blip_model, run_llava_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Image Captioning Pipeline")
    parser.add_argument('--project_dir', type=str, default='/projects/crunchie/boyanova/EEG_Things/Grouping-Embeddings', help="Path to the project directory")
    parser.add_argument('--image_dir', type=str, default='/projects/crunchie/boyanova/EEG_Things/data_set/Images', help="Path to the image directory")  # Fixed typo here
    parser.add_argument('--interactive', action='store_true', help="Enable interactive mode to choose language model step")

    args = parser.parse_args()

    if args.interactive:
        print("Interactive mode enabled.")
        # Adjust steps to handle user interaction
        steps = {
            "BLIP model": False,
            "LLAVA model": False,
        }

        # Ask user for each model step
        for step_name in steps:
            user_input = input(f"Do you want to run {step_name}? (yes/no): ").strip().lower()
            steps[step_name] = user_input in ["yes", "y"]
    else:
        print("Enable interactive mode (use --interactive)")


    # Execute the chosen steps
    if steps["BLIP model"]:
        print("<<< Running BLIP model >>>")
        run_blip_model(args.project_dir, args.image_dir)

    if steps["LLAVA model"]:
        print("<<< Running LLAVA model >>>")
        run_llava_model(args.project_dir, args.image_dir)

if __name__ == "__main__":
    main()

