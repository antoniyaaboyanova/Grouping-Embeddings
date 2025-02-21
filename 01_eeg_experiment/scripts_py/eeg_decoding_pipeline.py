from eeg_decoding_utils import *
import argparse
import mne
import multiprocessing

# Suppress MNE output
mne.set_log_level('error')

# Define Temporal Generalisation function wrapper
def run_temp_gen(func, sub):
    func(sub)

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="EEG Decoding Pipeline")
    parser.add_argument('--sub', type=int, required=True, help="Subject ID")
    parser.add_argument('--interactive', action='store_true', help="Enable interactive mode to choose steps")
    parser.add_argument('--preprocess', action='store_true', help="Run preprocessing step")
    parser.add_argument('--edi', action='store_true', help="Run EDI analysis")
    parser.add_argument('--svm', action='store_true', help="Run SVM analysis")
    parser.add_argument('--svm_att', action='store_true', help="Run SVM analysis - attention")
    parser.add_argument('--svm_exp', action='store_true', help="Run SVM analysis - expectation")
    parser.add_argument('--temp-gen', action='store_true', help="Run Temporal Generalisation steps")

    args = parser.parse_args()
    sub = args.sub

    # If no specific step is chosen and not in interactive mode, default to all steps
    run_all = not (args.preprocess or args.edi or args.svm or args.temp_gen or args.interactive)

    if args.interactive:
        print("Interactive mode enabled.")
        steps = {
            "Preprocessing": args.preprocess,
            "EDI analysis": args.edi,
            "SVM analysis": args.svm,
            "SVM analysis - attention": args.svm_att,
            "SVM analysis - expectation": args.svm_att,
            "Temporal Generalisation": args.temp_gen,
        }

        # Ask user for each step
        for step_name in steps:
            user_input = input(f"Do you want to perform {step_name}? (yes/no): ").strip().lower()
            steps[step_name] = user_input in ["yes", "y"]
    else:
        steps = {
            "Preprocessing": args.preprocess or run_all,
            "EDI analysis": args.edi or run_all,
            "SVM analysis": args.svm or run_all,
            "SVM analysis - attention": args.svm_att or run_all,
            "SVM analysis - expectation": args.svm_exp or run_all,
            "Temporal Generalisation": args.temp_gen or run_all,
        }

    # Execute pipeline steps
    if steps["Preprocessing"]:
        print("<<< EEG Preprocessing >>>")
        conds = ["fix", "img"]
        for cond in conds:
            preprocess_eeg(sub, cond)

    if steps["EDI analysis"]:
        print("<<< EDI analysis >>>")
        run_edi(sub)

    if steps["SVM analysis"]:
        print("<<< SVM analysis >>>")
        run_svm(sub)

    if steps["SVM analysis - attention"]:
        print("<<< SVM analysis - attention >>>")
        run_svm_att(sub)

    if steps["SVM analysis - expectation"]:
        print("<<< SVM analysis - attention >>>")
        run_svm_exp(sub)

    if steps["Temporal Generalisation"]:
        print("<<< Starting Temporal Generalisation >>>")
        functions = [TempGen_within, TempGen_att, TempGen_exp]
        processes = []

        for func in functions:
            p = multiprocessing.Process(target=run_temp_gen, args=(func, sub))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

if __name__ == "__main__":
    main()


