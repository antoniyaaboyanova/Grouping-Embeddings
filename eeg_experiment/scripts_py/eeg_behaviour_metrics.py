import os
import pandas as pd
import pickle 
from eeg_decoding_utils import dump_data

def sep_conditions(files_list):
    group_beh = pd.concat(files_list, axis=0)
    
    conditions_dict = {
        "fix_rand": {},
        "fix_det": {},
        "att_rand": {},
        "att_det": {}
    }

    conditions_dict["att_rand"]["df"] = group_beh[
        (group_beh['block_type'] == 'random') & 
        (group_beh['block_task'] == 'Pay attention to the images!')
    ].reset_index()

    conditions_dict["att_det"]["df"] = group_beh[
        (group_beh['block_type'] == 'determ') & 
        (group_beh['block_task'] == 'Pay attention to the images!')
    ].reset_index()

    conditions_dict["fix_rand"]["df"] = group_beh[
        (group_beh['block_type'] == 'random') & 
        (group_beh['block_task'] == 'Pay attention to the fixation cross!')
    ].reset_index()

    conditions_dict["fix_det"]["df"] = group_beh[
        (group_beh['block_type'] == 'determ') & 
        (group_beh['block_task'] == 'Pay attention to the fixation cross!')
    ].reset_index()

    return conditions_dict

def compute_metrics(conditions_dict):
    metrics = {}
    for key in conditions_dict.keys():
        metrics[key] = {}
        df = conditions_dict[key]["df"]
        response_trials = df[df['key_resp_3.keys'].notna()]

        # Calculate Mean RT for correct responses
        metrics[key]["rt"] = response_trials[response_trials['key_resp_3.corr'] == 1]['key_resp_3.rt'].mean()

        # Total non-catch trials and hit trials
        non_catch_trials = df[df['catch_trial'] == True]
        hit_trials = non_catch_trials[non_catch_trials['key_resp_3.corr'] == 1]
        metrics[key]["hit"] = len(hit_trials) / len(non_catch_trials) if len(non_catch_trials) > 0 else 0

        # Total catch trials and false alarms
        catch_trials = df[df['catch_trial'] == False]
        false_alarm_trials = catch_trials[catch_trials['key_resp_3.keys'].notna()]
        metrics[key]["fa"] = len(false_alarm_trials) / len(catch_trials) if len(catch_trials) > 0 else 0
        
    return metrics

def behavioural_metrics(subjects, project_path):
    """
    Computes behavioral metrics for the given list of subjects and saves the results.

    Args:
        subjects (list): List of subject numbers to process.
        project_path (str): Base path for the project data files.
    """
    conds = ["fix", "img"]
    for sub in subjects:
        print(f"Processing Subject {sub:02d}...")
        sub_files = []
        
        for cond in conds:
            try:
                file_path = os.path.join(project_path, f'beh/sub{sub:02d}/{sub}_eeg_exp_{cond}.csv')
                sub_files.append(pd.read_csv(file_path))
            except FileNotFoundError:
                print(f"File not found: {file_path}. Skipping...")
                continue
        
        if not sub_files:
            print(f"No data found for Subject {sub:02d}. Skipping...")
            continue
        
        sub_dict = sep_conditions(sub_files)
        sub_metrics = compute_metrics(sub_dict)
        
        metric_file_name = os.path.join(project_path, f'beh_metrics/metrics_{sub:02d}.pkl')
        dump_data(sub_metrics, metric_file_name)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute behavioral metrics.")
    parser.add_argument(
        '--subjects', 
        nargs='+', 
        type=int, 
        required=True, 
        help="List of subject numbers to process (e.g., --subjects 1 2 3)."
    )
    parser.add_argument(
        '--project_path', 
        type=str, 
        default='/projects/crunchie/boyanova/EEG_Things/eeg_experiment/', 
        help="Base path for the project files (default: /projects/crunchie/boyanova/EEG_Things/eeg_experiment/)."
    )
    args = parser.parse_args()
    
    behavioural_metrics(args.subjects, args.project_path)