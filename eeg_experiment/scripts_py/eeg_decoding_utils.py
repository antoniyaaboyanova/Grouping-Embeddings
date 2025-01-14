
import mne
import pickle
import logging
import scipy
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.discriminant_analysis import _cov
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

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

def average_across_points(dat, window_size=10):
    dshape = dat['eeg'].shape
    new_length = dshape[-1] // window_size
    eeg_reshaped = dat['eeg'][:, :, :new_length * window_size].reshape(dshape[0], dshape[1], new_length, window_size)
    dat['eeg'] = eeg_reshaped.mean(axis=-1)
    dat['time'] = dat['time'][:new_length * window_size].reshape(new_length, window_size).mean(axis=-1)
    return dat

def separate_odd_even(indices):
    odd_numbers = np.array([index for index in indices if index % 2 != 0])
    even_numbers = np.array([index for index in indices if index % 2 == 0])
    return odd_numbers, even_numbers

def select_partition(data, cond):
        
        ### Select condition: det vs rand  
        image_labels = [1,2,3,4]
        if cond == "rand":
            image_labels = [im + 10 for im in image_labels]
    
        mask = np.isin(data["ids"], image_labels)
    
        eeg = data["eeg"][mask]
        ids = data["ids"][mask]
        return eeg, ids
    
def random_eeg_pick(eeg, ids, trial_lim=150):
        eeg_svm = np.full((len((np.unique(ids))), trial_lim, eeg.shape[1], eeg.shape[2]), np.nan)
        for idx, x in enumerate(np.unique(ids)):
            
            total_num_trials = len(ids[ids == x])
            range_array = np.arange(0, total_num_trials)
            random_numbers = np.random.choice(range_array, trial_lim, replace=False)
            # Select
            eeg_svm[idx, :, :, :] = eeg[ids == x][random_numbers, :, :]
            
        return eeg_svm

def get_pseudotrials(eeg_dat, tr_num):
    shape = eeg_dat.shape
    k = shape[1]
    l = int(shape[1] / k)
    
    while l < int(tr_num):
        k -= 1
        l = int(shape[1] / k)

    eeg_dat = eeg_dat[:, np.random.permutation(shape[1]), :, :]
    eeg_dat = eeg_dat[:, :l*k, :, :]
    pst = np.reshape(eeg_dat, (shape[0], k, l, shape[2], shape[3]))
    pst = pst.mean(axis=1)
    return pst, k

def get_pseudotrials_TG(eeg_dat, tr_num):
        """
        Applies pseudotrial creation independently for each slice of the first leading dimension.
        
        Parameters:
            eeg_dat (numpy.ndarray): EEG data with shape (2, 4, 150, 64, 90).
            tr_num (int): Desired number of trials.
        
        Returns:
            pst (numpy.ndarray): Pseudotrials with shape (2, 4, tr_num, 64, 90).
            k (int): Number of chunks each trial is divided into.
        """
        # Prepare to store results
        results = []
        k_values = []
    
        # Iterate over the first leading dimension
        for i in range(eeg_dat.shape[0]):
            single_data = eeg_dat[i]  # Extract slice with shape (4, 150, 64, 90)
            
            shape = single_data.shape
            k = shape[1]  # Start with the number of trials
            l = int(shape[1] / k)
            
            # Adjust k and l to fit tr_num
            while l < int(tr_num):
                k = k - 1
                l = int(shape[1] / k)
    
            # Shuffle and reshape for pseudotrials
            single_data = single_data[:, np.random.permutation(shape[1]), :, :]
            single_data = single_data[:, :l*k, :, :]
            single_data = single_data.reshape(shape[0], k, l, shape[2], shape[3])
            
            # Average across the pseudotrial axis
            pst_single = single_data.mean(axis=1)  # Resulting shape: (4, l, 64, 90)
    
            results.append(pst_single)  # Append results for this slice
            k_values.append(k)
    
        # Stack results along the first axis to combine (2, 4, l, 64, 90)
        pst = np.stack(results, axis=0)
        k = k_values[0]  # Assuming k is the same across all slices
        
        return pst, k

#### Preprocessing ####
def preprocess_eeg(sub, cond, highpass=None, lowpass=40, trialwin=(-0.2, 0.7)):
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Paths
    vhdr_file = f'/projects/crunchie/boyanova/EEG_Things/eeg_experiment/eeg_raw/sub{sub:02d}/eeg_things_{cond}_{sub:04d}.vhdr'
    beh_file = pd.read_csv(f'/projects/crunchie/boyanova/EEG_Things/eeg_experiment/beh/sub{sub:02d}/{sub}_eeg_exp_{cond}.csv')

    # Load the raw data
    raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)

    # Filter
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    raw = raw.copy().filter(l_freq=highpass, h_freq=lowpass, picks=eeg_picks)

    # Get events from eeg
    events, events_id = mne.events_from_annotations(raw)
    events  = events[2:, :]
    allowed_events = np.array([1, 2, 3, 4, 11, 12, 13, 14, 201, 202, 203, 204, 211, 212, 213, 214]) 
    events_filtered = events[np.isin(events[:, 2], allowed_events)]

    # Get epochs
    epochs = mne.Epochs(raw, events_filtered, tmin=trialwin[0], tmax=trialwin[1],
                        picks='eeg', baseline=(None, 0), preload=True, reject=None)

    # Get data
    dat = {"eeg": epochs.get_data(),
           "time": epochs.times,
           "ids": events_filtered[:, 2],
           "channels": epochs.ch_names,
           "button_press_mask": np.isin(beh_file['key_resp_3.keys'].values[0:-1], 'space'),
           "block_type": beh_file['block_type'].values[0:-1] ,
           "block_num": np.array([int(x.split("_")[-1].split(".")[0]) for x in beh_file['block_name'].values[0:-1]])}

    # Save data
    dat_name = f"/projects/crunchie/boyanova/EEG_Things/eeg_experiment/eeg_epoched/eeg_things_{sub:04d}_{cond}.pickle"
    print(f"Total accepted trials: {dat['eeg'].shape[0]}")
    dump_data(dat, dat_name)
##### EDI func #####
def run_edi(sub, conditions_1=["fix", "img"], conditions_2 =["det", "rand"], trial_num=12, img_nperms=20, trial_lim=150):
    edi_data = {}
    for cond in conditions_1:
        for cond2 in conditions_2:
            cond_name = f"{cond}_{cond2}"
            edi_data[cond_name] = []

            # Load and preprocess data
            dat_name = f"/projects/crunchie/boyanova/EEG_Things/eeg_experiment/eeg_epoched/eeg_things_{sub:04d}_{cond}.pickle"
            dat = load_data(dat_name)
            dat = average_across_points(dat, window_size=10)

            # Button press mask
            bt_press = dat["button_press_mask"]
            dat["eeg"] = dat["eeg"][~bt_press]
            dat["ids"] = dat["ids"][~bt_press]

            # Select condition
            eeg_, ids_ = select_partition(dat, cond2)

            # Get variables
            n_conditions = len(np.unique(ids_))
            n_time = eeg_.shape[-1]

            # DA matrix
            TG = np.full((n_conditions, n_conditions, n_time), np.nan)

            # Randomly pick trials
            eeg_svm = np.full((len(np.unique(ids_)), trial_lim, eeg_.shape[1], eeg_.shape[2]), np.nan)

            for p in tqdm(range(img_nperms)):
                for idx, x in enumerate(np.unique(ids_)):
                    total_num_trials = len(ids_[ids_ == x])
                    range_array = np.arange(0, total_num_trials)
                    random_numbers = np.random.choice(range_array, trial_lim, replace=False)
                    eeg_svm[idx, :, :, :] = eeg_[ids_ == x][random_numbers, :, :]

                odd, even = separate_odd_even(np.arange(0, trial_lim))
                eeg_odd = eeg_svm[:, odd, :, :]
                eeg_even = eeg_svm[:, even, :, :]

                # Calculate RDMs - Mahalanobis distance
                for cA in range(n_conditions):
                    for cB in range(cA, n_conditions):
                        for t in range(n_time):
                            cA_values = eeg_odd[cA, :, :, t]
                            cB_values = eeg_even[cB, :, :, t]
                            mah_dist = np.mean(cdist(cA_values, cB_values, 'mahalanobis'))
                            TG[cA, cB, t] = np.nansum(np.array((TG[cA, cB, t], mah_dist)))

            TG = TG / img_nperms
            edi_data[cond_name] = TG

    dump_data(edi_data, f"/projects/crunchie/boyanova/EEG_Things/eeg_experiment/eeg_decoding/eeg_mahlanobis_{sub:04d}.pickle")

##### SVM func #####
def run_svm(sub, conditions_1=["fix", "img"], conditions_2 =["det", "rand"], trial_num=12, img_nperms=100, trial_lim=150, testsize=0.2):
    decoding_data = {}
    

    for cond in conditions_1:
        for cond2 in conditions_2:
            cond_name = f"{cond}_{cond2}"
            decoding_data[cond_name] = []
            logging.info('Processing condition: %s', cond_name)

            # Load and preprocess data
            dat_name = f"/projects/crunchie/boyanova/EEG_Things/eeg_experiment/eeg_epoched/eeg_things_{sub:04d}_{cond}.pickle"
            dat = load_data(dat_name)
            dat = average_across_points(dat, window_size=10)
            
            # Apply button press mask
            bt_press = dat["button_press_mask"]
            dat["eeg"] = dat["eeg"][~bt_press]
            dat["ids"] = dat["ids"][~bt_press]
            
            # Select condition
            eeg_, ids_ = select_partition(dat, cond2)

            # Get variables
            n_conditions = len(np.unique(ids_))
            n_sensors = eeg_.shape[1]
            n_time = eeg_.shape[-1]

            TG = np.full((n_conditions, n_conditions, n_time), np.nan)
            eeg_svm = np.full((n_conditions, trial_lim, n_sensors, n_time), np.nan)

            # Loop over permutations
            for p in tqdm(range(img_nperms), desc="Permutations"):
                for idx, x in enumerate(np.unique(ids_)):
                    range_array = np.arange(len(ids_[ids_ == x]))
                    random_numbers = np.random.choice(range_array, trial_lim, replace=False)
                    eeg_svm[idx, :, :, :] = eeg_[ids_ == x][random_numbers, :, :]

                # Prepare pseudotrials and cross-validation
                pstrials, _ = get_pseudotrials(eeg_svm, trial_num)
                n_pstrials = pstrials.shape[1]
                n_test = int(n_pstrials * testsize)
                ps_ixs = np.arange(n_pstrials)
                cvs = int(n_pstrials / n_test)

                for cv in range(cvs):
                    test_ix = np.arange(n_test) + (cv * n_test)
                    train_ix = np.delete(ps_ixs.copy(), test_ix)
                    ps_train = pstrials[:, train_ix, :, :]
                    ps_test = pstrials[:, test_ix, :, :]

                    # Whitening 
                    sigma_ = np.array([np.mean([_cov(ps_train[c, :, :, t], shrinkage='auto') for t in range(n_time)], axis=0) for c in range(n_conditions)])
                    sigma = sigma_.mean(axis=0)
                    sigma_inv = scipy.linalg.fractional_matrix_power(sigma, -0.5)
                    ps_train = (ps_train.swapaxes(2, 3) @ sigma_inv).swapaxes(2, 3)
                    ps_test = (ps_test.swapaxes(2, 3) @ sigma_inv).swapaxes(2, 3)

                    # Decoding using SVM
                    for cA in range(n_conditions):
                        for cB in range(cA + 1, n_conditions):
                            for t in range(n_time):
                                train_x = np.vstack((ps_train[cA, :, :, t], ps_train[cB, :, :, t]))
                                test_x = np.vstack((ps_test[cA], ps_test[cB]))[:, :, t]
                                train_y = np.array([1] * len(train_ix) + [2] * len(train_ix))
                                test_y = np.array([1] * len(test_ix) + [2] * len(test_ix))

                                classifier = LinearSVC(dual=True, penalty='l2', loss='hinge', C=0.5, max_iter=10000)
                                classifier.fit(train_x, train_y)
                                pred_y = classifier.predict(test_x)
                                acc_score = accuracy_score(test_y, pred_y)
                                TG[cA, cB, t] = np.nansum(np.array((TG[cA, cB, t], acc_score)))

            TG = TG / (img_nperms * cvs)
            decoding_data[cond_name] = TG

    output_file = f"/projects/crunchie/boyanova/EEG_Things/eeg_experiment/eeg_decoding/eeg_decoding_{sub:04d}.pickle"
    dump_data(decoding_data, output_file)

##### Temporal Generalization (all versions) #####
def TempGen_within(sub,  conditions_1 = ["fix", "img"], conditions_2 = ["det", "rand"], testsize = 0.2, trial_num = 12, img_nperms = 25, trial_lim = 150):

    decoding_data = {}

    for cond in conditions_1:
        for cond2 in conditions_2:
            cond_name = f"{cond}_{cond2}"
            decoding_data[cond_name] = []

            print(cond_name)
            dat_name = f"/projects/crunchie/boyanova/EEG_Things/eeg_experiment/eeg_epoched/eeg_things_{sub:04d}_{cond}.pickle"
            dat = load_data(dat_name)
            dat = average_across_points(dat, window_size=10)

            # Button press mask
            bt_press = dat["button_press_mask"]
            dat["eeg"] = dat["eeg"][~bt_press]
            dat["ids"] = dat["ids"][~bt_press]
            dat["block_num"] = dat["block_num"][~bt_press]

            # Select condition
            eeg_, ids_ = select_partition(dat, cond2)

            # Get variables
            n_conditions = len(np.unique(ids_))
            n_sensors = eeg_.shape[1]
            n_time = eeg_.shape[-1]
            TG = np.full((n_conditions, n_conditions, n_time, n_time), np.nan)

            eeg_svm = np.full((n_conditions, trial_lim, n_sensors, n_time), np.nan)

            for p in tqdm(range(img_nperms)):
                for idx, x in enumerate(np.unique(ids_)):
                    total_num_trials = len(ids_[ids_ == x])
                    range_array = np.arange(0, total_num_trials)
                    random_numbers = np.random.choice(range_array, trial_lim, replace=False)
                    eeg_svm[idx, :, :, :] = eeg_[ids_ == x][random_numbers, :, :]

                pstrials, binsize = get_pseudotrials(eeg_svm, trial_num)
                n_pstrials = pstrials.shape[1]
                n_test = int(n_pstrials * testsize)
                ps_ixs = np.arange(n_pstrials)
                cvs = int(n_pstrials / n_test)

                for cv in range(cvs):
                    # we take idxs for the test/train
                    test_ix = np.arange(n_test) + (cv * n_test)
                    train_ix = np.delete(ps_ixs.copy(), test_ix)

                    # subset idxs from the pseudotrials 
                    ps_train = pstrials[:,train_ix,:,:]
                    ps_test = pstrials[:,test_ix,:,:]

                    # Whitening using the Epoch method // multivariate noise norm - it uses the cov b/w channels
                    # https://www.sciencedirect.com/science/article/abs/pii/S1053811918301411
                    # https://doi.org/10.1016/j.neuroimage.2015.12.012
                    sigma_ = np.empty((n_conditions, n_sensors, n_sensors))
                    for c in range(n_conditions):
                        # compute sigma for each time point, then average across time
                        sigma_[c] = np.mean([_cov(ps_train[c, :, :, t], shrinkage='auto')
                                             for t in range(n_time)], axis=0)
                    sigma = sigma_.mean(axis=0)  # average across conditions
                    # the formula is sigma * -1/2 // reason for sigma_inv
                    sigma_inv = scipy.linalg.fractional_matrix_power(sigma, -0.5)

                    # apply sigma to pseudo trials 
                    ps_train = (ps_train.swapaxes(2, 3) @ sigma_inv).swapaxes(2, 3)
                    ps_test = (ps_test.swapaxes(2, 3) @ sigma_inv).swapaxes(2, 3)

                    # decoding: cA image vs cB (cA + 1 :) // then do it for each time point 
                    for cA in range(n_conditions):
                        #print('decoding image ' + str(cA))
                        for cB in range(cA+1, n_conditions):
                            for t in range(n_time):
                                
                                # retrieve the patterns from pseudotrials that correspond to cA and cB at time pt t
                                train_x = np.array((ps_train[cA,:,:,t], ps_train[cB,:,:,t]))
                                # concatinate them
                                train_x = np.reshape(train_x,(len(train_ix)*2, n_sensors))
                                
                                # do the same with the test set, but here we take all time points 
                                test_x = np.array((ps_test[cA], ps_test[cB]))
                                test_x = np.reshape(test_x,(len(test_ix)*2, n_sensors, n_time))
                                
                                # config labesls 1 for cA and 2 for cB
                                train_y = np.array([1] * len(train_ix) + [2] * len(train_ix))
                                test_y = np.array([1] * len(test_ix) + [2] * len(test_ix))

                                # instantiate a classifier 
                                classifier = LinearSVC(dual=True,
                                                        penalty = 'l2',
                                                        loss = 'hinge',
                                                        C = .5,
                                                        multi_class = 'ovr',
                                                        fit_intercept = True,
                                                        max_iter = 10000)
                                # train it
                                classifier.fit(train_x, train_y)
                                # temporal test
                                for tt in range(n_time):
                                    pred_y = classifier.predict(test_x[:,:,tt])
                                    acc_score = accuracy_score(test_y,pred_y)
                                    TG[cA,cB,t,tt] = np.nansum(np.array((TG[cA,cB,t,tt],acc_score)))


            TG = TG / (img_nperms * cvs)
            decoding_data[cond_name] = TG

    dump_data(decoding_data, f"/projects/crunchie/boyanova/EEG_Things/eeg_experiment/eeg_decoding/eeg_TG_within_{sub:04d}.pickle")

def TempGen_att(subject, subsample_factor=10, testsize=0.2, trial_num=12, img_nperms=25, trial_lim=150):
    """
    Perform temporal generalization analysis for EEG data.

    Parameters:
    - subject: int, the subject ID
    - subsample_factor: int, factor for subsampling data
    - testsize: float, proportion of data used for testing
    - trial_num: int, number of trials per condition
    - img_nperms: int, number of permutations for pseudotrials
    - trial_lim: int, maximum number of trials to process
    """
    sub = int(subject)
    decoding_data = {}

    pairs = [
        ("fix_det", "img_det"),
        ("fix_rand", "img_rand"),
        ("img_det", "fix_det"),
        ("img_rand", "fix_rand")
    ]

    for pair_train, pair_test in pairs:
        print(f"Training on: {pair_train}, Testing on: {pair_test}")
        cond_name = f"{pair_train}_{pair_test}"
        train_out, train_in = pair_train.split("_")
        test_out, test_in = pair_test.split("_")

        if train_out != test_out:
            dat_name_train = f"/projects/crunchie/boyanova/EEG_Things/eeg_experiment/eeg_epoched/eeg_things_{sub:04d}_{train_out}.pickle"
            dat_train = load_data(dat_name_train)
            dat_train = average_across_points(dat_train, window_size=subsample_factor)
            bt_press = dat_train["button_press_mask"]
            dat_train["eeg"] = dat_train["eeg"][~bt_press]
            dat_train["ids"] = dat_train["ids"][~bt_press]
            dat_train["block_num"] = dat_train["block_num"][~bt_press]

            dat_name_test = f"/projects/crunchie/boyanova/EEG_Things/eeg_experiment/eeg_epoched/eeg_things_{sub:04d}_{test_out}.pickle"
            dat_test = load_data(dat_name_test)
            dat_test = average_across_points(dat_test, window_size=subsample_factor)
            bt_press = dat_test["button_press_mask"]
            dat_test["eeg"] = dat_test["eeg"][~bt_press]
            dat_test["ids"] = dat_test["ids"][~bt_press]
            dat_test["block_num"] = dat_test["block_num"][~bt_press]

            n_conditions = len(range(4))
            n_sensors = dat_train["eeg"].shape[1]
            n_time = dat_train["eeg"].shape[-1]

            TG = np.full((n_conditions, n_conditions, n_time, n_time), np.nan)
            train_eeg, train_ids = select_partition(dat_train, train_in)
            test_eeg, test_ids = select_partition(dat_test, test_in)

            for p in tqdm(range(img_nperms)):
                eeg_svm_train = random_eeg_pick(train_eeg, train_ids, trial_lim)
                eeg_svm_test = random_eeg_pick(test_eeg, test_ids, trial_lim)

                eeg_general = np.stack((eeg_svm_train, eeg_svm_test), axis=0)
                pstrials, binsize = get_pseudotrials_TG(eeg_general, trial_num)

                n_pstrials = pstrials.shape[2]
                n_test = int(n_pstrials * testsize)
                ps_ixs = np.arange(n_pstrials)
                cvs = int(n_pstrials / n_test)

                for cv in range(cvs):
                    #print(f"cv: {cv + 1}, out of: {cvs}")

                    test_ix = np.arange(n_test) + (cv * n_test)
                    train_ix = np.delete(ps_ixs.copy(), test_ix)

                    ps_train = pstrials[0][:, train_ix, :, :]
                    ps_test = pstrials[1][:, test_ix, :, :]

                    sigma_ = np.empty((n_conditions, n_sensors, n_sensors))
                    for c in range(n_conditions):
                        sigma_[c] = np.mean([
                            _cov(ps_train[c, :, :, t], shrinkage='auto')
                            for t in range(n_time)
                        ], axis=0)

                    sigma = sigma_.mean(axis=0)
                    sigma_inv = scipy.linalg.fractional_matrix_power(sigma, -0.5)

                    ps_train = (ps_train.swapaxes(2, 3) @ sigma_inv).swapaxes(2, 3)
                    ps_test = (ps_test.swapaxes(2, 3) @ sigma_inv).swapaxes(2, 3)

                    for cA in range(n_conditions):
                        for cB in range(cA + 1, n_conditions):
                            for t in range(n_time):
                                train_x = np.reshape(
                                    np.array((ps_train[cA, :, :, t], ps_train[cB, :, :, t])),
                                    (len(train_ix) * 2, n_sensors)
                                )
                                test_x = np.reshape(
                                    np.array((ps_test[cA], ps_test[cB])),
                                    (len(test_ix) * 2, n_sensors, n_time)
                                )
                                train_y = np.array([1] * len(train_ix) + [2] * len(train_ix))
                                test_y = np.array([1] * len(test_ix) + [2] * len(test_ix))

                                classifier = LinearSVC(
                                    dual=True,
                                    penalty='l2',
                                    loss='hinge',
                                    C=0.5,
                                    multi_class='ovr',
                                    fit_intercept=True,
                                    max_iter=10000
                                )
                                classifier.fit(train_x, train_y)
                                for tt in range(n_time):
                                    pred_y = classifier.predict(test_x[:, :, tt])
                                    acc_score = accuracy_score(test_y, pred_y)
                                    TG[cA, cB, t, tt] = np.nansum(np.array((TG[cA, cB, t, tt], acc_score)))

            TG = TG / (img_nperms * cvs)
            decoding_data[cond_name] = TG

    output_path = f"/projects/crunchie/boyanova/EEG_Things/eeg_experiment/eeg_decoding/eeg_TG_att_{sub:04d}.pickle"
    dump_data(decoding_data, output_path)

def TempGen_exp(sub,  testsize = 0.2, trial_num = 12, img_nperms = 25, trial_lim = 150 ):

    decoding_data = {}

    pairs = [("fix_det", "fix_rand"),
             ("fix_rand", "fix_det"),
             ("img_det", "img_rand"),
             ("img_rand", "img_det")]

    for pair_train, pair_test in pairs:
        print(f"Training on: {pair_train}, Testing on: {pair_test}")
        cond_name = f"{pair_train}_{pair_test}"
        train_out = pair_train.split("_")[0]
        train_in = pair_train.split("_")[1]

        test_out = pair_test.split("_")[0]
        test_in = pair_test.split("_")[1]

        if train_out == test_out:
            dat_name = f"/projects/crunchie/boyanova/EEG_Things/eeg_experiment/eeg_epoched/eeg_things_{sub:04d}_{train_out}.pickle"
            dat = load_data(dat_name)

            ### Subsample data
            dat = average_across_points(dat, window_size=10)

            ### Button press mask
            bt_press = dat["button_press_mask"]
            dat["eeg"] = dat["eeg"][~bt_press]
            dat["ids"] = dat["ids"][~bt_press]
            dat["block_num"] = dat["block_num"][~bt_press]

            ### Get vars
            n_conditions = len(range(4))
            n_sensors = dat["eeg"].shape[1]
            n_time = dat["eeg"].shape[-1]

            ### DA matrix 
            TG = np.full((n_conditions, n_conditions, n_time, n_time), np.nan)        
            train_eeg, train_ids = select_partition(dat, train_in)
            test_eeg, test_ids = select_partition(dat, test_in)

            for p in tqdm(range(img_nperms)):
                eeg_svm_train = random_eeg_pick(train_eeg, train_ids, trial_lim)
                eeg_svm_test = random_eeg_pick(test_eeg, test_ids, trial_lim)

                eeg_general = np.stack((eeg_svm_train, eeg_svm_test), axis=0)
        

                pstrials, _ = get_pseudotrials_TG(eeg_general, trial_num)
            
                n_pstrials = pstrials.shape[2]
                n_test = int(n_pstrials * testsize)
                ps_ixs = np.arange(n_pstrials)
                cvs = int(n_pstrials / n_test)

                for cv in range(cvs):
                    #print('cv: {}, out of: {}'.format(cv+1, cvs))

                    # we take idxs for the test/train
                    test_ix = np.arange(n_test) + (cv * n_test)
                    train_ix = np.delete(ps_ixs.copy(), test_ix)

                    # subset idxs from the pseudotrials
                    ps_train = pstrials[0]
                    ps_test = pstrials[1]
                    ps_train = ps_train[:,train_ix,:,:]
                    ps_test = ps_test[:,test_ix,:,:]

                    sigma_ = np.empty((n_conditions, n_sensors, n_sensors))
                    for c in range(n_conditions):
                        # compute sigma for each time point, then average across time
                        sigma_[c] = np.mean([_cov(ps_train[c, :, :, t], shrinkage='auto')
                                            for t in range(n_time)], axis=0)

                    # average across conditions
                    sigma = sigma_.mean(axis=0)  
                    # the formula is sigma * -1/2 // reason for sigma_inv
                    sigma_inv = scipy.linalg.fractional_matrix_power(sigma, -0.5)

                    # apply sigma to pseudo trials 
                    ps_train = (ps_train.swapaxes(2, 3) @ sigma_inv).swapaxes(2, 3)
                    ps_test = (ps_test.swapaxes(2, 3) @ sigma_inv).swapaxes(2, 3)

                    # decoding: cA image vs cB (cA + 1 :) // then do it for each time point 
                    for cA in range(n_conditions):
                        #print('decoding image ' + str(cA))
                        for cB in range(cA+1, n_conditions):
                            for t in range(n_time):
                                # retrieve the patterns from pseudotrials that correspond to cA and cB at time pt t
                                train_x = np.array((ps_train[cA,:,:,t], ps_train[cB,:,:,t]))
                                # concatinate them
                                train_x = np.reshape(train_x,(len(train_ix)*2, n_sensors))
                                # do the same with the test set, but here we take all time points 
                                test_x = np.array((ps_test[cA], ps_test[cB]))
                                test_x = np.reshape(test_x,(len(test_ix)*2, n_sensors, n_time))
                                # config labesls 1 for cA and 2 for cB
                                train_y = np.array([1] * len(train_ix) + [2] * len(train_ix))
                                test_y = np.array([1] * len(test_ix) + [2] * len(test_ix))

                                # instantiate a classifier 
                                classifier = LinearSVC(dual=True,
                                                        penalty = 'l2',
                                                        loss = 'hinge',
                                                        C = .5,
                                                        multi_class = 'ovr',
                                                        fit_intercept = True,
                                                        max_iter = 10000)
                                # train it
                                classifier.fit(train_x, train_y)
                                for tt in range(n_time):
                                    pred_y = classifier.predict(test_x[:,:,tt])
                                    acc_score = accuracy_score(test_y,pred_y)
                                    # we store the acc score in the temp gen mattrix 
                                    TG[cA,cB,t,tt] = np.nansum(np.array((TG[cA,cB,t,tt],acc_score)))
            TG = TG / (img_nperms * cvs)
            decoding_data[cond_name] = TG

    dump_data(decoding_data, "/projects/crunchie/boyanova/EEG_Things/eeg_experiment/eeg_decoding/eeg_TG_exp_{:04d}.pickle".format(sub))