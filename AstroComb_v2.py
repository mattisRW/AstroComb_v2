import sys
import os
import matplotlib
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.fft
import scipy.optimize
import scipy.interpolate
import time
import multimodality as mm


# Utility class to allow dot notation for dictionaries
class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def main():
    # Clear screen equivalent
    os.system('cls' if os.name == 'nt' else 'clear')
    print("Starting AstroComb Analysis...")

    # ---------------------------
    # Define Inputs and Constants
    # ---------------------------
    in_params = DotDict({
        'dataFolder': './',     # Folder containing data file
        'dataFilename': 'DataFile.txt',            # Data file name
        'tempFolder': './',      # Folder to store files and figures
    })

    params = DotDict({
        'N1': 0,                            # First sample
        'N2': 1000000,                      # Last sample
        'selectBasic': [1],                 # Column indices of selected records (assumes first column is depth)
        'inputNames': [''],                 # Names of selected records
        'dz': 10,                           # Sampling distance [mm]
        'winLgt': 32000,                      # Window length [in samples]

        'rateMin': 1.0,                     # Min sed rate [mm/ky]
        'rateMax': 14.0,                    # Max sed rate [mm/ky]
        'rateStep': 0.05,                    # Sed rate step [mm/ky]

        'milancCyc': np.array([405, 99, 32.4, 19.5]),  # Milankovic cycles in ky
        'infoDepths': np.array([76, 82, 88, 94, 100, 106, 112]), #Depths to visualize spectral matches [m]

        'anchorDepth': 0,                   # [m]
        'anchorTime': 0,                    # [ky] 

        'detectLowLim': 1.0,
        'detectUpLim': 4.0,
        'pVal': 0.01,

        'postType': 'MAP',           # [Unimodal, Multimodal, MC, MAP] Type of posterior analysis to perform
        'mcCount': 1000,                    # Number of Monte Carlo simulations for timeline generation, default 1000
        'credInterval': 0.89,               # Range for credible interval, gets converted directly to variance for error calculation
        'mmPruneProbThreshold': 0.05,       # Probability threshold for pruning branches in multimodal analysis
        'mcPersistStrength': 0.5,           # Correlation strength for MC simulation
        'mcSigmaFactor': 10,                # Factor to determine Gaussian smoothing sigma for MC simulation
        })

    # ---------------------------
    # Read Data
    # ---------------------------
    data = read_data(in_params, params)

    # ---------------------------
    # Compute Spectra
    # ---------------------------
    spectral_info = calc_spectra(in_params, data, params)

    # ---------------------------
    # Compute Posterior Distributions (Sed Rates)
    # ---------------------------
    sed_rate_results = compute_sed_rate_pdfs(params, in_params, spectral_info)

    # ---------------------------
    # Compute Timescale
    # ---------------------------
    sed_rate_results = post_timescale(params, data, sed_rate_results)

    # ---------------------------
    # Plot Results
    # ---------------------------
    plot_results(in_params, params, data, spectral_info, sed_rate_results)
    
    print("\nAnalysis Complete.")
    plt.show()


# -------------------------------------------------------------------------
# FUNCTIONS
# -------------------------------------------------------------------------

def read_data(in_params, params):
    print('Reading data...')
    
    # Python 0-based indexing adjustments
    select_basic = params.selectBasic
    N1 = params.N1
    N2 = params.N2
    dz = params.dz * 0.001 # Convert mm to m for resampling
    
    # Adjust window length to closest odd value > winLgt
    win_lgt = 2 * round(params.winLgt / 2) + 1
    win_lgt = int(win_lgt)
    
    filepath = os.path.join(in_params.dataFolder, in_params.dataFilename)
    
    try:
        # Assuming space or tab delimited file
        data_file = np.loadtxt(filepath, skiprows=1)
    except OSError:
        print(f"Error: File {filepath} not found. Please ensure the data file exists, it's accessible and tab-separated.")
        sys.exit(1)
    
    max_idx = min(N2, data_file.shape[0])
    z_vals_full = data_file[N1:max_idx, 0]

    # Extract records
    # Handle scalar or list for selectBasic
    if isinstance(select_basic, int):
        cols = [select_basic]
    else:
        cols = select_basic
        
    data_records = data_file[:, cols]
    
    # Limit by N1, N2
    data_records = data_records[N1:max_idx, :]

    if np.arange(z_vals_full[0], z_vals_full[-1] + dz, dz).shape[0] != z_vals_full.shape[0]:
        print('Data continuity check: Resampling data to regular intervals...')
        z_vals_full, data_records = data_continuity(data_records, z_vals_full, dz)
    
    # Normalize data for better numerical stability in spectral analysis
    data_records = normalize_data(data_records)

    # Calculate analysis range (removing half window from ends)
    half_win = (win_lgt - 1) // 2
    
    if half_win > 0:
        z_vals_analy = z_vals_full[half_win:-half_win]
    else:
        z_vals_analy = z_vals_full
    
    print('Reading data completed.')

    return DotDict({
        'dataRecords': data_records,
        'zValsFull': z_vals_full,
        'zValsAnaly': z_vals_analy,
        'winLgt': win_lgt,
        'milancCyc': params.milancCyc,
        'rateStep': params.rateStep
    })


def calc_spectra(in_params, data, params):
    filepath = in_params.tempFolder
    dz = params.dz          # in mm for kyr/mm units in frequencies
    rate_min = params.rateMin
    rate_max = params.rateMax
    milanc_cyc = params.milancCyc
    
    data_records = data.dataRecords
    win_lgt = data.winLgt
    
    N_full, n_records = data_records.shape
    
    taper = 1.0
    taperfac = 1.0 / (1.0 - taper / 2.0)
    win_vec = scipy.signal.windows.tukey(win_lgt, alpha=taper)
    pad_size = np.min([48000, np.ceil(win_lgt * 10).astype(int)])

    # # Padding that allows for efficient FFT computation (next power of two)
    # close_pow2 = 1 << (win_lgt * 2 - 1).bit_length()     # Closest power of two greater than or equal to win_lgt * 2
    # pad_size = int((close_pow2 - win_lgt)/2)             # Adjust pad_size to ensure total length is a power of two

    # Spectral Parameters
    k_vals, _ = spectral_param(dz, win_lgt, pad_size)

    # Calculate indices for frequency band of interest
    min_freq_limit = 0.8 * np.min(1.0 / (milanc_cyc * rate_max))
    max_freq_limit = 1.2 * np.max(1.0 / (milanc_cyc * rate_min))

    if min_freq_limit < k_vals[0] or max_freq_limit > k_vals[-1]:
        print(f"Error: Frequency limits ({min_freq_limit:.4f}, {max_freq_limit:.4f}) are outside the computed k_vals range ({k_vals[0]:.4f}, {k_vals[-1]:.4f}).")
        # Calculate sed rate limits based on k_vals range
        sed_rate_min = 1.2 * np.max(1.0 / (k_vals[-1] * milanc_cyc))
        print(f"Smallest sedimentation rate that can be resolved: {sed_rate_min:.4f} mm/ky")
        sys.exit(1)

    mink = find_closest_index(k_vals, min_freq_limit)
    maxk = find_closest_index(k_vals, max_freq_limit)
    
    max_n = N_full - win_lgt + 1

    # Pre-allocate sliding spectrum array
    # Shape: (Depths, Frequencies, Records)
    n_freqs = maxk - mink + 1
    sliding_spec = np.zeros((max_n, n_freqs, n_records))
    
    sliding_spec_file = os.path.join(filepath, "slidingSpec.npy")
    background_spec_file = os.path.join(filepath, "backgroundSpec.npy")
    k_vals_file = os.path.join(filepath, "kVals.npy")
    # ---------------------------
    # 1. Compute Sliding Spectra
    # ---------------------------
    if os.path.isfile(sliding_spec_file):
        print(f"Loading {sliding_spec_file}...")
        sliding_spec = np.load(sliding_spec_file)
    else:
        print('Calculating spectra...')
        start_time = time.time()
        
        full_pad_len = pad_size * 2 + win_lgt
        
        for nr in range(n_records):
            print(f'Processing record {nr + 1}/{n_records}...')
            rec_data = data_records[:, nr]
            for n in tqdm(range(max_n), desc=f'Record {nr + 1} Spectra', unit='segment'):                
                # Extract window
                segment = rec_data[n : n + win_lgt]
                
                # Detrend
                segment_det = scipy.signal.detrend(segment, type='linear')
                
                # Taper
                segment_taper = taperfac * (segment_det * win_vec)
                
                # Pad
                # Efficient padding using zeros
                data_pad = np.zeros(full_pad_len)
                data_pad[pad_size : pad_size + win_lgt] = segment_taper
                
                # FFT
                # amplitudeSpectrum returns single sided amplitude
                f, amp_spec = amplitude_spectrum(data_pad, dz)
                
                # Store interesting band
                sliding_spec[n, :, nr] = amp_spec[mink : maxk + 1]
                
        np.save(sliding_spec_file, sliding_spec)
        np.save(k_vals_file, k_vals[mink : maxk + 1])
        print(f'Calculation of spectra complete. Time: {time.time() - start_time:.2f}s')
    # ---------------------------
    # 2. Identify and Remove Background
    # ---------------------------
    if os.path.isfile(background_spec_file):
        print(f"Loading {background_spec_file}...")
        background_specs = np.load(background_spec_file)
    else:
        print('\nIdentify and remove spectral background...')
        start_time = time.time()
        
        background_specs = np.zeros((max_n, n_freqs, n_records))
        v_freqs = k_vals[mink : maxk + 1]
        
        for nr in range(n_records):
            print(f'Processing record {nr + 1}/{n_records} (Background Removal)...')
            for n in tqdm(range(max_n), desc=f'Record {nr + 1} Background Removal', unit='segment'):
                w_spec = sliding_spec[n, :, nr]
                
                # Passing x and y data
                bg_fit, _ = SWA_fit(v_freqs, w_spec, window_counts=(11, 55))

                background_specs[n, :, nr] = bg_fit
                
        np.save(background_spec_file, background_specs)
        print(f'Spectra background removal complete. Time: {time.time() - start_time:.2f}s')

    # Initial spectral information saved to file for reference
    info_file = os.path.join(filepath, "spectral_info.txt")
    with open(info_file, "w") as f:
        f.write("Hyper parameters used: \n")
        f.write(f"Selected Records: {params.selectBasic}\n")
        f.write(f"Input Names: {params.inputNames}\n")
        f.write(f"Sampling Distance (mm): {params.dz}\n")
        f.write(f"Window Length: {params.winLgt}\n")
        f.write(f"Min Sed Rate (mm/ky): {params.rateMin}\n")
        f.write(f"Max Sed Rate (mm/ky): {params.rateMax}\n")
        f.write(f"Rate Step (mm/ky): {params.rateStep}\n")
        f.write(f"Detection Lower Limit: {params.detectLowLim}\n")
        f.write(f"Detection Upper Limit: {params.detectUpLim}\n")
        f.write(f"P-Value: {params.pVal}\n")
        f.write(f"Milankovic Cycles (ky): {params.milancCyc}\n")
        f.write("\nSpectral Information: \n")
        f.write(f"Padding Size: {pad_size}\n")
        f.write(f"Frequency range (k): {k_vals[mink]} to {k_vals[maxk]}\n")
        f.write(f"Frequency resolution (dk): {k_vals[1] - k_vals[0]}\n")
        f.write(f"Number of frequency bins: {n_freqs}\n")


    return DotDict({
        'backgroundSpec': background_specs,
        'slidingSpec': sliding_spec,
        'kVals': k_vals,
        'mink': mink,
        'maxk': maxk
    })


def compute_sed_rate_pdfs(params, in_params, spectral_info):
    print('\nCompute posterior distributions of sedimentation rates...')
    start_time = time.time()

    milanc_cyc = params.milancCyc
    rate_min = params.rateMin
    rate_max = params.rateMax
    rate_step = params.rateStep
    detect_low_lim = params.detectLowLim
    detect_up_lim = params.detectUpLim
    p_val = params.pVal

    file_path = in_params.tempFolder

    background_specs = spectral_info.backgroundSpec
    k_vals = spectral_info.kVals
    
    # Calculate spectral lines for potential sedimentation rates
    # Using numpy arange but being careful with float precision
    sed_rates = np.arange(rate_min, rate_max + rate_step/10.0, rate_step)
    n_sed_rates = len(sed_rates)
    n_spec_lines = len(milanc_cyc)
    
    # ---------------------------
    # Precompute indices (Vectorized)
    # ---------------------------
    # Grid of target frequencies: Shape (n_sed_rates, n_spec_lines)
    # k = 1 / (rate * cycle)
    target_k = 1.0 / (sed_rates[:, np.newaxis] * milanc_cyc[np.newaxis, :]) # k is frequencies for cycles given sed rates
    
    # Find closest indices in k_vals for all targets
    spec_lines_idx = np.abs(k_vals[np.newaxis, np.newaxis, :] - target_k[:, :, np.newaxis]).argmin(axis=2)
    
    # ---------------------------
    # Compute PDFs (Vectorized)
    # ---------------------------
    # background_specs: (nLayerRange, nFreqsSelected, nRecords)
    
    mink = spectral_info.mink
    spec_lines_idx_rel = spec_lines_idx - mink  # Now relative to background_specs frequency axis
    
    # Clip indices to be valid within the stored window
    n_freqs_stored = background_specs.shape[1]
    spec_lines_idx_rel = np.clip(spec_lines_idx_rel, 0, n_freqs_stored - 1)

    n_layer_range, _, n_records = background_specs.shape
    
    # Standard deviation per row (per depth, per record)
    # Shape: (nLayerRange, nRecords)
    std_row = np.std(background_specs, axis=1)
    mean_row = np.mean(background_specs, axis=1) # goes from (nLayerRange, nFreqs, nRecords) to (nLayerRange, nRecords)
    detect_lim = mean_row + detect_low_lim * std_row  # (nLayerRange, nRecords)
    detect_lim_upper = mean_row + detect_up_lim * std_row

    # Extracted Values Shape: (nLayerRange, nSedRates, nSpecLines, nRecords)
    # background_specs dimensions: [Depth, Freq, Record]
    bg_T = background_specs.transpose(1, 0, 2)  # [Freq, Depth, Record]
    
    # Extract
    # spec_lines_idx_rel is [nSedRates, nSpecLines])
    pdfs_file = os.path.join(file_path, "pdfs.npy")    
    sed_rate_pdfs = np.zeros((n_sed_rates, n_layer_range))
    records_used = np.zeros((n_sed_rates, n_layer_range, n_records))

    x0, k = sigmoid_params(detect_lim, detect_lim_upper, p_val)
    
    for r in range(n_sed_rates):
        indices = spec_lines_idx_rel[r, :] # (nSpecLines,)
        
        # bg_T shape: [Freq, Depth, Record]
        # vals shape: [nSpecLines, Depth, Record]
        vals = bg_T[indices, :, :]
        
        # Calculate sigmoid 
        probs = sigmoid_vec(vals, detect_lim, x0, k, p_val)  # Shape: (Lines, Depth, Rec)
 
        # Collapse Lines and Records dimensions
        combined_prob = np.prod(probs, axis=(0, 2)) # Result shape: (Depth,)
        
        sed_rate_pdfs[r, :] = combined_prob

        # Collapse lines to get probability of detection for each record at each depth
        records_used[r, :, :] = np.prod(probs, axis=0)  # Shape: (Depth, Rec) 


    # Change shape to (n_layer_range, n_records) with mean over rates to see how much each record contributes at each depth
    records_used = np.max(records_used, axis=0)  # Shape: (Depth, Rec) - max probability across rates for each record at each depth
    records_used = records_used / np.sum(records_used, axis=1, keepdims=True)  # Normalize to sum to 1 across records for each depth

    # Normalize PDFs to sum to 1
    col_sums = np.sum(sed_rate_pdfs, axis=0)
    # Avoid divide by zero
    mask = col_sums > 0
    sed_rate_pdfs[:, mask] = sed_rate_pdfs[:, mask] / col_sums[mask]

    print(f'Computation of posterior distributions completed. Time: {time.time() - start_time:.2f}s')
    np.save(pdfs_file, sed_rate_pdfs)

    return DotDict({
        'sedRatePdfs': sed_rate_pdfs,
        'sedRates': sed_rates,
        'specLinesIdx': spec_lines_idx,
        'recordsUsed': records_used,
        'detectLim': detect_lim,
        'filePath': file_path
    })


def post_timescale(params, data, sed_rate_results):
    dz = params.dz 
    milanc_cyc = params.milancCyc
    anchor_depth = params.anchorDepth
    anchor_time = params.anchorTime
    post_type = params.postType
    cred_interval = params.credInterval
    
    z_vals_analy = data.zValsAnaly
    sed_rate_pdfs = sed_rate_results.sedRatePdfs
    sed_rates = sed_rate_results.sedRates
    records_used = sed_rate_results.recordsUsed
    file_path = sed_rate_results.filePath
    
    n_rates, n_layers = sed_rate_pdfs.shape

    # Check modality type is valid
    if post_type not in ['Unimodal', 'Multimodal', 'MC', 'MAP']:
        raise ValueError(f"Invalid post_type '{post_type}'. Must be one of ['Unimodal', 'Multimodal', 'MC', 'MAP'].")

    # Regrid (Invert Rates) to get PDFs over slowness (time per depth) instead of sedimentation rate. Non-linear transformation
    inv_sed_rate_pdfs, inv_sed_rates = regrid_data(sed_rate_pdfs, z_vals_analy, sed_rates, n_rates)
    
    # Normalize inverted PDFs
    col_sums = np.sum(inv_sed_rate_pdfs, axis=0)
    mask = col_sums > 0
    inv_sed_rate_pdfs[:, mask] = inv_sed_rate_pdfs[:, mask] / col_sums[mask]
    np.save(os.path.join(file_path, "invSedRatePdfs.npy"), inv_sed_rate_pdfs)

    # Compute Times
    times = np.zeros(n_layers)
    times[0] = 0.0
    
    anchor_idx = find_closest_index(z_vals_analy, anchor_depth)
    
    mean_inv_rates = np.zeros(n_layers)
    var_inv_rates = np.zeros(n_layers)
    var_times = np.zeros(n_layers)
    idx_map_rates = np.zeros(n_layers, dtype=int)

    if post_type == 'Unimodal':
        mean_inv_rates[0], var_inv_rates[0] = HDI_from_MCMC(inv_sed_rate_pdfs[:, 0], inv_sed_rates, cred_interval)
        # Forward Calculation, time per layer
        for n in range(1, n_layers):
            # Mean Slowness
            # E[1/v] = sum( t * P(t) )
            var, mean = HDI_from_MCMC(inv_sed_rate_pdfs[:, n], inv_sed_rates, cred_interval)

            mean_inv_rates[n] = mean
            var_inv_rates[n] = var * dz  # transformation to time variance, dz is the depth increment

            # Integrate Time from rates with mean posterior probability
            times[n] = times[n-1] + dz * mean_inv_rates[n-1]
            idx_map_rates[n] = find_closest_index(sed_rates, 1/mean_inv_rates[n])
    
    if post_type == 'Multimodal':
        time_, var_, rate_, layers = mm.run_multimodal_analysis(inv_sed_rate_pdfs, inv_sed_rates, dz, params) # time_ and var_ in kyr
        times[:layers] = time_
        var_inv_rates[:layers] = var_
        mean_inv_rates[:layers] = rate_
        idx_map_rates[:layers] = [find_closest_index(sed_rates, 1/r) for r in rate_]
        
        # Run unimodal approach for remaining layers if any
        for n in range(layers, n_layers):
            # Mean Slowness
            # E[1/v] = sum( t * P(t) )
            var, mean = HDI_from_MCMC(inv_sed_rate_pdfs[:, n], inv_sed_rates, cred_interval)

            mean_inv_rates[n] = mean
            var_inv_rates[n] = var * dz 

            # Integrate Time from rates with mean posterior probability
            times[n] = times[n-1] + dz * mean_inv_rates[n-1]
            idx_map_rates[n] = find_closest_index(sed_rates, 1/mean_inv_rates[n])

    if post_type == 'MC':
        mc_count = params.mcCount
        mc_sigma_factor = params.mcSigmaFactor
        gauss_sigma = abs(np.diff(inv_sed_rates)[0])*mc_sigma_factor # Standard deviation for Gaussian smoothing, can be adjusted based on expected sedimentation rate variability
        correlation_probability = params.mcPersistStrength   # Correlation strength for Monte Carlo simulation, can be adjusted based on expected sedimentation rate persistence

        inv_rates = np.empty((mc_count, n_layers), dtype=np.float64)
        mem_usage = inv_rates.nbytes / (1024 ** 3)  # Memory usage in GB
        if mem_usage > 4.0:
            print(f"Warning: Monte Carlo simulation may consume significant memory ({mem_usage:.2f} GB). Consider reducing mcCount, using a machine with more memory or using another approach.")
        
        inv_rates[:, 0] = np.random.choice(
            inv_sed_rates,
            size=mc_count,
            p=inv_sed_rate_pdfs[:, 0])
        
        for layer in tqdm(range(1, n_layers), desc='Monte Carlo Simulation', unit='layer'):
            previous_rates = inv_rates[:, layer - 1]
            layer_pdf = inv_sed_rate_pdfs[:, layer]

            gauss = np.exp(
                -0.5 * ((inv_sed_rates[None, :] - previous_rates[:, None]) / gauss_sigma) ** 2)
            
            # normalise to sum to correlation_probability
            gauss /= np.sum(gauss, axis=1, keepdims=True)

            combined_pdf = layer_pdf[None, :] + gauss * correlation_probability
            combined_pdf /= np.sum(combined_pdf, axis=1, keepdims=True)

            cdf = np.cumsum(combined_pdf, axis=1)
            random_values = np.random.random(mc_count)
            indices = np.argmax(cdf >= random_values[:, None], axis=1) # Index of the first element in each row that is greater than or equal to the corresponding random value

            inv_rates[:, layer] = inv_sed_rates[indices]
        
        times = np.cumsum(inv_rates, axis=1) * dz
        mean_inv_rates = np.mean(inv_rates, axis=0)
        idx_map_rates = [find_closest_index(sed_rates, 1/r) for r in mean_inv_rates]

    if post_type == 'MAP':
        # Maximum a posteriori estimation
        for n in range(n_layers):
            idx_map_rates[n] = np.argmax(inv_sed_rate_pdfs[:, n])
            mean_inv_rates[n] = np.sum(inv_sed_rates * inv_sed_rate_pdfs[:, n])
            var_inv_rates[n] = np.sum(((inv_sed_rates - mean_inv_rates[n])**2) * inv_sed_rate_pdfs[:, n]) * dz  # variance transformation to time variance
            if n > 0:
                times[n] = times[n-1] + dz * mean_inv_rates[n-1]
            idx_map_rates[n] = find_closest_index(sed_rates, 1/mean_inv_rates[n])
    
    # Backward from anchor, 
    if anchor_idx > 0:
        var_inv_subset = var_inv_rates[:anchor_idx+1]
        # cumsum from end to start
        rev_cumsum = np.cumsum(var_inv_subset[::-1])[::-1]
        var_times[:anchor_idx+1] = rev_cumsum
        
    # Forward from anchor
    var_times[anchor_idx:] = np.cumsum(var_inv_rates[anchor_idx:])
    
    std_times = np.sqrt(var_times)
    
    # Compute mean sedimentation rates
    mean_rates = np.sum(sed_rates[:, np.newaxis] * sed_rate_pdfs, axis=0) # not used currently
    
    # Smoothing mean rates
    k = len(z_vals_analy) // 20
    filt = np.ones(k) / k
    mean_rates = np.convolve(mean_rates, filt, mode='same')
    # NaN edges
    half_k = (k + 1) // 2
    mean_rates[:half_k] = np.nan
    mean_rates[-half_k:] = np.nan

    #idx_map_rates = find_closest_index(sed_rates, 1/mean_inv_rates)
    map_rates = sed_rates[idx_map_rates]
    
    min_map_rates = np.min(map_rates)
    max_k_result = 1.0 / (min_map_rates * np.min(milanc_cyc)) # Max frequency corresponding to min sedimentation rate at shortest cycle, used for plotting
    
    
    return DotDict({
        'times': times,
        'stdTimes': std_times,
        'idxMapRates': idx_map_rates,
        'maxkResult': max_k_result,
        'sedRatePdfs': sed_rate_results.sedRatePdfs, # carry over
        'sedRates': sed_rate_results.sedRates, 
        'specLinesIdx': sed_rate_results.specLinesIdx,
        'recordsUsed': sed_rate_results.recordsUsed,
        'detectLim': sed_rate_results.detectLim,
        'postType': post_type,
    })


def plot_results(in_params, params, data, spectral_info, posterior_results):
    file_path = in_params.tempFolder
    rate_step = params.rateStep
    info_depths = params.infoDepths
    milanc_cyc = params.milancCyc
    anchor_time = params.anchorTime
    anchor_depth = params.anchorDepth
    
    times = posterior_results.times
    std_times = posterior_results.stdTimes
    post_type = posterior_results.postType
    z_vals_analy = data.zValsAnaly
    sed_rate_pdfs = posterior_results.sedRatePdfs
    sed_rates = posterior_results.sedRates
    records_used = posterior_results.recordsUsed 
    input_names = params.inputNames

    # Plotting aesthetics
    #plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'

    # ---------------------------
    # Plot 1: Information Content & PDFs
    # ---------------------------
    n_layers = len(z_vals_analy)
    n_rates = len(sed_rates)
    
    # Information Content
    null_distr = np.ones(n_rates) / n_rates
    # Kullback-Leibler divergence approx
    # inform = sum( p * log(p/q) ) * dx
    # Broadcasting
    # P: (nRates, nLayers)
    # Q: (nRates, 1)
    
    # Add epsilon to avoid log(0)
    eps = 1e-12
    p = sed_rate_pdfs + eps
    q = null_distr[:, np.newaxis]
    
    term = p * np.log(p / q) #* rate_step
    inform = np.sum(term, axis=0)
    
    # Smooth
    k = len(z_vals_analy) // 20
    filt = np.ones(k) / k
    inform = np.convolve(inform, filt, mode='same')

    if records_used.shape[1] > 1:
        fig, ax = plt.subplots(1, 3, figsize=(18, 8), gridspec_kw={'width_ratios': [1, 1, 5], 'wspace': 0.05}, sharey=True)
        ax_info = ax[0]
        ax_cont = ax[1]
        ax_img = ax[2]
    else:
        fig, ax = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [1, 5], 'wspace': 0.05}, sharey=True)
        ax_info = ax[0]
        ax_img = ax[1]
    

    # Plot Information Content
    ax_info.plot(inform, z_vals_analy, 'k-', linewidth=2)
    ax_info.invert_yaxis()
    ax_info.set_ylabel('Stratigraphic Depth [m]', fontsize=14)
    ax_info.set_title('Information', fontsize=10)
    ax_info.grid(True, which='both', linestyle='--', alpha=0.5, color = 'black')
    

    # Plot Record Contribution as stacked area plot
    if records_used.shape[1] > 1:
        records_prob = np.cumsum(records_used, axis=1)  # Cumulative sum for stacked area plot
        colors = plt.cm.tab20(np.linspace(0, 1, records_used.shape[1]))

        for r in range(records_used.shape[1]):
            x1 = 0 if r == 0 else records_prob[:, r-1]
            x2 = records_prob[:, r]
            ax_cont.fill_betweenx(z_vals_analy, x1, x2, step='post', alpha=0.5, label=input_names[r], color=colors[r])

        ax_cont.set_xlim(0, 1)
        ax_cont.legend(title="Records", loc='upper right', fontsize=8)
        ax_cont.set_title('Record Contribution', fontsize=10)
        ax_cont.grid(True, which='both', linestyle='--', alpha=0.5, color = 'black')
        plt.setp(ax_cont.get_yticklabels(), visible=False)

    # Plot Image
    # extent = [left, right, bottom, top]
    # Note: z is on y-axis (inverted), rates on x-axis
    im = ax_img.imshow((sed_rate_pdfs.T), aspect='auto', 
                    extent=[sed_rates[0], sed_rates[-1], z_vals_analy[-1], z_vals_analy[0]],
                    cmap='turbo')
    im.set_clim(0, 1)
    
    ax_img.set_xlabel('Sedimentation Rate [mm/ky]', fontsize=14)
    ax_img.set_title('Probability Distributions of Sedimentation Rates', fontsize=16)
    plt.colorbar(im, ax=ax_img, label='Probability Density')
    plt.setp(ax_img.get_yticklabels(), visible=False)
    plt.savefig(os.path.join(file_path, 'SedRatePDFs.png'), dpi=300)
    
    # ---------------------------
    # Plot 2: Time Scale
    # ---------------------------
    fig2 = plt.figure(figsize=(10, 8))

    anchor_idx = find_closest_index(z_vals_analy, anchor_depth)

    # Plotting is handled differently for MC vs Unimodal/Multimodal
    if post_type == 'MC':
        # Shift each Monte Carlo realization to anchor depth
        if anchor_time != 0:
            times_shifted = 0.001 * (times - times[:, anchor_idx][:, np.newaxis] + anchor_time)  # Convert to Ma and anchor
        else:
            times_shifted = 0.001 * times  # Convert to Ma without anchoring
        # depth-time txt file for MC
        complete_array = np.column_stack((z_vals_analy, times_shifted.T))
        np.savetxt(os.path.join(file_path, 'DepthTime_MC.txt'), complete_array, header='Depth_m\t' + '\t'.join([f'Age_Ma_MC_{i+1}' for i in range(times_shifted.shape[0])]))

        # plot all MC realizations with low alpha for visibility
        plt.plot(times_shifted.T, z_vals_analy, color='blue', alpha=0.1, label='Estimated Time (MC)')
        plt.plot(np.mean(times_shifted, axis=0), z_vals_analy, color='black', linestyle='-', linewidth=2, label='Mean Estimated Time')
        plt.plot(times_shifted[0, anchor_idx], z_vals_analy[anchor_idx], 'ro', label='Anchor Point')

    else:
        #depth-time txt file
        if anchor_time != 0:
            times_shifted = 0.001 * (times - times[anchor_idx] + anchor_time)  # Convert to Ma and anchor
        else:
            times_shifted = 0.001 * times  # Convert to Ma without anchoring

        with open(os.path.join(file_path, 'DepthTime_' + post_type + '.txt'), 'w') as f:
            f.write('Depth_m\tAge_Ma\tStd_Age_Ma\n')
            for z, t, std_t in zip(z_vals_analy, times_shifted, 0.001 * std_times):
                f.write(f'{z}\t{t}\t{std_t}\n')

        plt.plot(times_shifted, z_vals_analy, '-o', markersize=2, label='Estimated Time')
        
        # Error bars
        # subsample for visibility
        step = max(1, n_layers // 50)
        idx = np.arange(0, n_layers, step)
        plt.errorbar(times_shifted[idx], z_vals_analy[idx], 
                    xerr=0.001 * std_times[idx], fmt='none', ecolor='k', capsize=3)
        plt.plot(times_shifted[anchor_idx], z_vals_analy[anchor_idx], 'ro', label='Anchor Point')

    
    plt.gca().invert_yaxis()
    plt.xlabel('Age [Ma]', fontsize=14)
    plt.ylabel('Stratigraphic depth [m]', fontsize=14)
    plt.title('Estimated Relative Geological Time Scale, Posterior Type: ' + post_type, fontsize=16)
    plt.grid(True)
    plt.savefig(os.path.join(file_path, 'EstimatedTimeScale.png'), dpi=300)
    
    # ---------------------------
    # Plot 3: Spectral Matches
    # ---------------------------
    # This part visualizes the fit at specific depths
    # Reconstructing data for plot
    
    sliding_spec = spectral_info.backgroundSpec
    mink = spectral_info.mink
    k_vals = spectral_info.kVals
    idx_map_rates = posterior_results.idxMapRates
    max_k_result = posterior_results.maxkResult
    spec_lines_idx = posterior_results.specLinesIdx
    detect_lowlim = params.detectLowLim
    input_names = params.inputNames
    
    idx_max_k = find_closest_index(k_vals, max_k_result)
    
    # Limit to relevant band
    # sliding_spec is stored from mink to maxk
    # We want to plot from mink to idx_max_k (global index)
    # The local index is relative to mink
    
    local_max_k = idx_max_k - mink
    if local_max_k >= sliding_spec.shape[1]:
        local_max_k = sliding_spec.shape[1] - 1
        
    k_interval_vals = k_vals[mink : mink + local_max_k + 1]
    
    n_records = sliding_spec.shape[2]
    
    for nr in range(n_records):
        fig = plt.figure(figsize=(8, 10))

        # Determine a scale for offset
        # Get max amplitude in the slices we care about
        max_amp = 0
        depth_indices = []
        sed_rates_analy = []
        for d in info_depths:
            idx = find_closest_index(z_vals_analy, d)
            depth_indices.append(idx)
            sed_rates_analy.append(sed_rates[idx_map_rates[idx]])
            max_amp = max(max_amp, np.max(sliding_spec[idx, :local_max_k+1, nr]))
            
        offset_step = 1.5 * max_amp if max_amp > 0 else 1.0
        current_offset = 0
        
        for i, d_val in enumerate(reversed(info_depths)):
            d_idx = find_closest_index(z_vals_analy, d_val)
            
            # Data 1: Spectrum
            spec_slice = sliding_spec[d_idx, :local_max_k+1, nr]

            # Normalize by mean? MATLAB: dataMatrix1 = dataMatrix1 / mean(...)
            #spec_slice = spec_slice / np.mean(spec_slice)

            # Add detection limit for visualization
            std = np.std(spec_slice)
            mean = np.mean(spec_slice)
            detect_lim_slice = mean + detect_lowlim * std
                        
            # Re-eval offset based on normalized data
            if i == 0: offset_step = 1.5 * max(detect_lim_slice, (np.max(spec_slice)))
            
            plt.plot(k_interval_vals, spec_slice + current_offset, 'b', linewidth=1)
            plt.axhline(y=detect_lim_slice + current_offset, color='r', linestyle='--', linewidth=0.5)
            
            # Data 2: Lines
            # Reconstruct lines
            map_rate_idx = idx_map_rates[d_idx]
            line_indices_global = spec_lines_idx[map_rate_idx, :]
            
            # Filter lines within range
            valid_lines = line_indices_global[(line_indices_global >= mink) & (line_indices_global <= (mink + local_max_k))]
            
            line_y = np.zeros_like(k_interval_vals)
            # Find local indices
            valid_lines_local = valid_lines - mink
            
            # Set height to max of spectrum
            h = np.max(spec_slice)*0.8
            
            for vl in valid_lines_local:
                 # Map index back to k value
                 k_loc = k_interval_vals[vl]
                 plt.plot([k_loc, k_loc], [current_offset, current_offset + h], 'k', linewidth=1)
                
            plt.text(k_interval_vals[-1], current_offset + h, f'Depth = {d_val} m \n Sed Rate = {sed_rates_analy[len(info_depths)-1-i]:.2f} mm/ky', 
                     ha='right', va='top', color='b')
            
            current_offset += offset_step
            
        plt.xlabel('Wave Number [m^-1]')
        plt.yticks([])
        plt.title(f'Spectral Match at Selected Depths (Record: {input_names[nr]})')
        plt.savefig(os.path.join(file_path, f'SpectralMatch_Record{nr+1}.png'), dpi=300)

    # Open spectral information file and append details
    info_file = os.path.join(file_path, "spectral_info.txt")
    with open(info_file, "a") as f:
        # If more information is added to algorithm, it can be appended here for analysis
        f.write("\nPosterior Analysis Information: \n")
        f.write(f"Average information content: {np.mean(inform):.4f}\n")
        f.write(f"\t Maximum information content: {np.max(inform):.4f}\n")
        f.write(f"\t Minimum information content: {np.min(inform):.4f}\n")

# -------------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------------

def find_closest_index(v, x):
    """Find index of element in v closest to x"""
    return (np.abs(np.array(v) - x)).argmin()


def amplitude_spectrum(x, dx):
    """
    Single-sided amplitude spectrum.
    x: signal
    dx: sampling interval
    """
    N = len(x)
    Fs = 1.0 / dx
    X = scipy.fft.fft(x)
    P2 = np.abs(X / N)    

    # Single-sided spectrum
    if N % 2 == 0:
        P1 = P2[:N // 2 + 1]
        P1[1:-1] = 2 * P1[1:-1]
        f = Fs * np.arange(N // 2 + 1) / N
    else:
        P1 = P2[:(N + 1) // 2]
        P1[1:] = 2 * P1[1:]
        f = Fs * np.arange((N + 1) // 2) / N
    
    return f, P1


def spectral_param(dz, win_lgt, pad_size):
    Nk = win_lgt + 2 * pad_size  # total number of points after padding
    dk = 1.0 / (Nk * dz) 
    limit = Nk // 2  # Nyquist frequency in cycles per unit length
    k_vals = np.arange(limit) * dk
    return k_vals, Nk


def SWA_fit(freqs, data, window_counts=(11, 99)):
    log_data = np.log(data)

    # Candidate window counts to test
    windows = np.arange(window_counts[0], window_counts[1] + 1, 2)  # odd numbers from 11 to 99

    fits = []
    rmse = []

    # Precompute prefix sums once
    freq_cumsum = np.concatenate(([0.0], np.cumsum(freqs)))
    log_cumsum = np.concatenate(([0.0], np.cumsum(log_data)))

    n = len(freqs)

    for window_count in windows:
        # Split indices for this candidate
        edges = np.linspace(0, n, window_count + 1, dtype=int)
        lengths = np.diff(edges)

        # Remove empty bins if they occur
        valid = lengths > 0
        if valid.sum() < 2:
            continue

        starts = edges[:-1][valid]
        ends = edges[1:][valid]
        lengths = lengths[valid]

        # Vectorized window means
        window_centers = (freq_cumsum[ends] - freq_cumsum[starts]) / lengths
        log_power_spectra = (log_cumsum[ends] - log_cumsum[starts]) / lengths

        # Main interpolation
        linear_interp = np.interp(freqs, window_centers, log_power_spectra)

        # Low-frequency extrapolation
        low_mask = freqs < window_centers[0]
        if len(window_centers) > 1:
            first_slope = (log_power_spectra[1] - log_power_spectra[0]) / (
                window_centers[1] - window_centers[0])

            linear_interp[low_mask] = (
                first_slope * (freqs[low_mask] - window_centers[0]) + log_power_spectra[0])
                
        else:
            linear_interp[low_mask] = log_power_spectra[0]

        # High-frequency tail
        high_mask = freqs > window_centers[-1]
        high_freqs = freqs[high_mask]
        y_tail = linear_interp[high_mask]

        if len(y_tail) > 2: # if tail long enough, quadratic fit
            white_noise_level = np.mean(y_tail)

            coeffs = np.polyfit(high_freqs, y_tail, 2)
            quadratic_vals = np.polyval(coeffs, high_freqs)

            rmse_white = np.sqrt(np.mean((y_tail - white_noise_level) ** 2))
            rmse_quadratic = np.sqrt(np.mean((y_tail - quadratic_vals) ** 2))

            if rmse_white < rmse_quadratic:
                linear_interp[high_mask] = white_noise_level
            else:
                linear_interp[high_mask] = quadratic_vals
        elif len(y_tail) > 0:
            linear_interp[high_mask] = np.mean(y_tail)

        fits.append(linear_interp)
        rmse.append(np.sqrt(np.mean((log_data - linear_interp) ** 2)))

    if not rmse:
        residual = data - np.mean(data)
        return residual, 0

    best_fit_index = np.argmin(rmse)
    swa_fit = fits[best_fit_index]
    residual = data - np.exp(swa_fit)

    return residual, windows[best_fit_index]


def sigmoid_params(a, b, p):
    '''Calculate the parameters for the sigmoid function given lower and upper detection limits and a probability value.
    This is calculated once for each layer and then used in the sigmoid function for all records in that layer.
    a: lower detection limit (scalar)
    b: upper detection limit (scalar)
    p: probability value (scalar)
    Returns:
        x0: midpoint of the sigmoid function (scalar)
        k: steepness of the sigmoid function (scalar)
    '''
    diff = np.maximum(b - a, 1e-12)
    x0 = 0.5 * (a + b)
    k = (2.0 / diff) * np.log((1.0 - p) / p)

    return x0, k


def sigmoid_vec(x, a, x0, k, p):
    '''Uses the parameters calculated from sigmoid_params to calculate the sigmoid function for all records in a layer.
    This is translated to a probability value for each record in the layer.
    x: input array (Lines, Depth, Rec)
    a: lower detection limit (scalar)
    x0: midpoint of the sigmoid function (scalar)
    k: steepness of the sigmoid function (scalar)
    p: probability value (scalar)
    Returns:
        probs: array of probabilities for each record in the layer (Lines, Depth, Rec)
    '''
    arg = np.clip(-k * (x - x0), -100, 100)
    s = 1.0 / (1.0 + np.exp(arg))

    return np.where(x <= a, p, s)


def normalize_data(data):
    # Normalize each column to be between 0 and 1
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    # Avoid divide by zero
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1e-12
    normalized = (data - min_vals) / ranges
    return normalized


def regrid_data(A, d, s, n_t):
    """
    Interpolate A(d, s) to A_T(d, t=1/s)
    """
    # s is rates. t = 1/s (slowness).
    t_raw = 1.0 / s
    
    # Sort t ascending
    idx_sorted = np.argsort(t_raw)
    t_sorted = t_raw[idx_sorted]
    A_sorted = A[idx_sorted, :]
    
    # Linear grid for t
    t_new = np.linspace(t_sorted[0], t_sorted[-1], n_t)
    
    # Interpolate
    f = scipy.interpolate.interp1d(t_sorted, A_sorted, kind='linear', axis=0, fill_value="extrapolate")
    A_T = f(t_new)
    
    return A_T, t_new


def data_continuity(data, depth, dz):
    '''Check data continuity and regrid to regular depth intervals, filling gaps with zeroes if large, otherwise interpolating'''

    depth_regular = np.arange(depth[0], depth[-1], dz)
    n_records = data.shape[1]
    data_regular = np.zeros((len(depth_regular), n_records))

    large_gap_tolerance = 10 * dz  # 10 times dz for zero filling

    for rec in range(n_records):
        for i in range(len(depth_regular)):

            idx = np.argmin(np.abs(depth - depth_regular[i]))

            # Check if the closest depth is within dz (no gap)
            if np.abs(depth[idx] - depth_regular[i]) <= dz + 1e-4:
                data_regular[i, rec] = data[idx, rec]
            
            # Check if it's a small gap (interpolation if not at edges)
            elif np.abs(depth[idx] - depth_regular[i]) <= large_gap_tolerance:
                if idx > 0 and idx < len(depth) - 1:
                    data_regular[i, rec] = np.interp(depth_regular[i], depth[idx-1:idx+2], data[idx-1:idx+2, rec])
            
            else:
                data_regular[i, rec] = 0  # Fill gaps with zeroes
    return depth_regular, data_regular



def HDI_from_MCMC(post, sed_rates, credible_mass):
    # Source - https://stackoverflow.com/a/31917867
    # Posted by user72564
    # Retrieved 2026-03-12, License - CC BY-SA 3.0
    # Minor updates and modifications made by Mattis Westergaard for speed-up and specificity
    posterior_samples = np.random.choice(sed_rates, size = 5000, p = post)

    sorted_points = np.sort(posterior_samples)
    ciIdxInc = np.ceil(credible_mass * len(sorted_points)).astype('int')
    nCIs = len(sorted_points) - ciIdxInc
    ciWidth = sorted_points[ciIdxInc:] - sorted_points[:nCIs]
    
    HDImin = sorted_points[ciWidth.argmin()]
    HDImax = sorted_points[ciWidth.argmin()+ciIdxInc]
    HDI_samples = posterior_samples[(posterior_samples >= HDImin) & (posterior_samples <= HDImax)]

    mean = np.mean(HDI_samples)
    # Weighted variance to capture the spread of the posterior distribution within the HDI
    var = np.sum(post * (sed_rates - mean) ** 2) / np.sum(post)
    return var, mean



# ------------------
#    ENTRY POINT
# ------------------
if __name__ == "__main__":
    main()