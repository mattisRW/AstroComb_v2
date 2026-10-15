import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from scipy import stats
import pandas as pd

#### Settings ####

file_path = './'        # Path to folder containing backgroundSpec.npy and kVals.npy
milancCentres = np.array([405, 99, 32.4, 19.5])  # Milankovitch cycles in kyr
milancLimits = np.array([0, 1, 1, 1])  # Uncertainty in Milankovitch cycles in kyr

mcCount = 1000
burnIn = 200

sedMin = 2.0
sedMax = 18.0
sedStep = 0.1

detectLimitLower = 1
detectLimitUpper = 4
pVal = 0.01

sigma = 0.1
sigma_likelihood = 0.02

subset_count = 200

### Main Function ###

def main():
    spectra = np.load(file_path + 'backgroundSpec.npy')  # shape: (frequencies, layers, records)
    k_vals = np.load(file_path + 'kVals.npy')  # shape: (frequencies,)
    sed_rates = np.arange(sedMin, sedMax + sedStep, sedStep)  # shape: (sed_rates,)
    milancCyc = np.stack((milancCentres, milancLimits), axis=1)  # shape: (cycles, 2)

    # Create a subset of the spectra for analysis
    idx_list = np.linspace(0, spectra.shape[0] - 1, subset_count, dtype=int)
    if spectra.shape[2] > 1:
        spectra = np.sum(spectra, axis=2, keepdims=True) # robust stacking over records, sum over the third axis (records), avoids losing information if record with noise is present
        print("Averaged spectra over records, new shape:", spectra.shape)

    spectra_subset = spectra[idx_list, :, :]  # Select a subset of layers and add a new axis for records
    spectra_subset = spectra_subset.transpose(1, 0, 2)  # Now shape is (frequencies, layers, records) to match with AstroComb

    milanc_samples, mc_sed_rates, mc_sed_rate_probs, likelihoods, accepts = metropolis_sampling(
        spectra_subset, k_vals, sed_rates, milancCyc, np.array([detectLimitLower, detectLimitUpper]),
         pVal, mcCount, burnIn, sigma, sigma_likelihood)

    plot_results(milanc_samples, mc_sed_rates, mc_sed_rate_probs, likelihoods, accepts, sed_rates)

    diagnostics(milanc_samples, mc_sed_rates, mc_sed_rate_probs, likelihoods, accepts)


#### Functions ####

def k_lines(k_vals, milanc_cyc, sed_ranges):
    '''
    Calculate the k values for the Milankovitch cycles across a range of sedimentation rates.

    Parameters:
        k_vals (np.ndarray): 1D array of frequency values, assumed to be sorted.
        milanc_cyc (np.ndarray): 1D array of Milankovitch cycle periods.
        sed_ranges (np.ndarray): 1D array of sedimentation rates.
        
    Returns:
        np.ndarray: An array of indices into k_vals for each combination of sed rate and cycle.
    '''
    # Calculate target frequencies (k values)
    target_k = 1.0 / (milanc_cyc[np.newaxis, :] * sed_ranges[:, np.newaxis])
    
    # Find the insertion points for target_k in k_vals
    indices = np.searchsorted(k_vals, target_k)

    # Handle boundary cases where indices are 0 or len(k_vals)
    indices = np.clip(indices, 1, len(k_vals) - 1)
    
    # Get the k_vals candidates (indices - 1 and indices)
    k_vals_before = k_vals[indices - 1]
    k_vals_after = k_vals[indices]
    
    # Find which candidate is closer
    is_before_closer = (target_k - k_vals_before) < (k_vals_after - target_k)
    
    # Return the indices of the closer k_vals
    return np.where(is_before_closer, indices - 1, indices)  # shape: (sed_rates, milanc_cyc)


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


# pipeline vectorised across layers
def create_spectra_vectorized(k_indices, spectra_subset):
    '''Create a synthetic spectra based on all sedimentation rates and Milankovitch cycles.'''
    spectra_subset = np.asarray(spectra_subset) # shape: (frequency, records)
    synthetic_spectra = np.zeros((spectra_subset.shape[0], k_indices.shape[0]))  # shape: (frequencies, sed_rates)
    power = np.max(spectra_subset) 

    for k in range(k_indices.shape[0]):
        synthetic_spectra[k_indices[k, :], k] = power  # Assign values to the synthetic spectra at the k_indices for each sed rate
    return synthetic_spectra


def misfit_calc_vectorized(synth_spectra, spectra_subset):
    # synth_spectra:  (F, S, L)
    # spectra_subset: (F, L, R)
    # output:         (S, L)

    R = spectra_subset.shape[2]

    synth_sq = np.sum(synth_spectra**2, axis=0)           # (S, L)
    data_sum = np.sum(spectra_subset, axis=2)             # (F, L)
    data_sq_sum = np.sum(spectra_subset**2, axis=(0, 2))  # (L,)

    cross = np.einsum(
        'fsl,fl->sl',
        synth_spectra,
        data_sum,
        optimize=True
    )
    return R * synth_sq + data_sq_sum - 2 * cross


def sed_rate_pipeline_vectorized(spectra_subset, k_vals, detect_limit_lower, x0, k, p_val, sed_rates, milanc_cyc):
    # First calculate the PDF for the sampled milanc cycles and sed rates across all layers
    k_indices = k_lines(k_vals, milanc_cyc, sed_rates)
    pdf_probs = sigmoid_vec(spectra_subset[k_indices, :, :], detect_limit_lower, x0, k, p_val) # shape: (sed_rates, layers, records)
    pdf_ = np.prod(pdf_probs, axis=(1, 3))  # collapse lines and records. shape: (sed_rates, layers)
    pdf_normalized = pdf_ / np.sum(pdf_, axis=0, keepdims=True)  # Normalize the pdf for each layer
    
    argmax_sed_rate = np.zeros(spectra_subset.shape[1])  # shape: (layers)
    sed_rate_prob = np.zeros(spectra_subset.shape[1])  # shape: (sed rate probability, layers)
    synthetic_spectra_all = np.zeros((len(k_vals), 1, spectra_subset.shape[1]))  # shape: (frequencies, sampled sed rates, layers)

    # Mask the PDF for each layer and sample sed rates from the credible region, and create synthetic spectra for each sampled sed rate
    for layer_idx in range(spectra_subset.shape[1]):
        # Sample from the normalized PDF for each layer
        argmax_sed_rate[layer_idx] = sed_rates[np.argmax(pdf_normalized[:, layer_idx])]  # MAP sed rate
        sed_rate_prob[layer_idx] = np.max(pdf_normalized[:, layer_idx])  # Probability of the MAP sed rate
        k_indices_sampled = k_lines(k_vals, milanc_cyc, np.array([argmax_sed_rate[layer_idx]]))  # Get the k indices for the sampled sed rates
        synth_spectra = create_spectra_vectorized(k_indices_sampled, spectra_subset[:, layer_idx, :])  
        synthetic_spectra_all[:, :, layer_idx] = synth_spectra
    
    # calculate misfit between synthetic and observed spectra for all layers and sampled sed rates across records 
    misfits = misfit_calc_vectorized(synthetic_spectra_all, spectra_subset)  # shape: (sed_rates, layers)
    return argmax_sed_rate, misfits, sed_rate_prob


def metropolis_sampling(spectra_subset, k_vals, sed_rates_all, milanc_cyc, detect_lims, p_val, mc_count, burn_in, sigma, likelihood_sigma):
    # Create the lower and upper detection limits for the sigmoid function based on the spectra subset
    detect_limit_lower = detect_lims[0] * np.std(spectra_subset, axis = 0) + np.mean(spectra_subset, axis = 0)  # shape: (layers)
    detect_limit_upper = detect_lims[1] * np.std(spectra_subset, axis = 0) + np.mean(spectra_subset, axis = 0)  # shape: (layers)
    x0, k = sigmoid_params(detect_limit_lower, detect_limit_upper, p_val)


    # Initialize arrays to store the results of the Monte Carlo sampling
    n_total = mc_count + burn_in + 1
    mc_sed_rates = np.zeros((spectra_subset.shape[1], n_total))  # shape: (layers, MC samples)
    mc_sed_rate_probs = np.zeros((spectra_subset.shape[1], n_total))  # shape: (layers, MC samples)
    likelihoods = np.zeros(n_total)  # shape: (MC samples) 

    milanc_samples = np.zeros((n_total, len(milanc_cyc[:, 0])))  # shape: (MC samples, cycles)
    milanc_limits = np.array([milanc_cyc[:, 0] - milanc_cyc[:, 1], milanc_cyc[:, 0] + milanc_cyc[:, 1]]).T   # set up so that not restricted to 4 cycles, but can be any number of cycles in milanc_cyc array
    milanc_samples[0, :] = np.random.uniform(milanc_limits[:, 0], milanc_limits[:, 1])  # first row is random samples within the ranges

    accepts = np.zeros(n_total)  # counter for accepted Milankovitch cycles

    # Run the MH algorithm for sampling Milankovitch cycles and estimating sedimentation rates
    for mc in range(n_total):
        if mc % 100 == 0 and mc > burn_in:
            print(f"MC iteration: {mc - burn_in}/{mc_count}", f"Acceptance rate: {np.mean(accepts[burn_in:mc]):.2%}")
        
        # sample Milankovitch cycles around previous cycles for the current MC iteration
        if mc > 0:
            perturbation = np.random.normal(0, sigma, size=len(milanc_cyc[:, 0])) * milanc_cyc[:, 1]
            new_samples = milanc_samples[mc - 1, :] + perturbation
            
            if np.all((new_samples[1:] >= milanc_limits[1:, 0]) & (new_samples[1:] <= milanc_limits[1:, 1])):
                milanc_samples[mc, :] = new_samples
            else:
                # rebound exceeding samples back into the allowed range
                # samples outside the range are reflected back into the range
                for i in range(len(new_samples)):
                    if new_samples[i] < milanc_limits[i, 0]:
                        new_samples[i] = milanc_limits[i, 0] + (milanc_limits[i, 0] - new_samples[i])  # reflect back into range
                    elif new_samples[i] > milanc_limits[i, 1]:
                        new_samples[i] = milanc_limits[i, 1] - (new_samples[i] - milanc_limits[i, 1])  # reflect back into range
                milanc_samples[mc, :] = new_samples

        else:
            milanc_samples[mc, :] = milanc_samples[mc, :]  # first iteration already initialized

        best_sed_rates, misfits, sed_rate_prob = sed_rate_pipeline_vectorized(
            spectra_subset[:, :, :], k_vals, detect_limit_lower, x0, k, p_val, sed_rates_all, milanc_samples[mc, :])
        
        mc_sed_rates[:, mc] = best_sed_rates
        mc_sed_rate_probs[:, mc] = sed_rate_prob

        #misfit into likelihood
        likelihoods[mc] = -1/(2 * likelihood_sigma**2) * np.sum(misfits)  # current likelihood based on the scores for all layers

        likelihood_previous = likelihoods[mc - 1]
        log_acceptance_ratio = likelihoods[mc] - likelihood_previous

        # accept/reject
        if (
            log_acceptance_ratio >= 0
            or np.log(np.random.rand()) < log_acceptance_ratio
            or mc == 0  # always accept the first sample
        ):
            accepts[mc] = 1
        else:
            likelihoods[mc] = likelihood_previous
            milanc_samples[mc, :] = milanc_samples[mc - 1, :]
            mc_sed_rates[:, mc] = mc_sed_rates[:, mc - 1]

    print(f"Acceptance rate of Milankovitch cycle sampling: {np.mean(accepts[burn_in:]):.2%}")
    return milanc_samples, mc_sed_rates, mc_sed_rate_probs, likelihoods, accepts


def plot_results(milanc_samples, mc_sed_rates, mc_sed_rate_probs, likelihoods, accepts, sed_rates_all):
    # Plot the results of the Monte Carlo sampling
    ### plot 1: Trace of Milankovitch cycle samples
    fig, axs = plt.subplots(len(milanc_samples[0, :]), 1, figsize=(10, 8), sharex=True)
    for i in range(len(milanc_samples[0, :])):
        axs[i].plot(milanc_samples[burnIn:, i], label=f'Milankovitch Cycle {i + 1}', color='blue')
        axs[i].set_ylabel(f'Cycle {i + 1} (kyr)')
        axs[i].axvline(x = np.argmin(likelihoods[burnIn:]), color='red', linestyle='--', label='Best Likelihood')
        axs[i].legend()
    axs[-1].set_xlabel('MC Iteration')
    axs[0].set_title('MCMC Traces of Milankovitch Cycles')
    plt.savefig(file_path + '/MCMC_Traces_Milankovitch_Cycles.png', dpi=300, bbox_inches='tight')


    #
    ### plot 2: Highest likelihood sedimentation rates and mean across iterations
    mean_sed_rates = np.mean(mc_sed_rates[:, burnIn:], axis=1)
    std_sed_rates = np.std(mc_sed_rates[:, burnIn:], axis=1)
    best_sed_rates = mc_sed_rates[:, np.argmax(likelihoods[burnIn:])]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(subset_count), mean_sed_rates, label='Mean Sedimentation Rate', color='blue')
    ax.fill_between(np.arange(subset_count), mean_sed_rates - std_sed_rates, mean_sed_rates + std_sed_rates, color='blue', alpha=0.2, label='Std Dev')
    ax.plot(np.arange(subset_count), best_sed_rates, label='Best Likelihood Sedimentation Rate', color='red')
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Sedimentation Rate (mm/kyr)')
    ax.set_title('Sedimentation Rates Across Layers')
    ax.legend()
    plt.savefig(file_path + '/Sedimentation_Rates_Across_Layers.png', dpi=300, bbox_inches='tight')


def diagnostics(milanc_samples, mc_sed_rates, mc_sed_rate_probs, likelihoods, accepts):
    chain = milanc_samples[burnIn:, :]
    loglike = likelihoods[burnIn:]

    parameter_names = [
    f"cycle_{i}" for i in range(chain.shape[1])]

    n = chain.shape[0]

    print("=" * 65)
    print("MCMC THESIS DIAGNOSTIC")
    print("=" * 65)

    print(f"Post-burn-in iterations : {n}")
    print(f"Number of parameters   : {chain.shape[1]}")

    accept_post = accepts[burnIn:]

    acceptance_rate = np.mean(accept_post)

    print("\nAcceptance rate")
    print("----------------")
    print(f"Overall: {acceptance_rate:.3f}")

    print("\nSummary statistics")
    posterior = {
    parameter_names[i]: chain[:, i][None, :]
    for i in range(chain.shape[1])}

    idata = az.from_dict({
        "posterior": posterior})

    summary = az.summary(
    idata,
    kind="all")

    parameter_diagnostics = summary[
        ["mean", "sd", "mcse_mean", "ess_bulk", "ess_tail"]].copy()

    print("\nParameter diagnostics")
    print("---------------------")
    print(parameter_diagnostics)

    parameter_diagnostics["MCSE/SD"] = (parameter_diagnostics["mcse_mean"]/ parameter_diagnostics["sd"])

    print("\nRelative Monte Carlo error")
    print("--------------------------")
    print(parameter_diagnostics[["mcse_mean", "sd", "MCSE/SD"]])

    parameter_diagnostics["IACT"] = (n / parameter_diagnostics["ess_bulk"])

    print("\nApproximate integrated autocorrelation time")
    print("--------------------------------------------")
    print(parameter_diagnostics[["ess_bulk", "IACT"]])

    corr = np.corrcoef(chain, rowvar=False)

    corr_df = pd.DataFrame(corr,
        index=parameter_names,
        columns=parameter_names)

    print("\nParameter correlation matrix")
    print("----------------------------")
    print(corr_df)

    def geweke_z(x, first=0.10, last=0.50):
        n = len(x)

        n_first = int(first * n)
        n_last = int(last * n)

        x_first = x[:n_first]
        x_last = x[-n_last:]

        # Newey-West style long-run variance estimate
        def spectral_variance(x):
            x = np.asarray(x)
            n = len(x)

            # Conservative lag choice
            max_lag = min(int(np.sqrt(n)), n - 1)

            x_centered = x - np.mean(x)
            gamma0 = np.mean(x_centered**2)
            spectral = gamma0

            for lag in range(1, max_lag + 1):
                gamma = np.mean(x_centered[:-lag]* x_centered[lag:])

                # Bartlett window
                weight = 1 - lag / (max_lag + 1)

                spectral += 2 * weight * gamma
            return spectral

        var_first = spectral_variance(x_first)
        var_last = spectral_variance(x_last)

        z = (np.mean(x_first) - np.mean(x_last)) / np.sqrt(var_first / n_first+ var_last / n_last)

        return z

    geweke_results = {}

    for i, name in enumerate(parameter_names):
        z = geweke_z(chain[:, i])
        geweke_results[name] = z
    
    geweke_df = pd.DataFrame.from_dict(geweke_results, orient="index", columns=["Geweke_Z"])

    geweke_df["p_value"] = (2 * stats.norm.sf(np.abs(geweke_df["Geweke_Z"])))

    print("\nGeweke diagnostic")
    print("-----------------")
    print(geweke_df)

    print("\nParameter-likelihood correlations")
    print("---------------------------------")

    for i, name in enumerate(parameter_names):
        r = np.corrcoef(chain[:, i], loglike)[0, 1]
        print(f"{name}: r = {r:.4f}")

### entry point
if __name__ == "__main__":
    main()
