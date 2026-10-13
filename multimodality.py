import numpy as np
from treelib import Node, Tree
from scipy import signal
from tqdm import tqdm
import copy

### This file contains the functions necessary to perform the multimodal analysis available in AstroComb ###
# customization parameters (move to config file later)

prob_limit = 0.001  # limit for probability distance in layer likeness comparison, adjust based on data

class dataNode:
    def __init__(self, layer, prob, sed_rate, var):
        self.layer = int(layer)       # layer to which the node belongs, if grouped, first layer
        self.prob = prob              # probability of the node, gets updated during the algorithm. used for pruning and renormalization
        self.cum_prob = prob          # cumulative probability, gets updated during the algorithm
        self.sed_rate = sed_rate      # sedimentation rate associated with node, only changed upon initialization
        self.var = var                # time variance associated with the node, gets updated during the algorithm. used for error bars in timeline
        self.time_cum = 0             # cumulative time associated with the node, gets calculated during the algorithm
        self.layer_count = 0          # number of layers in the original data that are grouped together in this node, used for time conversion and plotting
        self.total_idx = 0            # index of layer in the original data, used for plotting 



def Hyndman_CDR(pdfs, idx, alpha):
    """
    Calculates the highest density credible region for a given PDF.
    pdfs: 2D array of shape (n_points, n_parameters) containing the PDF values for each parameter.
    idx: index of the parameter for which to calculate the credible region.
    alpha: desired probability mass to be included in the credible region (e.g., 0.95 for a 95% credible region).
    Returns a boolean mask indicating which points are in the credible region and the threshold eta.
    """
    # Sort the PDF values in descending order
    sorted_pdf = np.sort(pdfs[:, idx])[::-1]
    
    # Calculate the cumulative sum of the sorted probabilities
    cumulative_prob = np.cumsum(sorted_pdf)
    
    try:
        credible_points_idx = np.where(cumulative_prob >= alpha)[0][0]
    except IndexError:
        credible_points_idx = len(sorted_pdf) - 1
        
    # The threshold eta is the smallest value in the credible region
    eta = sorted_pdf[credible_points_idx]
    
    # Create a mask for values greater than or equal to eta
    mask = pdfs[:, idx] >= eta
    return mask, eta



def peak_detection(pdfs, sed_rates, eta, idx):
    '''
    Finds indices of peaks in the PDF above the threshold eta, determined by Hyndman's method.
    '''
    height_limit = eta
    distance_limit = np.round(len(sed_rates) * 0.01)
    if distance_limit < 1:
        distance_limit = 1
    peaks, _ = signal.find_peaks(pdfs[:, idx], height = height_limit, distance = distance_limit)
    if len(peaks) == 0:
        if pdfs[0, idx] >= eta:
            peaks = np.append(peaks, 0)
        else:
            print("No peaks found above the threshold.")
    return peaks



def peak_properties(pdfs, idx, sed_rates, eta, peaks):
    '''
    Creates nodes based on the eta and the peaks. The data is divided into regions and sorted before nodes are created and probabilities are renormalised.
    pdfs: 2D array of shape (inverse sed rates, layers) containing the PDF values for each parameter.
    idx: index of the layer for which to calculate the nodes.
    sed_rates: 1D array of sedimentation rates corresponding to the PDF values.
    eta: threshold value for peak detection.
    peaks: array of indices of detected peaks.
    '''

    idx_over_eta = np.where(pdfs[:, idx] >= eta)[0]  # mask CDR region
    idx_region_split = np.split(idx_over_eta, np.where(np.diff(idx_over_eta) != 1)[0] + 1)   # split into regions of consecutive indices over eta

    nodes = []
    region_with_peaks = np.where([any(peak in region for peak in peaks) for region in idx_region_split])[0]
    if len(peaks) == 0:
        idx_region_split = pdfs[idx_over_eta, idx]  # if no peaks, use all indices over eta
    else:
        for i, region in enumerate(idx_region_split):
            if not any(peak in region for peak in peaks): # if no peak in region, move indices to closest region
                # identify closest region with peak
                nearest_peak = peaks[np.argmin(np.abs(peaks - region.mean()))]
                closest_region = next(
                    region_index
                    for region_index in region_with_peaks
                    if nearest_peak in idx_region_split[region_index])
                # copy region to closest region with peak
                idx_region_split[closest_region] = np.append(idx_region_split[closest_region], region)
                idx_region_split[closest_region] = np.sort(idx_region_split[closest_region])
                # empty region to delete later
                idx_region_split[i] = np.array([])

        # delete empty regions
        idx_region_split = [region for region in idx_region_split if len(region) > 0]

    for i, region in enumerate(idx_region_split):
        data = dataNode(layer = idx,
                        prob = np.sum(pdfs[region, idx]),
                        sed_rate = np.mean(sed_rates[region]),
                        var = np.var(sed_rates[region]))
        nodes.append(data)
    
    #renormalise probabilities of nodes in the layer
    total_prob = sum(node.prob for node in nodes)
    if total_prob > 0:
        for node in nodes:
            node.prob = node.prob / total_prob
            node.cum_prob = node.prob  # initialize cumulative probability to the normalized probability of the node
    return nodes 



def renormalise_tree(tree):
    '''
    Renormalises cumulative probabilities of all nodes based on the probabilities of the level. Child node probabilities are updated based on the new parent node probabilities.
    '''
    # renormalise layer by layer
    for layer in range(tree.depth()):
        nodes_in_layer = list(tree.filter_nodes(lambda node: node.data.layer == layer))
        total_prob = sum(node.data.cum_prob for node in nodes_in_layer)
        if total_prob > 0:
            for node in nodes_in_layer:
                node.data.cum_prob = node.data.cum_prob / total_prob  # renormalise cumulative probability
                # update child node probabilities based on new parent node probability
                for child in tree.children(node.identifier):
                    child.data.cum_prob = child.data.prob * node.data.cum_prob  # update child node cumulative probability based on new parent node cumulative probability



def prune_tree(tree, prune_prob_threshold): #NOTE currently unused
    '''
    Prune dead branches with leaves not at maximum depth. Renormalise tree after pruning.
    '''
    # find all leaves not at maximum depth
    max_depth = tree.depth() - 1
    leaves_to_prune = [node for node in tree.leaves() if node.data.layer < max_depth or node.data.cum_prob < prune_prob_threshold]
    if len(leaves_to_prune) == len(tree.leaves()):
        return False  # if all leaves are to be pruned, skip pruning to avoid empty tree
    leaves_to_prune = sorted(leaves_to_prune, key=lambda node: node.data.cum_prob)  # sort leaves by cumulative probability (lowest first)

    for leaf in leaves_to_prune:
        branch = list(tree.rsearch(leaf.identifier))  # get branch from leaf to root
        for node_id in branch:
            # check if node has siblings before pruning
            has_siblings = len(tree.siblings(node_id)) > 0
            tree.remove_node(node_id)  # prune node
            if has_siblings:
                break  # stop pruning the branch if there are siblings
        
    return True  # indicate that pruning was performed



def prune_tree_leaf(tree, prune_prob_threshold):
    '''
    Prune leaf nodes with low cumulative probabilities.
    '''
    # find all leaves not at maximum depth
    max_depth = tree.depth() - 1
    leaves_to_prune = [node for node in tree.leaves() if node.data.layer < max_depth or node.data.cum_prob < prune_prob_threshold]
    if len(leaves_to_prune) == len(tree.leaves()):
        return False  # if all leaves are to be pruned, skip pruning to avoid empty tree
    leaves_to_prune = sorted(leaves_to_prune, key=lambda node: node.data.cum_prob)  # sort leaves by cumulative probability (lowest first)

    for leaf in leaves_to_prune:
        tree.remove_node(leaf.identifier)  # prune leaf node
        
    return True  # indicate that pruning was performed



### The following functions are used to calculate layer likeness and combine data within some user defined threshold ###
def get_all_layer_properties(pdfs, sed_rates, alpha):
    """
    Pre-computes the properties (nodes) for all layers.
    """
    all_nodes = []
    for i in range(pdfs.shape[1]):
        _, eta = Hyndman_CDR(pdfs, i, alpha)
        peaks = peak_detection(pdfs, sed_rates, eta, i)
        nodes = peak_properties(pdfs, i, sed_rates, eta, peaks)
        all_nodes.append(nodes)
    return all_nodes



def simple_layer_likeness(nodes_i, nodes_j, sed_rate_limit, prob_limit = prob_limit):
    """
    Compares two layers based on their pre-computed properties.
    """
    if len(nodes_i) != len(nodes_j):
        return False

    # Use vectorized operations for faster comparison
    sed_rates_i = np.array([node.sed_rate for node in nodes_i])
    sed_rates_j = np.array([node.sed_rate for node in nodes_j])
    probs_i = np.array([node.prob for node in nodes_i])
    probs_j = np.array([node.prob for node in nodes_j])

    sed_rate_distance = np.sum(np.abs(sed_rates_i - sed_rates_j))
    prob_distance = np.sum(np.abs(probs_i - probs_j))

    sed_rate_limit = sed_rate_limit * len(nodes_i)  # Adjust limit based on number of nodes
    prob_limit = prob_limit * len(nodes_i)

    return sed_rate_distance < sed_rate_limit and prob_distance < prob_limit



def combine_layers(pdfs, sed_rates, alpha):
    '''
    Combines similar layers based on the properties of their nodes. Layers are compared sequentially, and if they are deemed similar 
    based on the simple_layer_likeness function, their PDFs are combined by taking the mean across the layers. The number of layers combined 
    is tracked to account for increased uncertainty in the timeline calculation and for accurate integration as dz increases with layer count.
    '''
    pdfs_combined = []
    layer_count = []
    limit_max = pdfs.shape[1]
    combination_maximum = pdfs.shape[1]/100
    i = 0

    all_layer_nodes = get_all_layer_properties(pdfs, sed_rates, alpha)

    while i < limit_max:
        start_layer_idx = i
        # Find the end of the sequence of similar layers
        while (i + 1 < limit_max and 
            simple_layer_likeness(all_layer_nodes[i], all_layer_nodes[i + 1], sed_rate_limit)) and (i - start_layer_idx < combination_maximum):
            i += 1
        
        # The sequence of similar layers is from start_layer_idx to i (inclusive)
        num_layers = i - start_layer_idx + 1
        
        # Extract the pdfs for the combined layers and calculate the mean
        combined_pdfs_mean = np.mean(pdfs[:, start_layer_idx : i + 1], axis=1)
        
        pdfs_combined.append(combined_pdfs_mean)
        layer_count.append(num_layers)
        
        i += 1
    print(f"Combined {len(pdfs_combined)} layers from {limit_max} original layers.")
    # Convert lists to numpy arrays
    pdfs_combined = np.array(pdfs_combined).T
    layer_count = np.array(layer_count)
    return pdfs_combined, layer_count



### The following functions are used to create the tree and calculate the time-depth relationship ###
def create_tree(pdfs_combined, sed_rates, layer_count, dz, prune_prob_threshold):
    '''
    Creates a tree based on the combined layers.
     - For each layer, nodes are created based on the peaks in the PDF and their properties (probability, sedimentation rate, variance).
     - Nodes are attached to each leaf of the tree, with probability calculated as sum of pdf in region * probability of parent node. Variance is scaled by the number of layers combined to account for increased uncertainty.
     - Time is calculated by integrating the sedimentation rate over the number of layers combined, added to the parent node time.
     - After each layer, the tree is renormalised and pruned based on the cumulative probability to remove unlikely branches. Renormalisation is performed again after pruning to account for removed probability mass.
    '''
    dataTree = Tree()
    dataTree.create_node(identifier=None, tag = "root", data=dataNode(layer=-1, prob=1.0, sed_rate=0, var=0))  # root node

    pbar = tqdm(range(0, len(pdfs_combined[0])), desc="Processing layers")
    for idx in pbar:
        mask, eta = Hyndman_CDR(pdfs_combined, idx, 0.70)
        peaks = peak_detection(pdfs_combined, sed_rates, eta, idx)
        nodes = peak_properties(pdfs_combined, idx, sed_rates, eta, peaks)
        
        leaves = dataTree.leaves()
        alive_leaves = [leaf for leaf in leaves if dataTree.depth(leaf) == idx]  # only consider leaves with cumulative probability above threshold and at the current depth
        new_node_added_to_leaf = False  # flag to check if a new node was added to any leaf
        for leaf in alive_leaves:
            # attach nodes to each leaf, with probability calculated as sum of pdf in region * probability of parent node
            parent_node = dataTree.get_node(leaf.identifier)
            for node_data in nodes:
                new_node_data = copy.copy(node_data)
                new_node_data.cum_prob = new_node_data.prob * parent_node.data.cum_prob
                new_node_data.var = new_node_data.var * (layer_count[idx] * dz)**2                              # variance scaled by number of layers combined
                new_node_data.layer_count = layer_count[idx]                                          # number of layers in the original data that are grouped together in this node, used for time conversion and plotting
                new_node_data.total_idx = parent_node.data.total_idx + layer_count[idx]   # first layer index of the node in the original data
                new_node_data.time_cum = dz * new_node_data.sed_rate * new_node_data.layer_count + parent_node.data.time_cum              # integrate sedimentation rate over the number of layers to get time, add to parent node time

                if new_node_data.cum_prob > prune_prob_threshold: # only create node if cumulative probability above threshold
                    dataTree.create_node(identifier=None, tag = f"{idx}", parent=leaf.identifier, data=new_node_data)
                    new_node_added_to_leaf = True
        
        if not new_node_added_to_leaf:
            print(f"Layer {idx}: no new nodes added to any leaf, skipping pruning to avoid empty tree")
            break  # skip pruning if no new nodes were added to any leaf
        renormalise_tree(dataTree)

        status = prune_tree_leaf(dataTree, prune_prob_threshold = prune_prob_threshold)
        if status == False:
            print(f"Layer {idx}: pruning skipped to avoid empty tree")
            break

        renormalise_tree(dataTree) # renormalise after pruning to account for removed probability mass
    return dataTree



def timeline(tree, dz, layer_count):
    '''
    Calculates the timeline based on the tree structure. Averages time and sedimentation rate of nodes that contribute to each layer, 
    and calculates outer edges of variance across branches to get error bars. Interpolates to avoid gaps caused by combining layers.
     - layer_limit is set to the maximum depth of the tree to match the range of the AC data. Only layers that contribute to the tree are included in the timeline.
     - For each layer, nodes that contribute to the layer are identified based on their total_idx and layer_count. The time, sedimentation rate, and variance of 
        these nodes are averaged to get the timeline for the layer. The variance is calculated based on the maximum and minimum time across branches, accounting 
        for the variance of each node and the number of layers combined.
     - Interpolation is performed to fill in any gaps in the timeline caused by combining layers, which can lead to some layers not contributing to the tree and therefore having no nodes.
    '''
    layer_limit = tree.depth()  # to output
    all_nodes = list(tree.all_nodes())
    
    layers_total = np.sum(layer_count[:layer_limit])
    layer_times = np.zeros([layers_total])
    layer_vars = np.zeros([layers_total])
    layer_rate = np.zeros([layers_total])
    
    for layer in range(layers_total):
        layer_nodes = [node for node in all_nodes if node.data.total_idx == layer]
        times_total = np.array([])
        vars_total = np.array([])
        rate_total = np.array([])
        prob_total = np.array([])

        for node in layer_nodes:
            if node.data.time_cum == 0:
                continue  # skip nodes with zero cumulative time
            times_total = np.append(times_total, node.data.time_cum)
            vars_total = np.append(vars_total, node.data.var)
            rate_total = np.append(rate_total, node.data.sed_rate)
            prob_total = np.append(prob_total, node.data.cum_prob)


        if times_total.size == 0:
            # skip iteration if no nodes contribute to the layer (necessary as many layers are combined)
            continue
        # weighted mean and times based on cumulative probability of nodes that contribute to the layer
        times_weighted = np.average(times_total, weights=prob_total/np.sum(prob_total))
        rate_weighted = np.average(rate_total, weights=prob_total/np.sum(prob_total))

        layer_times[layer] = times_weighted   # average time of nodes that contribute to the layer
        layer_rate[layer] = rate_weighted    # average sedimentation rate of nodes that contribute to the layer
        layer_vars[layer] = np.sum((prob_total/np.sum(prob_total))**2 * vars_total)  # weighted variance of nodes that contribute to the layer

    # interpolate to avoid gaps caused by combining layers 
    layer_vars = np.insert(layer_vars, 0, 1e-15)  # add zero at the start to avoid interpolation issues
    layer_times = np.insert(layer_times, 0, 1e-15)  # add zero at the start to avoid interpolation issues
    layer_rate = np.insert(layer_rate, 0, 1e-15)  # add zero at the start to avoid interpolation issues

    mask = layer_times > 0
    vars_interp = np.interp(np.arange(layers_total+1), np.arange(layers_total+1)[mask], layer_vars[mask])
    times_interp = np.interp(np.arange(layers_total+1), np.arange(layers_total+1)[mask], layer_times[mask])
    rate_interp = np.interp(np.arange(layers_total+1), np.arange(layers_total+1)[mask], layer_rate[mask])
    # remove the first element which was added for interpolation

    return times_interp[1:], vars_interp[1:], rate_interp[1:], layers_total  # times and vars in kyr



def run_multimodal_analysis(pdfs, sed_rates, dz, params):
    '''
    Main function to run the multimodal analysis. Combines similar layers, creates the tree, and calculates the timeline.
    '''
    global prune_prob_threshold, sed_rate_limit, alpha
    alpha = params.credInterval
    prune_prob_threshold = params.mmPruneProbThreshold
    sed_rate_limit = np.diff(sed_rates)[0] 

    pdfs_combined, layer_count = combine_layers(pdfs, sed_rates, alpha)
    
    dataTree = create_tree(pdfs_combined, sed_rates, layer_count, dz, prune_prob_threshold)
    
    times_interp, vars_interp, rate_interp, layers_total = timeline(dataTree, dz, layer_count)

    print(f"Multimodal analysis completed. Total layers processed: {layers_total}")

    return times_interp, vars_interp, rate_interp, layers_total

