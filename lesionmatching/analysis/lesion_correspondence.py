"""
Script to establish lesion correspondence, since there isn't necessarily 1:1 corr. between objects in the
predicted and reference segmentations

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import os
from lesionmatching.util_scripts.image_utils import find_individual_lesions
import networkx as nx
import numpy as np
from networkx.algorithms import bipartite
from networkx.exception import NetworkXPointlessConcept
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class Lesion():

    def __init__(self, lesion:np.ndarray, idx:int=-1, prefix:str='Baseline', center=None, diameter:float=-1.0):

        self.lesion = lesion

        self.idx = idx

        self.name = '{}_lesion_{}'.format(prefix, idx)

        self.label = -1

        self.diameter = diameter

        self.center = np.asarray(center)

    def get_lesion(self):
        return self.lesion

    def get_name(self):
        return self.name

    def get_idx(self):
        return self.idx

    def get_center(self):
        return self.center

    def get_diameter(self):
        return self.diameter

    # Use this method only for predicted lesion!!
    def set_label(self, label):
        self.label = label

    def get_label(self):
        return self.label


def create_correspondence_graph_from_list(pred_lesions,
                                          gt_lesions,
                                          seg,
                                          gt,
                                          min_distance=10.00,
                                          min_diameter=10.00,
                                          verbose=False):


    assert(isinstance(pred_lesions, list))
    assert(isinstance(gt_lesions, list))

    # Create a directed bipartite graph
    dgraph = nx.DiGraph()

    # In case of no overlap between 2 lesion, we add an edge with weight 0
    # so that the graph has no disconnected nodes
    # Create forward edges (partition 0 -> partition 1)
    for p_idx, pred_lesion in enumerate(pred_lesions):
        seg_lesion_volume = pred_lesion.get_lesion()
        # Iterate over GT lesions
        seg_lesion_center = pred_lesion.get_center()
        for g_idx, gt_lesion in enumerate(gt_lesions):
            if gt_lesion.get_diameter() <= min_diameter:
                continue
            gt_lesion_volume = gt_lesion.get_lesion()
            gt_lesion_center = gt_lesion.get_center()
            distance = np.sqrt(np.dot((seg_lesion_center-gt_lesion_center),
                                      (seg_lesion_center-gt_lesion_center)))

            # Compute distance between lesion centers
            if distance <= min_distance:
                dgraph.add_weighted_edges_from([(pred_lesion, gt_lesion, 1)])
                if verbose is True:
                    print('Follow-up lesion {} matches baseline lesion {}'.format(p_idx, g_idx))
            else: # False positive
                dgraph.add_weighted_edges_from([(pred_lesion, gt_lesion, 0)])

    # Create backward edges (partition 1 -> partition 0)
    for g_idx, gt_lesion in enumerate(gt_lesions):
        if gt_lesion.get_diameter() <= min_diameter:
            continue
        gt_lesion_volume = gt_lesion.get_lesion()
        gt_lesion_center = gt_lesion.get_center()
        # Iterate over pred lesions
        for p_idx, pred_lesion in enumerate(pred_lesions):
            seg_lesion_volume = pred_lesion.get_lesion()
            seg_lesion_center = pred_lesion.get_center()
            distance = np.sqrt(np.dot((seg_lesion_center-gt_lesion_center),
                                      (seg_lesion_center-gt_lesion_center)))
            if distance <= min_distance:
                dgraph.add_weighted_edges_from([(gt_lesion, pred_lesion, 1)])
                if verbose is True:
                    print('Baseline lesion {} matches follow-up lesion {}'.format(p_idx, g_idx))
            else:
                dgraph.add_weighted_edges_from([(gt_lesion, pred_lesion, 0)])

    # Check if the constructed graph is bipartite
    assert(bipartite.is_bipartite(dgraph))

    return dgraph


def filter_edges(dgraph, min_weight=0.0):
    """

    Function to remove edges with zero weight (for better viz)

    """
    try:
        pred_lesion_nodes, gt_lesion_nodes = bipartite.sets(dgraph)
    except NetworkXPointlessConcept:
        return None

    # Create a dummy graph that has disconnected nodes for better visualization

    dgraph_viz = nx.DiGraph()

    # Create forward connections
    for pred_node in pred_lesion_nodes:
        weights = []
        for gt_node in gt_lesion_nodes:
            edge_weight = dgraph[pred_node][gt_node]['weight']
            weights.append(edge_weight)
            if edge_weight > min_weight:
                dgraph_viz.add_weighted_edges_from([(pred_node, gt_node, edge_weight)])

        max_weight = np.amax(np.array(weights))

        if max_weight == 0:
            dgraph_viz.add_node(pred_node) # False positive

    # Create backward connections
    for gt_node in gt_lesion_nodes:
        weights = []
        for pred_node in pred_lesion_nodes:
            edge_weight = dgraph[gt_node][pred_node]['weight']
            weights.append(edge_weight)
            if edge_weight > min_weight:
                dgraph_viz.add_weighted_edges_from([(gt_node, pred_node, edge_weight)])

        max_weight = np.amax(np.array(weights))

        if max_weight == 0:
            dgraph_viz.add_node(gt_node) # False negative

    return dgraph_viz


def visualize_lesion_correspondences(dgraph, fname=None, remove_list=None, min_weight=0.0):

    try:
        pred_lesion_nodes, gt_lesion_nodes = bipartite.sets(dgraph)
        dgraph_viz = filter_edges(dgraph, min_weight=min_weight)

        if remove_list is not None:
            dgraph_viz.remove_nodes_from(remove_list)

            # Get rid of the of the lesions from the list
            for pred_node in remove_list:
                pred_lesion_nodes.remove(pred_node)

        # Create a color map
        color_map = []
        label_dict = {}
        for node in dgraph_viz:
            node_name = node.get_name()

            if "predicted" in node_name.lower() or "moving" in node_name.lower():
                color_map.append('tab:red')
                label_dict[node] = '{}{}'.format(node_name[0].upper(), node.get_idx())
            else:
                color_map.append('tab:green')
                label_dict[node] = '{}{}'.format(node_name[0].upper(), node.get_idx())

        pos = nx.bipartite_layout(dgraph_viz, gt_lesion_nodes)

        fig, ax = plt.subplots()

        nx.draw(dgraph_viz,
                pos=pos,
                labels=label_dict,
                with_labels=True,
                node_color=color_map,
                ax=ax,
                fontsize=18,
                nodesize=800,
                font_color='white')


        fig.savefig(fname,
                    bbox_inches='tight')

        plt.close()
    except NetworkXPointlessConcept:
        print('No measurable lesions found')


def construct_dict_from_graph(dgraph, min_diameter=10.00):
    """
    Function to construct a list of matched lesions
    Example ouput : [(fixed_lesion_id_0, moving_lesion_id_0), ..., (fixed_lesion_id_n, moving_lesion_id_n)]

    """
    moving_lesion_nodes, fixed_lesion_nodes = bipartite.sets(dgraph)

    assert('moving' in list(moving_lesion_nodes)[0].get_name().lower())
    assert('fixed' in list(fixed_lesion_nodes)[0].get_name().lower())

    predicted_matches_dict = {}
    for f_node in fixed_lesion_nodes:
        predicted_matches_dict[f_node.get_name().lower()] = []
        f_diameter = f_node.get_diameter()
        for m_node in moving_lesion_nodes:
            edge_weight = dgraph[f_node][m_node]['weight']
            if edge_weight > 0: # If a match, each edge is weighted by 1/distance
                if f_diameter >= min_diameter:
                    predicted_matches_dict[f_node.get_name().lower()].append(m_node.get_name().lower())

    return predicted_matches_dict


def find_unmatched_lesions_in_moving_images(dgraph, min_overlap=0.5):

    unmatched_lesions = 0

    moving_lesion_nodes, fixed_lesion_nodes = bipartite.sets(dgraph)

    assert('moving' in list(moving_lesion_nodes)[0].get_name().lower())
    assert('fixed' in list(fixed_lesion_nodes)[0].get_name().lower())

    for m_node in moving_lesion_nodes:
        match_counter = 0
        for f_node in fixed_lesion_nodes:
            edge_weight = dgraph[m_node][f_node]['weight']
            if edge_weight > min_overlap:
                match_counter += 1
        # Check match counter, if it hasn't incremented => No match found!
        if match_counter == 0:
            unmatched_lesions += 1


    return unmatched_lesions

def preprocess_gt_dict(gt_dict):
    """

    Preprocess gt_dict so that each key (fixed_lesion name) contains a
    list of matching moving lesion names

    a typical dictionary entry would look like:
        fixed_lesion_n : [moving_lesion_a, ... moving_lesion_d]
    In case of no matches,
        fixed_lesion_n : []

    """

    full_gt_dict = {}
    forward_gt_dict = {} # 'fixed_lesion_*' : [moving_lesion_*]
    backward_gt_dict = {} # 'moving_lesion_*' : [fixed_lesion_*]


    # Construct empty lists
    for fixed_lesion in list(gt_dict.keys()):
        forward_gt_dict[fixed_lesion] = []

    moving_lesion_list = list(gt_dict[list(gt_dict.keys())[0]].keys())


    for moving_lesion in moving_lesion_list:
        backward_gt_dict[moving_lesion] = []

    # Construct forward matching dict
    for fixed_lesion, match_dict in gt_dict.items():
        for moving_lesion, match in match_dict.items():
            if match is True:
                forward_gt_dict[fixed_lesion].append(moving_lesion)

    # Construct backward matching dict
    for moving_lesion in moving_lesion_list:
        for fixed_lesion, match_dict in gt_dict.items():
            if match_dict[moving_lesion] is True:
                backward_gt_dict[moving_lesion].append(fixed_lesion)

    full_gt_dict['forward_dict'] = forward_gt_dict
    full_gt_dict['backward_dict'] = backward_gt_dict

    full_gt_dict['new lesions'] = []
    full_gt_dict['disappearing lesions'] = []

    for fixed_lesion, moving_list in forward_gt_dict.items():
        if len(moving_list) == 0:
            full_gt_dict['disappearing lesions'].append(fixed_lesion)

    for moving_lesion, fixed_list in backward_gt_dict.items():
        if len(fixed_list) == 0:
            full_gt_dict['new lesions'].append(moving_lesion)

    return full_gt_dict

def compute_detection_metrics(pred_dict,
                              gt_dict):

    # Compare prediciton to ground truth
    true_positive_matches = 0
    false_positive_matches = 0
    false_negatives = 0
    true_negatives = 0

    forward_gt_dict = gt_dict['forward_dict']
    backward_gt_dict = gt_dict['backward_dict']

    # NOTE: We count a split (i.e. a single fixed lesion split into multiple moving lesions) as a single true positive
    # We count a merge (i.e. multiple fixed lesions merge into a single moving lesion) as multiple true positives
    for fixed_lesion, moving_lesions in pred_dict.items():
        matches = 0
        if len(moving_lesions) > 0:
            for moving_lesion in moving_lesions:
                if moving_lesion in forward_gt_dict[fixed_lesion]:
                    matches += 1
                else:
                    false_positive_matches += 1
            if matches == len(moving_lesions): # All lesions matched
                true_positive_matches += 1
            elif matches < len(moving_lesions): # Partial matches!
                false_negatives += len(moving_lesions) - matches
        else: # Empty list for the lesion in the baseline image!
            if len(forward_gt_dict[fixed_lesion]) == 0:
                true_negatives += 1
            else:
                false_negatives += 1

    count_dict = {}
    count_dict['TP'] = true_positive_matches
    count_dict['FP'] = false_positive_matches
    count_dict['FN'] = false_negatives
    count_dict['TN'] = true_negatives

    return count_dict

