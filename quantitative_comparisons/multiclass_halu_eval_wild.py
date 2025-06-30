import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from neural_controllers import NeuralController
from utils import load_model

import json
import numpy as np
import torch
import pickle
from sklearn.model_selection import train_test_split
import random
random.seed(0)

NEURAL_CONTROLLERS_DIR = os.environ['NEURAL_CONTROLLERS_DIR']
TYPES = ['confused / erroneous queries', 'inappropriate content', 'complex reasoning', 
         'out-of-scope information', 'beyond-modality interaction', 'other types']
TYPE_MAP = {
    'confused': 'confused / erroneous queries',
    'inappropriate': 'inappropriate content',
    'complex': 'complex reasoning',
    'out-of-scope': 'out-of-scope information',
    'beyond-modality': 'beyond-modality interaction',
    'other': 'other types'
}


def read_json_to_list(file_path):
    """
    Reads a JSON file and returns its contents as a list of dictionaries.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    if isinstance(data, list):
        return data
    else:
        raise ValueError("JSON content is not a list.")

def get_multiclass_halu_eval_wild_data(prompt_version='v1'):
    data_path = f'{NEURAL_CONTROLLERS_DIR}/data/hallucinations/halu_eval_wild/HaluEval_Wild_6types.json'
    entries = read_json_to_list(data_path)
    
    # Get unique classes from TYPE_MAP
    classes = TYPES
    num_classes = len(classes)

    if prompt_version == 'v1':
        template = "Queries that induce hallucinations consist of the following six types. "
        template += "(1) Confused / Erroneous queries: Queries that contain errors in the entity, relation, or sentence. "
        template += "(2) Inappropriate content: Queries that request inappropriate content. "
        template += "(3) Complex reasoning: Queries that require complex reasoning. "
        template += "(4) Out-of-scope information: Queries that ask for information out-of-scope for the LLM. "
        template += "(5) Beyond-modality interaction: Queries that require modalities beyond the abilities of the language model being queried. "
        template += "(6) Other types: Queries that are not out-of-scope, do not require complex reasoning, are not beyond-modality, are not inappropriate, and are not confused or erroneous. " 
        template += "Based on the above definitions, which single category does the following query fall into? Respond just with a number between 1 and 6. "
        template += "For example, your response would be just 'N.' if the query belongs to category N.\n\n"
        template += "Query: {query}"
    elif prompt_version == 'empty':
        template = "{query}"

    inputs = []
    ohe_labels = []
    
    for entry in entries:
        query = entry['query']
        qtype = entry['query_type']
        inputs.append(template.format(query=query))
        
        label = [0] * num_classes
        class_idx = classes.index(qtype)
        label[class_idx] = 1
        ohe_labels.append(torch.tensor(label))
    
    ohe_labels = torch.stack(ohe_labels).reshape(-1, num_classes).cuda().float()

    return inputs, ohe_labels   

def get_kfold_splits(k_folds, n_total):
    """
    Creates k-fold cross validation splits.
    Args:
        k_folds (int): Number of folds for cross validation
        n_total (int): Total number of samples
    Returns:
        List of dictionaries containing fold indices
    """
    results_dir = f'{NEURAL_CONTROLLERS_DIR}/results/halu_eval_wild_results'
    os.makedirs(results_dir, exist_ok=True)
    out_name = f'{results_dir}/kfold_splits_k_{k_folds}_ntotal_{n_total}.pkl'
    try:
        with open(out_name, 'rb') as f:
            splits = pickle.load(f)
            return splits
    except:
        indices = np.random.permutation(n_total)
        splits = []
        fold_size = n_total // k_folds
        for fold in range(k_folds):
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < k_folds - 1 else n_total
            test_indices = indices[start_idx:end_idx]
            train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
            train_indices, val_indices = train_test_split(train_indices, test_size=0.2, random_state=0, shuffle=True)
            splits.append({
                'train_indices': train_indices,
                'val_indices': val_indices,
                'test_indices': test_indices,
                'fold': fold
            })
    with open(out_name, 'wb') as f:
        pickle.dump(splits, f)
    return splits

def split_states_on_idx(inputs, split):
    train_inputs, val_inputs, test_inputs = {}, {}, {}
    for layer_idx, layer_states in inputs.items():
        train_inputs[layer_idx] = layer_states[split['train_indices']]
        val_inputs[layer_idx] = layer_states[split['val_indices']]
        test_inputs[layer_idx] = layer_states[split['test_indices']]
    return train_inputs, val_inputs, test_inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control_method', type=str, default='rfm')
    parser.add_argument('--model_name', type=str, default='llama_3.3_70b_4bit_it', choices=['llama_3.3_70b_4bit_it', 'llama_3_8b_it'])
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--n_components', type=int, default=6)
    parser.add_argument('--rfm_iters', type=int, default=10)
    parser.add_argument('--prompt_version', type=str, default='v1')
    parser.add_argument('--tuning_metric', type=str, default='top_agop_vectors_ols_auc')
    args = parser.parse_args()
    for n_, v_ in args.__dict__.items():
        print(f"{n_:<20} : {v_}")  

    control_method = args.control_method
    model_name = args.model_name
    n_components = args.n_components
    prompt_version = args.prompt_version
    k_folds = args.k_folds
    tuning_metric = args.tuning_metric

    if control_method not in ['rfm']:
        n_components = 6
        tuning_metric = 'auc'

    language_model, tokenizer = load_model(model=model_name)
    controller = NeuralController(
        language_model,
        tokenizer,
        control_method=control_method,
        rfm_iters=args.rfm_iters,
        batch_size=2,
        n_components=n_components
    )  

    unformatted_inputs, labels = get_multiclass_halu_eval_wild_data(prompt_version)
    inputs = []
    for prompt in unformatted_inputs:
        chat = [
            {
                "role": "user", 
                "content": prompt
            },
        ]
        inputs.append(tokenizer.apply_chat_template(chat, tokenize=False))

    print("="*100)
    print(inputs[0])
    print("="*100)


    # Precompute and cache hidden states
    hidden_states_path = os.path.join(f'{NEURAL_CONTROLLERS_DIR}', f'hidden_states', 
                                    f'halu_eval_wild_{model_name}_prompt_{prompt_version}.pth')
    if os.path.exists(hidden_states_path):
        with open(hidden_states_path, 'rb') as f:
            hidden_states = pickle.load(f)
    else:
        from direction_utils import get_hidden_states
        hidden_states = get_hidden_states(inputs, language_model, tokenizer, 
                                      controller.hidden_layers, 
                                      controller.hyperparams['forward_batch_size'])
        os.makedirs(os.path.dirname(hidden_states_path), exist_ok=True)
        with open(hidden_states_path, 'wb') as f:
            pickle.dump(hidden_states, f)


    # K-fold splits
    splits = get_kfold_splits(k_folds, len(labels))
    results_dir = f'{NEURAL_CONTROLLERS_DIR}/results/halu_eval_wild_results'
    os.makedirs(results_dir, exist_ok=True)

    all_test_predictions = []
    all_test_labels = []
    for fold_data in splits:
        print(f'Fold {fold_data["fold"]+1} of {k_folds}')
        fold = fold_data['fold']
        val_indices = fold_data['val_indices']
        train_indices = fold_data['train_indices']
        test_indices = fold_data['test_indices']
        

        # Split hidden states and labels
        train_hidden_states = {layer: states[train_indices] for layer, states in hidden_states.items()}
        val_hidden_states = {layer: states[val_indices] for layer, states in hidden_states.items()}
        test_hidden_states = {layer: states[test_indices] for layer, states in hidden_states.items()}

        train_labels = labels[train_indices]
        val_labels = labels[val_indices]
        test_labels = labels[test_indices]

        try:
            print("Loading controller")
            controller.load(concept=f'halu_eval_wild_multiclass_fold_{fold}_prompt_{prompt_version}_tuning_metric_{tuning_metric}_top_k_{n_components}', model_name=model_name, path=f'{NEURAL_CONTROLLERS_DIR}/directions/')
        except:
            print("Loading failed, computing directions")
            controller.compute_directions(train_hidden_states, train_labels, val_hidden_states, val_labels, tuning_metric=tuning_metric)
            controller.save(concept=f'halu_eval_wild_multiclass_fold_{fold}_prompt_{prompt_version}_tuning_metric_{tuning_metric}_top_k_{n_components}', model_name=model_name, path=f'{NEURAL_CONTROLLERS_DIR}/directions/')

        print("Evaluating directions")
        _, _, _, test_predictions = controller.evaluate_directions(
            train_hidden_states, train_labels,
            val_hidden_states, val_labels,
            test_hidden_states, test_labels,
            n_components=n_components,
            agg_model=control_method,
        )

        # Store predictions and labels for this fold
        all_test_predictions.append((test_indices, test_predictions))
        all_test_labels.append((test_indices, test_labels.cpu() if hasattr(test_labels, 'cpu') else test_labels))

    # Aggregate predictions and labels over all folds
    # Sort by original indices to match dataset order
    all_indices = torch.cat([torch.from_numpy(idx) for idx, _ in all_test_predictions])
    aggregated_preds = torch.cat([pred['aggregation'] for _, pred in all_test_predictions])
    best_layer_preds = torch.cat([pred['best_layer'] for _, pred in all_test_predictions])
    sort_idx = torch.argsort(all_indices).to(aggregated_preds.device)

    aggregated_preds_sorted = aggregated_preds[sort_idx]
    best_layer_preds_sorted = best_layer_preds[sort_idx]

    # Compute overall metrics
    from direction_utils import compute_prediction_metrics
    aggregated_metrics = compute_prediction_metrics(aggregated_preds_sorted, labels)
    best_layer_metrics = compute_prediction_metrics(best_layer_preds_sorted, labels)
    
    # out_name = f'{results_dir}/{source_ds}-{model_name}-{control_method}-prompt_{prompt_version}-tuning_metric_{tuning_metric}-top_k_{n_components}-best_layer_metrics.pkl'
    # out_name = f'{results_dir}/{source_ds}-{model_name}-{control_method}-prompt_{prompt_version}-tuning_metric_{tuning_metric}-top_k_{n_components}-aggregated_metrics.pkl'
    agg_metrics_file = f'{results_dir}/halu_eval_wild-{model_name}-{control_method}-prompt_{prompt_version}-tuning_metric_{tuning_metric}-top_k_{n_components}-aggregated_metrics.pkl'
    with open(agg_metrics_file, 'wb') as f:
        pickle.dump(aggregated_metrics, f)

    best_layer_metrics_file = f'{results_dir}/halu_eval_wild-{model_name}-{control_method}-prompt_{prompt_version}-tuning_metric_{tuning_metric}-top_k_{n_components}-best_layer_metrics.pkl'
    with open(best_layer_metrics_file, 'wb') as f:
        pickle.dump(best_layer_metrics, f)

    # Save predictions
    predictions_file = f'{results_dir}/halu_eval_wild-{model_name}-{control_method}-prompt_{prompt_version}-tuning_metric_{tuning_metric}-top_k_{n_components}-predictions.pkl'
    with open(predictions_file, 'wb') as f:
        pickle.dump({
            'aggregation': aggregated_preds_sorted,
            'best_layer': best_layer_preds_sorted,
        }, f)

    print('Aggregated k-fold metrics:')
    print(aggregated_metrics)

    print('Best layer k-fold metrics:')
    print(best_layer_metrics)
        
if __name__ == '__main__':              
    main()