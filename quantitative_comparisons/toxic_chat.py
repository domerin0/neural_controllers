import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from neural_controllers import NeuralController
from utils import load_model
import numpy as np
import pickle
import torch

from datasets import load_dataset
import random
random.seed(0)

NEURAL_CONTROLLERS_DIR = os.environ['NEURAL_CONTROLLERS_DIR']

def get_prompt(prompt_version):
    if prompt_version == 'v1':
        return 'Toxic content mainly includes offensiveness, sexism, and hateful speech. Does the following query contain toxic content? State yes or no. \n{query}'
    elif prompt_version == 'empty':
        return '{query}'
    else:
        raise ValueError(f"Invalid prompt version: {prompt_version}")

def get_data(controller, prompt_version):
    # Load the dataset
    ds = load_dataset("lmsys/toxic-chat", "toxicchat1123")

    train_inputs = [x['user_input'] for x in ds['train']]
    test_inputs = [x['user_input'] for x in ds['test']]

    prompt = get_prompt(prompt_version)

    # Wrap inputs with prompt
    train_inputs = [prompt.format(query=x) for x in train_inputs]
    test_inputs = [prompt.format(query=x) for x in test_inputs]
        
    # Format prompts with chat template
    train_inputs = [controller.format_prompt(x) for x in train_inputs]
    test_inputs = [controller.format_prompt(x) for x in test_inputs]
    
    # Extract labels
    train_labels = [x['toxicity'] for x in ds['train']]
    test_labels = [x['toxicity'] for x in ds['test']]
    
    return train_inputs, np.array(train_labels), test_inputs, np.array(test_labels)


def split_states_on_idx(inputs, indices):
    if isinstance(inputs, list):
        return [inputs[i] for i in indices]
    elif isinstance(inputs, dict):
        split_states = {}
        for layer_idx, layer_states in inputs.items():
            split_states[layer_idx] = layer_states[indices]
        return split_states
    else:
        return inputs[indices]


def get_splits(k_folds, n_total):
    """
    Creates k-fold cross validation splits with multiple random seeds.
    
    Args:
        k_folds (int): Number of folds for cross validation
        n_total (int): Total number of samples
        
    Returns:
        List of dictionaries containing fold indices for each seed
    """
    results_dir = f'{NEURAL_CONTROLLERS_DIR}/results/toxic_chat_results'
    os.makedirs(results_dir, exist_ok=True)
    out_name = f'{results_dir}/kfold_splits_k_{k_folds}_ntotal_{n_total}.pkl'
    
    try:
        with open(out_name, 'rb') as f:
            splits = pickle.load(f)
            return splits
    except:
        # Create random permutation of indices
        indices = np.random.permutation(n_total)

        splits = []
        fold_size = n_total // k_folds

        # For each fold
        for fold in range(k_folds):
            # Calculate fold size
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < k_folds - 1 else n_total
            
            val_indices = indices[start_idx:end_idx]
            train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
            
            splits.append({
                'val_indices': val_indices,
                'train_indices': train_indices,
                'fold': fold
            })

    with open(out_name, 'wb') as f:
        pickle.dump(splits, f)

    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control_method', type=str, default='rfm')
    parser.add_argument('--model_name', type=str, default='llama_3.3_70b_4bit_it')
    parser.add_argument('--n_components', type=int, default=3)
    parser.add_argument('--k_folds', type=int, default=10)
    parser.add_argument('--rfm_iters', type=int, default=8)
    parser.add_argument('--prompt_version', type=str, default='v1')
    parser.add_argument('--tuning_metric', type=str, default='top_agop_vectors_ols_auc')
    args = parser.parse_args()
    for n_, v_ in args.__dict__.items():
        print(f"{n_:<20} : {v_}")
    
    control_method = args.control_method
    model_name = args.model_name
    n_components = args.n_components
    k_folds = args.k_folds
    prompt_version = args.prompt_version
    tuning_metric = args.tuning_metric

    if control_method not in ['rfm']:
        n_components=1
        tuning_metric = 'auc'
                            
    language_model, tokenizer = load_model(model=model_name)
    controller = NeuralController(
        language_model,
        tokenizer,
        control_method=control_method,
        rfm_iters=args.rfm_iters,
        batch_size=1,
        n_components=n_components,
    )  
    controller.name = model_name
    
    # Get base data first
    train_inputs, train_labels, test_inputs, test_labels = get_data(controller, prompt_version) 


    print("="*100)
    print(train_inputs[0])
    print("="*100)
    
    # Precompute and cache hidden states
    train_hidden_states_path = os.path.join(f'{NEURAL_CONTROLLERS_DIR}', f'hidden_states', 
                                   f'toxic_chat_train_{model_name}_prompt_{prompt_version}.pth')
    test_hidden_states_path = os.path.join(f'{NEURAL_CONTROLLERS_DIR}', f'hidden_states', 
                                  f'toxic_chat_test_{model_name}_prompt_{prompt_version}.pth')
    
    if os.path.exists(train_hidden_states_path) and os.path.exists(test_hidden_states_path):
        print("Loading cached hidden states...")
        with open(train_hidden_states_path, 'rb') as f:
            train_hidden_states = pickle.load(f)
        with open(test_hidden_states_path, 'rb') as f:
            test_hidden_states = pickle.load(f)
    else:
        print("Computing hidden states...")
        from direction_utils import get_hidden_states
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(train_hidden_states_path), exist_ok=True)
        
        # Compute train hidden states
        train_hidden_states = get_hidden_states(train_inputs, language_model, tokenizer, 
                                         controller.hidden_layers, 
                                         controller.hyperparams['forward_batch_size'])
        with open(train_hidden_states_path, 'wb') as f:
            pickle.dump(train_hidden_states, f)
            
        # Compute test hidden states
        test_hidden_states = get_hidden_states(test_inputs, language_model, tokenizer, 
                                        controller.hidden_layers, 
                                        controller.hyperparams['forward_batch_size'])
        with open(test_hidden_states_path, 'wb') as f:
            pickle.dump(test_hidden_states, f)
    
    # Convert train_labels to tensor for proper indexing
    train_labels = torch.tensor(train_labels).cuda().float()
    test_labels = torch.tensor(test_labels).cuda().float()
    
    # Get k-fold splits
    splits = get_splits(k_folds, len(train_labels))
    
    results_dir = f'{NEURAL_CONTROLLERS_DIR}/results/toxic_chat_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Store predictions from each fold
    all_test_predictions = []
    
    # For each fold
    for fold_data in splits:
        fold = fold_data['fold']
        print(f"Fold {fold+1} out of {k_folds}")

        val_indices = fold_data['val_indices']
        train_indices = fold_data['train_indices']
        
        # Split the hidden states
        val_hidden_states = split_states_on_idx(train_hidden_states, val_indices)
        train_hidden_states_split = split_states_on_idx(train_hidden_states, train_indices)
        
        # Split the labels
        val_labels = train_labels[val_indices]
        train_labels_split = train_labels[train_indices]
          
        try:
            controller.load(concept=f'toxic_chat_fold_{fold}_prompt_{prompt_version}_tuning_metric_{tuning_metric}_top_k_{n_components}', model_name=model_name, path=f'{NEURAL_CONTROLLERS_DIR}/directions/')
        except:
            controller.compute_directions(train_hidden_states_split, train_labels_split, 
                                          val_hidden_states, val_labels, 
                                          tuning_metric=tuning_metric)
            controller.save(concept=f'toxic_chat_fold_{fold}_prompt_{prompt_version}_tuning_metric_{tuning_metric}_top_k_{n_components}', model_name=model_name, path=f'{NEURAL_CONTROLLERS_DIR}/directions/')

        # Evaluate on validation and test sets
        _, _, _, test_predictions = controller.evaluate_directions(
            train_hidden_states_split, train_labels_split,
            val_hidden_states, val_labels,
            test_hidden_states, test_labels,
            n_components=n_components,
            agg_model=control_method,
        )
        
        # Store test predictions for this fold
        all_test_predictions.append(test_predictions)
    

    all_agg_predictions = [p['aggregation'].cpu().numpy() for p in all_test_predictions] 
    all_best_layer_predictions = [p['best_layer'].cpu().numpy() for p in all_test_predictions]

    avg_agg_predictions = np.mean(all_agg_predictions, axis=0)
    avg_best_layer_predictions = np.mean(all_best_layer_predictions, axis=0)

    # Compute and save AUC score
    from direction_utils import compute_prediction_metrics
    agg_metrics = compute_prediction_metrics(avg_agg_predictions, test_labels)
    best_layer_metrics = compute_prediction_metrics(avg_best_layer_predictions, test_labels)

    # Save predictions
    predictions_file = f'{results_dir}/toxic_chat-{model_name}-{control_method}-prompt_{prompt_version}-tuning_metric_{tuning_metric}-top_k_{n_components}-predictions.pkl'
    with open(predictions_file, 'wb') as f:
        pickle.dump({
            'aggregation': avg_agg_predictions,
            'best_layer': avg_best_layer_predictions,
        }, f)
    
    agg_metrics_file = f'{results_dir}/toxic_chat-{model_name}-{control_method}-prompt_{prompt_version}-tuning_metric_{tuning_metric}-top_k_{n_components}-aggregated_metrics.pkl'
    with open(agg_metrics_file, 'wb') as f:
        pickle.dump(agg_metrics, f)

    best_layer_metrics_file = f'{results_dir}/toxic_chat-{model_name}-{control_method}-prompt_{prompt_version}-tuning_metric_{tuning_metric}-top_k_{n_components}-best_layer_metrics.pkl'
    with open(best_layer_metrics_file, 'wb') as f:
        pickle.dump(best_layer_metrics, f)

    print("Final metrics of aggregated predictions bagged over folds:")
    print(f"AUC score: {agg_metrics['auc']:.4f}")
    print(f"F1 score: {agg_metrics['f1']:.4f}")
    print(f"Precision: {agg_metrics['precision']:.4f}")
    print(f"Recall: {agg_metrics['recall']:.4f}")
    print(f"Accuracy: {agg_metrics['acc']:.4f}")
    print(f"MSE: {agg_metrics['mse']:.4f}")

    print("Final metrics of best layer predictions bagged over folds:")
    print(f"AUC score: {best_layer_metrics['auc']:.4f}")
    print(f"F1 score: {best_layer_metrics['f1']:.4f}")
    print(f"Precision: {best_layer_metrics['precision']:.4f}")
    print(f"Recall: {best_layer_metrics['recall']:.4f}")
    print(f"Accuracy: {best_layer_metrics['acc']:.4f}")
    print(f"MSE: {best_layer_metrics['mse']:.4f}")

if __name__ == '__main__':              
    main()