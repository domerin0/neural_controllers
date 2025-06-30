import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from neural_controllers import NeuralController
from utils import load_model
import re
import json
import torch
import numpy as np
import random
random.seed(0)

import pickle
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup

NEURAL_CONTROLLERS_DIR = os.environ['NEURAL_CONTROLLERS_DIR']
_TAGS = ["entity", "relation", "sentence", "invented", "subjective", "unverifiable"]
TEMPLATES = {
    'empty' : "{statement}",
    'v1' : ("Consider hallucinations of the following types:\n"
            "(1a) Entity : Contradictory entity errors are a sub-category within Type 1, "
            "where an entity in a statement is incorrect and changing that single entity "
            "can make the entire sentence factually correct.\n"
            
            "(1b) Relation : Contradictory relation errors are another sub-category within "
            "contradictory statements where a semantic relationship (e.g., verbs, prepositions, "
            "or adjectives) in a statement is incorrect.\n"
            
            "(1c) Sentence : Contradictory sentence errors refer to cases where a full statement "
            "entirely contradicts relevant evidence from the web, and cannot be solved via "
            "phrase-level edits.\n"
            
            "(2) Invented : Invented errors refer to statements where the LM generates an entirely "
            "fabricated entity that doesn't exist based on world knowledge. Fictional entities in "
            "creative work aren't included.\n"
            
            "(3) Subjective : Subjective errors refer to expressions about existing entities that "
            "lack universal validity. These statements often do not contain facts and are influenced "
            "by personal beliefs or opinions.\n"
            
            "(4) Unverifiable : These are statements where the LM output contains facts, but no "
            "retrieved.\n\n"

            'Based on the above definition, does the following statement contain a hallucination? '
            'State yes or no.\n\nStatement: {statement}'
        )
}

def remove_deleted_text(text):
    # Use regex to match text between <delete> and </delete> tags
    regex = r'<delete>.*?</delete>'
    
    # Replace all matches with empty string
    # re.DOTALL flag (s) allows matching across multiple lines
    return re.sub(regex, '', text, flags=re.DOTALL)

def remove_empty_tags(html_content):
    # Pattern to match empty tags with optional whitespace between them
    pattern = r'<(\w+)>\s*</\1>'
    
    # Keep removing empty tags until no more changes are made
    prev_content = None
    current_content = html_content
    
    while prev_content != current_content:
        prev_content = current_content
        current_content = re.sub(pattern, '', current_content)
    
    return current_content

def modify(s):
    s = remove_deleted_text(s)
    s = remove_empty_tags(s)

    indicator = [0, 0, 0, 0, 0, 0]
    soup = BeautifulSoup(s, "html.parser")
    s1 = ""
    for t in range(len(_TAGS)):
        indicator[t] = len(soup.find_all(_TAGS[t]))
    # print(soup.find_all(text=True))
    for elem in soup.find_all(text=True):
        if elem.parent.name != "delete":
            s1 += elem
    return s1, int(sum(indicator)>0)

def get_fava_annotated_data(prompt_version='v1'):
    # Specify the path to your JSON file
    file_path = f'{NEURAL_CONTROLLERS_DIR}/data/hallucinations/fava/annotations.json'

    # Open and read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

    template = TEMPLATES[prompt_version]
    inputs = []
    labels = []
    for d in data:
        s = d['annotated']
        i, label = modify(s)
        inputs.append(template.format(statement=i))
        labels.append(label)
    return inputs, labels

def get_fava_training_data(tokenizer, max_n=10000, prompt_version='v1'):
    # Load training data from FAVA dataset
    from datasets import load_dataset
    ds = load_dataset("fava-uw/fava-data")
    completions = ds['train']['completion']

    # Try to load saved shuffle indices
    shuffle_indices_path = f'{NEURAL_CONTROLLERS_DIR}/results/fava_annotated_results/fava_shuffle_indices.pkl'
    try:
        with open(shuffle_indices_path, 'rb') as f:
            indices = pickle.load(f)
            completions = [completions[i] for i in indices]
    except:
        # If loading fails, create new shuffle indices
        random.seed(0)
        indices = list(range(len(completions)))
        
        random.shuffle(indices)
        completions = [completions[i] for i in indices]
        
        # Save the shuffle indices
        os.makedirs(os.path.dirname(shuffle_indices_path), exist_ok=True)
        with open(shuffle_indices_path, 'wb') as f:
            pickle.dump(indices, f)

    template = TEMPLATES[prompt_version]
    inputs = []
    labels = []
    for d in completions:
        i, label = modify(d)
        chat = [{'role':'user','content':template.format(statement=i)}]
        inputs.append(tokenizer.apply_chat_template(chat, tokenize=False))
        labels.append(label)
    
    return inputs[:max_n], labels[:max_n]

def get_splits(k_folds, n_total):
    """
    Creates k-fold cross validation splits and saves/loads them from disk.
    """
    results_dir = f'{NEURAL_CONTROLLERS_DIR}/results/fava_annotated_results'
    os.makedirs(results_dir, exist_ok=True)
    out_name = f'{results_dir}/kfold_splits_k_{k_folds}_ntotal_{n_total}.pkl'
    try:
        with open(out_name, 'rb') as f:
            splits = pickle.load(f)
            return splits
    except:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=0)
        indices = np.arange(n_total)
        splits = []
        for fold, (val_idx, test_idx) in enumerate(kf.split(indices)):
            splits.append({
                'val_indices': val_idx,
                'test_indices': test_idx,
                'fold': fold
            })
        with open(out_name, 'wb') as f:
            pickle.dump(splits, f)
        return splits

def split_test_states_on_idx(inputs, split):
    val_inputs, test_inputs = {}, {}
    for layer_idx, layer_states in inputs.items():
        val_inputs[layer_idx] = layer_states[split['val_indices']]
        test_inputs[layer_idx] = layer_states[split['test_indices']]
    return val_inputs, test_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control_method', type=str, default='rfm')
    parser.add_argument('--model_name', type=str, default='llama_3.3_70b_4bit_it')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--n_components', type=int, default=3)
    parser.add_argument('--prompt_version', type=str, default='v1')
    parser.add_argument('--tuning_metric', type=str, default='top_agop_vectors_ols_auc')
    args = parser.parse_args()
    for n_, v_ in args.__dict__.items():
        print(f"{n_:<20} : {v_}")
    
    control_method = args.control_method
    model_name = args.model_name
    n_folds = args.n_folds
    n_components = args.n_components
    prompt_version = args.prompt_version
    tuning_metric = args.tuning_metric

    if control_method not in ['rfm']:
        n_components = 1
        tuning_metric = 'auc'                        
        
    print("Num components:", n_components)
    print("Tuning metric:", tuning_metric)
                        
    language_model, tokenizer = load_model(model=model_name)
    # Get annotated data for val/test
    unformatted_inputs, labels = get_fava_annotated_data(prompt_version)
    inputs = []
    for unformatted_input in unformatted_inputs:
        chat = [{'role':'user','content':unformatted_input}]
        inputs.append(tokenizer.apply_chat_template(chat, tokenize=False))

    print("="*100)
    print(inputs[0])
    print("="*100)

    controller = NeuralController(
        language_model,
        tokenizer,
        control_method=control_method,
        batch_size=1,
        rfm_iters=5,
        n_components=n_components,
    )
    hidden_states_path = os.path.join(f'{NEURAL_CONTROLLERS_DIR}', f'hidden_states', 
                                      f'fava_annotated_{model_name}_prompt_{prompt_version}.pth')
    if os.path.exists(hidden_states_path):
        with open(hidden_states_path, 'rb') as f:
            hidden_states = pickle.load(f)
    else:
        from direction_utils import get_hidden_states
        hidden_states = get_hidden_states(inputs, language_model, tokenizer, 
                                      controller.hidden_layers, 
                                      controller.hyperparams['forward_batch_size'])
    
        with open(hidden_states_path, 'wb') as f:
            pickle.dump(hidden_states, f)

    train_inputs, train_labels = get_fava_training_data(tokenizer, prompt_version=prompt_version)

    train_hidden_states_path = os.path.join(f'{NEURAL_CONTROLLERS_DIR}', f'hidden_states', 
                                f'fava_training_{model_name}_prompt_{prompt_version}.pth')
    if os.path.exists(train_hidden_states_path):
        with open(train_hidden_states_path, 'rb') as f:
            train_hidden_states = pickle.load(f)
    else:
        from direction_utils import get_hidden_states
        train_hidden_states = get_hidden_states(train_inputs, language_model, tokenizer, 
                                    controller.hidden_layers, 
                                    controller.hyperparams['forward_batch_size'])

        with open(train_hidden_states_path, 'wb') as f:
            pickle.dump(train_hidden_states, f)

    # K-Fold Cross Validation on annotated data for val/test
    all_aggregated_predictions = []
    all_best_layer_predictions = []
    all_indices = []
    splits = get_splits(n_folds, len(labels))
    for split in splits:
        fold = split['fold']
        print(f"Fold {fold+1}/{n_folds}")
        
        val_indices = split['val_indices']
        test_indices = split['test_indices']

        val_labels = [labels[i] for i in val_indices]
        test_labels = [labels[i] for i in test_indices]

        # Compute hidden states for val and test splits
        val_hidden_states = {layer_idx: layer_states[val_indices] for layer_idx, layer_states in hidden_states.items()}
        test_hidden_states = {layer_idx: layer_states[test_indices] for layer_idx, layer_states in hidden_states.items()}

        results_dir = f'{NEURAL_CONTROLLERS_DIR}/results/fava_annotated_results'
        os.makedirs(results_dir, exist_ok=True)
        
        try:
            controller.load(concept=f'fava_training_prompt_{prompt_version}_fold_{fold}_tuning_metric_{tuning_metric}_top_k_{n_components}', model_name=model_name, path=f'{NEURAL_CONTROLLERS_DIR}/directions/')
        except:
            # Create new controller and compute directions using the fixed training set
            controller = NeuralController(
                language_model,
                tokenizer,
                control_method=control_method,
                batch_size=2,
                rfm_iters=5,
                n_components=n_components
            )

            

            controller.compute_directions(train_hidden_states, train_labels, 
                                          val_hidden_states, val_labels,
                                          tuning_metric=tuning_metric)
            controller.save(concept=f'fava_training_prompt_{prompt_version}_fold_{fold}_tuning_metric_{tuning_metric}_top_k_{n_components}', model_name=model_name, path=f'{NEURAL_CONTROLLERS_DIR}/directions/')
              

        # Split val set into train and sub-val sets
        sub_train_indices, sub_val_indices = train_test_split(range(len(val_labels)), test_size=0.2, random_state=0, shuffle=True)
        sub_train_indices = torch.tensor(sub_train_indices)
        sub_val_indices = torch.tensor(sub_val_indices)

        train_sub_hidden_states = {layer_idx: layer_states[sub_train_indices] for layer_idx, layer_states in val_hidden_states.items()}
        val_sub_hidden_states = {layer_idx: layer_states[sub_val_indices] for layer_idx, layer_states in val_hidden_states.items()}

        train_sub_labels = [val_labels[i] for i in sub_train_indices]
        val_sub_labels = [val_labels[i] for i in sub_val_indices]
        
        _, _, _, test_predictions = controller.evaluate_directions(
            train_sub_hidden_states, train_sub_labels,
            val_sub_hidden_states, val_sub_labels,
            test_hidden_states, test_labels,
            n_components=n_components,
            agg_model=control_method,
        )

        all_aggregated_predictions.append(test_predictions['aggregation'])
        all_best_layer_predictions.append(test_predictions['best_layer'])
        all_indices.append(torch.from_numpy(test_indices))
        
    all_aggregated_predictions = torch.cat(all_aggregated_predictions)
    all_best_layer_predictions = torch.cat(all_best_layer_predictions)
    all_indices = torch.cat(all_indices)

    # Sort predictions according to index order
    sorted_order = torch.argsort(all_indices)
    all_best_layer_predictions = all_best_layer_predictions[sorted_order]
    all_aggregated_predictions = all_aggregated_predictions[sorted_order]

    # Compute and save AUC score
    from direction_utils import compute_prediction_metrics
    labels = torch.tensor(labels)
    agg_metrics = compute_prediction_metrics(all_aggregated_predictions, labels)
    best_layer_metrics = compute_prediction_metrics(all_best_layer_predictions, labels)

    agg_metrics_file = f'{results_dir}/fava-{model_name}-{control_method}-prompt_{prompt_version}-tuning_metric_{tuning_metric}-top_k_{n_components}-aggregated_metrics.pkl'
    with open(agg_metrics_file, 'wb') as f:
        pickle.dump(agg_metrics, f)

    best_layer_metrics_file = f'{results_dir}/fava-{model_name}-{control_method}-prompt_{prompt_version}-tuning_metric_{tuning_metric}-top_k_{n_components}-best_layer_metrics.pkl'
    with open(best_layer_metrics_file, 'wb') as f:
        pickle.dump(best_layer_metrics, f)

    # Save predictions
    predictions_file = f'{results_dir}/fava-{model_name}-{control_method}-prompt_{prompt_version}-tuning_metric_{tuning_metric}-top_k_{n_components}-predictions.pkl'
    with open(predictions_file, 'wb') as f:
        pickle.dump({
            'aggregation': all_aggregated_predictions.cpu(),
            'best_layer': all_best_layer_predictions.cpu(),
        }, f)
    
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