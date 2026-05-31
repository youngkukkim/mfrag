import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    
    root_path = os.path.abspath(os.path.dirname(__file__))
    preprocessed_data_path = os.path.join(root_path, "data")
    result_path = os.path.join(root_path, "result_mols")
    
    os.makedirs(preprocessed_data_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    
                                        
    parser.add_argument("--input_raw_data", type=str)
    parser.add_argument("--target", type=str)
    parser.add_argument("--off_target_list", type=str, nargs='+')
    parser.add_argument("--label_name", type=str, default="label")
    parser.add_argument("--smiles_col", type=str, default="Canonical_SMILES")
    
                                        
    parser.add_argument("--labeled_data", type=str, default=os.path.join(preprocessed_data_path, "labeled_data.csv"))
    parser.add_argument("--label_type", type=str, choices=['regression','binary'])
    parser.add_argument("--label_threshold", type=float, default=-99999999)
    parser.add_argument("--frag_data", type=str, default=os.path.join(preprocessed_data_path, "zinc250k_frag.pt"))
    parser.add_argument('-v', "--vocab", type=str)
    
                      
    parser.add_argument("-g", "--gpu_id", type=int, default=-1)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size_mfrag", type=int, default=2048) 
    parser.add_argument("--save_epoch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=1e-1)
    parser.add_argument("--beta", type=float, default=1e-1)
    parser.add_argument("--gamma", type=float, default=0)
    parser.add_argument("--delta", type=float, default=1e-2)
    parser.add_argument("--output_model", type=str)
    
                                             
    parser.add_argument('--num_mols', type=int, default=3000)
    parser.add_argument('-m', '--input_model', type=str)

    parser.add_argument('--batch_size_rl', type=int, default=256)     
    parser.add_argument('--start_steps', type=int, default=40)
    parser.add_argument('--update_after', type=int, default=30)
    parser.add_argument('--update_every', type=int, default=256)
    
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--num_layer', type=int, default=3)
    
    parser.add_argument('--tau', type=float, default=1e-1)
    parser.add_argument('--target_entropy', type=float, default=1.)
    parser.add_argument('--init_alpha', type=float, default=1.)
    parser.add_argument('--init_pi_lr', type=float, default=1e-4)
    parser.add_argument('--init_q_lr', type=float, default=1e-4)
    parser.add_argument('--init_alpha_lr', type=float, default=5e-4)
    parser.add_argument('--alpha_max', type=float, default=20.)
    parser.add_argument('--alpha_min', type=float, default=.05)

    parser.add_argument('--population_size', type=int, default=100)
    parser.add_argument('--mutation_rate', type=float, default=0.1)

    parser.add_argument('--max_vocab_update', type=int, default=50)
    parser.add_argument('--max_vocab_size', type=int, default=1000)
    parser.add_argument('--region_score_mode', type=str, default='knn',
                        choices=['knn', 'min_distance'])

    parser.add_argument('--output_mols', type=str, default=os.path.join(result_path, "output_mols.csv"))
    
    return parser.parse_args()
