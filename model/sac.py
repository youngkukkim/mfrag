import time
import csv
import json
import os
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
import numpy as np
import pandas as pd
from rdkit import Chem

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, lr_scheduler

from torch_geometric.data import Batch

from model.ac import GCNActorCritic
from utils_sac.utils import get_att_points, delete_multiple_element
from utils_ga.ga import reproduce
from utils_mfrag.utils import get_sanitize_error_frags
from utils_mfrag.data import get_graph, get_graph_from_frag


DOCKING_TARGETS = ['parp1', 'fa7', '5ht1b', 'braf', 'jak2']


class ReplayBuffer:
    def __init__(self, size):
        self.obs_buf = []                                      
        self.obs2_buf = []                                      
        self.act_buf = np.zeros((size, 3), dtype=np.int32)      
        self.rew_buf = np.zeros(size, dtype=np.float32)        
        self.done_buf = np.zeros(size, dtype=np.float32)       
        
        self.ac_prob_buf = []
        self.log_ac_prob_buf = []
        
        self.ac_first_buf = []
        self.ac_second_buf = []
        self.ac_third_buf = []
        self.o_embeds_buf = []
        
        self.ptr, self.size, self.max_size = 0, 0, size
        self.done_location = []

    def store(self, obs, act, rew, next_obs, done, ac_prob, log_ac_prob,
              ac_first_prob, ac_second_hot, ac_third_prob, o_embeds):
        if self.size == self.max_size:
            self.obs_buf.pop(0)
            self.obs2_buf.pop(0)
            
            self.ac_prob_buf.pop(0)
            self.log_ac_prob_buf.pop(0)
            
            self.ac_first_buf.pop(0)
            self.ac_second_buf.pop(0)
            self.ac_third_buf.pop(0)

            self.o_embeds_buf.pop(0)

        self.obs_buf.append(obs)
        self.obs2_buf.append(next_obs)
        
        self.ac_prob_buf.append(ac_prob)
        self.log_ac_prob_buf.append(log_ac_prob)
        
        self.ac_first_buf.append(ac_first_prob)
        self.ac_second_buf.append(ac_second_hot)
        self.ac_third_buf.append(ac_third_prob)
        self.o_embeds_buf.append(o_embeds)

        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        if done:
            self.done_location.append(self.ptr)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def rew_store(self, rew):
        done_location_np = np.array(self.done_location)
        zeros = np.where(rew == 0.0)[0]
        nonzeros = np.where(rew != 0.0)[0]
        zero_ptrs = done_location_np[zeros]

        done_location_np = done_location_np[nonzeros]
        rew = rew[nonzeros]

        if len(self.done_location) > 0:
            self.rew_buf[done_location_np] = self.rew_buf[done_location_np] + rew
            self.done_location = []

        self.act_buf = np.delete(self.act_buf, zero_ptrs, axis=0)
        self.rew_buf = np.delete(self.rew_buf, zero_ptrs)
        self.done_buf = np.delete(self.done_buf, zero_ptrs)
        delete_multiple_element(self.obs_buf, zero_ptrs.tolist())
        delete_multiple_element(self.obs2_buf, zero_ptrs.tolist())

        delete_multiple_element(self.ac_prob_buf, zero_ptrs.tolist())
        delete_multiple_element(self.log_ac_prob_buf, zero_ptrs.tolist())
        
        delete_multiple_element(self.ac_first_buf, zero_ptrs.tolist())
        delete_multiple_element(self.ac_second_buf, zero_ptrs.tolist())
        delete_multiple_element(self.ac_third_buf, zero_ptrs.tolist())

        delete_multiple_element(self.o_embeds_buf, zero_ptrs.tolist())

        self.size = min(self.size - len(zero_ptrs), self.max_size)
        self.ptr = (self.ptr - len(zero_ptrs)) % self.max_size
        
    def sample_batch(self, device, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        obs_batch = [self.obs_buf[idx] for idx in idxs]
        obs2_batch = [self.obs2_buf[idx] for idx in idxs]

        ac_prob_batch = [self.ac_prob_buf[idx] for idx in idxs]
        log_ac_prob_batch = [self.log_ac_prob_buf[idx] for idx in idxs]
        
        ac_first_batch = torch.stack([self.ac_first_buf[idx].to(device) for idx in idxs]).squeeze(1)
        ac_second_batch = torch.stack([self.ac_second_buf[idx].to(device) for idx in idxs]).squeeze(1)
        ac_third_batch = torch.stack([self.ac_third_buf[idx].to(device) for idx in idxs]).squeeze(1)
        o_g_emb_batch = torch.stack([self.o_embeds_buf[idx][2] for idx in idxs]).squeeze(1)

        act_batch = torch.as_tensor(self.act_buf[idxs], dtype=torch.float32).unsqueeze(-1).to(device)
        rew_batch = torch.as_tensor(self.rew_buf[idxs], dtype=torch.float32).to(device)
        done_batch = torch.as_tensor(self.done_buf[idxs], dtype=torch.float32).to(device)

        batch = dict(obs=obs_batch,
                     obs2=obs2_batch,
                     act=act_batch,
                     rew=rew_batch,
                     done=done_batch,
                     ac_prob=ac_prob_batch,
                     log_ac_prob=log_ac_prob_batch,
                     ac_first=ac_first_batch,
                     ac_second=ac_second_batch,
                     ac_third=ac_third_batch,
                     o_g_emb=o_g_emb_batch)
        return batch


def xavier_uniform_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)


class SAC:
    def __init__(self, args, vocab, env_fn,
                 replay_size=int(1e6), gamma=0.99, polyak=0.995, train_alpha=True):
        super().__init__()
        
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        self.device = args.device
        self.num_mols = args.num_mols
        self.gamma = gamma
        self.polyak = polyak
        self.args = args
        
        tm = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        desc_suffix = self._build_desc_suffix(args)
        self.run_id = f'{tm}_{args.target}_{args.seed}_{desc_suffix}'
        self.fname = f'results/{self.run_id}.csv'
        self.attempt_fname = f'results/{self.run_id}_attempts.csv'
        print(f'\033[92m{self.fname}\033[0m')
        print(f'\033[92m{self.attempt_fname}\033[0m')
        self._init_attempt_log()
        
        self.batch_size = args.batch_size
        self.start_steps = args.start_steps
        self.update_after = args.update_after
        self.update_every = args.update_every
        self.docking_every = int(args.update_every / 2)
        self.train_alpha = train_alpha

        self.env = env_fn
        self.vocab = vocab

        self.obs_dim = args.emb_size * 2
        self.action_dims = [40, len(vocab['FRAG']), 40]
        
        self.target_entropy = args.target_entropy

        self.log_alpha = torch.tensor([np.log(args.init_alpha)], requires_grad=train_alpha) 

        self.ac = GCNActorCritic(self.env, args, vocab).to(args.device)
        self.ac_targ = deepcopy(self.ac).to(args.device).eval()

                                                                                              
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        for q in self.ac.parameters():
            q.requires_grad = True

        self.replay_buffer = ReplayBuffer(size=replay_size)

        pi_lr = args.init_pi_lr
        q_lr = args.init_q_lr
        alpha_lr = args.init_alpha_lr
    
        self.pi_params = list(self.ac.pi.parameters()) 
        self.q_params = list(self.ac.q1.parameters()) + list(self.ac.q2.parameters()) + list(self.ac.embed.parameters())
        
        self.pi_optimizer = Adam(self.pi_params, lr=pi_lr, weight_decay=1e-4)
        self.q_optimizer = Adam(self.q_params, lr=q_lr, weight_decay=1e-4)
        self.alpha_optimizer = Adam([self.log_alpha], lr=alpha_lr, eps=1e-4)

        self.q_scheduler = lr_scheduler.ReduceLROnPlateau(self.q_optimizer, factor=0.1, patience=768) 
        self.pi_scheduler = lr_scheduler.ReduceLROnPlateau(self.pi_optimizer, factor=0.1, patience=768)        

        self.alpha_start = self.start_steps
        self.alpha_end = self.start_steps + 30000
        self.alpha_max = args.alpha_max
        self.alpha_min = args.alpha_min
        
        self.population_size = args.population_size
        self.mutation_rate = args.mutation_rate
        self.population = []
        self.population_score = []
        self.ga_smiles_list = []
        
        self.max_vocab_update = args.max_vocab_update
        self.max_vocab_size = args.max_vocab_size

        self.t = 0
        self.ac.apply(xavier_uniform_init)
        
        from model.mfrag import MFRAG
        ckpt = self._resolve_mfrag_ckpt(args)
        self.mfrag_ckpt_path = ckpt
        ckpt_obj = torch.load(ckpt, map_location=self.device)
        state_dict = ckpt_obj['state_dict'] if isinstance(ckpt_obj, dict) and 'state_dict' in ckpt_obj else ckpt_obj
        mfrag_arch = self._resolve_mfrag_arch(args, state_dict, ckpt_obj)
        self.mfrag = MFRAG(device=self.device, model_arch=mfrag_arch).to(self.device)
        print(f'[MFRAG] loading ckpt={ckpt} arch={mfrag_arch}')
        self.mfrag.load_state_dict(state_dict)
        self.mfrag.eval()
        self.ac.set_mfrag(self.mfrag)
        self.ac_targ.set_mfrag(self.mfrag)
        self.region_generated_bank = {}
        self.mfrag_finetune_count = 0
        self.mfrag_finetune_last_bank_size = 0
        self._mfrag_finetune_zinc_dataset = None
        self._mfrag_finetune_zinc_scores = None
        if not getattr(self.args, 'disable_region_guidance', False):
            self.region_train_embeddings, self.region_train_scores, self.region_train_high_mask = self._build_region_train_reference()
            self._refresh_region_reference()
        else:
            self.region_train_embeddings = torch.zeros((0, 128), dtype=torch.float32)
            self.region_train_scores = torch.zeros((0,), dtype=torch.float32)
            self.region_train_high_mask = torch.zeros((0,), dtype=torch.bool)
            self.region_train_smiles = []

        if getattr(args, 'enable_constrained_mfrag_guidance', False):
            self._configure_constraint_guidance()

    def _configure_constraint_guidance(self):
        from utils_mfrag.benchmark import load_benchmark_reference, load_scalar_mfrag

        for name in ('qed', 'sa'):
            checkpoint = getattr(self.args, name + '_mfrag_ckpt')
            reference = os.path.join(self.args.constraint_embedding_dir, name + '.npz')
            embeddings, scores = load_benchmark_reference(reference, checkpoint, name)
            model = load_scalar_mfrag(checkpoint, self.device, self.args.mfrag_model_arch)
            self.ac.pi.set_constraint_region_guide(
                name, model, embeddings, scores,
                getattr(self.args, name + '_region_cutoff'), self.args.constraint_knn_k)
            print('[MFRAG benchmark] {} references={} cutoff={}'.format(
                name, len(scores), getattr(self.args, name + '_region_cutoff')))

    def _ga_parent_mask(self, qed_scores, sa_scores):
        qed_scores = np.asarray(qed_scores, dtype=np.float32)
        sa_scores = np.asarray(sa_scores, dtype=np.float32)
        if qed_scores.shape != sa_scores.shape or qed_scores.ndim != 1:
            raise ValueError('GA QED and SA arrays must have matching one-dimensional shapes')
        if not getattr(self.args, 'ga_qed_sa_gate', False):
            return np.ones(qed_scores.shape, dtype=bool)
        return (np.isfinite(qed_scores) & np.isfinite(sa_scores)
                & (qed_scores > self.args.ga_qed_threshold)
                & (sa_scores > self.args.ga_sa_threshold))

    def _update_ga_population(self, mols, learning_scores, qed_scores, sa_scores):
        parent_mask = self._ga_parent_mask(qed_scores, sa_scores)
        if len(mols) != len(parent_mask) or len(mols) != len(learning_scores):
            raise ValueError('GA population inputs must have equal lengths')
        if getattr(self.args, 'ga_qed_sa_gate', False):
            parent_mask &= np.asarray([mol is not None for mol in mols], dtype=bool)
        self.population.extend(mol for mol, keep in zip(mols, parent_mask) if keep)
        self.population_score.extend(score for score, keep in zip(learning_scores, parent_mask) if keep)
        ranked = sorted(zip(self.population, self.population_score),
                        key=lambda pair: pair[1], reverse=True)[:self.population_size]
        self.population = [pair[0] for pair in ranked]
        self.population_score = [pair[1] for pair in ranked]

    def _resolve_mfrag_ckpt(self, args):
        explicit_ckpt = getattr(args, 'mfrag_ckpt', '')
        if explicit_ckpt:
            return explicit_ckpt
        return os.path.join(
            getattr(args, 'mfrag_root', 'ckpt'),
            getattr(args, 'mfrag_label_mode', 'reg'),
            args.target,
            getattr(args, 'mfrag_ckpt_name', 'best.pt'),
        )

    def _resolve_mfrag_arch(self, args, state_dict, ckpt_obj):
        requested_arch = getattr(args, 'mfrag_model_arch', 'auto')
        if requested_arch != 'auto':
            return requested_arch

        if any(k.startswith('mol_encoder.') or k.startswith('frag_encoder.') for k in state_dict):
            return 'dual'
        if any(k.startswith('gather.') or k.startswith('embed.') for k in state_dict):
            return 'shared'

        ckpt_args = ckpt_obj.get('args') if isinstance(ckpt_obj, dict) else None
        ckpt_arch = getattr(ckpt_args, 'model_arch', None)
        return ckpt_arch if ckpt_arch in ('shared', 'dual') else 'shared'

    def _build_desc_suffix(self, args):
        mode = getattr(args, 'frag_desc_mode', 'ecfp')
        dim = 1024 if mode == 'raw_ecfp' else getattr(args, 'frag_desc_dim', 128)
        ew = getattr(args, 'ecfp_weight', 1.0)
        sw = getattr(args, 'sp_weight', 1.0)
        reward_suffix = 'target'
        suffix = f'{mode}_d{dim}_ew{ew:g}_sw{sw:g}_rw{reward_suffix}'
        if getattr(args, 'ga_qed_sa_gate', False):
            suffix += '_benchmark'
        selection = getattr(args, 'fragment_selection_mode', 'hybrid')
        noise = getattr(args, 'gumbel_noise_scale', 1e-3)
        if selection != 'hybrid' or noise != 1e-3:
            suffix += f'_{selection}_g{noise:g}'
        return suffix

    def _canonicalize_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=False)

    def _target_to_learning_score(self, scores):
        scores = np.asarray(scores, dtype=np.float32)
        if self.args.target in DOCKING_TARGETS:
            return np.clip(scores, 0.0, 20.0) / 20.0
        return np.clip(scores, 0.0, 1.0)

    def _embed_region_smiles(self, smiles_list):
        if len(smiles_list) == 0:
            return torch.zeros((0, 128), dtype=torch.float32)

        graphs = [get_graph_from_frag(smiles) for smiles in smiles_list]
        embeddings = []
        batch_size = getattr(self.args, 'batch_size', 256)
        for start in range(0, len(graphs), batch_size):
            batch = Batch.from_data_list(graphs[start:start + batch_size]).to(self.device)
            with torch.no_grad():
                _, batch_embedding = self.mfrag(batch)
            embeddings.append(batch_embedding.detach().cpu())
        return torch.cat(embeddings, dim=0)

    def _build_region_train_reference(self):
        sample_size = getattr(self.args, 'region_train_sample_size', 1000)
        high_score_quantile = getattr(self.args, 'region_high_score_quantile', 0.95)

        train_df = pd.read_csv('data/zinc250k.csv')
        with open('data/valid_idx_zinc250k.json') as f:
            test_idx = set(json.load(f))
        train_idx = [i for i in range(len(train_df)) if i not in test_idx]
        train_df = train_df.iloc[train_idx].reset_index(drop=True)
        train_df = train_df[np.abs(train_df[self.args.target].to_numpy(dtype=np.float32)) > 1e-8].reset_index(drop=True)
        train_df['CANONICAL_SMILES'] = train_df['smiles'].apply(self._canonicalize_smiles)
        train_df = train_df.dropna(subset=['CANONICAL_SMILES']).drop_duplicates(subset=['CANONICAL_SMILES']).reset_index(drop=True)

        train_scores = self._target_to_learning_score(train_df[self.args.target].to_numpy(dtype=np.float32))
        high_thr = np.quantile(train_scores, high_score_quantile)
        high_idx = np.where(train_scores >= high_thr)[0]
        low_idx = np.where(train_scores < high_thr)[0]
        if sample_size is None or sample_size <= 0 or sample_size >= len(train_df):
            selected_idx = np.arange(len(train_df))
        else:
            rng = np.random.default_rng(self.args.seed)
            remaining_take = max(sample_size - len(high_idx), 0)
            selected_high = high_idx
            low_take = min(len(low_idx), remaining_take)
            selected_low = rng.choice(low_idx, size=low_take, replace=False) if low_take > 0 else np.array([], dtype=int)
            selected_idx = np.concatenate([selected_low, selected_high])

        selected_smiles = train_df.iloc[selected_idx]['CANONICAL_SMILES'].tolist()
        selected_scores = train_scores[selected_idx]
        selected_embeddings = self._embed_region_smiles(selected_smiles)
        selected_high_mask = torch.as_tensor(selected_scores >= high_thr, dtype=torch.bool)
        self.region_train_smiles = selected_smiles
        return selected_embeddings, torch.as_tensor(selected_scores, dtype=torch.float32), selected_high_mask

    def _refresh_region_reference(self):
        if getattr(self.args, 'disable_region_guidance', False):
            return
        if len(self.region_generated_bank) == 0:
            ref_embeddings = self.region_train_embeddings
            ref_scores = self.region_train_scores
            ref_high_mask = self.region_train_high_mask
        else:
            items = sorted(
                self.region_generated_bank.items(),
                key=lambda x: x[1]['score'],
                reverse=True,
            )[:getattr(self.args, 'region_generated_max', 5000)]
            gen_embeddings = torch.stack([item[1]['embedding'] for item in items], dim=0)
            gen_scores = torch.tensor([item[1]['score'] for item in items], dtype=torch.float32)
            gen_high_mask = torch.tensor([bool(item[1].get('is_high', False)) for item in items], dtype=torch.bool)
            ref_embeddings = torch.cat([self.region_train_embeddings, gen_embeddings], dim=0)
            ref_scores = torch.cat([self.region_train_scores, gen_scores], dim=0)
            ref_high_mask = torch.cat([self.region_train_high_mask, gen_high_mask], dim=0)

        knn_k = getattr(self.args, 'region_knn_k', 40)
        score_mode = getattr(self.args, 'region_score_mode', 'knn')
        self.ac.pi.set_region_reference(ref_embeddings, ref_scores, knn_k, ref_high_mask, score_mode)
        self.ac_targ.pi.set_region_reference(ref_embeddings, ref_scores, knn_k, ref_high_mask, score_mode)

    def _update_region_generated_bank(self, smiles_list, docking_scores):
        if getattr(self.args, 'disable_region_guidance', False) and not getattr(self.args, 'enable_mfrag_finetune', False):
            return
        if len(smiles_list) == 0:
            return

        batch_best = {}
        for smiles, score in zip(smiles_list, docking_scores):
            canonical = self._canonicalize_smiles(smiles)
            if canonical is None:
                continue
            score = float(score)
            if canonical not in batch_best or score > batch_best[canonical]:
                batch_best[canonical] = score

        if len(batch_best) == 0:
            return

        canonical_smiles = list(batch_best.keys())
        embeddings = self._embed_region_smiles(canonical_smiles)
        high_thr = float(np.quantile(self.region_train_scores.cpu().numpy(), getattr(self.args, 'region_high_score_quantile', 0.95)))\
            if len(self.region_train_scores) > 0 else float('inf')
        for idx, canonical in enumerate(canonical_smiles):
            score = batch_best[canonical]
            existing = self.region_generated_bank.get(canonical)
            if existing is None or score > existing['score']:
                self.region_generated_bank[canonical] = {
                    'score': score,
                    'embedding': embeddings[idx].clone(),
                    'is_high': score >= high_thr,
                }

        max_generated = getattr(self.args, 'region_generated_max', 5000)
        if len(self.region_generated_bank) > max_generated:
            high_items = [(k, v) for k, v in self.region_generated_bank.items() if v.get('is_high', False)]
            low_items = [(k, v) for k, v in self.region_generated_bank.items() if not v.get('is_high', False)]
            low_items = sorted(low_items, key=lambda x: x[1]['score'], reverse=True)
            keep_low = max(max_generated - len(high_items), 0)
            items = high_items + low_items[:keep_low]
            self.region_generated_bank = dict(items)

        self._refresh_region_reference()

    def _mfrag_finetune_dir(self):
        return os.path.join('ckpt', 'runtime_mfrag', self.args.target, self.run_id)

    def _load_mfrag_finetune_zinc_dataset(self):
        if self._mfrag_finetune_zinc_dataset is not None:
            return self._mfrag_finetune_zinc_dataset, self._mfrag_finetune_zinc_scores

        from train_mfrag import DockingDataset

        print('[MFRAG fine-tune] loading zinc anchor data=data/zinc250k_frag.pt', flush=True)
        train_frag, _ = torch.load('data/zinc250k_frag.pt', map_location='cpu')
        dataset = DockingDataset(train_frag, self.args.target)
        self._mfrag_finetune_zinc_dataset = dataset
        self._mfrag_finetune_zinc_scores = dataset.reg_target_values.astype(np.float32)
        print(
            f'[MFRAG fine-tune] zinc anchor loaded n={len(dataset)}',
            flush=True,
        )
        return dataset, self._mfrag_finetune_zinc_scores

    def _make_generated_mfrag_sample(self, smiles, score):
        try:
            frag_source = get_graph(smiles)
            if frag_source is None or not hasattr(frag_source, 'frags') or len(frag_source.frags) == 0:
                return None
            graph = get_graph_from_frag(smiles)
            frag_list = [get_graph_from_frag(frag_smiles) for frag_smiles in frag_source.frags]
            frag_list = [frag for frag in frag_list if frag is not None]
            if graph is None or len(frag_list) == 0:
                return None
        except Exception:
            return None

        score = float(score)
        return (
            graph,
            frag_list,
            len(frag_list),
            torch.tensor(score, dtype=torch.float32),
            torch.tensor(float(score > 0.5), dtype=torch.float32),
        )

    def _take_random_indices(self, candidates, n_take, rng):
        if n_take <= 0 or len(candidates) == 0:
            return []
        n_take = min(int(n_take), len(candidates))
        chosen = rng.choice(len(candidates), size=n_take, replace=False)
        return [candidates[int(i)] for i in chosen]

    def _build_mfrag_finetune_samples(self):
        generated_items = list(self.region_generated_bank.items())
        generated_items = [(smiles, item['score']) for smiles, item in generated_items]
        generated_items = [(smiles, score) for smiles, score in generated_items if np.isfinite(score)]
        if len(generated_items) == 0:
            return [], {}

        rng = np.random.default_rng(self.args.seed + self.mfrag_finetune_count + len(generated_items))
        sample_base = min(
            len(generated_items),
            int(getattr(self.args, 'mfrag_finetune_max_generated', 5000)),
        )
        frac_gen_top = float(getattr(self.args, 'mfrag_finetune_generated_top_frac', 0.4))
        frac_gen_random = float(getattr(self.args, 'mfrag_finetune_generated_random_frac', 0.2))
        frac_zinc_top = float(getattr(self.args, 'mfrag_finetune_zinc_top_frac', 0.2))
        frac_zinc_random = float(getattr(self.args, 'mfrag_finetune_zinc_random_frac', 0.2))

        gen_top_n = max(int(round(sample_base * frac_gen_top)), 0)
        gen_random_n = max(int(round(sample_base * frac_gen_random)), 0)
        zinc_top_n = max(int(round(sample_base * frac_zinc_top)), 0)
        zinc_random_n = max(int(round(sample_base * frac_zinc_random)), 0)

        generated_sorted = sorted(generated_items, key=lambda x: x[1], reverse=True)
        selected_gen_top = generated_sorted[:gen_top_n]
        selected_top_smiles = {smiles for smiles, _ in selected_gen_top}
        generated_rest = [(smiles, score) for smiles, score in generated_items if smiles not in selected_top_smiles]
        selected_gen_random = self._take_random_indices(generated_rest, gen_random_n, rng)

        zinc_dataset, zinc_scores = self._load_mfrag_finetune_zinc_dataset()
        zinc_order = np.argsort(-zinc_scores)
        selected_zinc_top_idx = [int(i) for i in zinc_order[:min(zinc_top_n, len(zinc_order))]]
        selected_zinc_top_set = set(selected_zinc_top_idx)
        zinc_rest_idx = [i for i in range(len(zinc_dataset)) if i not in selected_zinc_top_set]
        selected_zinc_random_idx = self._take_random_indices(zinc_rest_idx, zinc_random_n, rng)

        generated_samples = []
        for smiles, score in selected_gen_top + selected_gen_random:
            sample = self._make_generated_mfrag_sample(smiles, score)
            if sample is not None:
                generated_samples.append(sample)

        zinc_samples = [zinc_dataset[int(i)] for i in selected_zinc_top_idx + selected_zinc_random_idx]
        samples = generated_samples + zinc_samples
        if len(samples) > 0:
            order = rng.permutation(len(samples))
            samples = [samples[int(i)] for i in order]
        composition = {
            'generated_bank': len(generated_items),
            'generated_top_requested': len(selected_gen_top),
            'generated_random_requested': len(selected_gen_random),
            'generated_usable': len(generated_samples),
            'zinc_top': len(selected_zinc_top_idx),
            'zinc_random': len(selected_zinc_random_idx),
            'total': len(samples),
        }
        return samples, composition

    def _reembed_region_reference_after_mfrag_update(self):
        if getattr(self.args, 'disable_region_guidance', False):
            return

        if getattr(self, 'region_train_smiles', None):
            self.region_train_embeddings = self._embed_region_smiles(self.region_train_smiles)

        if len(self.region_generated_bank) > 0:
            canonical_smiles = list(self.region_generated_bank.keys())
            embeddings = self._embed_region_smiles(canonical_smiles)
            for idx, smiles in enumerate(canonical_smiles):
                self.region_generated_bank[smiles]['embedding'] = embeddings[idx].clone()

        self._refresh_region_reference()

    def _run_mfrag_finetune(self, samples, composition):
        from train_mfrag import (
            build_property_loss, collate_mfrag_batch,
            mean_pool_frag_embeddings, ContinuousContrastiveLoss,
        )

        if len(samples) == 0:
            print('[MFRAG fine-tune] skip: no usable samples', flush=True)
            return None

        finetune_idx = self.mfrag_finetune_count + 1
        out_dir = self._mfrag_finetune_dir()
        os.makedirs(out_dir, exist_ok=True)

        print(
            '[MFRAG fine-tune] '
            f'idx={finetune_idx} bank={composition.get("generated_bank", 0)} '
            f'gen_top={composition.get("generated_top_requested", 0)} '
            f'gen_random={composition.get("generated_random_requested", 0)} '
            f'gen_usable={composition.get("generated_usable", 0)} '
            f'zinc_top={composition.get("zinc_top", 0)} '
            f'zinc_random={composition.get("zinc_random", 0)} '
            f'total={composition.get("total", len(samples))}',
            flush=True,
        )

        pred_loss_fn = build_property_loss(
            'reg', getattr(self.args, 'mfrag_finetune_loss', 'huber'),
            float(getattr(self.args, 'mfrag_finetune_delta', 1.0)),
        )
        ctr_loss_fn = ContinuousContrastiveLoss(temperature=0.1, label_temperature=0.1)
        optimizer = Adam(self.mfrag.parameters(), lr=float(getattr(self.args, 'mfrag_finetune_lr', 1e-4)))
        batch_size = int(getattr(self.args, 'mfrag_finetune_batch_size', 512))
        epochs = int(getattr(self.args, 'mfrag_finetune_epochs', 3))

        self.mfrag.train()
        rng = np.random.default_rng(self.args.seed + finetune_idx)
        for epoch in range(1, epochs + 1):
            order = np.arange(len(samples))
            rng.shuffle(order)
            total_loss = 0.0
            total_pred_loss = 0.0
            total_distance_loss = 0.0
            total_ctr_loss = 0.0
            n_batches = 0

            for start in range(0, len(order), batch_size):
                batch_indices = order[start:start + batch_size]
                batch_samples = [samples[int(i)] for i in batch_indices]
                graph_batch, frag_batch, frags_num, reg_target, _ = collate_mfrag_batch(batch_samples)
                if frag_batch is None:
                    continue
                graph_batch = graph_batch.to(self.device)
                frag_batch = frag_batch.to(self.device)
                frags_num = frags_num.to(self.device)
                reg_target = reg_target.to(self.device).view(-1, 1)

                optimizer.zero_grad(set_to_none=True)
                pred, graph_embedding = self.mfrag(graph_batch)
                pred_loss = pred_loss_fn(pred, reg_target)
                frag_embeddings = self.mfrag.encode_frag(frag_batch)
                pooled_frag_embeddings = mean_pool_frag_embeddings(frag_embeddings, frags_num)
                graph_embedding_target = graph_embedding.detach()
                distance_loss = F.pairwise_distance(graph_embedding_target, pooled_frag_embeddings).mean()
                ctr_embeddings = torch.cat([graph_embedding_target, pooled_frag_embeddings], dim=0)
                ctr_targets = torch.cat([reg_target.view(-1), reg_target.view(-1)], dim=0)
                ctr_loss = ctr_loss_fn(ctr_embeddings, ctr_targets)
                loss = pred_loss + 0.1 * distance_loss + 0.1 * ctr_loss
                loss.backward()
                clip_grad_norm_(self.mfrag.parameters(), 5.0)
                optimizer.step()

                total_loss += float(loss.item())
                total_pred_loss += float(pred_loss.item())
                total_distance_loss += float(distance_loss.item())
                total_ctr_loss += float(ctr_loss.item())
                n_batches += 1

            denom = max(n_batches, 1)
            print(
                '[MFRAG fine-tune] '
                f'idx={finetune_idx} epoch={epoch}/{epochs} '
                f'loss={total_loss / denom:.6f} '
                f'pred_loss={total_pred_loss / denom:.6f} '
                f'distance_loss={total_distance_loss / denom:.6f} '
                f'ctr_loss={total_ctr_loss / denom:.6f}',
                flush=True,
            )

        self.mfrag.eval()
        ckpt_path = os.path.join(
            out_dir,
            f'finetune_{finetune_idx:03d}_n{len(self.region_generated_bank)}.pt',
        )
        torch.save(
            {
                'state_dict': self.mfrag.state_dict(),
                'args': vars(self.args).copy(),
                'source_ckpt': self.mfrag_ckpt_path,
                'composition': composition,
                'finetune_idx': finetune_idx,
                'generated_bank_size': len(self.region_generated_bank),
            },
            ckpt_path,
        )
        self.mfrag.load_state_dict(torch.load(ckpt_path, map_location=self.device)['state_dict'])
        self.mfrag_ckpt_path = ckpt_path
        self.mfrag_finetune_count = finetune_idx
        self.mfrag_finetune_last_bank_size = len(self.region_generated_bank)
        self.ac.set_mfrag(self.mfrag)
        self.ac_targ.set_mfrag(self.mfrag)
        self._reembed_region_reference_after_mfrag_update()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        print(f'[MFRAG fine-tune] saved={ckpt_path}', flush=True)
        return ckpt_path

    def _maybe_mfrag_finetune(self):
        if not getattr(self.args, 'enable_mfrag_finetune', False):
            return

        bank_size = len(self.region_generated_bank)
        trigger = int(getattr(self.args, 'mfrag_finetune_trigger', 1000))
        interval = int(getattr(self.args, 'mfrag_finetune_interval', 1000))
        if bank_size < trigger:
            return
        if self.mfrag_finetune_count == 0 and self.mfrag_finetune_last_bank_size > 0:
            if bank_size < self.mfrag_finetune_last_bank_size + interval:
                return
        if self.mfrag_finetune_count > 0 and bank_size < self.mfrag_finetune_last_bank_size + interval:
            return

        samples, composition = self._build_mfrag_finetune_samples()
        ckpt_path = self._run_mfrag_finetune(samples, composition)
        if ckpt_path is None:
            self.mfrag_finetune_last_bank_size = bank_size

    def _init_attempt_log(self):
        with open(self.attempt_fname, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                't',
                'target',
                'seed',
                'attempt_result',
                'result_type',
                'failure_case',
                'terminal_trigger',
                'stop',
                'new',
                'replay_stored',
                'terminal_reset',
                'atom_count_before',
                'atom_count_after',
                'att_count_before',
                'att_count_after',
                'reward_step',
                'final_smiles',
                'sac_pending',
                'ga_pending',
                'done_pending',
                'action_first',
                'action_second',
                'action_third',
            ])

    def _log_attempt(self, info, replay_stored, terminal_reset, reward, ac):
        if info.get('stop'):
            attempt_result = 'success'
            failure_case = ''
        else:
            attempt_result = 'failure'
            if info.get('terminal_trigger') == 'pre_max_atoms':
                failure_case = 'max_atoms_terminal_without_valid_append'
            elif 'pre_max_atoms' in info.get('terminal_trigger', ''):
                failure_case = 'max_atoms_mixed_terminal_without_valid_append'
            elif info.get('terminal_trigger') == 'pre_no_attachment':
                failure_case = 'no_attachment_terminal_without_valid_append'
            elif 'pre_no_attachment' in info.get('terminal_trigger', ''):
                failure_case = 'mixed_terminal_without_valid_append'
            elif not replay_stored:
                failure_case = 'invalid_next_observation'
            else:
                failure_case = 'terminal_without_valid_append'

        with open(self.attempt_fname, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.t,
                self.args.target,
                self.args.seed,
                attempt_result,
                info.get('result_type', ''),
                failure_case,
                info.get('terminal_trigger', ''),
                info.get('stop', False),
                info.get('new', False),
                replay_stored,
                terminal_reset,
                info.get('atom_count_before', ''),
                info.get('atom_count_after', ''),
                info.get('att_count_before', ''),
                info.get('att_count_after', ''),
                reward,
                info.get('smile', ''),
                len(self.env.smile_list),
                len(self.ga_smiles_list),
                len(self.replay_buffer.done_location),
                int(ac[0]),
                int(ac[1]),
                int(ac[2]),
            ])

                     
    def update_vocab(self, mol_list, scores_list):
        smiles_list = [Chem.MolToSmiles(m) for m in mol_list]
        batch = []
        for smiles in smiles_list:
            graph = get_graph(smiles)
            if graph is not None:
                batch.append(graph)
        if not batch:
            return
        
        batch = Batch.from_data_list(batch).to(self.device)
                            
                                  
                                                                          
        
                                                                                                                                     
                                                      
                                                                                                     
                                                                     
                                                           

        
                                                                                                  
                                                                             
        new_frags_smis = sum(batch.frags,[])
        
        new_frags_smis = list(set(new_frags_smis))
        
        error_frags = get_sanitize_error_frags(new_frags_smis)
        for frag in error_frags:
            new_frags_smis.remove(frag)

        new_frags = [get_graph_from_frag(i) for i in new_frags_smis]

        new_frags = Batch.from_data_list(new_frags).to(self.device)

        
        with torch.no_grad():
            new_frags_score, new_frags_embedding = self.mfrag(new_frags)


                                                                                
        new_frag_tuples = list(zip(new_frags_smis, [i.item() for i in new_frags_score]))
        print(len(new_frag_tuples))
        
        new_frag_tuples = sorted(new_frag_tuples, key=lambda x: x[1], reverse=True)[:self.max_vocab_update]
        new_frag_tuples = [(frag, score) for frag, score in new_frag_tuples if frag not in self.vocab['FRAG']]

        frag_tuples = new_frag_tuples
        print(len(new_frag_tuples))
            
        print(len(self.vocab['FRAG']))
        self.vocab['FRAG_QUEUE'].extend(frag_tuples)
             
                                                                                            
             
        print(len(self.vocab['FRAG_QUEUE']))
        print(self.vocab['FRAG_QUEUE'][0])
        print(self.vocab['FRAG_QUEUE'][-2])
        print(self.vocab['FRAG_QUEUE'][-1])
        self.vocab['FRAG_QUEUE'] = sorted(self.vocab['FRAG_QUEUE'], key=lambda x: x[1], reverse=True)[:self.max_vocab_size]
        print(len(self.vocab['FRAG_QUEUE']))
        print(self.vocab['FRAG_QUEUE'][0])
        print(self.vocab['FRAG_QUEUE'][-2])
        print(self.vocab['FRAG_QUEUE'][-1])
        self.vocab['FRAG'] = [frag for frag, score in self.vocab['FRAG_QUEUE']]
        self.vocab['FRAG_MOL'] = [Chem.MolFromSmiles(frag) for frag in self.vocab['FRAG']]
        self.vocab['FRAG_ATT'] = [get_att_points(mol) for mol in self.vocab['FRAG_MOL']]
        
        print(len(self.vocab['FRAG']))
        self.action_dims = [40, len(self.vocab['FRAG']), 40]
        self.env.update_vocab(self.vocab)
        self.ac.pi.update_vocab(self.vocab)
        torch.cuda.empty_cache()
    
    def compute_loss_q(self, data):
        ac_first, ac_second, ac_third = data['ac_first'], data['ac_second'], data['ac_third']             
        self.ac.q1.train()
        self.ac.q2.train()
        o = data['obs']
        _, _, o_g_emb = self.ac.embed(o)
        q1 = self.ac.q1(o_g_emb, ac_first, ac_second, ac_third).squeeze()
        q2 = self.ac.q2(o_g_emb.detach(), ac_first, ac_second, ac_third).squeeze()

                                                   
        o2 = data['obs2']
        r, d = data['rew'], data['done']
        
        with torch.no_grad():
            o2_g, o2_n_emb, o2_g_emb = self.ac.embed(o2)
            cands = self.ac.embed(self.ac.pi.cand)
            a2, (a2_prob, log_a2_prob), (ac2_first, ac2_second, ac2_third) = self.ac.pi(o2_g_emb, o2_n_emb, o2_g, cands)
                             
            q1_pi_targ = self.ac_targ.q1(o2_g_emb, ac2_first, ac2_second, ac2_third)
            q2_pi_targ = self.ac_targ.q2(o2_g_emb, ac2_first, ac2_second, ac2_third)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ).squeeze() 
            backup = r + self.gamma * (1 - d) * q_pi_targ

                                         
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q

    def compute_loss_pi(self, data):
        with torch.no_grad():
            o_embeds = self.ac.embed(data['obs'])   
            o_g, o_n_emb, o_g_emb = o_embeds
            cands = self.ac.embed(self.ac.pi.cand)

        _, (ac_prob, log_ac_prob), (ac_first, ac_second, ac_third) =\
            self.ac.pi(o_g_emb, o_n_emb, o_g, cands)

        q1_pi = self.ac.q1(o_g_emb, ac_first, ac_second, ac_third)
        q2_pi = self.ac.q2(o_g_emb, ac_first, ac_second, ac_third)
        q_pi = torch.min(q1_pi, q2_pi)

        ac_prob_sp = torch.split(ac_prob, self.action_dims, dim=1)
        log_ac_prob_sp = torch.split(log_ac_prob, self.action_dims, dim=1)
        
        loss_policy = torch.mean(-q_pi)        

                                         
        alpha = min(self.log_alpha.exp().item(), self.alpha_max)
        alpha = max(self.log_alpha.exp().item(), self.alpha_min)

        loss_entropy = 0
        loss_alpha = 0
        
        ac_prob_comb = torch.einsum('by, bz->byz', ac_prob_sp[1], ac_prob_sp[2]).reshape(self.batch_size, -1)                 
        ac_prob_comb = torch.einsum('bx, bz->bxz', ac_prob_sp[0], ac_prob_comb).reshape(self.batch_size, -1)                      
                                                         
        
        log_ac_prob_comb = log_ac_prob_sp[0].reshape(self.batch_size, self.action_dims[0], 1, 1).repeat(
                                    1, 1, self.action_dims[1], self.action_dims[2]).reshape(self.batch_size, -1)\
                            + log_ac_prob_sp[1].reshape(self.batch_size, 1, self.action_dims[1], 1).repeat(
                                    1, self.action_dims[0], 1, self.action_dims[2]).reshape(self.batch_size, -1)\
                            + log_ac_prob_sp[2].reshape(self.batch_size, 1, 1, self.action_dims[2]).repeat(
                                    1, self.action_dims[0], self.action_dims[1], 1).reshape(self.batch_size, -1)
        loss_entropy = (alpha * ac_prob_comb * log_ac_prob_comb).sum(dim=1).mean()
        loss_alpha = -(self.log_alpha.to(self.device) *\
                        ((ac_prob_comb * log_ac_prob_comb).sum(dim=1) + self.target_entropy).detach()).mean()

        return loss_entropy, loss_policy, loss_alpha

    def update(self, data):
                                                           
        ave_pi_grads, ave_q_grads = [], []
        
        loss_q = self.compute_loss_q(data)
        self.q_optimizer.zero_grad()
        loss_q.backward()
        clip_grad_norm_(self.q_params, 5)
        for q in list(self.q_params):
            if q.grad is not None:
                ave_q_grads.append(q.grad.abs().mean().item())
        
        self.q_optimizer.step()
        self.q_scheduler.step(loss_q)

                                                                    
                                                                       
        for q in self.q_params:
            q.requires_grad = False

        loss_entropy, loss_policy, loss_alpha = self.compute_loss_pi(data)
        loss_pi = loss_entropy + loss_policy
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        clip_grad_norm_(self.pi_params, 5)
        for p in self.pi_params:
            if p.grad is not None:
                ave_pi_grads.append(p.grad.abs().mean().item())
        
        self.pi_optimizer.step()
        self.pi_scheduler.step(loss_policy)
        
        if self.train_alpha:
            if self.alpha_start <= self.t < self.alpha_end:
                self.alpha_optimizer.zero_grad()
                loss_alpha.backward()
                self.alpha_optimizer.step()
        
                                                                       
        for p in self.q_params:
            p.requires_grad = True
        
                                                              
        with torch.no_grad():
            self.ac_targ.load_state_dict(self.ac.state_dict())
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def run(self):
        num_generated = 0
        pbar = tqdm(total=self.num_mols)
        o = self.env.reset()

        while True:
            with torch.no_grad():
                cands = self.ac.embed(self.ac.pi.cand)
                o_embeds = self.ac.embed([o])
                o_g, o_n_emb, o_g_emb = o_embeds

                if self.t >= self.start_steps:
                    ac, (ac_prob, log_ac_prob), (ac_first, ac_second, ac_third) =\
                    self.ac.pi(o_g_emb, o_n_emb, o_g, cands)
                else:
                    ac = self.env.sample_motif()[np.newaxis]
                    (ac_prob, log_ac_prob), (ac_first, ac_second, ac_third) =\
                    self.ac.pi.sample(ac[0], o_g_emb, o_n_emb, o_g, cands)

            o2, r, d, info = self.env.step(ac[0])

            r_d = info['stop']
                                                                         
            replay_stored = any(o2['att'])
            if replay_stored:
                if type(ac) == np.ndarray:
                    self.replay_buffer.store(o, ac, r, o2, r_d,
                                             ac_prob, log_ac_prob, ac_first, ac_second, ac_third,
                                             o_embeds)
                else:
                    self.replay_buffer.store(o, ac.detach().cpu().numpy(), r, o2, r_d,
                                             ac_prob, log_ac_prob, ac_first, ac_second, ac_third,
                                             o_embeds)

                                                                                                 
            o = o2

                                        
            if get_att_points(self.env.mol) == []:                                           
                d = True
            if not any(o2['att']):
                d = True

            if d:
                self._log_attempt(info, replay_stored=replay_stored, terminal_reset=True, reward=r, ac=ac[0])
                o = self.env.reset()
                print(
                    f'[terminal] t={self.t} stop={info["stop"]} '
                    f'sac_pending={len(self.env.smile_list)} '
                    f'ga_pending={len(self.ga_smiles_list)} '
                    f'done_pending={len(self.replay_buffer.done_location)}'
                )

                              
                if self.t >= self.start_steps and len(self.population) >= 2:
                    offspring = reproduce(self.population, self.population_score, self.mutation_rate)
                    if offspring is not None:
                        self.ga_smiles_list.append(Chem.MolToSmiles(offspring))

            if self.t > 1 and self.t % self.docking_every == 0:
                print(
                    f'[dock-check] t={self.t} '
                    f'sac_pending={len(self.env.smile_list)} '
                    f'ga_pending={len(self.ga_smiles_list)} '
                    f'will_flush={bool(self.env.smile_list or self.ga_smiles_list)}'
                )

            if self.t > 1 and self.t % self.docking_every == 0 and (self.env.smile_list != [] or self.ga_smiles_list != []):
                sac_smiles = list(self.env.smile_list)
                ga_smiles = list(self.ga_smiles_list)
                n_sac_smi = len(sac_smiles)
                n_ga_smi = len(ga_smiles)
                all_smiles = sac_smiles + ga_smiles
                n_smi = len(all_smiles)
                print('=================== num sac smiles : ', n_sac_smi)
                print('=================== num ga smiles : ', n_ga_smi)
                print('=================== num smiles : ', n_smi)
                print('=================== t : ', self.t)
                mol_type_list = ['sac' for i in range(n_sac_smi)] + ['ga' for i in range(n_ga_smi)]
                if n_smi > 0:
                    done_pending_before = len(self.replay_buffer.done_location)
                    rews, ext_rew = self.env.reward_batch_vqs(all_smiles)
                    learning_scores = np.asarray(ext_rew, dtype=np.float32)
                    self._update_region_generated_bank(all_smiles, learning_scores)
                    self._maybe_mfrag_finetune()

                    r_batch = learning_scores[:n_sac_smi]
                    if n_sac_smi > 0:
                        self.replay_buffer.rew_store(r_batch)
                    print(
                        f'[dock-flush] t={self.t} '
                        f'n_sac={n_sac_smi} n_ga={n_ga_smi} '
                        f'n_total={n_smi} ext_rew={len(ext_rew)} '
                        f'done_before={done_pending_before} '
                        f'done_after={len(self.replay_buffer.done_location)}'
                    )

                    with open(self.fname, 'a') as f:
                        for i in range(n_smi):
                            str = f'{all_smiles[i]},' + f'{mol_type_list[i]},'
                            for rew in rews:
                                str = str + f'{rew[i]},'
                            str = str + (f'{ext_rew[i]}' + '\n')
                            f.write(str)

                    mols = [Chem.MolFromSmiles(s) for s in all_smiles]

                                  
                    if self.t >= self.start_steps and n_ga_smi > 0:
                        self.update_vocab(mols[n_sac_smi:], learning_scores[n_sac_smi:])
                    print(self.t, self.start_steps, self.update_after)
                                            
                    self._update_ga_population(mols, learning_scores, rews[1], rews[2])

                    num_generated = num_generated + n_smi
                    pbar.update(n_smi)

                    if num_generated >= self.num_mols:
                        pbar.close()
                        break

                    self.env.reset_batch()
                    self.ga_smiles_list = []

                             
            if self.t >= self.update_after and self.t % self.update_every == 0:
                for j in range(self.update_every):
                    batch = self.replay_buffer.sample_batch(self.device, self.batch_size)
                    self.update(data=batch)
            
            self.t = self.t + 1
