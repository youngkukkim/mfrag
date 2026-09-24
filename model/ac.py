from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.nn.pytorch.glob import SumPooling

from rdkit import Chem
from torch_geometric.data import Batch
from utils_sac.utils import ecfp
from utils_mfrag.data import get_graph_from_frag


msg = fn.copy_src(src='x', out='m')


def reduce_mean(nodes):
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'x': accum}


def reduce_sum(nodes):
    accum = torch.sum(nodes.mailbox['m'], 1)
    return {'x': accum}  


class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, agg="sum", is_normalize=False, residual=True):
        super().__init__()
        self.residual = residual
        assert agg in ["sum", "mean"], "Wrong agg type"
        self.agg = agg
        self.is_normalize = is_normalize
        self.linear1 = nn.Linear(in_channels, out_channels, bias=False)
        self.activation = nn.ReLU()

    def forward(self, g):
        h_in = g.ndata['x']
        if self.agg == "sum":
            g.update_all(msg, reduce_sum)
        elif self.agg == "mean":
            g.update_all(msg, reduce_mean)
        h = self.linear1(g.ndata['x'])
        h = self.activation(h)
        if self.is_normalize:
            h = F.normalize(h, p=2, dim=1)
        if self.residual:
            h = h + h_in
        return h


class GCNPredictor(nn.Module):
    def __init__(self, args, atom_vocab):
        super().__init__()
        self.embed = GCNEmbed(args, atom_vocab)
        self.pred_layer = nn.Sequential(
                    nn.Linear(args.emb_size*2, args.emb_size, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(args.emb_size, 1, bias=True))

    def forward(self, o):
        _, _, graph_emb = self.embed(o)
        pred = self.pred_layer(graph_emb)
        return pred


class GCNQFunction(nn.Module):
    def __init__(self, args, override_seed=False):
        super().__init__()
        if override_seed:
            seed = args.seed + 1
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.batch_size = args.batch_size
        self.device = args.device
        self.emb_size = args.emb_size
        self.d = 3 * args.emb_size + 80
        self.frag_desc_dim = get_frag_desc_dim(args)

        self.frag_desc_emb = nn.Linear(self.frag_desc_dim, args.emb_size)
        self.qpred_layer = nn.Sequential(
                            nn.Linear(self.d, int(self.d // 2), bias=False),
                            nn.ReLU(inplace=False),
                            nn.Linear(int(self.d // 2), 1, bias=True))
    
    def forward(self, graph_emb, ac_first_prob, ac_second_desc, ac_third_prob):
        frag_desc_emb = self.frag_desc_emb(ac_second_desc)
        emb_state_action = torch.cat([graph_emb, ac_first_prob, frag_desc_emb, ac_third_prob], dim=-1).contiguous()
        qpred = self.qpred_layer(emb_state_action)
        return qpred


def get_frag_desc_dim(args):
    frag_desc_dim = getattr(args, 'frag_desc_dim', 128)
    frag_desc_mode = getattr(args, 'frag_desc_mode', 'ecfp')
    if frag_desc_mode == 'raw_ecfp':
        return 1024
    return frag_desc_dim * 2 if frag_desc_mode == 'concat' else frag_desc_dim


class SFSPolicy(nn.Module):
    def __init__(self, env, args, frag_vocab_mol):
        super().__init__()
        self.device = args.device
        self.batch_size = args.batch_size
        self.frag_vocab_mol = frag_vocab_mol
        self.emb_size = args.emb_size
        self.tau = args.tau
        self.frag_desc_mode = getattr(args, 'frag_desc_mode', 'ecfp')
        self.frag_desc_hidden_dim = getattr(args, 'frag_desc_dim', 128)
        self.ecfp_weight = getattr(args, 'ecfp_weight', 1.0)
        self.sp_weight = getattr(args, 'sp_weight', 1.0)
        self.frag_desc_dim = get_frag_desc_dim(args)
        self.mfrag = None
        self.fragment_selection_mode = getattr(args, 'fragment_selection_mode', 'hybrid')
        if getattr(args, 'disable_region_guidance', False):
            self.fragment_selection_mode = 'sac_only'
        self.gumbel_noise_scale = float(getattr(args, 'gumbel_noise_scale', 1e-3))
        self.region_guidance_enabled = self.fragment_selection_mode != 'sac_only'
        self.constraint_guidance_enabled = bool(getattr(args, 'enable_constrained_mfrag_guidance', False))
        self.constraint_region_guides = {}
        self.region_knn_k = getattr(args, 'region_knn_k', 40)
        self.region_score_mode = getattr(args, 'region_score_mode', 'knn')
        self.region_ref_embeddings = None
        self.region_ref_scores = None
        self.region_ref_high_mask = None
        
                              
        self.bond_type_num = 4

        self.env = env                                        
        self.ecfp_proj = nn.Linear(1024, self.frag_desc_hidden_dim, bias=False).to(self.device)
        self.sp_proj = nn.Linear(128, self.frag_desc_hidden_dim, bias=False).to(self.device)
        self.ecfp_norm = nn.LayerNorm(self.frag_desc_hidden_dim).to(self.device)
        self.sp_norm = nn.LayerNorm(self.frag_desc_hidden_dim).to(self.device)
        
        self.cand = self.create_candidate_motifs()
        self.motif_type_num = len(self.cand)
        self.cand_ecfp = self.get_candidate_ecfp()
        self.cand_mfrag = None
        self.ac3_att_len = torch.LongTensor([len(x['att']) 
                                for x in self.cand]).to(self.device)
        self.ac3_att_mask = torch.cat([torch.LongTensor([i]*len(x['att'])) 
                                for i, x in enumerate(self.cand)], dim=0).to(self.device)

        self.action1_layers = nn.ModuleList([nn.Bilinear(2*args.emb_size, 2*args.emb_size, args.emb_size).to(self.device),
                                nn.Linear(2*args.emb_size, args.emb_size, bias=False).to(self.device),
                                nn.Linear(2*args.emb_size, args.emb_size, bias=False).to(self.device), 
                                nn.Sequential(
                                nn.Linear(args.emb_size, args.emb_size//2, bias=False),
                                nn.ReLU(inplace=False),
                                nn.Linear(args.emb_size//2, 1, bias=True)).to(self.device)])
                       
        self.action2_layers = nn.ModuleList([nn.Bilinear(self.frag_desc_dim, args.emb_size, args.emb_size).to(self.device),
                                nn.Linear(self.frag_desc_dim, args.emb_size, bias=False).to(self.device),
                                nn.Linear(args.emb_size, args.emb_size, bias=False).to(self.device), 
                                nn.Sequential(
                                nn.Linear(args.emb_size, args.emb_size, bias=False),
                                nn.ReLU(inplace=False),
                                nn.Linear(args.emb_size, args.emb_size, bias=True),
                                nn.ReLU(inplace=False),
                                nn.Linear(args.emb_size, 1, bias=True))])

        self.action3_layers = nn.ModuleList([nn.Bilinear(2*args.emb_size, 2*args.emb_size, args.emb_size).to(self.device),
                                nn.Linear(2*args.emb_size, args.emb_size, bias=False).to(self.device),
                                nn.Linear(2*args.emb_size, args.emb_size, bias=False).to(self.device),
                                nn.Sequential(
                                nn.Linear(args.emb_size, args.emb_size//2, bias=False),
                                nn.ReLU(inplace=False),
                                nn.Linear(args.emb_size//2, 1, bias=True)).to(self.device)])

                                                 
        self.max_action = 40               
        
    def update_vocab(self, vocab):
        self.frag_vocab_mol = vocab['FRAG_MOL']
        self.cand = self.create_candidate_motifs()
        self.motif_type_num = len(self.cand)
        self.cand_ecfp = self.get_candidate_ecfp()
        self.cand_mfrag = self.get_candidate_mfrag_embedding() if self.mfrag is not None else None
        for guide in self.constraint_region_guides.values():
            guide['candidate_embeddings'] = self.get_candidate_mfrag_embedding(guide['model'])
        self.ac3_att_len = torch.LongTensor([len(x['att']) 
                                for x in self.cand]).to(self.device)
        self.ac3_att_mask = torch.cat([torch.LongTensor([i]*len(x['att'])) 
                                for i, x in enumerate(self.cand)], dim=0).to(self.device)

    def set_mfrag(self, mfrag):
        self.mfrag = mfrag
        self.cand_mfrag = self.get_candidate_mfrag_embedding()

    def set_score_predictor(self, score_predictor):
        self.set_mfrag(score_predictor)

    def set_constraint_region_guide(self, name, model, reference_embeddings,
                                    reference_scores, cutoff, knn_k=40):
        embeddings = torch.as_tensor(reference_embeddings, dtype=torch.float32, device=self.device)
        scores = torch.as_tensor(reference_scores, dtype=torch.float32, device=self.device).view(-1)
        candidates = self.get_candidate_mfrag_embedding(model)
        if (embeddings.ndim != 2 or embeddings.size(0) == 0
                or embeddings.size(0) != scores.numel()
                or embeddings.size(1) != candidates.size(1)
                or not torch.isfinite(embeddings).all() or not torch.isfinite(scores).all()):
            raise ValueError('Invalid {} region reference'.format(name))
        self.constraint_region_guides[name] = {
            'model': model, 'reference_embeddings': embeddings, 'reference_scores': scores,
            'candidate_embeddings': candidates, 'cutoff': float(cutoff), 'knn_k': int(knn_k),
        }

    def set_region_reference(self, embeddings, scores, knn_k=None, high_mask=None, score_mode=None):
        if embeddings is None or scores is None or len(embeddings) == 0:
            self.region_ref_embeddings = None
            self.region_ref_scores = None
            self.region_ref_high_mask = None
            return
        self.region_ref_embeddings = torch.as_tensor(embeddings, dtype=torch.float32, device=self.device)
        self.region_ref_scores = torch.as_tensor(scores, dtype=torch.float32, device=self.device)
        if knn_k is not None:
            self.region_knn_k = int(knn_k)
        if score_mode is not None:
            self.region_score_mode = str(score_mode)
        if high_mask is None:
            self.region_ref_high_mask = None
        else:
            high_mask = torch.as_tensor(high_mask, dtype=torch.bool, device=self.device).view(-1)
            if high_mask.numel() != self.region_ref_embeddings.size(0):
                high_mask = torch.zeros(self.region_ref_embeddings.size(0), dtype=torch.bool, device=self.device)
            self.region_ref_high_mask = high_mask

    def has_region_guidance(self):
        return (
            self.region_guidance_enabled
            and self.mfrag is not None
            and self.cand_mfrag is not None
            and self.region_ref_embeddings is not None
            and self.region_ref_scores is not None
            and self.region_ref_embeddings.size(0) > 0
        )

    def create_candidate_motifs(self):
        motif_gs = [self.env.get_observation_mol(mol) for mol in self.frag_vocab_mol]
        return motif_gs

    def get_candidate_ecfp(self):
        return torch.tensor(
            [ecfp(Chem.MolFromSmiles(x['smi'])) for x in self.cand],
            dtype=torch.float32,
            device=self.device,
        )

    def get_candidate_mfrag_embedding(self, model=None):
        model = self.mfrag if model is None else model
        if model is None:
            return torch.zeros((self.motif_type_num, 128), dtype=torch.float32, device=self.device)

        frag_graphs = [get_graph_from_frag(x['smi']) for x in self.cand]
        frag_batch = Batch.from_data_list(frag_graphs).to(self.device)
        with torch.no_grad():
            if hasattr(model, 'encode_frag'):
                frag_embedding = model.encode_frag(frag_batch)
            else:
                _, frag_embedding = model(frag_batch)
        return frag_embedding

    def get_candidate_descriptors(self):
        if self.frag_desc_mode == 'raw_ecfp':
            return self.ecfp_weight * self.cand_ecfp

        ecfp_branch = self.ecfp_norm(self.ecfp_proj(self.cand_ecfp))

        if self.frag_desc_mode == 'ecfp':
            return self.ecfp_weight * ecfp_branch

        if self.cand_mfrag is None:
            self.cand_mfrag = self.get_candidate_mfrag_embedding()
        sp_branch = self.sp_norm(self.sp_proj(self.cand_mfrag))

        if self.frag_desc_mode == 'sp':
            return self.sp_weight * sp_branch
        if self.frag_desc_mode == 'sum':
            return self.ecfp_weight * ecfp_branch + self.sp_weight * sp_branch
        if self.frag_desc_mode == 'concat':
            return torch.cat(
                [self.ecfp_weight * ecfp_branch, self.sp_weight * sp_branch],
                dim=-1,
            )
        raise ValueError(f'Unsupported frag_desc_mode: {self.frag_desc_mode}')

    def get_fragment_viability_mask(self, current_att_counts):
        if torch.is_tensor(current_att_counts):
            att_counts = current_att_counts.to(self.device).view(-1).long()
        else:
            att_counts = torch.tensor(current_att_counts, device=self.device, dtype=torch.long).view(-1)

        viable_mask = (att_counts.unsqueeze(1) + self.ac3_att_len.view(1, -1) - 2) > 0
        no_viable = ~viable_mask.any(dim=1)
        if torch.any(no_viable):
            viable_mask = viable_mask.clone()
            viable_mask[no_viable] = True
        return viable_mask

    def query_region_scores(self, query_embeddings):
        if query_embeddings.dim() == 1:
            query_embeddings = query_embeddings.unsqueeze(0)

        k = min(max(int(self.region_knn_k), 1), self.region_ref_embeddings.size(0))
        distances = torch.cdist(query_embeddings, self.region_ref_embeddings)
        if self.region_score_mode == 'min_distance':
            if self.region_ref_high_mask is not None and torch.any(self.region_ref_high_mask):
                distances = distances[:, self.region_ref_high_mask]
            return -distances.min(dim=-1).values
        knn_dist, knn_idx = torch.topk(distances, k=k, dim=-1, largest=False)
        weights = 1.0 / (knn_dist + 1e-6)
        neighbor_scores = self.region_ref_scores[knn_idx]
        return (weights * neighbor_scores).sum(dim=-1) / weights.sum(dim=-1)

    def get_region_guided_logits(self, current_att_count, fallback_logits):
        if not self.has_region_guidance():
            if self.region_guidance_enabled and self.fragment_selection_mode == 'mfrag_only':
                raise RuntimeError('M-FRAG-only selection requires an initialized region reference')
            return fallback_logits
        if self.constraint_guidance_enabled:
            for name in ('qed', 'sa'):
                guide = self.constraint_region_guides.get(name)
                if guide is None or guide['candidate_embeddings'].size(0) != self.motif_type_num:
                    raise RuntimeError('Missing or stale {} region guide'.format(name))

        try:
            _, current_final_smiles = self.env.get_final_smiles_mol()
            current_graph = get_graph_from_frag(current_final_smiles)
            current_batch = Batch.from_data_list([current_graph]).to(self.device)
            with torch.no_grad():
                _, current_emb = self.mfrag(current_batch)
                current_score = self.query_region_scores(current_emb).squeeze(0)
                fragment_count = max(int(getattr(self.env, 'counter', 0)) + 1, 1)
                mixed_emb = (
                    current_emb * float(fragment_count) + self.cand_mfrag
                ) / float(fragment_count + 1)
                candidate_scores = self.query_region_scores(mixed_emb)
                improvements = candidate_scores - current_score
                constraint_mask = torch.ones_like(improvements, dtype=torch.bool)
                if self.constraint_guidance_enabled:
                    for name in ('qed', 'sa'):
                        guide = self.constraint_region_guides[name]
                        _, current_constraint_emb = guide['model'](current_batch)
                        mixed_constraint_emb = (
                            current_constraint_emb * float(fragment_count)
                            + guide['candidate_embeddings']
                        ) / float(fragment_count + 1)
                        scores = self._query_reference_scores(
                            mixed_constraint_emb, guide['reference_embeddings'],
                            guide['reference_scores'], guide['knn_k'])
                        constraint_mask &= scores >= guide['cutoff']
        except Exception:
            if self.fragment_selection_mode == 'mfrag_only':
                viable = self.get_fragment_viability_mask([current_att_count]).squeeze(0)
                return torch.zeros_like(fallback_logits).masked_fill(~viable, -1e9)
            return fallback_logits

        viable_mask = self.get_fragment_viability_mask([current_att_count]).squeeze(0)
        improving_mask = improvements > 0
        guided_mask = viable_mask & improving_mask & constraint_mask

        guided_logits = torch.full_like(fallback_logits, -1e9)
        if torch.any(guided_mask):
            guided_logits[guided_mask] = improvements[guided_mask]
            return guided_logits
        if torch.any(viable_mask):
            if self.fragment_selection_mode == 'mfrag_only':
                guided_logits[viable_mask] = improvements[viable_mask]
            else:
                guided_logits[viable_mask] = fallback_logits[viable_mask]
            return guided_logits
        return fallback_logits

    @staticmethod
    def _query_reference_scores(query_embeddings, reference_embeddings, reference_scores, knn_k):
        k = min(max(int(knn_k), 1), reference_embeddings.size(0))
        distances = torch.cdist(query_embeddings, reference_embeddings)
        knn_dist, knn_idx = torch.topk(distances, k=k, dim=-1, largest=False)
        weights = 1.0 / (knn_dist + 1e-6)
        return (weights * reference_scores[knn_idx]).sum(dim=-1) / weights.sum(dim=-1)

    def gumbel_softmax(self, logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1,\
                    g_ratio=None) -> torch.Tensor:
        if g_ratio is None:
            g_ratio = self.gumbel_noise_scale
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )                
        gumbels = (logits + gumbels * g_ratio) / tau                       
        y_soft = gumbels.softmax(dim)
        
        if hard:
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
        return ret

    def forward(self, graph_emb, node_emb, g, cands):
        """
        graph_emb : bs x hidden_dim
        node_emb : (bs x num_nodes) x hidden_dim)
        g: batched graph
        att: indexs of attachment points, list of list
        """
        g.ndata['node_emb'] = node_emb
        cand_g, cand_node_emb, cand_graph_emb = cands
        cand_desc = self.get_candidate_descriptors()

                                                              
        ob_len = g.batch_num_nodes().tolist()
        att_mask = g.ndata['att_mask']                                         
        
        if g.batch_size != 1:
            att_mask_split = torch.split(att_mask, ob_len, dim=0)
            att_len = [torch.sum(x, dim=0) for x in att_mask_split]
            current_att_counts = [int(x.item()) for x in att_len]
        else:
            att_len = torch.sum(att_mask, dim=-1)                                   
            current_att_counts = [int(att_len.item())]

        cand_att_mask = cand_g.ndata['att_mask']

                                          
                               
                                          
                                                  
        att_emb = torch.masked_select(node_emb , att_mask.unsqueeze(-1))
        att_emb = att_emb.view(-1, 2*self.emb_size)
        
        if g.batch_size != 1:
            graph_expand = torch.cat([graph_emb[i].unsqueeze(0).repeat(att_len[i],1) for i in range(g.batch_size)], dim=0).contiguous()
        else:
            graph_expand = graph_emb.repeat(att_len, 1)

        att_emb = self.action1_layers[0](att_emb, graph_expand) + self.action1_layers[1](att_emb)\
                    + self.action1_layers[2](graph_expand)
        logits_first = self.action1_layers[3](att_emb)

        if g.batch_size != 1:
            ac_first_prob = [torch.softmax(logit, dim=0)
                            for i, logit in enumerate(torch.split(logits_first, att_len, dim=0))]
            ac_first_prob = [p+1e-8 for p in ac_first_prob]
            log_ac_first_prob = [x.log() for x in ac_first_prob]
        else:
            ac_first_prob = torch.softmax(logits_first, dim=0) + 1e-8
            log_ac_first_prob = ac_first_prob.log()

        if g.batch_size != 1:
            first_stack = []
            first_ac_stack = []
            for i, node_emb_i in enumerate(torch.split(att_emb, att_len, dim=0)):
                ac_first_hot_i = self.gumbel_softmax(ac_first_prob[i], tau=self.tau, hard=True, dim=0).transpose(0,1)
                ac_first_i = torch.argmax(ac_first_hot_i, dim=-1)
                first_stack.append(torch.matmul(ac_first_hot_i, node_emb_i))
                first_ac_stack.append(ac_first_i)

            emb_first = torch.stack(first_stack, dim=0).squeeze(1)
            ac_first = torch.stack(first_ac_stack, dim=0).squeeze(1)
            
            ac_first_prob = torch.cat([
                                torch.cat([ac_first_prob_i, ac_first_prob_i.new_zeros(
                                    max(self.max_action - ac_first_prob_i.size(0), 0), 1)]
                                        , 0).contiguous().view(1,self.max_action)
                                for i, ac_first_prob_i in enumerate(ac_first_prob)], dim=0).contiguous()

            log_ac_first_prob = torch.cat([
                                    torch.cat([log_ac_first_prob_i, log_ac_first_prob_i.new_zeros(
                                        max(self.max_action - log_ac_first_prob_i.size(0), 0), 1)]
                                            , 0).contiguous().view(1,self.max_action)
                                    for i, log_ac_first_prob_i in enumerate(log_ac_first_prob)], dim=0).contiguous()
            
        else:
            ac_first_hot = self.gumbel_softmax(ac_first_prob, tau=self.tau, hard=True, dim=0).transpose(0,1)
            ac_first = torch.argmax(ac_first_hot, dim=-1)
            emb_first = torch.matmul(ac_first_hot, att_emb)
            ac_first_prob = torch.cat([ac_first_prob, ac_first_prob.new_zeros(
                            max(self.max_action - ac_first_prob.size(0), 0), 1)]
                                , 0).contiguous().view(1,self.max_action)
            log_ac_first_prob = torch.cat([log_ac_first_prob, log_ac_first_prob.new_zeros(
                            max(self.max_action - log_ac_first_prob.size(0), 0), 1)]
                                , 0).contiguous().view(1,self.max_action)

                                         
                                                         
                                         
        emb_first_expand = emb_first.view(-1, 1, self.emb_size).repeat(1, self.motif_type_num, 1)
        cand_expand = cand_desc.unsqueeze(0).repeat(g.batch_size, 1, 1)
        
        emb_cat = self.action2_layers[0](cand_expand, emb_first_expand) +\
                    self.action2_layers[1](cand_expand) + self.action2_layers[2](emb_first_expand)

        logit_second = self.action2_layers[3](emb_cat).squeeze(-1)
        viable_mask = self.get_fragment_viability_mask(current_att_counts)
        if g.batch_size == 1 and self.region_guidance_enabled:
            fallback_logits = logit_second.squeeze(0).masked_fill(~viable_mask.squeeze(0), -1e9)
            logit_second = self.get_region_guided_logits(current_att_counts[0], fallback_logits).unsqueeze(0)
        else:
            logit_second = logit_second.masked_fill(~viable_mask, -1e9)
        ac_second_prob = F.softmax(logit_second, dim=-1) + 1e-8
        log_ac_second_prob = ac_second_prob.log()
        
        ac_second_hot = self.gumbel_softmax(ac_second_prob, tau=self.tau, hard=True)
        emb_second = torch.matmul(ac_second_hot, cand_graph_emb)
        ac_second = torch.argmax(ac_second_hot, dim=-1)
        ac_second_desc = torch.matmul(ac_second_hot, cand_desc)

                                         
                                        
                                         
                                          
        cand_att_emb = torch.masked_select(cand_node_emb, cand_att_mask.unsqueeze(-1))
        cand_att_emb = cand_att_emb.view(-1, 2*self.emb_size)

        ac3_att_mask = self.ac3_att_mask.repeat(g.batch_size, 1)                                  
        ac3_att_mask = torch.where(ac3_att_mask==ac_second.view(-1, 1),
                            1, 0).view(g.batch_size, -1)                          
        ac3_att_mask = ac3_att_mask.bool()

        ac3_cand_emb = torch.masked_select(cand_att_emb.view(1, -1, 2*self.emb_size), 
                                ac3_att_mask.view(g.batch_size, -1, 1)).view(-1, 2*self.emb_size)                              
        
        ac3_att_len = torch.index_select(self.ac3_att_len, 0, ac_second).tolist()
        emb_second_expand = torch.cat([emb_second[i].unsqueeze(0).repeat(ac3_att_len[i],1) for i in range(g.batch_size)]).contiguous()

        emb_cat_ac3 = self.action3_layers[0](emb_second_expand, ac3_cand_emb) + self.action3_layers[1](emb_second_expand)\
                  + self.action3_layers[2](ac3_cand_emb)
        
        logits_third = self.action3_layers[3](emb_cat_ac3)

                       
        if g.batch_size != 1:
            ac_third_prob = [torch.softmax(logit, dim=-1)
                            for i, logit in enumerate(torch.split(logits_third.squeeze(-1), ac3_att_len, dim=0))]
            ac_third_prob = [p+1e-8 for p in ac_third_prob]
            log_ac_third_prob = [x.log() for x in ac_third_prob]
        else:
            logits_third = logits_third.transpose(1,0)
            ac_third_prob = torch.softmax(logits_third, dim=-1) + 1e-8
            log_ac_third_prob = ac_third_prob.log()
        
                                                  
        if g.batch_size != 1:
            third_stack = []
            third_ac_stack = []
            for i, node_emb_i in enumerate(torch.split(emb_cat_ac3, ac3_att_len, dim=0)):
                ac_third_hot_i = self.gumbel_softmax(ac_third_prob[i], tau=self.tau, hard=True, dim=-1)
                ac_third_i = torch.argmax(ac_third_hot_i, dim=-1)
                third_stack.append(torch.matmul(ac_third_hot_i, node_emb_i))
                third_ac_stack.append(ac_third_i)

                del ac_third_hot_i
            ac_third = torch.stack(third_ac_stack, dim=0)
            ac_third_prob = torch.cat([
                                torch.cat([ac_third_prob_i, ac_third_prob_i.new_zeros(
                                    self.max_action - ac_third_prob_i.size(0))]
                                        , dim=0).contiguous().view(1,self.max_action)
                                for i, ac_third_prob_i in enumerate(ac_third_prob)], dim=0).contiguous()
            
            log_ac_third_prob = torch.cat([
                                    torch.cat([log_ac_third_prob_i, log_ac_third_prob_i.new_zeros(
                                        self.max_action - log_ac_third_prob_i.size(0))]
                                            , 0).contiguous().view(1,self.max_action)
                                    for i, log_ac_third_prob_i in enumerate(log_ac_third_prob)], dim=0).contiguous()

        else:
            ac_third_hot = self.gumbel_softmax(ac_third_prob, tau=self.tau, hard=True, dim=-1)
            ac_third = torch.argmax(ac_third_hot, dim=-1)
            
            ac_third_prob = torch.cat([ac_third_prob, ac_third_prob.new_zeros(
                                        1, self.max_action - ac_third_prob.size(1))] 
                                , -1).contiguous()
            log_ac_third_prob = torch.cat([log_ac_third_prob, log_ac_third_prob.new_zeros(
                                        1, self.max_action - log_ac_third_prob.size(1))]
                                , -1).contiguous()

        ac_prob = torch.cat([ac_first_prob, ac_second_prob, ac_third_prob], dim=1).contiguous()
        log_ac_prob = torch.cat([log_ac_first_prob,
                            log_ac_second_prob, log_ac_third_prob], dim=1).contiguous()
        ac = torch.stack([ac_first, ac_second, ac_third], dim=1)

        return ac, (ac_prob, log_ac_prob), (ac_first_prob, ac_second_desc, ac_third_prob)
    
    def sample(self, ac, graph_emb, node_emb, g, cands):
        g.ndata['node_emb'] = node_emb
        cand_g, cand_node_emb, cand_graph_emb = cands 
        cand_desc = self.get_candidate_descriptors()

                                                             
        att_mask = g.ndata['att_mask']                                                  
        att_len = torch.sum(att_mask, dim=-1)                                     
        current_att_counts = [int(att_len.item())]

        cand_att_mask = cand_g.ndata['att_mask']

                                          
                               
                                          
                                                  
        att_emb = torch.masked_select(node_emb, att_mask.unsqueeze(-1))
        att_emb = att_emb.view(-1, 2*self.emb_size)
        graph_expand = graph_emb.repeat(att_len, 1)
        
        att_emb = self.action1_layers[0](att_emb, graph_expand) + self.action1_layers[1](att_emb)\
                    + self.action1_layers[2](graph_expand)
        logits_first = self.action1_layers[3](att_emb).transpose(1,0)
            
        ac_first_prob = torch.softmax(logits_first, dim=-1) + 1e-8
        
        log_ac_first_prob = ac_first_prob.log()
        ac_first_prob = torch.cat([ac_first_prob, ac_first_prob.new_zeros(1,
                        max(self.max_action - ac_first_prob.size(1),0))]
                            , 1).contiguous()
        
        log_ac_first_prob = torch.cat([log_ac_first_prob, log_ac_first_prob.new_zeros(1,
                        max(self.max_action - log_ac_first_prob.size(1),0))]
                            , 1).contiguous()
        emb_first = att_emb[ac[0]].unsqueeze(0)
        
                                         
                                     
                                         
        emb_first_expand = emb_first.repeat(1, self.motif_type_num, 1)
        cand_expand = cand_desc.unsqueeze(0).repeat(g.batch_size, 1, 1)     
        
        emb_cat = self.action2_layers[0](cand_expand, emb_first_expand) +\
                    self.action2_layers[1](cand_expand) + self.action2_layers[2](emb_first_expand)
        
        logit_second = self.action2_layers[3](emb_cat).squeeze(-1)
        viable_mask = self.get_fragment_viability_mask(current_att_counts)
        logit_second = logit_second.masked_fill(~viable_mask, -1e9)
        ac_second_prob = F.softmax(logit_second, dim=-1) + 1e-8
        log_ac_second_prob = ac_second_prob.log()
        
        ac_second_hot = self.gumbel_softmax(ac_second_prob, tau=self.tau, hard=True)
        emb_second = torch.matmul(ac_second_hot, cand_graph_emb)
        ac_second_desc = torch.matmul(ac_second_hot, cand_desc)

                                           
                                        
                                         
                                           
        cand_att_emb = torch.masked_select(cand_node_emb, cand_att_mask.unsqueeze(-1))
        cand_att_emb = cand_att_emb.view(-1, 2*self.emb_size)

        ac3_att_mask = self.ac3_att_mask.repeat(g.batch_size, 1)                                  
                                                          
        
        ac3_att_mask = torch.where(ac3_att_mask==ac[1], 
                            1, 0).view(g.batch_size, -1)                          
        ac3_att_mask = ac3_att_mask.bool()

        ac3_cand_emb = torch.masked_select(cand_att_emb.view(1, -1, 2*self.emb_size), 
                                ac3_att_mask.view(g.batch_size, -1, 1)).view(-1, 2*self.emb_size)
        
        ac3_att_len = self.ac3_att_len[ac[1]]
        emb_second_expand = emb_second.repeat(ac3_att_len,1)
        emb_cat_ac3 = self.action3_layers[0](emb_second_expand, ac3_cand_emb) + self.action3_layers[1](emb_second_expand)\
                  + self.action3_layers[2](ac3_cand_emb)

        logits_third = self.action3_layers[3](emb_cat_ac3)
        logits_third = logits_third.transpose(1,0)
        ac_third_prob = torch.softmax(logits_third, dim=-1) + 1e-8
        log_ac_third_prob = ac_third_prob.log()

                                                  
        ac_third_prob = torch.cat([ac_third_prob, ac_third_prob.new_zeros(
                                        1, self.max_action - ac_third_prob.size(1))] 
                                , -1).contiguous()
        log_ac_third_prob = torch.cat([log_ac_third_prob, log_ac_third_prob.new_zeros(
                                        1, self.max_action - log_ac_third_prob.size(1))]
                                , -1).contiguous()

                                     
        ac_prob = torch.cat([ac_first_prob, ac_second_prob, ac_third_prob], dim=1).contiguous()
        log_ac_prob = torch.cat([log_ac_first_prob, 
                            log_ac_second_prob, log_ac_third_prob], dim=1).contiguous()

        return (ac_prob, log_ac_prob), (ac_first_prob, ac_second_desc, ac_third_prob)
        

class GCNEmbed(nn.Module):
    def __init__(self, args, atom_vocab):
        super().__init__()

        self.device = args.device
        self.bond_type_num = 4
        self.d_n = len(atom_vocab) + 18
        
        self.emb_size = args.emb_size * 2
        in_channels = 8
        self.emb_linear = nn.Linear(self.d_n, in_channels, bias=False)

        self.gcn_layers = nn.ModuleList([GCN(in_channels, self.emb_size, agg="sum", residual=False)])
        for _ in range(args.num_layer - 1):
            self.gcn_layers.append(GCN(self.emb_size, self.emb_size, agg="sum"))
        self.pool = SumPooling() 
        
    def forward(self, ob):
        ob_g = [o['g'] for o in ob]
        ob_att = [o['att'] for o in ob]

                                                 
        for i, x_g in enumerate(ob_g):
            att_onehot = F.one_hot(torch.LongTensor(ob_att[i]), 
                        num_classes=x_g.number_of_nodes()).sum(0)
            ob_g[i].ndata['att_mask'] = att_onehot.bool()

        g = deepcopy(dgl.batch(ob_g)).to(self.device)
        
        g.ndata['x'] = self.emb_linear(g.ndata['x'])

        for i, conv in enumerate(self.gcn_layers):
            h = conv(g)
            g.ndata['x'] = h
        
        emb_node = g.ndata['x']

                              
        emb_graph = self.pool(g, g.ndata['x'])
        
        return g, emb_node, emb_graph


class GCNActorCritic(nn.Module):
    def __init__(self, env, args, vocab, predictor=False):
        super().__init__()

        self.embed = GCNEmbed(args, vocab['ATOM'])
        self.pi = SFSPolicy(env, args, vocab['FRAG_MOL'])
        self.q1 = GCNQFunction(args)
        self.q2 = GCNQFunction(args, override_seed=True)
        if predictor:
            self.p = GCNPredictor(args, vocab['ATOM'])

    def set_mfrag(self, mfrag):
        self.pi.set_mfrag(mfrag)

    def set_score_predictor(self, score_predictor):
        self.set_mfrag(score_predictor)
