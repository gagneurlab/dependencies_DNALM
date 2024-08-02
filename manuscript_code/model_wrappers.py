from transformers import AutoTokenizer
from transformers import BertConfig
from transformers import AutoModelForSequenceClassification, DefaultDataCollator
from flash_attn.models.bert import BertModel, BertForPreTraining
import torch
import os
import torch.nn as nn
from torch.amp import autocast
import itertools
import tqdm
import numpy as np
import math



device='cuda'

class ReconstructionModel(nn.Module):
    
    def __init__(self, lm, tokenizer, device=device, 
                 kmer_size = 1, 
                 left_special_tokens = 2,
                 right_special_tokens = 1,
                 return_logits = False,
                 require_lm_grad = False):
        super().__init__()
        self.lm = lm
        self.tokenizer = tokenizer
        # mode
        if kmer_size > 1:
            assert not return_logits, "Not implemented"
        self.return_logits = return_logits
        self.require_lm_grad = require_lm_grad
        # constants
        self.kmer_size = kmer_size
        self.device = device
        self.left_special_tokens = left_special_tokens
        self.right_special_tokens = right_special_tokens
        # extra layers for the probability computation
        self.create_prb_filter()
        # components for the backward
        self.word_embeddings = None
        self.hook_dict = {}
        
    def set_lm_grad_computation(self, switch):
        self.require_lm_grad = switch
        
    def set_word_embedding_hook(self):
        def getHook():
            def hook(model, input, output):
                output.retain_grad()
                self.word_embeddings = output
            return hook
        self.hook_dict["words"] = self.lm.bert.embeddings.word_embeddings.register_forward_hook(getHook())
        
    def create_prb_filter(self):
        vocab = self.tokenizer.get_vocab()
        kmer_list = ["".join(x) for x in itertools.product("ACGT",repeat=self.kmer_size)]
        nt_mapping = {"A":0,"C":1,"G":2,"T":3}
        prb_filter = np.zeros((self.kmer_size, 4**self.kmer_size, 4))
        for kmer in kmer_list:
            token = vocab[kmer] - 5 # there are 5 special tokens
            for idx, nt in enumerate(kmer):
                nt_idx = nt_mapping[nt]
                prb_filter[(self.kmer_size-1)-idx, token, nt_idx] = 1
        prb_filter = torch.from_numpy(prb_filter)
        self.prb_filter = prb_filter.to(self.device).float() # k, 4**k, 
    
    @autocast(device)
    def forward(self, tokens):
        grad_context = contextlib.nullcontext() if self.require_lm_grad else torch.no_grad() 
        with grad_context:
            logits = self._get_lm_output(tokens) # n_masked, 4**k
            # reshape to b, k, 4**k (assumes at most one span is masked, so n_masked = b*k)
            logits = logits.reshape(tokens.shape[0], self.kmer_size, logits.shape[1])
            if self.return_logits:
                logits = logits.sum(axis=(1)) # b * 4
                # reorder from A T C G to A C G T
                #logits_reord = torch.zeros_like(logits)
                #logits_reord[:,:,0] = logits[:,:,0] # A
                #logits_reord[:,:,1] = logits[:,:,2] # C
                #logits_reord[:,:,2] = logits[:,:,3] # G
                #logits_reord[:,:,3] = logits[:,:,1] # T
                #logits = logits_reord
                return logits#_reord
            else:
                # softmax to get probabilities
                kmer_prbs = torch.softmax(logits,dim=2) # b, k, 4**k
                # average over the span and the kmer in each span
                nt_prbs = (kmer_prbs.unsqueeze(-1) * self.prb_filter)
                nt_prbs = nt_prbs.sum(axis=(1,2)) # B, 4
                nt_prbs = nt_prbs/self.kmer_size
                return nt_prbs
        
    def _get_lm_output(self, tokens):
        # get masked tokens
        labels = (tokens == 4).to(torch.int64).to(self.device) # 4 is the mask token
        # this automatically extracts only predictions for masked tokens
        #predictions = self.lm(tokens, labels = labels)["prediction_logits"] # Dim: n_masked_tokens, vocab_size
        predictions = self.lm(tokens)["prediction_logits"] # B, L_in, vocab_size
        predictions = predictions[labels.bool()] # n_masked, vocab_size
        logits = predictions[:,5:(5+self.prb_filter.shape[1])] # remove any non k-mer dims (there are 5 special tokens)
        return logits
    
    def mask_tokens(self, tokens):
        # create diagonal identity matrix of the same shape as basis for the masking
        diag_matrix = torch.eye(tokens.shape[1]).numpy()
        # propagate the ones to kmer-sized spans
        masked_indices = np.apply_along_axis(lambda m : np.convolve(m, [1] * self.kmer_size, mode = 'same'),axis = 1, arr = diag_matrix).astype(bool)
        masked_indices = torch.from_numpy(masked_indices)
        # do not mask special tokens and do not repeat at edges (this math works for k = 6, unclear if it generalizes)
        masked_indices = masked_indices[(math.ceil(self.kmer_size/2) - 1)+self.left_special_tokens:-((self.kmer_size//2)+self.right_special_tokens)]
        masked_tokens = tokens.unsqueeze(1).expand(tokens.shape[0],masked_indices.shape[0],tokens.shape[1]).clone() # reshape
        masked_tokens[:,masked_indices] = 4 # mask
        return masked_tokens
    
    def predict_all_from_dataloader(self, 
                                    data_loader,
                                    batch_size = 128):
        all_preds = []
        for i, group in tqdm.tqdm(enumerate(data_loader)):
            output_arrays = []
            # get some tokenized sequences (B, L_in)
            tokens = group['input_ids']
            # mask them
            masked_tokens = self.mask_tokens(tokens)
            # remember the number of sequences (shape[0] = B) and number of maskings (shape[1] = n_masked)
            group_shape = masked_tokens.shape
            # reshape to B*n_masked, L_in
            masked_tokens = masked_tokens.reshape(masked_tokens.shape[0]*masked_tokens.shape[1], masked_tokens.shape[2])
            token_loader = torch.utils.data.DataLoader(masked_tokens, batch_size=batch_size, 
                                                       num_workers = 4, shuffle = False, collate_fn = None)
            # predict
            for j, batch in enumerate(token_loader):
                outputs = self(batch.to(self.device)) # b, 4
                output_arrays.append(outputs.cpu()) # send to cpu to conserve memory
            # rebuild to B*n_masked, 4
            predictions = torch.concat(output_arrays, axis=0)
            # reshape to B, n_masked, 4
            predictions = predictions.reshape((group_shape[0],group_shape[1],predictions.shape[1]))
            all_preds.append(predictions)
        predictions = torch.concat(all_preds, axis=0)
        return predictions
