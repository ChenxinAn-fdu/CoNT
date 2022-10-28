import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from fairseq import search
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerModel,
    transformer_wmt_en_de,
    transformer_iwslt_de_en,
)
import logging

from fairseq.cont_generator import SequenceGenerator

logger = logging.getLogger(__name__)


@register_model("transformer_cont")
class TransformerCoNTModel(TransformerModel):
    """
    Contrastive learning based generation model
    """

    def __init__(self, generator, tgt_dict, cfg, args):
        super().__init__(args, generator.encoder, generator.decoder)
        self.cfg = cfg
        self.tgt_dict = tgt_dict
        self.args = args
        self.generator = generator

        self.pad_id = self.args.pad
        self.eos_id = self.args.eos
        self.hidden_size = self.args.encoder_embed_dim
        self.linear_layer = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.xavier_uniform_(self.linear_layer.weight)

        search_strategy_train = search.DiverseBeamSearch(
            self.tgt_dict, int(self.args.max_sample_num*self.args.from_hypo), self.args.diverse_bias
        )
        self.sampler_train = SequenceGenerator(
            [self.generator],
            tgt_dict=self.tgt_dict,
            max_len_a=self.args.max_len_a,
            max_len_b=self.args.max_len_b,
            len_penalty=self.args.lenpen,
            search_strategy=search_strategy_train,
            beam_size=int(self.args.max_sample_num*self.args.from_hypo),
            max_len=self.args.max_sample_len - 1
        )
        search_strategy_test = search.BeamSearch(
            self.tgt_dict
        )
        self.sampler_test = SequenceGenerator(
            [self.generator],
            tgt_dict=self.tgt_dict,
            max_len_a=self.args.max_len_a,
            max_len_b=self.args.max_len_b,
            len_penalty=self.args.lenpen,
            search_strategy=search_strategy_test,
            beam_size=self.args.beam_size,
        )

    @classmethod
    def build_model(cls, args, task):
        # set any default arguments
        transformer_cont(args)
        transformer_model = TransformerModel.build_model(args, task)
        return TransformerCoNTModel(
            transformer_model, task.target_dictionary, task.cfg, args
        )

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        super(TransformerCoNTModel, TransformerCoNTModel).add_args(parser)
        parser.add_argument('--lenpen', default=0.1, type=float)
        parser.add_argument('--max_len_a', default=1.0, type=float)
        parser.add_argument('--max_len_b', default=50.0, type=float)
        parser.add_argument('--max_input_len', default=96, type=int)
        parser.add_argument('--max_sample_len', default=64, type=int)
        parser.add_argument('--max_sample_num', default=16, type=int)
        parser.add_argument('--n_gram', default=2, type=int)
        parser.add_argument('--alpha', default=0.5, type=float)
        parser.add_argument('--diverse_bias', default=2.5, type=float)
        parser.add_argument('--beam_size', default=12, type=int)
        parser.add_argument('--from_hypo', default=0.75, type=float)
        parser.add_argument('--keep_dropout', default=0, type=int, choices=[0,1])
        parser.add_argument('--warmup', default=0, type=int, choices=[0,1])

    def form_ngram(self, input_tensor, n=4):
        """
        input_tensor: batch x sample_num x seq_len
        return: batch x seq_len-3 x 4
        """
        bsz, cand_num, seq_len = input_tensor.size(0), input_tensor.size(1), input_tensor.size(2)
        seq_len_clip = seq_len - n + 1
        input_tensor_repeated = input_tensor[:, :, None, :].repeat(1, 1, seq_len_clip, 1)
        help_matrix_1 = torch.triu(torch.ones(seq_len, seq_len))
        help_matrix_2 = torch.triu(torch.ones(seq_len, seq_len), diagonal=n)
        help_matrix = (help_matrix_1 - help_matrix_2)[:seq_len_clip].bool()[None, None, :, :]
        ret_tensor = torch.masked_select(input_tensor_repeated, help_matrix.to(input_tensor.device))
        return ret_tensor.view(bsz, cand_num, seq_len_clip, n)

    def torch_bleu(self, ref_tensor, sys_tensor):
        """
        ref_tensor: batch x seq_len1
        sys_tensor: batch x sample_num x seq_len2
        """
        sys_padding = (~(sys_tensor == self.pad_id)).float()
        ref_padding = (~(ref_tensor == self.pad_id)).float()
        # 将 ref 和 sys的pad_id 换成不一样的 防止pad_id 的影响
        n = min(min(self.args.n_gram, ref_tensor.size(-1)), sys_tensor.size(-1))
        ref_lengths = torch.sum(ref_padding, dim=-1) - n + 1
        ref_ones = torch.ones_like(ref_lengths, device=ref_lengths.device)
        ref_lengths = torch.where(ref_lengths > 0, ref_lengths, ref_ones)
        sys_lengths = torch.sum(sys_padding, dim=-1) - n + 1
        sys_ones = torch.ones_like(sys_lengths, device=sys_lengths.device)
        sys_lengths = torch.where(sys_lengths > 0, sys_lengths, sys_ones)
        ref_tensor = ref_tensor * ref_padding
        bsz, sample_num = sys_tensor.size(0), sys_tensor.size(1)
        ref_tensor = ref_tensor[:, None, :].repeat(1, sample_num, 1)
        input_tensor1_4gram = self.form_ngram(ref_tensor, n).float()
        input_tensor2_4gram = self.form_ngram(sys_tensor, n).float()  # batch x sample_num x seq_len-3 x 4
        sim_matrix = torch.cosine_similarity(input_tensor2_4gram.unsqueeze(3), input_tensor1_4gram.unsqueeze(2),
                                             dim=-1) >= 1.0
        sim_matrix = torch.sum(torch.max(sim_matrix, dim=-1).values, dim=-1)
        length = sys_lengths + ref_lengths.unsqueeze(1)
        return sim_matrix / length  # batch x sample_num

    def ranking_loss(self, cos_distance, bleu_distance):
        # equivalent to initializing TotalLoss to 0
        # here is to avoid that some special samples will not go into the following for loop
        margin = 0.01
        ones = torch.ones(cos_distance.size(), device=cos_distance.device)
        loss_func = torch.nn.MarginRankingLoss(0.0)
        total_loss = loss_func(cos_distance, cos_distance, ones)

        # candidate loss
        n = cos_distance.size(1)
        for i in range(1, n):
            pos_score = cos_distance[:, :-i]
            neg_score = cos_distance[:, i:]
            same_mask = (torch.abs(bleu_distance[:, :-i] - bleu_distance[:, i:]) > 0.001).float()
            ones = torch.ones(pos_score.size(), device=cos_distance.device)
            loss_func = torch.nn.MarginRankingLoss(margin * i, reduction='none')  # batch x i
            marginal_loss = loss_func(pos_score, neg_score, ones)
            total_loss += (marginal_loss * same_mask).sum()

        return total_loss

    def affine_transformation(self, input_features, padding_mask, axis):
        trans_tmp = F.relu(self.linear_layer(input_features))  # batch
        length = torch.sum(padding_mask, axis=1).unsqueeze(-1)
        if axis == 0:
            padding_mask = torch.transpose(padding_mask, 1, 0)
        trans_tmp = trans_tmp * padding_mask.unsqueeze(-1).float()
        trans_emb = torch.sum(trans_tmp, axis=axis)
        return trans_emb * (1 / length)

    @torch.no_grad()
    def sample_from_model(self, src_tokens, src_lengths, prev_output_tokens):
        self.generator.eval()
        self.generator.decoder.train()
        net_input = {"src_tokens": src_tokens, "src_lengths": src_lengths, "prev_output_tokens": prev_output_tokens}
        res, scores = self.sampler_train.generate([self.generator], {"net_input": net_input})
        cand_ids = torch.stack([torch.stack(cand) for cand in res])
        cand_mask = (cand_ids != self.pad_id).long()
        cand_len = torch.sum(cand_mask, dim=-1)
        max_len = torch.max(cand_len).item()
        self.generator.train()
        return cand_ids[:, :, :max_len]

    def pad2max_len(self, input_tensor, max_len):
        pad_size = max_len - input_tensor.shape[-1]
        pad_tensor = torch.ones([input_tensor.shape[0], input_tensor.shape[1], pad_size],
                                device=input_tensor.device).long()
        return torch.cat([input_tensor, pad_tensor], dim=-1)

    @torch.no_grad()
    def inference(self, sample):
        self.generator.eval()
        if self.args.keep_dropout == 1:
            self.generator.decoder.dropout_module.apply_during_inference = True
        res, scores = self.sampler_test.generate([self.generator], sample)
        cand_ids = torch.stack([torch.stack(cand) for cand in res])
        scores = torch.stack([torch.stack(score) for score in scores])

        if self.args.warmup == 0:
            cand_mask = (cand_ids != self.pad_id).long()
            cand_len = torch.sum(cand_mask, dim=-1)
            max_len = torch.max(cand_len).item()
            cand_ids = cand_ids[:, :, :max_len]
            src_tokens = sample["net_input"]["src_tokens"]
            src_pad_mask = (src_tokens != self.pad_id).long()
            encoder_out = self.generator.encoder(src_tokens, sample["net_input"]["src_lengths"])
            encoder_hidden_states = encoder_out["encoder_out"][0]  # src_len x batch x hidden
            encoder_feature = self.affine_transformation(encoder_hidden_states, src_pad_mask, 0)  # batch x h
            decoder_hidden_states = []
            for sample_idx in range(cand_ids.size(1)):
                sampled_input_dec = cand_ids[:, sample_idx]
                decoder_out = self.generator.decoder(sampled_input_dec, encoder_out, features_only=True)
                tgt_pad_mask = (sampled_input_dec != self.pad_id).long()
                decoder_feature = self.affine_transformation(decoder_out[0], tgt_pad_mask, 1)  # batch x h
                decoder_hidden_states.append(decoder_feature.unsqueeze(1))
            decoder_feature = torch.cat(decoder_hidden_states, dim=1)  # batch x sample_num x h
            cos_distance = torch.cosine_similarity(encoder_feature.unsqueeze(1), decoder_feature,
                                                   dim=-1)  # batch x sample_num
            normalize = torch.sum(0 - scores, keepdim=True, dim=-1)
            scores = (1 - self.args.alpha) * (scores / normalize) + self.args.alpha * cos_distance

        max_indices = torch.argmax(scores, dim=-1)[:, None, None]
        dummy = max_indices.repeat(1, 1, cand_ids.size(2))
        self.generator.train()
        return torch.gather(cand_ids, 1, dummy).squeeze(1)  # batch x seq_len

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        """
        cos_score distance of hypothesis to source
        bleu its actual bleu score
        """
        batch_size = src_tokens.size(0)
        src_pad_mask = ~(src_tokens == self.pad_id)

        encoder = self.generator.encoder
        decoder = self.generator.decoder
        encoder_out = encoder(src_tokens, src_lengths)
        encoder_hidden_states = encoder_out["encoder_out"][0]  # src_len x batch x hidden
        decoder_out = decoder(prev_output_tokens, encoder_out, features_only=True)
        decoder_hidden_states = decoder_out[0]  # batch x tgt_len x hidden
        lm_logits = decoder.output_layer(decoder_hidden_states)

        if self.args.warmup == 1:
            decoder_out = list(decoder_out)
            decoder_out[0] = lm_logits
            decoder_out = tuple(decoder_out)
            return decoder_out

        cand_ids = self.sample_from_model(src_tokens, src_lengths, prev_output_tokens)  # batch x beam_size x seq_len
        # prepare contrastive learning
        samples_from_batch = prev_output_tokens[None, :, :].repeat(batch_size, 1, 1)
        cand_len = cand_ids.size(2)
        samples_len = samples_from_batch.size(2)
        if samples_len < cand_len:
            samples_from_batch = self.pad2max_len(samples_from_batch, cand_len)
        else:
            samples_from_batch = samples_from_batch[:, :, :cand_len]

        samples_all = torch.cat([cand_ids, samples_from_batch], dim=1)  # batch x total_sample_num x seq_len
        bleu_distance = self.torch_bleu(prev_output_tokens, samples_all)  # batch x total_sample_num
        bleu_mask = (bleu_distance < 0.5)  # use to mask the gold
        bleu_distance_masked = bleu_distance * bleu_mask.float()
        sample_num = min(self.args.max_sample_num - 1, bleu_distance_masked.size(1) - 1)
        bleu_distance, bleu_indices = torch.sort(bleu_distance_masked, dim=-1, descending=True)
        sampled_bleu_distance = bleu_distance[:, :sample_num]
        sampled_bleu_indices = bleu_indices[:, :sample_num]
        self_indices = torch.arange(0, batch_size).reshape(batch_size, 1).to(
            sampled_bleu_indices.device) + cand_ids.size(1)  # manually add gold
        sampled_indices = torch.cat([self_indices, sampled_bleu_indices], dim=-1)
        self_bleu = torch.full([batch_size, 1], 0.5, device=sampled_bleu_distance.device)
        sampled_bleu_distance = torch.cat([self_bleu, sampled_bleu_distance], dim=-1)
        dummy = sampled_indices.unsqueeze(-1).repeat(1, 1, samples_all.size(2))
        sampled_input = torch.gather(samples_all, 1, dummy)  # batch x sample_num x seq_len
        encoder_feature = self.affine_transformation(encoder_hidden_states, src_pad_mask, 0)  # batch x h

        decoder_hidden_states = []
        for sample_idx in range(sampled_indices.size(-1)):
            sampled_input_dec = sampled_input[:, sample_idx, :self.args.max_sample_len]
            decoder_out = decoder(sampled_input_dec, encoder_out, features_only=True)
            tgt_pad_mask = ~(sampled_input_dec == self.pad_id)
            decoder_feature = self.affine_transformation(decoder_out[0], tgt_pad_mask, 1)  # batch x h
            decoder_hidden_states.append(decoder_feature.unsqueeze(1))

        decoder_feature = torch.cat(decoder_hidden_states, dim=1)  # batch x sample_num x h
        cos_distance = torch.cosine_similarity(encoder_feature.unsqueeze(1), decoder_feature,
                                               dim=-1)  # batch x samle_num

        cl_loss = self.ranking_loss(cos_distance, sampled_bleu_distance)
        decoder_out = list(decoder_out)
        decoder_out[0] = lm_logits
        decoder_out[1]["cl_loss"] = cl_loss
        decoder_out = tuple(decoder_out)
        return decoder_out


@register_model_architecture("transformer_cont", "transformer_cont_wmt")
def transformer_cont_wmt(args):
    transformer_wmt_en_de(args)


@register_model_architecture("transformer_cont", "transformer_cont_iwslt")
def transformer_cont(args):
    transformer_iwslt_de_en(args)

