import torch
import torch.nn as nn

import onmt.inputters as inputters
from onmt.utils.misc import aeq
from onmt.utils.loss import LossComputeBase
from onmt.utils.statistics import RougeCompute, Statistics


class CopyGenerator(nn.Module):
    """An implementation of pointer-generator networks (See et al., 2017)
    (https://arxiv.org/abs/1704.04368), which consider copying words
    directly from the source sequence.

    The copy generator is an extended version of the standard
    generator that computes three values.

    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of copying a word from
      the source
    * :math:`p_{copy}` the probility of copying a particular word.
      taken from the attention distribution directly.

    The model returns a distribution over the extend dictionary,
    computed as

    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`


    .. mermaid::

       graph BT
          A[input]
          S[src_map]
          B[softmax]
          BB[switch]
          C[attn]
          D[copy]
          O[output]
          A --> B
          A --> BB
          S --> D
          C --> D
          D --> O
          B --> O
          BB --> O


    Args:
       input_size (int): size of input representation
       output_size (int): size of output vocabulary
       pad_idx (int)
    """

    def __init__(self, input_size, output_size, pad_idx):
        super(CopyGenerator, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.linear_copy = nn.Linear(input_size, 1)
        self.pad_idx = pad_idx

    def forward(self, hidden, attn, src_map):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by compying
        source words.

        Args:
           hidden (`FloatTensor`): hidden outputs `[batch*tlen, input_size]`
           attn (`FloatTensor`): attn for each `[batch*tlen, input_size]`
           src_map (`FloatTensor`):
             A sparse indicator matrix mapping each source word to
             its index in the "extended" vocab containing.
             `[src_len, batch, extra_words]`
        Return:
            out_prob (`FloatTensor`): output probability `[batch*tlen, input_size]`
            copy_prob (`FloatTensor`): copy probability `[batch*tlen, input_size]`
        """
        # CHECKS
        batch_by_tlen, _ = hidden.size()
        batch_by_tlen_, slen = attn.size()
        slen_, batch, cvocab = src_map.size()
        aeq(batch_by_tlen, batch_by_tlen_)
        aeq(slen, slen_)

        # Original probabilities.
        logits = self.linear(hidden)
        logits[:, self.pad_idx] = -float('inf')
        prob = torch.softmax(logits, 1)

        # Probability of copying p(z=1) batch.
        p_copy = torch.sigmoid(self.linear_copy(hidden))
        # Probability of not copying: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(prob, 1 - p_copy)
        mul_attn = torch.mul(attn, p_copy)
        copy_prob = torch.bmm(
            mul_attn.view(-1, batch, slen).transpose(0, 1),
            src_map.transpose(0, 1)
        ).transpose(0, 1)
        copy_prob = copy_prob.contiguous().view(-1, cvocab)
        return torch.cat([out_prob, copy_prob], 1)


class CopyGeneratorLoss(nn.Module):
    """ Copy generator criterion """

    def __init__(self, vocab_size, force_copy, unk_index=0,
                 ignore_index=-100, eps=1e-20, end_token_idx=-100,
                 with_tag=False, bos_tag_idx=-100, end_tag_idx=-100, end_sent_idx=-100):
        super(CopyGeneratorLoss, self).__init__()
        self.force_copy = force_copy
        self.eps = eps
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.unk_index = unk_index
        self.end_token_idx = end_token_idx

        self.with_tag = with_tag
        self.bos_tag_idx = bos_tag_idx
        self.end_tag_idx = end_tag_idx
        self.end_sent_idx = end_sent_idx

    def forward(self, scores, align, target):
        """
        scores (FloatTensor): (batch_size*tgt_len) x dynamic vocab size
        align (LongTensor): (batch_size*tgt_len)
        target (LongTensor): (batch_size*tgt_len)
        """
        # probabilities assigned by the model to the gold targets
        vocab_probs = scores.gather(1, target.unsqueeze(1)).squeeze(1)

        # probability of tokens copied from source
        copy_ix = align.unsqueeze(1) + self.vocab_size
        copy_tok_probs = scores.gather(1, copy_ix).squeeze(1)
        # Set scores for unk to 0 and add eps
        copy_tok_probs[align == self.unk_index] = 0
        copy_tok_probs += self.eps  # to avoid -inf logs

        # find the indices in which you do not use the copy mechanism
        non_copy = align == self.unk_index
        if not self.force_copy:
            non_copy = non_copy | (target != self.unk_index)

        probs = torch.where(
            non_copy, copy_tok_probs + vocab_probs, copy_tok_probs
        )

        loss = -probs.log()  # just NLLLoss; can the module be incorporated?
        # Drop padding.
        loss[target == self.ignore_index] = 0
        return loss


class CopyGeneratorLossCompute(LossComputeBase):
    """
    Copy Generator Loss Computation.
    """

    def __init__(self, criterion, generator, tgt_vocab, normalize_by_length):
        super(CopyGeneratorLossCompute, self).__init__(criterion, generator)
        self.tgt_vocab = tgt_vocab
        self.normalize_by_length = normalize_by_length
        self.rouge_obj = RougeCompute(
            self.padding_idx, self.end_token_idx,
            with_tag=self.with_tag, end_sent_idx=self.end_sent_idx,
            bos_tag_idx=self.bos_tag_idx, end_tag_idx=self.end_tag_idx
        )

    @property
    def with_tag(self):
        return self.criterion.with_tag

    @property
    def end_sent_idx(self):
        return self.criterion.end_sent_idx

    @property
    def end_token_idx(self):
        return self.criterion.end_token_idx

    @property
    def bos_tag_idx(self):
        return self.criterion.bos_tag_idx

    @property
    def end_tag_idx(self):
        return self.criterion.end_tag_idx

    def _cal_rouge(self, pred_arr, target_arr):
        summaries, references = self.rouge_obj.generate_data(
            pred_arr, target_arr)
        # import pdb
        # pdb.set_trace()
        r_scores = self.rouge_obj.cal_rouge(summaries, references)
        return r_scores

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        batch_size = scores.size(1)
        scores = self._bottle(scores)
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target).masked_select(non_padding).sum().item()
        num_non_padding = non_padding.sum().item()

        # calculating rouge
        pred_arr = pred.view(-1, batch_size).contiguous().transpose(0, 1)
        target_arr = target.view(-1, batch_size).contiguous().transpose(0, 1)
        pred_arr = pred_arr.cpu().numpy()
        target_arr = target_arr.cpu().numpy()
        r_scores = self._cal_rouge(pred_arr, target_arr)

        stats = Statistics(loss.item(), num_non_padding, num_correct, r_scores)
        return stats

    def _make_shard_state(self, batch, output, range_, attns):
        """ See base class for args description. """
        if getattr(batch, "alignment", None) is None:
            raise AssertionError("using -copy_attn you need to pass in "
                                 "-dynamic_dict during preprocess stage.")

        return {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1]],
            "copy_attn": attns.get("copy"),
            "align": batch.alignment[range_[0] + 1: range_[1]]
        }

    def _compute_loss(self, batch, output, target, copy_attn, align):
        """
        Compute the loss. The args must match self._make_shard_state().
        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            copy_attn: the copy attention value.
            align: the align info.
        """
        target = target.view(-1)
        align = align.view(-1)
        scores = self.generator(
            self._bottle(output), self._bottle(copy_attn), batch.src_map
        )
        loss = self.criterion(scores, align, target)

        # this block does not depend on the loss value computed above
        # and is used only for stats
        scores_data = inputters.TextDataset.collapse_copy_scores(
            self._unbottle(scores.clone(), batch.batch_size),
            batch, self.tgt_vocab, batch.dataset.src_vocabs)
        # scores_data = self._bottle(scores_data)

        # this block does not depend on the loss value computed above
        # and is used only for stats
        # Correct target copy token instead of <unk>
        # tgt[i] = align[i] + len(tgt_vocab)
        # for i such that tgt[i] == 0 and align[i] != 0
        target_data = target.clone()
        unk = self.criterion.unk_index
        correct_mask = (target_data == unk) & (align != unk)
        offset_align = align[correct_mask] + len(self.tgt_vocab)
        target_data[correct_mask] += offset_align

        # Compute sum of perplexities for stats
        stats = self._stats(loss.sum().clone(), scores_data, target_data)

        # this part looks like it belongs in CopyGeneratorLoss
        if self.normalize_by_length:
            # Compute Loss as NLL divided by seq length
            tgt_lens = batch.tgt.ne(self.padding_idx).sum(0).float()
            # Compute Total Loss per sequence in batch
            loss = loss.view(-1, batch.batch_size).sum(0)
            # Divide by length of each sequence and sum
            loss = torch.div(loss, tgt_lens).sum()
        else:
            loss = loss.sum()

        return loss, stats
