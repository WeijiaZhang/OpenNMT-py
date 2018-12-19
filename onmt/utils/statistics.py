""" Statistics calculation utility """
from __future__ import division
import time
import math
import sys
import re

from torch.distributed import get_rank
from onmt.utils.distributed import all_gather_list
from onmt.utils.logging import logger
from onmt.utils.g_rouge import rouge


class RougeCompute(object):
    """
    Calculating google rouge evaluation for text sunmmarization
    """

    def __init__(self, padding_idx, bos_tag_idx, end_tag_idx):
        self.padding_idx = padding_idx
        self.bos_tag_idx = bos_tag_idx
        self.end_tag_idx = end_tag_idx

    @staticmethod
    def get_rouge_keys():
        keys = ['rouge_1/r_score', 'rouge_1/p_score', 'rouge_1/f_score',
                'rouge_2/r_score', 'rouge_2/p_score', 'rouge_2/f_score',
                'rouge_l/r_score', 'rouge_l/p_score', 'rouge_l/f_score']
        return keys

    def split_sentences(self, arr):
        """
        spliting sentences using given sentence tags
        ignore padding_idx
        """
        sentences = []
        bos_flag = False
        sent = []
        for ele in arr:
            if ele == self.padding_idx:
                continue
            elif ele == self.bos_tag_idx:
                bos_flag = True
            elif ele == self.end_tag_idx:
                if len(sent) > 1:
                    sent = " ".join(sent)
                    sentences.append(sent)
                bos_flag = False
                sent = []
            elif bos_flag:
                sent.append(str(ele))

        return sentences

    def generate_data(self, preds, target):
        """
        generating data format for google rouge evaluation
        """
        summaries = []
        references = []
        for pred_ele, tgt_ele in zip(preds, target):
            summary = self.split_sentences(pred_ele)
            reference = self.split_sentences(tgt_ele)
            if len(summary) > 0 and len(reference) > 0:
                summaries.append(summary)
                references.append(reference)
        return summaries, references

    def cal_rouge(self, summaries, references):
        g_scores = rouge(summaries, references)
        return g_scores


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """

    def __init__(self, loss=0, n_words=0, n_correct=0, rouge_scores=None):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.rouge_keys = RougeCompute.get_rouge_keys()
        if rouge_scores is None:
            self.rouge_scores = dict()
            for key in self.rouge_keys:
                self.rouge_scores[key] = 0.0

        self.start_time = time.time()

    @staticmethod
    def all_gather_stats(stat, max_size=4096):
        """
        Gather a `Statistics` object accross multiple process/nodes

        Args:
            stat(:obj:Statistics): the statistics object to gather
                accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            `Statistics`, the update stats object
        """
        stats = Statistics.all_gather_stats_list([stat], max_size=max_size)
        return stats[0]

    @staticmethod
    def all_gather_stats_list(stat_list, max_size=4096):
        """
        Gather a `Statistics` list accross all processes/nodes

        Args:
            stat_list(list([`Statistics`])): list of statistics objects to
                gather accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            our_stats(list([`Statistics`])): list of updated stats
        """
        # Get a list of world_size lists with len(stat_list) Statistics objects
        all_stats = all_gather_list(stat_list, max_size=max_size)

        our_rank = get_rank()
        our_stats = all_stats[our_rank]
        for other_rank, stats in enumerate(all_stats):
            if other_rank == our_rank:
                continue
            for i, stat in enumerate(stats):
                our_stats[i].update(stat, update_n_src_words=True)
        return our_stats

    def update(self, stat, update_n_src_words=False, update_rouge=False):
        """
        Update statistics by suming values with another `Statistics` object

        Args:
            stat: another statistic object
            update_n_src_words(bool): whether to update (sum) `n_src_words`
                or not

        """
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

        if update_n_src_words:
            self.n_src_words += stat.n_src_words
        if update_rouge:
            for key in self.rouge_keys:

    def rouge_n_r_score(self, n='1'):
        return 100 * self.rouge_scores['rouge_%s/r_score' % n]

    def rouge_n_p_score(self, n='1'):
        return 100 * self.rouge_scores['rouge_%s/p_score' % n]

    def rouge_n_f_score(self, n='1'):
        return 100 * self.rouge_scores['rouge_%s/f_score' % n]

    def accuracy(self):
        """ compute accuracy """
        return 100 * (self.n_correct / self.n_words)

    def xent(self):
        """ compute cross entropy """
        return self.loss / self.n_words

    def ppl(self):
        """ compute perplexity """
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        """ compute elapsed time """
        return time.time() - self.start_time

    def output(self, step, num_steps, learning_rate, start):
        """Write out statistics to stdout.

        Args:
           step (int): current step
           n_batch (int): total batches
           start (int): start time of step.
        """
        t = self.elapsed_time()
        logger.info(
            ("Step %2d/%5d; acc: %6.2f; ppl: %5.2f; xent: %4.2f; " +
             "lr: %7.5f; %3.0f/%3.0f tok/s; %6.0f sec")
            % (step, num_steps,
               self.accuracy(),
               self.ppl(),
               self.xent(),
               learning_rate,
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log_tensorboard(self, prefix, writer, learning_rate, step):
        """ display statistics to tensorboard """
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", self.xent(), step)
        writer.add_scalar(prefix + "/ppl", self.ppl(), step)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), step)
        writer.add_scalar(prefix + "/tgtper", self.n_words / t, step)
        writer.add_scalar(prefix + "/lr", learning_rate, step)
