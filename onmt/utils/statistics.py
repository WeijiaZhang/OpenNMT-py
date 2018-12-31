""" Statistics calculation utility """
from __future__ import division
import time
import math
import sys
import re

from torch.distributed import get_rank
from onmt.utils.distributed import all_gather_list
from onmt.utils.logging import logger
from onmt.utils import rouge_score


class RougeCompute(object):
    """
    Calculating google rouge evaluation for text sunmmarization
    """

    def __init__(self, padding_idx, end_token_idx,
                 with_tag=False, end_sent_idx=-100,
                 bos_tag_idx=-100, end_tag_idx=-100):
        self.padding_idx = padding_idx
        self.end_token_idx = end_token_idx

        self.with_tag = with_tag
        self.bos_tag_idx = bos_tag_idx
        self.end_tag_idx = end_tag_idx
        self.end_sent_idx = end_sent_idx
        if with_tag:
            self.split_func = self.split_sentences_with_tag
        else:
            self.split_func = self.split_sentences_no_tag

    @staticmethod
    def get_rouge_keys():
        keys = ['rouge_1/r_score', 'rouge_1/p_score', 'rouge_1/f_score',
                'rouge_2/r_score', 'rouge_2/p_score', 'rouge_2/f_score',
                'rouge_l/r_score', 'rouge_l/p_score', 'rouge_l/f_score']
        return keys

    @staticmethod
    def init_rouge_scores():
        rouge_keys = RougeCompute.get_rouge_keys()
        rouge_scores = dict()
        for key in rouge_keys:
            rouge_scores[key] = 0.0
        return rouge_scores

    def split_sentences_with_tag(self, arr):
        """
        spliting sentences using given sentence tags
        ignore padding_idx
        """
        sentences = []
        bos_flag = False
        sent = []
        # counting element that has visited in for loop
        i = 0
        for ele in arr:
            if ele == self.padding_idx or ele == self.end_token_idx:
                i += 1
                continue
            elif ele == self.bos_tag_idx:
                i += 1
                bos_flag = True
            elif ele == self.end_tag_idx:
                i += 1
                if len(sent) > 1:
                    sent = " ".join(sent)
                    sentences.append(sent)
                bos_flag = False
                sent = []
            elif bos_flag:
                i += 1
                sent.append(str(ele))

        # last sentence without end tag
        if len(sent) > 1:
            sent = " ".join(sent)
            sentences.append(sent)

        # obtaining tokens that have no right format
        # if i < len(arr):
        #     sent = []
        #     for ele in arr[i:]:
        #         if ele not in [self.padding_idx, self.bos_tag_idx, self.end_tag_idx]:
        #             sent.append(str(ele))

        #     if len(sent) > 1:
        #         sent = " ".join(sent)
        #         sentences.append(sent)

        return sentences

    def split_sentences_no_tag(self, arr):
        """
        spliting sentences without tags
        ignore padding_idx
        """
        sentences = []
        sent = []
        for ele in arr:
            if ele == self.padding_idx or ele == self.end_token_idx:
                continue
            elif ele == self.end_sent_idx:
                sent.append(str(ele))
                if len(sent) > 1:
                    sent = " ".join(sent)
                    sentences.append(sent)
                sent = []
            else:
                sent.append(str(ele))

        # last sentence without end symbol
        if len(sent) > 1:
            sent = " ".join(sent)
            sentences.append(sent)

        return sentences

    def generate_data(self, preds, target):
        """
        generating data format for google rouge evaluation
        """
        summaries = []
        references = []
        # import pdb
        # pdb.set_trace()
        for pred_ele, tgt_ele in zip(preds, target):
            summary = self.split_func(pred_ele)
            reference = self.split_func(tgt_ele)
            if len(summary) > 0 and len(reference) > 0:
                summaries.append(summary)
                references.append(reference)
        return summaries, references

    def cal_rouge(self, summaries, references):
        if len(summaries) > 0 and len(references) > 0:
            r_scores = rouge_score.cal_rouge(summaries, references)
        else:
            r_scores = None
        return r_scores


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
            self.num_rouge = 0
            self.rouge_scores = RougeCompute.init_rouge_scores()
        else:
            self.num_rouge = 1
            self.rouge_scores = rouge_scores

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

    def update(self, stat, update_n_src_words=False, update_rouge=True):
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
            self.num_rouge += stat.num_rouge
            for key in self.rouge_keys:
                self.rouge_scores[key] += stat.rouge_scores[key]

    def rouge_n_r_score(self, n):
        r_score = self.rouge_scores['rouge_%s/r_score' %
                                    n] / (self.num_rouge + 1e-8)
        return 100 * r_score

    def rouge_n_p_score(self, n):
        p_score = self.rouge_scores['rouge_%s/p_score' %
                                    n] / (self.num_rouge + 1e-8)
        return 100 * p_score

    def rouge_n_f_score(self, n):
        f_score = self.rouge_scores['rouge_%s/f_score' %
                                    n] / (self.num_rouge + 1e-8)
        return 100 * f_score

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
        # logger.info(
        #     ("Step %2d/%5d; acc: %6.2f; ppl: %5.2f; xent: %4.2f; " +
        #      "lr: %7.5f; %3.0f/%3.0f tok/s; %6.0f sec")
        #     % (step, num_steps,
        #        self.accuracy(),
        #        self.ppl(),
        #        self.xent(),
        #        learning_rate,
        #        self.n_src_words / (t + 1e-5),
        #        self.n_words / (t + 1e-5),
        #        time.time() - start))
        logger.info(
            ("Step %2d/%5d; Acc: %.2f; PPL: %.2f; ROUGE-1-F: %.2f; ROUGE-2-F: %.2f; ROUGE-L-F: %.2f; " +
             "lr: %.4f; %3.0f/%3.0f tok/s; %4.0f sec")
            % (step, num_steps,
               self.accuracy(),
               self.ppl(),
               self.rouge_n_f_score('1'),
               self.rouge_n_f_score('2'),
               self.rouge_n_f_score('l'),
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
        for n in ['1', '2', 'l']:
            writer.add_scalar(prefix + "/rouge_%s_r_score" % n,
                              self.rouge_n_r_score(n), step)
            writer.add_scalar(prefix + "/rouge_%s_p_score" % n,
                              self.rouge_n_p_score(n), step)
            writer.add_scalar(prefix + "/rouge_%s_f_score" % n,
                              self.rouge_n_f_score(n), step)
