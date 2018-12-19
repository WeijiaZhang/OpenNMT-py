# -*- encoding: utf-8 -*-
import argparse
import re
import os
import time
import pyrouge
import shutil
import sys
import codecs

from g_rouge import rouge
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)
from onmt.utils.logging import init_logger, logger


def split_sentences(article, sentence_start_tag='<t>', sentence_end_tag='</t>'):
    bare_sents = re.findall(r'%s (.+?) %s' %
                            (sentence_start_tag, sentence_end_tag), article)
    return bare_sents

# convenient decorator


def register_to_registry(registry):
    def _register(func):
        registry[func.__name__] = func
        return func
    return _register


baseline_registry = {}
register = register_to_registry(baseline_registry)

# baseline methods


@register
def all_sentences(article, sentence_start_tag='<t>', sentence_end_tag='</t>'):
    sents = split_sentences(article, sentence_start_tag, sentence_end_tag)
    return sents


@register
def first_sentence(article, sentence_start_tag='<t>', sentence_end_tag='</t>'):
    ''' use sentence tags to output the first sentence of an article as its summary. '''
    sents = split_sentences(article, sentence_start_tag, sentence_end_tag)
    return sents[:1]


@register
def first_three_sentences(article, sentence_start_tag='<t>', sentence_end_tag='</t>'):
    sents = split_sentences(article, sentence_start_tag, sentence_end_tag)
    return sents[:3]


@register
def first_two_sentences(article, sentence_start_tag='<t>', sentence_end_tag='</t>'):
    sents = split_sentences(article, sentence_start_tag, sentence_end_tag)
    return sents[:2]


@register
def verbatim(article, sentence_start_tag='<t>', sentence_end_tag='</t>'):
    sents = split_sentences(article, sentence_start_tag, sentence_end_tag)
    return sents


@register
def pre_sent_tag_verbatim(article):
    sents = article.split('<t>')
    good_sents = []
    for sent in sents:
        sent = sent.strip()
        if len(sent.split()) > 0:
            good_sents.append(sent)
    # print(good_sents)
    return good_sents


@register
def sent_tag_verbatim(article):
    sents = split_sentences(article, '<t>', '</t>')
    # print(sents)
    return sents


@register
def sent_tag_p_verbatim(article):
    bare_article = article.strip()
    bare_article += ' </t>'
    sents = split_sentences(bare_article, '<t>', '</t>')
    # print(sents)
    return sents


@register
def adhoc_old0(article):
    sents = split_sentences(article, '<t>', '</t>')
    good_sents = []
    for sent in sents:
        # Remove <unk>
        tokens = [x for x in sent.split() if x != '<unk>']
        # Ignore length 1 sententces
        if len(tokens) > 1:
            good_sents.append(' '.join(tokens))
    return good_sents


@register
def full(article):
    return [article]


@register
def adhoc_base(article):
    article += ' </t> </t>'
    first_end = article.index(' </t> </t>')
    article = article[:first_end] + ' </t>'
    sents = split_sentences(article)
    good_sents = []
    for sent in sents:
        # Remove <unk>
        tokens = [x for x in sent.split() if x != '<unk>']
        # Ignore length 1 sententces
        if len(tokens) > 1:
            good_sents.append(' '.join(tokens))
    return good_sents


@register
def no_sent_tag(article):
    article = article.strip()
    try:
        if article[-1] != '.':
            article += ' .'
    except:
        article += ' .'
    good_sents = list(re.findall(r'.+?\.', article))
    return good_sents


@register
def second_sentence(article, sentence_start_tag='<t>', sentence_end_tag='</t>'):
    sents = split_sentences(article, sentence_start_tag, sentence_end_tag)
    return sents[1:2]


def test_rouge(summaries, references, rouge_args=[]):
    """Calculate ROUGE scores of sequences passed as an iterator
       e.g. a list of str, an open file, StringIO or even sys.stdin
    """
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = ".rouge-tmp-{}".format(current_time)
    system_dir = os.path.join(tmp_dir, 'system')
    model_dir = os.path.join(tmp_dir, 'model')
    args_str = ' '.join(map(str, rouge_args))
    try:
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
            os.mkdir(system_dir)
            os.mkdir(model_dir)

        candidates = [" ".join(ele) for ele in summaries]
        references = [" ".join(ele[0]) for ele in references]
        assert len(candidates) == len(references)
        cnt = len(candidates)
        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(os.path.join(system_dir, "cand.%i.txt" % i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(os.path.join(model_dir, "ref.%i.txt" % i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])
        r = pyrouge.Rouge155()
        r.system_dir = system_dir
        r.model_dir = model_dir
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate()
        results_dict = r.output_to_dict(rouge_results)
        return results_dict
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)


def eval_rouge_by_sentence(summaries, references, rouge_args=[]):
    '''
    Args:
        summaries: [[sentence]]. Each summary is a list of strings (sentences)
        references: [[[sentence]]]. Each reference is a list of candidate summaries.
        rouge_args: [string]. A list of arguments to pass to the ROUGE CLI.
    '''
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = ".rouge-tmp-{}".format(current_time)
    system_dir = os.path.join(tmp_dir, 'system')
    model_dir = os.path.join(tmp_dir, 'model')
    args_str = ' '.join(map(str, rouge_args))
    try:
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
            os.mkdir(system_dir)
            os.mkdir(model_dir)
        assert len(summaries) == len(references)

        for i, (summary, candidates) in enumerate(zip(summaries, references)):
            summary_file = '%i.txt' % i
            for j, candidate in enumerate(candidates):
                candidate_file = '%i.%i.txt' % (i, j)
                with open(os.path.join(model_dir, candidate_file), 'w', encoding="utf-8") as f:
                    f.write('\n'.join(candidate))

            with open(os.path.join(system_dir, summary_file), 'w', encoding="utf-8") as f:
                f.write('\n'.join(summary))

        r = pyrouge.Rouge155()
        r.system_dir = system_dir
        r.model_dir = model_dir
        r.system_filename_pattern = '(\d+).txt'
        r.model_filename_pattern = '#ID#.\d+.txt'
        rouge_results = r.convert_and_evaluate()
        results_dict = r.output_to_dict(rouge_results)
        return results_dict
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)


def print_rouge_results(results_dict):
    for idx in ['1', '2', 'l']:
        head_prefix = "ROUGE-%s" % idx
        val_prefix = "rouge_%s" % idx
        res = ">> {}-R: {:.2f}, {}-P: {:.2f}, {}-F: {:.2f}".format(
            head_prefix, results_dict["%s_recall" % val_prefix] * 100,
            head_prefix, results_dict["%s_precision" % val_prefix] * 100,
            head_prefix, results_dict["%s_f_score" % val_prefix] * 100)
        logger.info(res)
    return res


def print_goolge_rouge(summaries, references):
    n_target = len(references)
    t0 = time.time()
    g_scores = rouge(summaries, [candidates[0] for candidates in references])
    dt = time.time() - t0

    # g_headers = ['rouge_1/r_score', 'rouge_1/p_score', 'rouge_1/f_score', 'rouge_2/r_score',
    #              'rouge_2/p_score', 'rouge_2/f_score', 'rouge_l/r_score', 'rouge_l/p_score', 'rouge_l/f_score']

    # print(g_headers)
    for idx in ['1', '2', 'l']:
        head_prefix = "ROUGE-%s" % idx
        val_prefix = "rouge_%s" % idx
        res = ">> {}-R: {:.2f}, {}-P: {:.2f}, {}-F: {:.2f}".format(
            head_prefix, g_scores["%s/r_score" % val_prefix] * 100,
            head_prefix, g_scores["%s/p_score" % val_prefix] * 100,
            head_prefix, g_scores["%s/f_score" % val_prefix] * 100)
        logger.info(res)

    logger.info('>> evaluated {} samples, took {:.3f}s, averaging {:.3f}s/sample'.format(
        n_target, dt, dt / (n_target + 1e-8)))


def main():
    init_logger('test_rouge.log')
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', required=True,
                        help='Path to the tokenized source file. One sample per line with sentence tags.')
    parser.add_argument('-t', '--target', required=True,
                        help='Path to the tokenized target file. One sample per line with sentence tags.')
    parser.add_argument('-ms', '--method-src', required=False, default='all_sentences',
                        choices=baseline_registry.keys(), help='Baseline method for source to use.')
    parser.add_argument('-mt', '--method-tgt', required=False, default='all_sentences',
                        choices=baseline_registry.keys(), help='Baseline method for target to use.')
    parser.add_argument('--no-stemming', dest='stemming',
                        action='store_false', help='Turn off stemming in ROUGE.')
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                        help='The number of bootstrap samples used in ROUGE.')
    parser.add_argument('-g', '--google', dest='run_google_rouge', action='store_true',
                        help='Evaluate with the ROUGE implementation from google/seq2seq.')
    args = parser.parse_args()

    process_src = baseline_registry[args.method_src]
    process_tgt = baseline_registry[args.method_tgt]

    source_file = codecs.open(args.source, encoding="utf-8")
    target_file = codecs.open(args.target, encoding="utf-8")

    # Read and preprocess generated summary
    n_source = 0
    references = []
    summaries = []
    for i, article in enumerate(source_file):
        summary = process_src(article)
        summaries.append(summary)
        n_source += 1

    n_target = 0
    for i, article in enumerate(target_file):
        candidate = process_tgt(article)
        references.append([candidate])
        n_target += 1

    source_file.close()
    target_file.close()
    assert n_source == n_target, 'Source and target must have the same number of samples.'
    rouge_args = [
        '-c', 95,  # 95% confidence intervals, necessary for the dictionary conversion routine
        '-n', 2,  # up to bigram
        '-a',
        '-r', args.n_bootstrap,  # the number of bootstrap samples for confidence bounds
    ]
    if args.stemming:
        # add the stemming flag
        rouge_args += ['-m']

    if args.run_google_rouge:
        print_goolge_rouge(summaries, references)
    else:
        t0 = time.time()
        # evaluate with official ROUGE script v1.5.5
        # results_dict = test_rouge(summaries, references, rouge_args=rouge_args)
        results_dict = eval_rouge_by_sentence(
            summaries, references, rouge_args=rouge_args)

        dt = time.time() - t0

        logger.info('>> method_src: %s, method_tgt: %s' %
                    (args.method_src, args.method_tgt))
        print_rouge_results(results_dict)
        logger.info('>> evaluated {} samples, took {:.3f}s, averaging {:.3f}s/sample'.format(
            n_target, dt, dt / (n_target + 1e-3)))


if __name__ == "__main__":
    main()
