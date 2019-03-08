#    Copyright 2018 AimBrain Ltd.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import json
import collections
import argparse
import string
from tqdm import tqdm
from spacy.tokenizer import Tokenizer
import en_core_web_sm

try:
    import cPickle as pickle
except:
    import pickle

nlp = en_core_web_sm.load()
tokenizer = Tokenizer(nlp.vocab)
exclude = set(string.punctuation)


def process_answers(q, phase, n_answers=3000):

    # find the n_answers most common answers
    counts = {}
    for row in q:
        counts[row['answer']] = counts.get(row['answer'], 0) + 1

    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)

    vocab = [w for c, w in cw[:n_answers]]

    # a 0-indexed vocabulary translation table
    itow = {i: w for i, w in enumerate(vocab)}
    wtoi = {w: i for i, w in enumerate(vocab)}  # inverse table
    pickle.dump({'itow': itow, 'wtoi': wtoi}, open(phase + '_a_dict.p', 'wb'))

    for row in q:
        accepted_answers = 0
        for w, c in row['answers']:
            if w in vocab:
                accepted_answers += c

        answers_scores = []
        for w, c in row['answers']:
            if w in vocab:
                answers_scores.append((w, c / accepted_answers))

        row['answers_w_scores'] = answers_scores

    json.dump(q, open('vqa_' + phase + '_final_3000.json', 'w'))


def process_questions(q):
    # build question dictionary
    def build_vocab(questions):
        count_thr = 0
        # count up the number of times a word is used
        counts = {}
        for row in questions:
            for word in row['question_toked']:
                counts[word] = counts.get(word, 0) + 1
        cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
        print('top words and their counts:')
        print('\n'.join(map(str, cw[:10])))

        # print some stats
        total_words = sum(counts.values())
        print('total words:', total_words)
        bad_words = [w for w, n in counts.items() if n <= count_thr]
        vocab = [w for w, n in counts.items() if n > count_thr]
        bad_count = sum(counts[w] for w in bad_words)
        print('number of bad words: %d/%d = %.2f%%' %
              (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
        print('number of words in vocab would be %d' % (len(vocab), ))
        print('number of UNKs: %d/%d = %.2f%%' %
              (bad_count, total_words, bad_count*100.0/total_words))

        return vocab

    vocab = build_vocab(q)
    # a 1-indexed vocab translation table
    itow = {i+1: w for i, w in enumerate(vocab)}
    wtoi = {w: i+1 for i, w in enumerate(vocab)}  # inverse table
    pickle.dump({'itow': itow, 'wtoi': wtoi}, open(phase + '_q_dict.p', 'wb'))


def tokenize_questions(qa, phase):
    qas = len(qa)
    for i, row in enumerate(tqdm(qa)):
        row['question_toked'] = [t.text if '?' not in t.text else t.text[:-1]
                                 for t in tokenizer(row['question'].lower())]  # get spacey tokens and remove question marks
        if i == qas - 1:
            json.dump(qa, open('vqa_' + phase + '_toked.json', 'w'))


def combine_qa(questions, annotations, phase):
    # Combine questions and answers in the same json file
    # 443757 questions
    data = []
    for i, q in enumerate(tqdm(questions['questions'])):
        row = {}
        # load questions info
        row['question'] = q['question']
        row['question_id'] = q['question_id']
        row['image_id'] = str(q['image_id'])

        # load answers
        assert q['question_id'] == annotations[i]['question_id']
        row['answer'] = annotations[i]['multiple_choice_answer']

        answers = []
        for ans in annotations[i]['answers']:
            answers.append(ans['answer'])
        row['answers'] = collections.Counter(answers).most_common()

        data.append(row)

    json.dump(data, open('vqa_' + phase + '_combined.json', 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        description='Preprocessing for VQA v2 text data')
    parser.add_argument('--data', nargs='+', help='train, val and/or test, list of data phases to be processed', required=True)
    parser.add_argument('--nanswers', default=3000, help='number of top answers to consider for classification.')
    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0:
        raise SystemExit('Unknown argument: {}'.format(unparsed))

    phase_list = args.data

    for phase in phase_list:

        print('processing ' + phase + ' data')
        if phase != 'test':
            # Combine Q and A
            print('Combining question and answer...')
            question = json.load(
                open('raw/v2_OpenEnded_mscoco_' + phase + '2014_questions.json'))
            answers = json.load(open('raw/v2_mscoco_' + phase + '2014_annotations.json'))
            combine_qa(question, answers['annotations'], phase)

            # Tokenize
            print('Tokenizing...')
            t = json.load(open('vqa_' + phase + '_combined.json'))
            tokenize_questions(t, phase)
        else:
            print ('Tokenizing...')
            t = json.load(open('raw/v2_OpenEnded_mscoco_' + phase + '2015_questions.json'))
            t = t['questions']
            tokenize_questions(t, phase)

        # Build dictionary for question and answers
        print('Building dictionary...')
        t = json.load(open('vqa_' + phase + '_toked.json'))
        if phase == 'train':
            process_questions(t)
        if phase != 'test':
            process_answers(t, phase, n_answers=args.nanswers)

    print('Done')
