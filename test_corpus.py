import pytest

from spacy import attrs

import corpus as c
from corpus import *

@pytest.fixture(scope="module")
def corpus():
	return c.Corpus('challenge_data/testing_dialog.txt', 'challenge_data/testing_missing.txt', training=True)

@pytest.fixture(scope="module")
def dialog(corpus):
	# take the first dialog in the list.
	return corpus.dialogs.itervalues().next()


# Corpus class
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def test_corpus_load(corpus):
	assert len(corpus.raw_missing) == 2
	assert len(corpus.raw_dialogs) == 2
	assert len(corpus.dialogs) == 2
	assert corpus.length == 2

def test_random_utterance(corpus):
	assert corpus.random_utterance("12345") != corpus.random_utterance("12345") or corpus.random_utterance("12345") != corpus.random_utterance("12345")

def test_stats(corpus):
	dialog, utters = corpus.stats()

	print(dialog, utters)

	assert dialog['min'] == 4
	assert dialog['max'] == 6
	assert dialog['count'] == 10
	assert dialog['mean'] == 5

	assert utters['min'] == 2
	assert utters['max'] == 27
	assert utters['count'] == 106
	assert utters['mean'] == 10

def test_sentence_triples_cbow(corpus):
	training, cross_validation = corpus.sentence_triples_cbow()
	assert len(training) == len(cross_validation) == 2
	assert len(training[0]) != len(cross_validation[0])
	assert len(training[0]) + len(cross_validation[0]) == (4 + 2) * 2

	train, test = training
	assert type(train) == list
	assert type(test) == list

	train, test = cross_validation
	assert type(train) == list
	assert type(test) == list

def test_all_dialog_missing_pair_as(corpus):
	xDialogs, xMissing, Y = corpus.all_dialog_missing_pair_as()
	assert len(xDialogs) == len(xMissing) == len(Y) == 4  # 2 dialogs, positive and negative cases
	assert len(xDialogs[0]) == 140
	assert type(xDialogs[0][0]) == numpy.int32
	assert len(xMissing[0]) == 28
	assert type(xMissing[0][0]) == numpy.int32

def  test_corpus_split(corpus):
	c1, c2  =corpus.split(cv_split=0.5)
	assert len(c1.dialogs) == len(c2.dialogs)

# Static methods...

def test_split_sets():
	(X_train, y_train), (X_test, y_test) = c.Corpus.split_sets(range(101), range(101), 0.2)
	assert len(X_train) + len(X_test) == 101
	assert len(y_train) + len(y_test) == 101
	assert len(X_train) == 80
	assert len(X_train) == len(y_train)
	assert len(X_test) == 21
	assert len(X_test) == len(y_test)


# Dialog class
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def test_dialogs_load(dialog):
	assert len(dialog.raw_dialog) == 5
	assert type(dialog[0]) == spacy.tokens.Doc
	assert type(dialog.missing) == spacy.tokens.Doc
	assert type(dialog.missing[0]) == spacy.tokens.Token
	assert dialog.missing[0].orth_ == "I"

def test_dialogs_complete(dialog):
	assert len(dialog.all) == 6
	assert len(dialog.dialog) == 5
	assert len(dialog.missing) == 23

def test_dialogs_all_triples_vectors(dialog):
	assert len(dialog.all_triples_vectors()) == 4

def test_dialogs_random_utterance(dialog):
	# Should mostly never pull an error :)
	assert dialog.random_utterance() != dialog.random_utterance() or dialog.random_utterance() != dialog.random_utterance()
	assert type(dialog.random_utterance()) == spacy.tokens.Doc

def test_Dialog_blob(dialog):
	b = dialog.blob()
	assert len(b) == sum([len(utter) for utter in dialog.dialog])  # should be the concatenation of all the utterances

# Formare class
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def test_Formare_do(corpus):
	f1 = Formare(max_length=5)
	doc = corpus.nlp(u"This sentence has 8 tokens in it!")
	fed = f1.do([doc])
	assert len(fed[0]) == 5  # cropped to length 5
	print(fed[0])
	assert not False in [type(item) == int for item in fed[0].tolist()]  # All of type int
	f2 = Formare(attr=attrs.IS_DIGIT, max_length=20)
	fed2 = f2.do([doc])[0]
	assert len(fed2) == 20
	assert 1 in fed2
	unique, counts = numpy.unique(fed2, return_counts=True)
	counts = dict(zip(unique, counts))
	assert counts[1] == 1
	assert counts[0] == 19

# Static methods...

def test_Formare_conformare():
	assert len(Formare.conformare(range(666), length=0)) == 666  # do nothing if length is zero
	conformed = Formare.conformare([range(10)], 12)
	assert len(conformed) == 1
	shorty = conformed[0]
	assert len(shorty) == 12  # new extended length should be 12
	assert shorty[0] == shorty[1] == 0  # start of the list should be padded with zeros
	assert len(Formare.conformare([range(10)], 5)[0]) == 5  # new shortened length should be 5


