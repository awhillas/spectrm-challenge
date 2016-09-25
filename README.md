# Solution

## 1. Describe your approach. Which methods did you chose and why?

I tried 3 approaches over 2 weeks.

### Approach 1: Word Embeddings, No learning.
My first attempt to get a base line. The basic idea is to use the word embeddings that come with [spaCy](http://spacy.io/), which are "Levy and Goldberg (2014) **dependency-based** word2vec model" so encode parsing relationships, to get similarity between the dialogs and the missing utterances. 

The problem seems to be an unsupervised clustering task so i was looking for the best distance metric in this approach. Taking the hint from  Danescu-Niculescu-Mizil & Lillian Lee (2011) I grouped the words by POS tag and then summed the word vectors (CBOW). Then the _cosine distance_ between these POS vectors and the missing word POS vectors were taken and then simply summed together to give a distance. Weights for each POS tag could be added latter and learnt (see approach 3)

spaCy also has "unigram log-probability of the words, estimated from counts from a large corpus, smoothed using Simple Good Turing estimation" and i used these to filter out anything with a log-probability greater than -8.0. This helped a lot.

### Approach 2: DeepNets, CNN and RNNs
I have some experience with multi-layered perceptrons (MLP) but not with Convolutional Neural Networks (CNN) or Recurrent NNs (RNN). I'd been reading a lot of research papers on them so I thought I'd give it a try with the Keras library interfacing with Tensorflow.

The main gist of it was, to compare two (or three) sentences by feeding each into a 1D CNN or RNN (LSTM or ) and then feeding the output of these into a MLP. I got this going but it was slow and gave poor performance. 

I had to generate negative examples which were just random utterances taken from the entire corpus. This seemed to be flawed as there was no guarantee these would be bad examples as *some of the utterances were so general* they might have worked.

Things I might have tried if given more time (and hardware):

- Different data combinations:
	- just compare two sentences instead of the whole dialog with the missing. Use every sentence pair in the training corpus to significantly increase training data.
	- before, missing and after utterances, using every 
		- or split the utterances and just use before and after sentences
- Different network architectures:
	- RNN x 3 -> MLP
	- CNN x 3 -> LSTM -> MLP
	- same as above but share the layers
- Use the spaCy pre-trained word embeddings

On the last day my attempts at experimenting with DeepNets came to a frustrating end as the learning was taking too long on my poor MacBook Air and I was out of time. I need a GPU if i'm going to explore DeepNets more.

I got a very mild success, 5% improvement in accuracy above random.

### Approach 3: Genetic Algorithm (GA)
Thinking back to the original clustering approach i'd had i thought if i could train some weights for the POS tags using a GA, which is fair simple perhaps i could improve the first approach.

Almost got it working but ran out of time. One day wasn't enough :(

---

Sadly approach 1 was the only one that worked the best as i need more time to experiment with the other two approaches.

# How do you evaluate your performance?

I was just using accuracy on 10% of the training data.

# Where are the weaknesses of your approach? What has to be considered when applying an approach like this in practice?

I'm assuming by "practice" you mean a chat bot? There are dozens of application of this task.

1. Approach 1: Its comparing every missing utterance to every dialog, which is O(n^2) and fairly slow. In practice it would be better the group the types of replies and then generate general word embeddings for each group. I'm guessing that 
2. DeepNets are quite quick to evaluate (given the right hardware). Again grouping the replies would help and you could then train a DeepNet for each level.
3. Once the GA has leant the parameters for the first approach it should be quite fast. GA are slow to train depending on the size of the problem space. I's also throw in parsing relations from a dependancy parse as well as the POS tags.

# Feedback to the challenge itself is appreciated as well. 

The challenge was quite fun and i sent it to some colleagues who are also researching in the NLP field and who might also have a go at it (for fun).

The data is a little dirty. I did a quick analysis of the `test_missing.txt` file which can be seen in `challenge_data/test_missing_uniqness.txt`. Basically 7.5% of the missing utterances are duplicates. They are also so general that they could be used in almost all the dialogs.

Yours

[Alexander Whillas](mailto:whillas@gmail.com)

handy: +49 178 332 6680

