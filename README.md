# Research Journal


The task is to choose a sentence from a long list of sentences. The minimum complexity will be `O(n^2)` as for every case one needs to look at every solution. Eliminating chosen solutions will reduce this but also runs the risk of doubling error rates (as choosing the wrong sentence removes it for it's correct partner as well).

Approaches which are based around the *Danescu-Niculescu-Mizil, Lee (2011)* paper's idea that interlocutors unconsciously engage in coordination and this can be detected with the *convergence* measure they have come up with.

## Day 1

Use a Maximum Entropy (MaxEnt) model to put this theory to the test. Feature detectors will be used to detect the presence of each of the nine "trigger family"'s presence between pairs of utterances (note that the data has multiple sentences per utterance but will all be treated as the same. Possibly in the future test weather it is better the just look at the last sentence?). These trigger families are:

-	Negation - not, never
-	Articles - an, the
-	Auxiliary verbs - will, have
-	High frequency adverbs - really, quickly
-	Conjunctions - but, whereas
-	Indefinite pronouns - it, those
-	Personal pronouns - them, her
-	Prepositions - to, with
-	Quantifiers - few, much

Some extra features that might be good features are:

-	matching nouns: detect if the same noun is used in any of the previous sentences
-	matching verbs: same as nouns
-	matching punctuation: look for matching exclamation/question marks in previous utterances of the same speaker.
-	Difference in the length of sentence of `a` to `b` as shorter sentences seem to induce short replys.

Training on every pair of utterances in the training data and let the MaxEnt model determine the relative weight of each feature.

## Day 2: Real AI instead of Kluge Hans

I might drop the 'Clever Hans' approach which i feel the bag of words appoch to NLP is about, training for the task instead of trying to understand intelligence. I'm _more interested in_ trying to use parsed sentences to "understand" what the dialog is about and match on higher level stuff like subject / object (co-references), sentiment analysis, etc of the dialogs

Perhaps also look at structural patterns. There has been some work on

-	discourse coherence
	-	[Structural Parallelism and Discourse Coherence: A Test of Centering Theory](http://www.sciencedirect.com/science/article/pii/S0749596X9892575X)
	-	[Representing Discourse Coherence: A Corpus-Based Study](http://www.mitpressjournals.org/doi/pdf/10.1162/0891201054223977)

Possibly thinking about using FrameNet to fill-in some blanks when comparing subjects and related verbs around a topic.

This is a very hard approch and will require some long term research. Perhaps too much for this challenge.

## Day 3: Base line and dialog frames


Going to start with a baseline test which just matches NOUNs+PROPNs and VERBs and chooses based on the total matching score.

Then i might augment this with word embeddings to see if that lifts the score. Will also try this on matching all types in the missing sentence with those in the dialog.

Could then try to apply a neural network on top of this to see if it does any better.

### Theory: Types of conversations

I'm thinking that the conversations fall into some sort of grouping i.e. interrogation (a question/answer pattern). The sentiment of the individual speakers might also be gauged to try and match tone of speaker. These would match a frame from FrameNet and a preprocessing task might be the match a frame to a dialog and then look for the missing sentence. Out of scope for this challenge.

## Day 4: Deep net

The plan is to use word embeddings (300 dimensional vector) for training a multi-layed perceptron [MLP], a _deep net_ as they are called these days, with a variety training approches including: 

- the raw sum of all the dialog sentences 
- just the previous sentences by the same speaker 
- just the pervious and following sentences to the missing one
- linguistic coordination elements i.e. for each type of POS tag in the missing sentence calculate the distance for each of the above, using them as additional input? 
	* Or instead of the distance just sum their vectors and concatenate them to be used as input?

Might also use the **log likelyhood** information that is supplied by the spaCy toolkit. This will help with filtering nouns and verbs as _the less likely a word is the more information in carries_.

The MLP will have no special architecture (not convolutional or recurrent) with two output nodes i.e. one for each class and then turn this into a confidence using a _softmax_ function.

### Read two papers on understanding (future directions)

- [Representing discourse coherence: A corpus-based study](http://www.mitpressjournals.org/doi/pdf/10.1162/0891201054223977)(2005) which offers a non-heirarchical representation of representing discourse cohesion.
- [Frame semantics for text understanding](http://www.ccs.neu.edu/course/csg224/resources/framenet/framenet.pdf)(2001) which talks about how one might use FrameNet to understand text by mapping to frames. The approch they suggest would be:

	1. choose a word (starting from the highest semanticallyrelevant predicate in a given sentence),
	2. determining the frames that it is capable of evoking, noticing the semantic roles of the props and participants in each such frame, trying to match the semantic needs associated with each such frame (and hence with each sense of the word) with phrases found in the sentence at hand
	3. choosing the one which makes the most coherent fit, and 
	4. entering the semantic structures associated with the dependent constituents into slots provided by the selected frame.
	
	Which again would be a research project in itself.
