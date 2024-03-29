# HMM-Part-of-Speech-Tagging

Natural Language Processing (NLP) is a subset of AI that focuses on the understanding and generation of written and spoken language. This involves a series of tasks from low-level speech recognition on audio signals to high-level semantic understanding and inferencing on the parsed sentences.

One task within this spectrum is Part-Of-Speech (POS) tagging. Every word and punctuation symbol is understood to have a syntactic role in its sentence, such as nouns (denoting people, places or things), verbs (denoting actions), adjectives (which describe nouns) and adverbs (which describe verbs), to name a few. Each word in a text is therefore associated with a part-of-speech tag (usually assigned by hand), where the total number of tags can depend on the organization tagging the text.

Our task for this assignment is to create a hidden Markov model (HMM) for POS tagging.

1. Creating the initial, transition and observation probability tables in the HMM model by using the text-tag pairs in the training files.
2. Predicting POS tags for untagged text by performing inference with the HMM model created in step 1. 
