# import nltk
# nltk.download('punkt')
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Input text to be summarized
input_text = """
I recently had a lot of fun making simulations for how the electric field responds to a moving charge.
It was all for a longer video on YouTube that covers the fundamentals of what light is,
but here I just thought I'd share with you some of the pretty animations that came out of it.
The full electric field is 3-dimensional. It associates each point in space,
with a little vector telling you what force would be applied to a unit charge if it were sitting at that
point in space. The 3D fields are fun to look at, but they are a bit busy, so often it's easier to just
show a slice of that field, say on the xy plane. In the main video I talk about the underlying
law used to make these simulations, it's not actually Maxwell's equations, which are the fundamental
laws underlying electricity and magnetism, instead I'm coding in another intermediate law,
something that you can derive from Maxwell's equations, but which is a little easier to wrap your
mind around, both for the sake of making the animations, and also for the sake of the student
trying to understand what's going on.
"""

# Parse the input text
parser = PlaintextParser.from_string(input_text, Tokenizer("english"))

# Create an LSA summarizer
summarizer = LsaSummarizer()

# Generate the summary
summary = summarizer(parser.document, sentences_count=3)  # You can adjust the number of sentences in the summary

# Output the summary
print("Original Text:")
print(input_text)
print("\nSummary:")
for sentence in summary:
    print(sentence)