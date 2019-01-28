from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


class PorterPreprocessor():
    def __init__(self, stoplist):
        self._stemmer = PorterStemmer()
        self._stoplist = stopwords.words(stoplist)

    def process_term(self, term):
        stem_term = None
        term = term.lower()
        stops = set(self._stoplist)
        if term is not None and term not in stops:
            stem_term = self._stemmer.stem(term)
        return stem_term


class StringPreprocessor(PorterPreprocessor):
    #
    # String query preprocessor

    def __init__(self, stoplist='english'):
        super(StringPreprocessor, self).__init__(stoplist)

    def __call__(self, *args):
        terms = []
        data = args[0]
        for term in data:
            processed_term = self.process_term(term)
            if processed_term is not None:
                terms.append(processed_term)
        return terms


class StringPreprocessorAdapter(StringPreprocessor):
    #
    # String query preprocessor

    def __init__(self, stoplist='english'):
        super(StringPreprocessorAdapter, self).__init__(stoplist)

    def __call__(self, *args):
        terms = []
        data = args[0].split(' ')
        return ' '.join(super(StringPreprocessorAdapter, self).__call__(data))