from spacy.language import Language
from spacy.tokens import Doc
from spacy.tokens import Span


def doc_nouns_getter(doc: Doc) -> str:
    '''
    Function that returns all nouns for the doc.

    Parameters:
    doc(Doc): The document to analyze.

    Yields:
    str: The nouns.
    '''
    for sent in doc._.non_empty_sentences:
        for token in sent._.nouns:
            yield token

def doc_verbs_getter(doc: Doc) -> str:
    '''
    Function that returns all verbs for the doc.

    Parameters:
    doc(Doc): The document to analyze.

    Yields:
    str: The verbs.
    '''
    for sent in doc._.non_empty_sentences:
        for token in sent._.verbs:
            yield token

def doc_adjectives_getter(doc: Doc) -> str:
    '''
    Function that returns all adjectives for the doc.

    Parameters:
    doc(Doc): The document to analyze.

    Yields:
    str: The adjectives.
    '''
    for sent in doc._.non_empty_sentences:
        for token in sent._.adjectives:
            yield token

def doc_adverbs_getter(doc: Doc) -> str:
    '''
    Function that returns all adverbs for the doc.

    Parameters:
    doc(Doc): The document to analyze.

    Yields:
    str: The adverbs.
    '''
    for sent in doc._.non_empty_sentences:
        for token in sent._.adverbs:
            yield token

def doc_pronouns_getter(doc: Doc) -> str:
    '''
    Function that returns all pronouns for the doc.

    Parameters:
    doc(Doc): The document to analyze.

    Yields:
    str: The pronouns.
    '''
    for sent in doc._.non_empty_sentences:
        for token in sent._.pronouns:
            yield token

def doc_pronouns_singular_first_person_getter(doc: Doc) -> str:
    '''
    Function that returns all pronouns singular first person for the doc.

    Parameters:
    doc(Doc): The document to analyze.

    Yields:
    str: The pronouns_singular_first_person.
    '''
    for sent in doc._.non_empty_sentences:
        for token in sent._.pronouns_singular_first_person:
            yield token

def doc_pronouns_plural_first_person_getter(doc: Doc) -> str:
    '''
    Function that returns all pronouns_plural_first_person for the doc.

    Parameters:
    doc(Doc): The document to analyze.

    Yields:
    str: The pronouns_plural_first_person.
    '''
    for sent in doc._.non_empty_sentences:
        for token in sent._.pronouns_plural_first_person:
            yield token

def doc_pronouns_singular_second_person_getter(doc: Doc) -> str:
    '''
    Function that returns all pronouns_singular_second_person for the doc.

    Parameters:
    doc(Doc): The document to analyze.

    Yields:
    str: The pronouns_singular_second_person.
    '''
    for sent in doc._.non_empty_sentences:
        for token in sent._.pronouns_singular_second_person:
            yield token

def doc_pronouns_plural_second_person_getter(doc: Doc) -> str:
    '''
    Function that returns all pronouns_plural_second_person for the doc.

    Parameters:
    doc(Doc): The document to analyze.

    Yields:
    str: The pronouns_plural_second_person.
    '''
    for sent in doc._.non_empty_sentences:
        for token in sent._.pronouns_plural_second_person:
            yield token

def doc_pronouns_singular_third_person_getter(doc: Doc) -> str:
    '''
    Function that returns all pronouns_singular_third_person for the doc.

    Parameters:
    doc(Doc): The document to analyze.

    Yields:
    str: The pronouns_singular_third_person.
    '''
    for sent in doc._.non_empty_sentences:
        for token in sent._.pronouns_singular_third_person:
            yield token

def doc_pronouns_plural_third_person_getter(doc: Doc) -> str:
    '''
    Function that returns all pronouns_plural_third_person for the doc.

    Parameters:
    doc(Doc): The document to analyze.

    Yields:
    str: The pronouns_plural_third_person.
    '''
    for sent in doc._.non_empty_sentences:
        for token in sent._.pronouns_plural_third_person:
            yield token


class InformativeWordTagger:
    '''
    This pipe identifies nouns, verbs, adverbs, adjectives and pronouns (of different types) per sentence.
    '''
    name = 'informative_word_tagger'

    def __init__(self, nlp: Language) -> None:
        '''
        This constructor will initialize the object that counts the words before the main verb of sentences.

        Parameters:
        language: The language that this pipeline will be used in.

        Returns:
        None.
        '''
        required_pipes = ['paragraphizer', 'alphanumeric_word_identifier']
        if not all((
            pipe in nlp.pipe_names
            for pipe in required_pipes
        )):
            message = 'Informative word tagger pipe need the following pipes: ' + ', '.join(required_pipes)
            raise AttributeError(message)
        
        self._nlp = nlp

        Span.set_extension('nouns', default=[]) # Count of adjectives in a noun phrase
        Span.set_extension('nouns_count', default=0)
        Span.set_extension('verbs', default=[])
        Span.set_extension('verbs_count', default=0)
        Span.set_extension('adjectives', default=[])
        Span.set_extension('adjectives_count', default=0)
        Span.set_extension('adverbs', default=[])
        Span.set_extension('adverbs_count', default=0)
        Span.set_extension('pronouns', default=[])
        Span.set_extension('pronouns_count', default=0)
        Span.set_extension('pronouns_singular_first_person', default=[])
        Span.set_extension('pronouns_singular_first_person_count', default=0)
        Span.set_extension('pronouns_plural_first_person', default=[])
        Span.set_extension('pronouns_plural_first_person_count', default=0)
        Span.set_extension('pronouns_singular_second_person', default=[])
        Span.set_extension('pronouns_singular_second_person_count', default=0)
        Span.set_extension('pronouns_plural_second_person', default=[])
        Span.set_extension('pronouns_plural_second_person_count', default=0)
        Span.set_extension('pronouns_singular_third_person', default=[])
        Span.set_extension('pronouns_singular_third_person_count', default=0)
        Span.set_extension('pronouns_plural_third_person', default=[])
        Span.set_extension('pronouns_plural_third_person_count', default=0)
        Doc.set_extension('nouns', getter=doc_nouns_getter) # Count of adjectives in a noun phrase
        Doc.set_extension('nouns_count', default=0)
        Doc.set_extension('verbs', getter=doc_verbs_getter)
        Doc.set_extension('verbs_count', default=0)
        Doc.set_extension('adjectives', getter=doc_adjectives_getter)
        Doc.set_extension('adjectives_count', default=0)
        Doc.set_extension('adverbs', getter=doc_adverbs_getter)
        Doc.set_extension('adverbs_count', default=0)
        Doc.set_extension('pronouns', getter=doc_pronouns_getter)
        Doc.set_extension('pronouns_count', default=0)
        Doc.set_extension('pronouns_singular_first_person', getter=doc_pronouns_singular_first_person_getter)
        Doc.set_extension('pronouns_singular_first_person_count', default=0)
        Doc.set_extension('pronouns_plural_first_person', getter=doc_pronouns_plural_first_person_getter)
        Doc.set_extension('pronouns_plural_first_person_count', default=0)
        Doc.set_extension('pronouns_singular_second_person', getter=doc_pronouns_singular_second_person_getter)
        Doc.set_extension('pronouns_singular_second_person_count', default=0)
        Doc.set_extension('pronouns_plural_second_person', getter=doc_pronouns_plural_second_person_getter)
        Doc.set_extension('pronouns_plural_second_person_count', default=0)
        Doc.set_extension('pronouns_singular_third_person', getter=doc_pronouns_singular_third_person_getter)
        Doc.set_extension('pronouns_singular_third_person_count', default=0)
        Doc.set_extension('pronouns_plural_third_person', getter=doc_pronouns_plural_third_person_getter)
        Doc.set_extension('pronouns_plural_third_person_count', default=0)

    def __call__(self, doc: Doc) -> Doc:
        '''
        This method will find all noun phrases and store them in an iterable.

        Parameters:
        doc(Doc): A Spacy document.

        Returns:
        Doc: The spacy document analyzed.
        '''
        # Iterate every non empty sentence of the document
        for sent in doc._.non_empty_sentences:
            sent._.nouns = [token for token in sent._.alpha_words if token.pos_ in ['NOUN', 'PROPN']]# Count of adjectives in a noun phrase
            sent._.verbs = [token for token in sent._.alpha_words if (token.pos_ == 'VERB') or (token.pos_ == 'AUX' and 'VerbForm' in str(token.morph))]
            sent._.adjectives = [token for token in sent._.alpha_words if token.pos_ == 'ADJ']
            sent._.adverbs = [token for token in sent._.alpha_words if token.pos_ == 'ADV']
            sent._.pronouns = [token for token in sent._.alpha_words if token.pos_ == 'PRON']
            sent._.pronouns_singular_first_person = [token for token in sent._.pronouns if 'Number=Sing' in token.morph and 'Person=1' in token.morph]
            sent._.pronouns_plural_first_person = [token for token in sent._.pronouns if 'Number=Plur' in token.morph and 'Person=1' in token.morph]
            sent._.pronouns_singular_second_person = [token for token in sent._.pronouns if 'Number=Sing' in token.morph and 'Person=2' in token.morph]
            sent._.pronouns_plural_second_person = [token for token in sent._.pronouns if 'Number=Plur' in token.morph and 'Person=2' in token.morph]
            sent._.pronouns_singular_third_person = [token for token in sent._.pronouns if 'Number=Sing' in token.morph and 'Person=3' in token.morph]
            sent._.pronouns_plural_third_person = [token for token in sent._.pronouns if 'Number=Plur' in token.morph and 'Person=3' in token.morph]
            sent._.nouns_count = len(sent._.nouns)
            sent._.verbs_count = len(sent._.verbs)
            sent._.adjectives_count = len(sent._.adjectives)
            sent._.adverbs_count = len(sent._.adverbs)
            sent._.pronouns_count = len(sent._.pronouns)
            sent._.pronouns_singular_first_person_count = len(sent._.pronouns_singular_first_person)
            sent._.pronouns_plural_first_person_count = len(sent._.pronouns_plural_first_person)
            sent._.pronouns_singular_second_person_count = len(sent._.pronouns_singular_second_person)
            sent._.pronouns_plural_second_person_count = len(sent._.pronouns_plural_second_person)
            sent._.pronouns_singular_third_person_count = len(sent._.pronouns_singular_third_person)
            sent._.pronouns_plural_third_person_count = len(sent._.pronouns_plural_third_person)
            doc._.nouns_count += sent._.nouns_count
            doc._.verbs_count += sent._.verbs_count
            doc._.adjectives_count += sent._.adjectives_count
            doc._.adverbs_count += sent._.adverbs_count
            doc._.pronouns_count += sent._.pronouns_count
            doc._.pronouns_singular_first_person_count += sent._.pronouns_singular_first_person_count
            doc._.pronouns_plural_first_person_count += sent._.pronouns_plural_first_person_count
            doc._.pronouns_singular_second_person_count += sent._.pronouns_singular_second_person_count
            doc._.pronouns_plural_second_person_count += sent._.pronouns_plural_second_person_count
            doc._.pronouns_singular_third_person_count += sent._.pronouns_singular_third_person_count
            doc._.pronouns_plural_third_person_count += sent._.pronouns_plural_third_person_count

        return doc