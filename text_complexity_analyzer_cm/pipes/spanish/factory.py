from spacy.lang.es import Spanish
from spacy.language import Language

from text_complexity_analyzer_cm.pipes.auxiliaries.alphanumeric_word_identifier import AlphanumericWordIdentifier
from text_complexity_analyzer_cm.pipes.auxiliaries.noun_phrase_tagger import NounPhraseTagger
from text_complexity_analyzer_cm.pipes.auxiliaries.paragraphizer import Paragraphizer
from text_complexity_analyzer_cm.pipes.auxiliaries.syllablelizer import Syllablelizer
from text_complexity_analyzer_cm.pipes.auxiliaries.content_word_identifier import ContentWordIdentifier
from text_complexity_analyzer_cm.pipes.auxiliaries.words_before_main_verb_counter import WordsBeforeMainVerbCounter
from text_complexity_analyzer_cm.pipes.coh_metrix_indices.descriptive_indices import DescriptiveIndices
from text_complexity_analyzer_cm.pipes.coh_metrix_indices.lexical_diversity_indices import LexicalDiversityIndices
from text_complexity_analyzer_cm.pipes.coh_metrix_indices.readability_indices import ReadabilityIndices
from text_complexity_analyzer_cm.pipes.coh_metrix_indices.syntactic_complexity_indices import SyntacticComplexityIndices

@Spanish.factory('alphanumeric_word_identifier')
def create_es_alphanumeric_word_identifier(nlp: Language, name: str) -> AlphanumericWordIdentifier:
    '''
    Function that creates an alphanumeric word identifier for spanish.
    
    Paramters:
    nlp(Language): Spacy model that will be used for the pipeline.
    name(str): Name of the pipe.

    Returns:
    AlphanumericWordIdentifier: The pipe that finds alphanumeric words.
    '''
    return AlphanumericWordIdentifier(nlp)

@Spanish.factory('paragraphizer', default_config={'paragraph_delimiter': '\n\n'})
def create_es_paragraphizer(nlp: Language, name: str, paragraph_delimiter: str) -> Paragraphizer:
    '''
    Function that creates a paragraph splitter for spanish.
    
    Paramters:
    nlp(Language): Spacy model that will be used for the pipeline.
    name(str): Name of the pipe.
    paragraph_delimiter(str): Character used to split the paragraphs.

    Returns:
    Paragraphizer: The pipe that separates the text into paragraphs.
    '''
    return Paragraphizer(nlp, paragraph_delimiter)

@Spanish.factory('syllablelizer', default_config={'language': 'es'})
def create_es_syllablelizer(nlp: Language, name: str, language: str) -> Syllablelizer:
    '''
    Function that creates a syllable splitter for spanish.
    
    Paramters:
    nlp(Language): Spacy model that will be used for the pipeline.
    name(str): Name of the pipe.
    language(str): Character used to split the paragraphs

    Returns:
    Syllablelizer: The pipe that finds divides the words by syllables.
    '''
    return Syllablelizer(nlp, language)

@Spanish.factory('descriptive_indices')
def create_es_descriptive_indices(nlp: Language, name: str) -> DescriptiveIndices:
    '''
    Function that creates descriptive indices pipe.
    
    Paramters:
    nlp(Language): Spacy model that will be used for the pipeline.
    name(str): Name of the pipe.

    Returns:
    ParagraphSplitter: The pipe that finds descriptive indices.
    '''
    return DescriptiveIndices(nlp)

@Spanish.factory('content_word_identifier')
def create_es_content_word_identifier(nlp: Language, name: str) -> ContentWordIdentifier:
    '''
    Function that creates content word identifier pipe.
    
    Paramters:
    nlp(Language): Spacy model that will be used for the pipeline.
    name(str): Name of the pipe.

    Returns:
    ContentWordIdentifier: The pipe that finds content words.
    '''
    return ContentWordIdentifier(nlp)

@Spanish.factory('lexical_diversity_indices')
def create_es_lexical_diversity_indices(nlp: Language, name: str) -> LexicalDiversityIndices:
    '''
    Function that creates lexical diversity indices pipe.
    
    Paramters:
    nlp(Language): Spacy model that will be used for the pipeline.
    name(str): Name of the pipe.

    Returns:
    ParagraphSplitter: The pipe that finds the lexical diversity indices.
    '''
    return LexicalDiversityIndices(nlp)

@Spanish.factory('readability_indices')
def create_es_readability_indices(nlp: Language, name: str) -> ReadabilityIndices:
    '''
    Function that creates readability indices pipe.
    
    Paramters:
    nlp(Language): Spacy model that will be used for the pipeline.
    name(str): Name of the pipe.

    Returns:
    ReadabilityIndices: The pipe that finds the readability indices.
    '''
    return ReadabilityIndices(nlp)

@Spanish.factory('noun_phrase_tagger')
def create_es_noun_phrase_tagger(nlp: Language, name: str) -> NounPhraseTagger:
    '''
    Function that creates a noun phrase tagger.
    
    Paramters:
    nlp(Language): Spacy model that will be used for the pipeline.
    name(str): Name of the pipe.

    Returns:
    NounPhraseTagger: The pipe that tags the noun phrases.
    '''
    return NounPhraseTagger(nlp)

@Spanish.factory('syntactic_complexity_indices')
def create_es_syntactic_complexity_indices(nlp: Language, name: str) -> SyntacticComplexityIndices:
    '''
    Function that creates a syntactic complexity indices pipe.
    
    Paramters:
    nlp(Language): Spacy model that will be used for the pipeline.
    name(str): Name of the pipe.

    Returns:
    SyntacticComplexityIndices: The pipe that calculates the syntactic complexity indices.
    '''
    return SyntacticComplexityIndices(nlp)

@Spanish.factory('words_before_main_verb_counter')
def create_es_words_before_main_verb_counter(nlp: Language, name: str) -> WordsBeforeMainVerbCounter:
    '''
    Function that creates words before main verb counter pipe.
    
    Paramters:
    nlp(Language): Spacy model that will be used for the pipeline.
    name(str): Name of the pipe.

    Returns:
    SyntacticComplexityIndices: The pipe that calculates the amount of words before the main verb of every sentence.
    '''
    return WordsBeforeMainVerbCounter(nlp)