from spacy.language import Language
from spacy.tokens import Doc
from time import time


class SyntacticPatternDensityIndices:
    '''
    This class will handle all operations to find the synthactic pattern density indices of a text according to Coh-Metrix.
    '''

    name = 'syntactic_pattern_density_indices'

    def __init__(self, nlp: Language) -> None:
        '''
        The constructor will initialize this object that calculates the synthactic pattern density indices for a specific language of those that are available.

        Parameters:
        nlp: The spacy model that corresponds to a language.
         
        Returns:
        None.
        '''
        required_pipes = ['negative_expression_tagger', 'noun_phrase_tagger', 'alphanumeric_word_identifier']
        if not all((
            pipe in nlp.pipe_names
            for pipe in required_pipes
        )):
            message = 'Syntatic pattern density indices pipe need the following pipes: ' + ', '.join(required_pipes)
            raise AttributeError(message)
        
        self._nlp = nlp
        self._incidence = 1000
        Doc.set_extension('syntactic_pattern_density_indices', default={})

    def __call__(self, doc: Doc) -> Doc:
        '''
        This method calculates the syntatic pattern density indices.

        Parameters:
        doc(Doc): A Spacy document.

        Reeturns:
        Doc: The spacy document that was analyzed
        '''
        if len(doc.text) == 0:
            raise ValueError('The text is empty.')
        
        print('Analyzing syntactic pattern density indices.')
        start = time()
        doc._.syntactic_pattern_density_indices['DRNP'] = self.__get_noun_phrase_density(doc)
        doc._.syntactic_pattern_density_indices['DRVP'] = self.__get_verb_phrase_density(doc)
        doc._.syntactic_pattern_density_indices['DRNEG'] = self.__get_negation_expressions_density(doc)
        end = time()
        print(f'Syntactic pattern density indices analyzed in {end - start} seconds.')
        return doc

    def __get_noun_phrase_density(self, doc: Doc) -> float:
        '''
        This function obtains the incidence of noun phrases that exist on a text per {self._incidence} words.

        Parameters:
        doc(Doc): The text to be analized.
        
        Returns:
        float: The incidence of noun phrases per {self._incidence} words.
        '''
        return (doc._.noun_phrases_count / doc._.alpha_words_count) * self._incidence

    def __get_verb_phrase_density(self, doc: Doc) -> float:
        '''
        This function obtains the incidence of verb phrases that exist on a text per {self._incidence} words.

        Parameters:
        doc(Doc): The text to be analized.
        
        Returns:
        float: The incidence of verb phrases per {self._incidence} words.
        '''
        return (doc._.verb_phrases_count / doc._.alpha_words_count) * self._incidence

    def __get_negation_expressions_density(self, doc: Doc) -> float:
        '''
        This function obtains the incidence of negation expressions that exist on a text per {self._incidence} words.

        Parameters:
        doc(Doc): The text to be analized.
        
        Returns:
        float: The incidence of negation expressions per {self._incidence} words.
        '''
        return (doc._.negative_expressions_count / doc._.alpha_words_count) * self._incidence