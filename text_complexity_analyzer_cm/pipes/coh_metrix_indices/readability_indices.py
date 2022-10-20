from spacy.language import Language
from spacy.tokens import Doc
from time import time


class ReadabilityIndices:
    '''
    This pipe will handle all operations to find the readability indices of a text according to Coh-Metrix. It needs the descriptive indices pipe to be added before it.
    '''
    name = 'readability_indices'

    def __init__(self, nlp: Language) -> None:
        '''
        The constructor will initialize this object that calculates the readability indices for a specific language of those that are available.

        Parameters:
        nlp(Language): The spacy model that corresponds to a language.
        
        Returns:
        None.
        '''
        required_pipes = ['descriptive_indices']
        if not all((
            pipe in nlp.pipe_names
            for pipe in required_pipes
        )):
            message = 'Readability diversity indices pipe need the following pipes: ' + ', '.join(required_pipes)
            raise AttributeError(message)
        
        self._nlp = nlp
        Doc.set_extension('readability_indices', default={})

    def __call__(self, doc: Doc) -> Doc:
        '''
        This method will calculate the readability indices.

        Parameters:
        doc(Doc): A Spacy document.

        Returns:
        Doc: The processed doc.
        '''
        if len(doc.text) == 0:
            raise ValueError('The text is empty.')

        print('Analyzing readability indices')
        start = time()
        doc._.readability_indices['RDFHGL'] = self.__calculate_fernandez_huertas_grade_level(doc)
        end = time()
        print(f'Readability indices analyzed in {end - start} seconds.')
        return doc

    def __calculate_fernandez_huertas_grade_level(self, doc: Doc) -> float:
        '''
        This function obtains the Fernández-Huertas readability index for a text.

        Parameters:
        doc(Doc): The text to be analized.
        
        Returns:
        float: The Fernández-Huertas readability index for a text.
        '''
        return 206.84 - 0.6 * doc._.descriptive_indices['DESWLsy'] - 1.02 * doc._.descriptive_indices['DESSL']