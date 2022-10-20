from typing import Text
from text_complexity_analyzer_cm.text_complexity_analyzer import TextComplexityAnalyzer
from text_complexity_analyzer_cm.utils.utils import preprocess_text_spanish

if __name__ == '__main__':
    texts = ['''Ellos jugaron todo el día. Asimismo, ellas participaron en el juego.


Yo corro con el hermoso gato. A nosotros no nos gusta el gato.
Ella tiene mascotas. Aunque, ella tiene plantas y él no.

Tú jamás dijiste que no, porque ustedes debieron salir temprano al mismo tiempo.''']

    texts = []
    with open('/home/hans/Documentos/Books/Enrique Meiggs', 'r') as book:
        content = book.read() * 3
        texts.append(content)
    
    texts = texts * 16
    #texts = ['El zorro pequeño saltó por encima del perro sin tocarlo.']
    tca = TextComplexityAnalyzer('es', False, preprocessing_func=preprocess_text_spanish)
    metrics = tca.calculate_all_indices_for_texts(texts, workers=16, batch_size=1)
    #for metric in metrics:
        #print(metric)
