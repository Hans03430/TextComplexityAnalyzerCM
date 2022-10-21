import os
from text_complexity_analyzer_cm.constants import BASE_DIRECTORY
 
def write_documentation(directory):
    elements = os.listdir(directory)
    # Iterate over the elements found
    for element in elements:
        module_name = '.'.join(directory.split('/')[7:])
        print(module_name, directory)
        if os.path.isfile(f'{directory}/{element}'):
            if element == '__init__.py': # Write documentation for module
                os.system(f'python -m pydoc -w {module_name}')
            else: # Write documentation for file
                file_name_no_extension = element.replace('.py', '')
                os.system(f'python -m pydoc -w {module_name}.{file_name_no_extension}')
        elif os.path.isdir(f'{directory}/{element}') and element != '__pycache__':
            write_documentation(f'{directory}/{element}')

if __name__ == "__main__":
    print(BASE_DIRECTORY)
    os.chdir(f'{BASE_DIRECTORY}')
    modules_path = f'{BASE_DIRECTORY}'
    write_documentation(modules_path)