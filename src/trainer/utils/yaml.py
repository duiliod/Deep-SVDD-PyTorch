import os
import yaml

from torch import Tensor

from .string_processing import list_to_string


def get_root():
    """Directory that the '{root}' placeholder resolves to.

    It is the DEEPSVDD_ROOT environment variable when set (it must contain the
    data/, split/ and configs/ folders), otherwise the root of this repository.
    This keeps configs and splits free of machine-specific absolute paths.
    """
    root = os.environ.get('DEEPSVDD_ROOT')
    if root:
        return os.path.abspath(root)
    here = os.path.dirname(os.path.abspath(__file__))  # <repo>/src/trainer/utils
    return os.path.abspath(os.path.join(here, '..', '..', '..'))


def _substitute_placeholders(node, structure):
    """Recursively replace the '{root}' and '{structure}' placeholders in any
    string found inside a (possibly nested) dict/list structure, so that a
    single config or split file can be shared across brain structures (e.g.
    THA, PAL, PUT, HIP, AMY) and across machines, and resolve to the right
    paths at load time.
    """
    if isinstance(node, str):
        node = node.replace('{root}', get_root())
        if structure is not None:
            node = node.replace('{structure}', structure)
        return node
    if isinstance(node, dict):
        return {k: _substitute_placeholders(v, structure) for k, v in node.items()}
    if isinstance(node, list):
        return [_substitute_placeholders(v, structure) for v in node]
    return node


def load(file_path, structure=None):
    calc_path = lambda path: os.path.join(os.path.dirname(file_path), path)

    def loadTensor(loader, node):
        return Tensor(loader.construct_sequence(node, deep=True))

    def loadInclude(loader, node):
        path = loader.construct_python_str(node)
        path = calc_path(path)
        return load(path, structure=structure)
    
    class Loader(yaml.FullLoader):
        pass
    
    Loader.add_constructor(u'!Tensor', loadTensor)
    Loader.add_constructor(u'!include', loadInclude)
    
    with open(file_path, 'r') as f:
        content = yaml.load(f, Loader=Loader)

    return _substitute_placeholders(content, structure)



def save_split_file_in_yaml(split_filename : str, dataset_path : str, training_filenames : list, validation_filenames : list, test_filenames : list, verbose : bool = False):
    """Generates a YAML file with a training, validation and test partition

    Args:
        split_filename (str): full path to the file to be saved
        dataset_path (str): path where the files are stored
        training_filenames (list): names of the files for the training set
        validation_filenames (list): names of the files for the validation set
        test_filenames (list): names of the files for the test set
        verbose (bool, optional): use this parameter to show the number of images in train, val and test. Defaults to False.

    Returns:
        dict: dictionary with all the content of the YAML file
    """

    # make a dictionary with the content for the file
    split_file_content = {"path": dataset_path,
                          "training" : list_to_string(training_filenames),
                          "validation" : list_to_string(validation_filenames),
                          "test" : list_to_string(test_filenames)}

    # export the split
    with open(split_filename, "w") as file:
        yaml.dump(split_file_content, file)

    # print the split
    if verbose:
        print("- Training set: {} files".format(len(training_filenames)))
        print("- Validation set: {} files".format(len(validation_filenames)))
        print("- Test set: {} files".format(len(test_filenames)))

    return split_file_content
