from collections import defaultdict
from itertools import count
import torch

BOS_SYM = '<BOS>'
EOS_SYM = '<EOS>'

def build_vocab(char_set, pad_syms=True):
    """
    Build an exhaustive character inventory for the corpus, and return dictionaries
    that can be used to map characters to indicies and vice-versa.
    
    Make sure to include BOS_SYM and EOS_SYM!
    
    :param corpus: a corpus, represented as an iterable of strings
    :returns: two dictionaries, one mapping characters to vocab indicies and another mapping idicies to characters.
    :rtype: dict-like object, dict-like object
    """
    if pad_syms:
        char_set|=set([BOS_SYM, EOS_SYM])
    sym_to_num = {}
    num_to_sym = {}
    
    char_set=sorted(char_set)
    for i, c in enumerate(char_set):
        sym_to_num[c]=i
        num_to_sym[i]=c
    
    return sym_to_num, num_to_sym
    
def sentence_to_vector(s, vocab, pad_with_bos=False):
    """
    Turn a string, s, into a list of indicies in from `vocab`. 
    
    :param s: A string to turn into a vector
    :param vocab: the mapping from characters to indicies
    :param pad_with_bos: Pad the sentence with BOS_SYM/EOS_SYM markers
    :returns: a list of the character indicies found in `s`
    :rtype: list
    """
    vector=[]
    if pad_with_bos:
        vector.append(vocab[BOS_SYM])
    for c in s:
        vector.append(vocab[c])
        
    if pad_with_bos:
        vector.append(vocab[EOS_SYM])
    
    return vector
    
def sentence_to_tensor(s, vocab, pad_with_bos=False):
    """
    :param s: A string to turn into a tensor
    :param vocab: the mapping from characters to indicies
    :param pad_with_bos: Pad the sentence with BOS_SYM/EOS_SYM markers
    :returns: (1, n) tensor where n=len(s) and the values are character indicies
    :rtype: torch.Tensor
    """
    vector=sentence_to_vector(s, vocab, pad_with_bos)
    tensor=torch.tensor([vector])
    return tensor
    
def build_label_vocab(labels):
    """
    Similar to build_vocab()- take a list of observed labels and return a pair of mappings to go from label to numeric index and back.
    
    The number of label indicies should be equal to the number of *distinct* labels that we see in our dataset.
    
    :param labels: a list of observed labels ("y" values)
    :returns: two dictionaries, one mapping label to indicies and the other mapping indicies to label
    :rtype: dict-like object, dict-like object
    """
    lab_set = set(labels)
    sym_to_num = {}
    num_to_sym = {}
    
    for i, l in enumerate(lab_set):
        sym_to_num[l]=i
        num_to_sym[i]=l
    
    return sym_to_num, num_to_sym
