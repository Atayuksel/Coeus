import re, itertools
import nltk.tag
from nltk.tree import Tree


def filter_insignificant(chunk, tag_suffixes=['DT', 'CC']):

    good = []

    for word, tag in chunk:
        ok = True

        for suffix in tag_suffixes:
            if tag.endswith(suffix):
                ok = False
                break

        if ok:
            good.append((word, tag))

    return good


def tag_startswith(prefix):
    def f(wt):
        return wt[1].startswith(prefix)

    return f


def tag_equals(tag):
    def f(wt):
        return wt[1] == tag

    return f


def first_chunk_index(chunk, pred, start=0, step=1):

    l = len(chunk)
    end = l if step > 0 else -1

    for i in range(start, end, step):
        if pred(chunk[i]):
            return i

    return None


plural_verb_forms = {
    ('is', 'VBZ'): ('are', 'VBP'),
    ('was', 'VBD'): ('were', 'VBD')
}

singular_verb_forms = {
    ('are', 'VBP'): ('is', 'VBZ'),
    ('were', 'VBD'): ('was', 'VBD')
}


def correct_verbs(chunk):

    vbidx = first_chunk_index(chunk, tag_startswith('VB'))
    # if no verb found, do nothing
    if vbidx is None:
        return chunk

    verb, vbtag = chunk[vbidx]
    nnpred = tag_startswith('NN')
    # find nearest noun to the right of verb
    nnidx = first_chunk_index(chunk, nnpred, start=vbidx + 1)
    # if no noun found to right, look to the left
    if nnidx is None:
        nnidx = first_chunk_index(chunk, nnpred, start=vbidx - 1, step=-1)
    # if no noun found, do nothing
    if nnidx is None:
        return chunk

    noun, nntag = chunk[nnidx]
    # get correct verb form and insert into chunk
    if nntag.endswith('S'):
        chunk[vbidx] = plural_verb_forms.get((verb, vbtag), (verb, vbtag))
    else:
        chunk[vbidx] = singular_verb_forms.get((verb, vbtag), (verb, vbtag))

    return chunk


def swap_verb_phrase(chunk):


    # find location of verb
    def vbpred(wt):
        word, tag = wt
        return tag != 'VBG' and tag.startswith('VB') and len(tag) > 2

    vbidx = first_chunk_index(chunk, vbpred)

    if vbidx is None:
        return chunk

    return chunk[vbidx + 1:] + chunk[:vbidx]


def swap_noun_cardinal(chunk):

    cdidx = first_chunk_index(chunk, tag_equals('CD'))
    # cdidx must be > 0 and there must be a noun immediately before it
    if not cdidx or not chunk[cdidx - 1][1].startswith('NN'):
        return chunk

    noun, nntag = chunk[cdidx - 1]
    chunk[cdidx - 1] = chunk[cdidx]
    chunk[cdidx] = noun, nntag
    return chunk


def swap_infinitive_phrase(chunk):


    def inpred(wt):
        word, tag = wt
        return tag == 'IN' and word != 'like'

    inidx = first_chunk_index(chunk, inpred)

    if inidx is None:
        return chunk

    nnidx = first_chunk_index(chunk, tag_startswith('NN'), start=inidx, step=-1) or 0
    return chunk[:nnidx] + chunk[inidx + 1:] + chunk[nnidx:inidx]


def singularize_plural_noun(chunk):

    nnsidx = first_chunk_index(chunk, tag_equals('NNS'))

    if nnsidx is not None and nnsidx + 1 < len(chunk) and chunk[nnsidx + 1][1][:2] == 'NN':
        noun, nnstag = chunk[nnsidx]
        chunk[nnsidx] = (noun.rstrip('s'), nnstag.rstrip('S'))

    return chunk


def transform_chunk(chunk,
                    chain=[filter_insignificant, swap_verb_phrase, swap_infinitive_phrase, singularize_plural_noun],
                    trace=0):

    for f in chain:
        chunk = f(chunk)

        if trace:
            print('%s : %s' % (f.__name__, chunk))

    return chunk


punct_re = re.compile(r'\s([,\.;\?])')


def chunk_tree_to_sent(tree, concat=' '):

    s = concat.join(nltk.tag.untag(tree.leaves()))
    return re.sub(punct_re, r'\g<1>', s)


def flatten_childtrees(trees):
    children = []

    for t in trees:
        if not isinstance(t, Tree):
            children.append(t)
        elif t.height() < 3:
            children.extend(t.pos())
        elif t.height() == 3:
            children.append(Tree(t.label(), t.pos()))
        else:
            children.extend(flatten_childtrees([c for c in t]))

    return children


def flatten_deeptree(tree):

    return Tree(tree.label(), flatten_childtrees([c for c in tree]))


def shallow_tree(tree):

    children = []

    for t in tree:
        if t.height() < 3:
            children.extend(t.pos())
        else:
            children.append(Tree(t.label(), t.pos()))

    return Tree(tree.label(), children)


def convert_tree_labels(tree, mapping):

    children = []

    for t in tree:
        if isinstance(t, Tree):
            children.append(convert_tree_labels(t, mapping))
        else:
            children.append(t)

    label = mapping.get(tree.label(), tree.label())
    return Tree(label, children)


if __name__ == '__main__':
    import doctest

    doctest.testmod()