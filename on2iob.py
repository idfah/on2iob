#!/usr/bin/env python3

import collections, copy
from glob import glob
import json, os, random

# use element tree to parse xml
import xml.etree.ElementTree as ETree

# use click for command line interface
import click

# constants for In, Out and Begin prefix
# tags will be in format [IOB]-{type}, e.g., `B-person`
I, O, B = 'I', 'O', 'B'

def xml2iob(et_node, data):
    '''Recursive function for converting an Ontonotes 5 XML file into IOB format.

    Args:
        et_node:  xml.etree.ElementTree.Element
            An element tree node, generally starting with the root node of the XML document
            and then recursing on each child node.
        data:  list(list(tuple(str, str))
            Sentences of words of tag/token 2-tuples.  This is the IOB data and the outer
            list will be extended by this function.  Initially, data should start as an
            empty `[[]]`.
    '''
    def tag_text(text, tag_type):
        '''Create `list((word, tag))` where the outer list represents sentences, separated
        by `\n` newlines and the `(word, tag)` 2-tuples are string token-tag pairs.

        text:  str
            The text to tag.  Tokens are separated by whitespace and lines by the `\n`
            newline character.
        tag_type: str
            The entity type for the tag, e.g., `person`.  The `IOB` part will be determined
            by the sequence of tokens in the text.
        '''
        lines = text.split('\n')
        nlines = len(lines)

        # for each line in the text
        for i, line in enumerate(lines):
            # start with the begin position
            tag_pos = B

            # remove leading and trailing whitespace
            line = line.strip(' ')

            # for each whitespace delimited token
            for word in line.split(' '):
                # skip empty tokens and repeated spaces
                if not word: continue

                # build the tag from position and type
                tag = tag_pos + '-' + tag_type if tag_type else O

                # append to the sentence
                data[-1].append((word, tag))

                # remaining tokens are inside
                tag_pos = I

            # split off new sentence
            if i < nlines - 1:
                data.append([])

    # Main text of the node, not inside subtags.  Note that a node may not have text,
    # e.g., for the root `<DOC>` tag if the first word is an entity.
    # For entity, `ENAMEX` tags, this will contain the entity tokens.
    if et_node.text:
        # `ENAMEX` tags (whatever that manes) in `.name` files surround entity spans.
        # The `TYPE` attribute contains the type of the entity.
        tag_type = et_node.attrib['TYPE'].lower() if et_node.tag == 'ENAMEX' else ''
        tag_text(et_node.text, tag_type)

    # Recurse, depth first, over each child.
    for et_child in et_node:
        xml2iob(et_child, data)

    # The tail is the text *after* the tag.  Since we don't support nested `ENAMEX` tags,
    # we just assume this is outside of any entity and is tagged as `O`.
    if et_node.tail:
        tag_text(et_node.tail, '')

def parse_iob_files(filenames, verbose):
    '''Convert a list of `.name` files into iob format.
    '''
    iob_data = []
    for filename in filenames:
        if verbose:
            print(filename, '- ', end='')

        # open the current file for reading, UTF8 or bust!
        with open(filename, mode='rt', encoding='utf8') as fh:
            raw_data = fh.read()

        # get the root node from the string file contents
        et_root = ETree.fromstring(raw_data)

        # the IOB data for this file is extracted recursively
        local_iob_data = [[]] # this is what's built up
        xml2iob(et_root, local_iob_data)

        # filter any empties
        # FIXME: why is this necessary?
        local_iob_data = list(filter(lambda sent: sent, local_iob_data))

        # verify that the number of sentences matches the number of lines in the document
        # this appears to be how the `.name` files are structured.
        nsent = len(local_iob_data)
        nraw_sent = len(list(filter(lambda line: line.strip(), raw_data.split('\n')))) - 2
        if nsent != nraw_sent:
            raise RuntimeError(
                f'Parsed sentences `{nsent}` does not match number of lines `{nraw_sent}` in `{filename}`')

        # append to the full IOB data
        iob_data.extend(local_iob_data)

        if verbose:
            print(nsent)

    return iob_data

def write_iob(filename, data):
    '''Write IOB sentence data to file.
    '''
    nsent = len(data)
    with open(filename, mode='wt', encoding='utf8') as fh:
        for i, sent in enumerate(data):
            assert sent
            fh.writelines([' '.join(token_tag) + '\n' for token_tag in sent])
            if i < nsent - 1:
                fh.write('\n')

def build_random_partitions(iob_data, valid_frac=0.07, test_frac=0.07, seed=42):
    '''Random train/validation/test split with seeded number generator.

    Notes:
        This is not the standard approach used in most literature for SOTA benchmarks!
    '''
    # FIXME: make another function to do this in the standard way
    rng = random.Random(seed)

    # copy the data to prevent shuffling as a side-effect
    iob_data = copy.deepcopy(iob_data)

    # shuffle the sentences
    rng.shuffle(iob_data)

    nsent = len(iob_data)
    valid_end = int(valid_frac * nsent)
    test_end = valid_end + int(test_frac * nsent)

    train_data = iob_data[:valid_end]
    valid_data = iob_data[valid_end:test_end]
    test_data = iob_data[test_end:]

    return train_data, valid_data, test_data

def print_metrics(iob_data):
    '''Print some useful metrics about the data to the console.
    '''
    print('=======')
    print('total sentences:', len(iob_data))
    print('total tokens:', sum(len(sent) for sent in iob_data))

    labels = sorted(set(label for sent in iob_data for word, label in sent))
    print('total labels:', len(labels), '-', labels)

    tag_counts = collections.defaultdict(int)
    for sent in iob_data:
        for word, tag in sent:
            if tag is O:
                continue

            tag_pos, tag_type = tag.split('-')
            if tag_pos == B:
                tag_counts[tag_type] += 1

    tag_counts = sorted(tag_counts.items(), key=lambda pair: pair[1], reverse=True)

    print('total types:', len(tag_counts))
    print('\n'.join(str(tag_count) for tag_count in tag_counts))

@click.command('on2iob', context_settings={'show_default': True})
@click.option('-d', '--data-dir', 'data_dir', default='data/files/data/english/annotations/', help='Path containing subdirectories with the Ontonoes `.name` annotation XML files.')
@click.option('--iob-file', 'iobfile', default='iob.txt', help='Name of the output file containing _all_ of the data in IOB format.')
@click.option('--train-file', 'trainfile', default='train.txt', help='Name of the output training data file in IOB format.')
@click.option('--valid-file', 'validfile', default='valid.txt', help='Name of the output validation data file in IOB format.')
@click.option('--test-file', 'testfile', default='test.txt', help='Name of the output test data file in IOB format.')
@click.option('-v', '--verbose', 'verbose', is_flag=True, default=True, help='Print verbose information to the console.')
def main(data_dir, iobfile, trainfile, validfile, testfile, verbose):
    # path containing subdirectories with the `.name` annotation XML files
    # these are provided with the Ontonotes 5 dataset, which must be acquired from the LDC
    datadir = os.path.realpath(os.path.expanduser(data_dir))

    # recursively glob all the file names
    filenames = sorted(glob(os.path.join(datadir, '**/*.name'), recursive=True))

    # parse the files
    iob_data = parse_iob_files(filenames, verbose)
    if verbose:
        print_metrics(iob_data)

    # write the full data to `iob.txt`
    write_iob(iobfile, iob_data)

    # build and write the train, valid and test partitions
    train_data, valid_data, test_data = build_random_partitions(iob_data)
    write_iob(validfile, train_data)
    write_iob(testfile, valid_data)
    write_iob(trainfile, test_data)

if __name__ == '__main__':
    main()
