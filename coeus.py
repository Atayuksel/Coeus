import pickle
import numpy as np
import nltk
from collections import deque
from progressbar import ProgressBar, Percentage, Bar


class Protein(object):

    def __init__(self, id):
        self.proteinID = id
        self.accessionNumbers = []
        self.proteinNames = []
        self.proteinGO = {}
        self.source = ' '

    def addAccessionNumber(self, number):
        self.accessionNumbers.append(number)

    def addProteinName(self, name):
        self.proteinNames.append(name)

    def addGO(self, key, value):
        self.proteinGO[key] = value

    def getGO(self):
        return self.proteinGO

    def getAccessionNumbers(self):
        return self.accessionNumbers

    def getProteinNames(self):
        return self.proteinNames

    def getProteinID(self):
        return self.proteinID

    def setSource(self, source):
        self.source = source


class WordDataset(object):
    def __init__(self, tra_xfile_name, test_xfile_name,
                 tra_yfile_name, test_yfile_name):

        self.batch_training_data = []
        self.batch_training_label = []
        self.batch_training_seqlen = []

        self.test_data = []
        self.test_labels = []
        self.test_seqlen = []

        self.max_seq_len = 0
        self.batch_id = 0
        self.classlength = 0
        self.datasetsize = 0

        with open(tra_xfile_name, 'rb') as f:
            self.full_training_protein = pickle.load(f)
        with open(tra_yfile_name, 'rb') as f:
            self.full_training_label = pickle.load(f)
        with open(test_xfile_name, 'rb') as f:
            self.full_test_protein = pickle.load(f)
        with open(test_yfile_name, 'rb') as f:
            self.full_test_label = pickle.load(f)

        # create character dictionary
        print("Creating character dictionary...")
        self.char_to_id, self.max_seq_len = self.__createchardict(self.full_training_protein)

        for label in self.full_training_label:
            self.datasetsize += 1
            if self.classlength < label:
                self.classlength = label
        self.classlength += 1

    def next(self, batch_size):
        # reset batch data
        self.batch_training_data = []
        self.batch_training_label = []
        self.batch_training_seqlen = []

        # reset batch_id
        if self.batch_id == len(self.full_training_protein):
            self.batch_id = 0

        # create batch input data
        # print("Creating batch input data...")
        batch_data = (self.full_training_protein[self.batch_id:min(self.batch_id + batch_size,
                                                                   len(self.full_training_protein))])

        self.batch_training_data, self.batch_training_seqlen = self.__createinputdata(batch_data, self.char_to_id,
                                                                                      self.max_seq_len)

        # create batch label data
        # print("Creating batch label data...")
        batch_label = (self.full_training_label[self.batch_id:min((self.batch_id + batch_size),
                                                                  len(self.full_training_label))])
        self.batch_training_label = self.__createlabeldata(batch_label, self.classlength)

        # set batch_id
        self.batch_id = min(self.batch_id + batch_size, len(self.full_training_protein))

        # return batch data
        return self.batch_training_data, self.batch_training_label, self.batch_training_seqlen

    def gettestset(self):

        self.test_data, self.test_seqlen = self.__createinputdata(self.full_test_protein, self.char_to_id,
                                                                  self.max_seq_len)
        self.test_labels = self.__createlabeldata(self.full_test_label, self.classlength)

        return self.test_data, self.test_labels, self.test_seqlen

    def getmaxseqlen(self):
        return self.max_seq_len

    def getclasslength(self):
        return self.classlength

    def getdictionarysize(self):
        return len(self.char_to_id)

    def getdatasetsize(self):
        return self.datasetsize

    def gettestsize(self):
        return len(self.test_data)

    def get_class_words(self, class_label):
        if class_label in self.full_training_label and (class_label+1) in self.full_training_label:
            start_index = self.full_training_label.index(class_label)
            end_index = self.full_training_label.index(class_label+1)
            training_labels = self.full_training_protein[start_index:end_index]
        elif class_label == self.classlength:
            start_index = self.full_training_label.index(class_label)
            training_labels = self.full_training_protein[start_index:]
        elif class_label in self.full_training_label and (class_label+1) not in self.full_training_label:
            start_index = self.full_training_label.index(class_label)
            tmp = 1
            while (class_label+tmp) not in self.full_training_label:
                tmp = tmp + 1
            end_index = self.full_training_label.index(class_label + tmp)
            training_labels = self.full_training_protein[start_index:end_index]
        return training_labels

    def get_test_words(self):
        return self.full_test_protein

    def get_test_labels(self):
        return self.full_test_label

    @staticmethod
    def __createchardict(wordlist):
        pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(wordlist)).start()
        char_to_id = {'UNK': 0}
        count = 1
        wordcount = 0
        max_seq_len = 0
        for word in wordlist:
            if len(word) > max_seq_len:
                max_seq_len = len(word)
            for char in word:
                if char not in char_to_id:
                    char_to_id[char] = count
                    count += 1
            wordcount += 1
            pbar.update(wordcount)
        pbar.finish()
        return char_to_id, max_seq_len

    @staticmethod
    def __createinputdata(wordlist, char_to_id, max_seq_len):
        #pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(wordlist)).start()
        inputdata = []
        seq_len = []
        dictionary_size = len(char_to_id)
        wordcount = 0
        for word in wordlist:
            if len(word) > max_seq_len:
                word = word[:max_seq_len]
            word_representation = np.zeros((max_seq_len, dictionary_size), dtype=float)
            seq_len.append(len(word))
            for idx in range(len(word)):
                char = word[idx]
                if char in char_to_id:
                    char_id = char_to_id[char]
                else:
                    char_id = 0
                word_representation[idx, char_id] = 1
            inputdata.append(word_representation)
            wordcount += 1
            #pbar.update(wordcount)
        #pbar.finish()
        return inputdata, seq_len

    @staticmethod
    def __createlabeldata(labellist, classlength):
        #pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(labellist)).start()
        max_functionid = 0
        labeldata = []
        instancecount = 0
        for label in labellist:
            label_representation = np.zeros(classlength, dtype=float)
            label_representation[label] = 1
            labeldata.append(label_representation)
            instancecount += 1
            #pbar.update(instancecount)
        #pbar.finish()
        return labeldata


class FamilyNode(object):
    def __init__(self, family_name):
        self.children = []
        self.name = family_name
        self.proteins = []

    def addChild(self, node):
        self.children.append(node)

    def addProtien(self, protein):
        self.proteins.append(protein)


class FamilyTree(object):
    def __init__(self):
        self.root = FamilyNode('root')

    def get_superfamilies(self, contain_none):
        superfamilies = self.root.children
        for node in superfamilies:
            if node.name == 'NoneSuperFamily' and not contain_none:
                superfamilies.remove(node)
        return superfamilies

    def get_families(self, contain_none):
        superfamilies = self.get_superfamilies(True)
        families = []
        for superfamily in superfamilies:
            for family in superfamily.children:
                if family.name != 'NoneFamily' or contain_none:
                    families.append(family)
        return families

    def get_subfamilies(self):
        families = self.get_families(True)
        subfamilies = []
        for family in families:
            for subfamily in family.children:
                subfamilies.append(subfamily)
        return subfamilies

    def get_subsubfamilies(self):
        subfamilies = self.get_subfamilies()
        subsubfamilies = []
        for subfamily in subfamilies:
            for subsubfamily in subfamily.children:
                subsubfamilies.append(subsubfamily)
        return subsubfamilies


    def find_node(self, key):
        nodeQueue = deque()
        nodeQueue.append(self.root)
        while len(nodeQueue) != 0:
            curNode = nodeQueue.pop()
            if curNode.name == key:
                return curNode
            else:
                for node in curNode.children:
                    nodeQueue.append(node)
        return -1

    def _create_path(self, pathdic):
        tmp = pathdic
        if 'superfamily' in pathdic:
            return tmp
        else:
            tmp['superfamily'] = 'NoneSuperFamily'
            if 'family' not in pathdic:
                tmp['family'] = 'NoneFamily'
                return tmp
            else:
                return tmp

    def _add_to_path(self, pathdic, protein_list):

        superFamilyName = pathdic['superfamily']
        superFamilies = self.root.children
        superFamilyNode = None
        for node in superFamilies:
            if node.name == superFamilyName:
                superFamilyNode = node
                break
        if superFamilyNode is None:
            superFamilyNode = FamilyNode(superFamilyName)
            superFamilies.append(superFamilyNode)
        for protein in protein_list:
            superFamilyNode.proteins.append(protein)

        if 'family' in pathdic:
            family_name = pathdic['family']
            family_node = None
            families = superFamilyNode.children
            for node in families:
                if node.name == family_name:
                    family_node = node
                    break
            if family_node is None:
                family_node = FamilyNode(family_name)
                families.append(family_node)
            for protein in protein_list:
                family_node.proteins.append(protein)

        if 'subfamily' in pathdic:
            subfamily_name = pathdic['subfamily']
            subfamily_node = None
            subfamilies = family_node.children
            for node in subfamilies:
                if node.name == subfamily_name:
                    subfamily_node = node
                    break
            if subfamily_node is None:
                subfamily_node = FamilyNode(subfamily_name)
                subfamilies.append(subfamily_node)
            for protein in protein_list:
                subfamily_node.proteins.append(protein)

        if 'sub-subfamily' in pathdic:
            sub_subfamily_name = pathdic['sub-subfamily']
            sub_subfamily_node = None
            sub_subfamilies = subfamily_node.children
            for node in sub_subfamilies:
                if node.name == sub_subfamily_name:
                    sub_subfamily_node = node
                    break
            if sub_subfamily_node is None:
                sub_subfamily_node = FamilyNode(sub_subfamily_name)
                sub_subfamilies.append(sub_subfamily_node)
            for protein in protein_list:
                sub_subfamily_node.proteins.append(protein)

    def add_proteins_to_tree(self, familyInfo, protein_list):
        pathdic = self._create_path(familyInfo)
        self._add_to_path(pathdic, protein_list)


class GeniaTagger(object):
    def __init__(self, abstract_location):
        self.abstract_dir = abstract_location
        self.sentences = []

    def create_tagger_input(self):
        with open(self.abstract_dir, 'r', encoding="utf8") as file:
            lines = file.readlines()
            for line in lines:
                line = line.split('\t')
                abstractID = line[0]
                title = line[1]
                abstract = line[2]
                text = title + ' ' + abstract
                sentences = nltk.sent_tokenize(text)
                for sentence in sentences:
                    self.sentences.append(sentence)
        return self.sentences