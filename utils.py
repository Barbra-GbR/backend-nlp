import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk import word_tokenize
from nltk import pos_tag
from difflib import SequenceMatcher
from urllib.parse import quote
import wikipedia
import warnings
import requests
import time

# from html.parser import HTMLParser
# class MLStripper(HTMLParser):
#     def __init__(self):
#         self.reset()
#         self.strict = False
#         self.convert_charrefs= True
#         self.fed = []
#     def handle_data(self, d):
#         self.fed.append(d)
#     def get_data(self):
#         return ''.join(self.fed)
# def strip_tags(html):
#     s = MLStripper()
#     s.feed(html)
#     return s.get_data()

def print_time(start_time, description=None):
    """
    Prints total time taken given a start time point.
    :param start_time: Time, time.time()
    :param description: String, description of action which was timed
    :return void
    """
    end_time = time.time()
    total_time = end_time - start_time
    print("Total time taken {}: ".format(description), time.strftime("%H:%M:%S", time.gmtime(int(total_time))))

def read_text_file(file):
    """
    Reads a file line by line and concatenates each line into a single string, which is then filtered for irrelevant
    characters.
    :param file: String, file name or absolute path
    :return: String, whole text in a document
    """
    text_string = ''
    with open(file, 'r',
              encoding='latin-1') as f:  # 'utf-8' codec can't decode byte 0x92 in position 2: invalid start byte
        reader = f.readlines()
        for line in reader:
            text_string = text_string + line + ' '
    return text_string

def write_suggestions_to_json(mapping):
    root = []

    for item in mapping:
        root_list = {}

        root_list["article_url"] = item[1]
        root_list["content"] = quote(str(item[2]))
        root_list["title"] = quote(str(list_to_string(item[1].split('/')[-1].split('_'))))
        root_list["topic"] = quote(str(item[3]))

        root.append(root_list)

    return root

def list_to_string(lst):
    str = ''
    for word in lst:
        str += word + ' '
    return str

def get_shortest_in_list(phrases_list, title_phrase):
    prev = phrases_list[0]
    for phrase in phrases_list:
        for sub_phrase in phrase.split(' '):
            # print('sub phrase: ', sub_phrase.lower(), '| title phrase: ', title_phrase.lower().split(' '))
            if sub_phrase.lower() in title_phrase.lower().split(' ') and len(str(phrase)) < len(str(prev)):
                prev = phrase
    return prev

def calc_character_similarity(a, b):
    """
    Measures the similarity of two strings and returns a score.
    :param a: String, comparing string a
    :param b: String, comparing string b
    :return: Float, a float in range (0.0 no similarity;:1.0 identical), representing the similarity between a and b
    """
    measure = SequenceMatcher(None, a, b).ratio()
    return measure

def get_wiki_url_and_content_by_keyphrase(phrase):
    with warnings.catch_warnings():  # TODO warning suppression
        warnings.simplefilter("ignore")
        wiki_page = wikipedia.page(phrase)
    return wiki_page.url, wiki_page.summary, wiki_page.categories

def search_wiki(phrase, results=10):
    wiki_search = wikipedia.search(phrase, results)
    return wiki_search

def get_content_from_url(url):
    return requests.get(url).content

def train_doc2vec_model(train_text, test_text):
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()),
                                  tags=str(i)) for i, _d in enumerate(train_text)]

    max_epochs = 10
    vec_size = 20
    alpha = 0.025

    model = Doc2Vec(size=vec_size,
                    alpha=alpha,
                    min_alpha=0.025,
                    min_count=1,
                    dm =1)
    model.build_vocab(tagged_data)

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    test_data = word_tokenize(test_text.lower())
    v1 = model.infer_vector(test_data)
    print("V1_infer", v1)

    similar_doc = model.docvecs.most_similar('1')
    print(similar_doc)

def surr_tok_list_to_string(lst):
    text = ''
    for item in lst[1]:
        text += item + ' '
    return text

def filter_pos(tagged_phrase_pairs, pos):
    """
    Filters everything but nouns out of a list in the tag phrase pair format.
    :param tagged_phrase_pairs: List, list of tagged phrase tuples with respective tf-idf scores
    :return: List, list of tagged, noun phrase tuples with respective tf-idf scores
    """
    tagged_noun_phrase_pairs = []
    pos_types = {"nouns": ['NN', 'NNS', 'NNP', 'NNPS']}
    for pair in tagged_phrase_pairs:
        word_noun_pair = pair[1]
        new_word_noun_pair = []
        for noun_word in word_noun_pair:
            if noun_word[1] in pos_types[pos]:
                new_word_noun_pair.append(noun_word)
        tagged_noun_phrase_pairs.append((pair[0], new_word_noun_pair))
    return tagged_noun_phrase_pairs

def get_sublist_index_in_list(sublist, l):  #TODO comment what happens in the search
    """
    Finds the starting and ending indexes of a sublist in a list.
    :param sublist: List, the sublist to be foud in list
    :param l: List, the list to look for sublist in
    :return: Tuple, a tuple of indexes representing starting and end index; or None if sublist index not found
    """
    results=[]
    sll=len(sublist)
    for ind in (i for i,e in enumerate(l) if e==sublist[0]):
        if l[ind:ind+sll]==sublist:
            results.append((ind,ind+sll-1))
    if results == []:
        # self.indexes_not_found += 1
        return None
    ret = results[0]
    return ret

def pos_tag_phrase_pairs(phrase_pairs_with_score):
    """
    Tokenizes the phrase and adds a pos tag to each token. Scores are discarded.
    :param phrase_pairs_with_score: List, list of phrase tuples with respective tf-idf scores
    :return: List, list of phrase list of tuples of word and pos tag pairs
    """
    tagged_text = []
    for pair in phrase_pairs_with_score:
        phrase = pair[1]
        tokens = word_tokenize(phrase)
        tags = pos_tag(tokens)
        tagged_text.append((pair[0], tags))

    return tagged_text
