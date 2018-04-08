from nltk.tokenize import casual_tokenize
from nltk.tokenize import word_tokenize
from urllib.parse import quote
from nltk.corpus import stopwords
from rake_nltk import Rake
from pathlib import Path
from lxml import html
import requests
import argparse
import time
import json
import os
import re

from utils import read_text_file
from utils import get_wiki_url_and_content_by_keyphrase
from utils import write_suggestions_to_json
from utils import print_time
from utils import filter_pos
from utils import list_to_string
from utils import calc_character_similarity
from utils import get_shortest_in_list
from utils import get_sublist_index_in_list
from utils import pos_tag_phrase_pairs

"""
Execution: python -W ignore Recommender.py --article_text article.txt
"""


class Recommender(object):
    def __init__(self):
        pass

    def tuple_list_to_string_list(self):  # TODO set up rules?
        """
        Combine the noun list in each tuple in a list to a string.
        Rules for which nouns to use? Database of noun keywords?
        :param alist: List, list of tagged, noun phrase tuples with respective tf-idf scores
        :return: List, list of string, noun tuples with respective tf-idf scores
        """
        out_alist = []
        for pair in self.all_phrases_tagged_nouns:
            string_nouns = ''
            nouns = pair[1]
            new_phrase = []
            count = 0
            for noun in nouns:
                string_nouns = string_nouns + noun[0]  # noun[0] gets the noun word, disregarding the pos tag
                count += 1
                if count != len(nouns):  # Add a space between nouns for all non-last nouns
                    string_nouns = string_nouns + ' '
            out_alist.append((pair[0], string_nouns))
        return out_alist

    def tokenize_phrases(self):
        ret_list = []
        for phrase in self.all_phrases:
            phrase_tokenized = casual_tokenize(phrase[1])
            ret_list.append([phrase[0], phrase_tokenized])
        return ret_list

    def get_surrounding_n_tokens(self, indexes, n):
        """
        Returns a list of the surrouding words of a keyphrase in original text.
        Indexes should have the following format: [(x,y)]
        :param text_as_tokens: List, a lit of the tokenized text
        :param indexes: Tuple, a tuple of integers representing starting and ending indexes
        :param n: Integer, number of steps to take and words to collect in each direction from the starting and ending indexes
        :return: List, a concatenated list of tokens surrounding the original keyphrase
        """
        surr_tokens = self.article_text_tokenized[indexes[0]-n: indexes[0]] + self.article_text_tokenized[indexes[1]:indexes[1]+n]
        whole_context = self.article_text_tokenized[indexes[0]-n: indexes[1]+n]
        return surr_tokens, whole_context

    def get_all_surrounding_tokens(self):
        # TODO instead of surrounding n tokens get surrounding context including phrases and get cosine similarity
        surrounding_list = []
        context_list = []
        for phrases_tokenized in self.all_phrases_tokenized:
            indexes = get_sublist_index_in_list(phrases_tokenized[1], self.article_text_tokenized)
            if indexes == None:
                continue
            surrounding_tokens, context_tokens = self.get_surrounding_n_tokens(indexes, 20)
            surrounding_list.append(phrases_tokenized + [surrounding_tokens])
            context_list.append(phrases_tokenized + [context_tokens])
        return surrounding_list, context_list

    def map_sim_to_suggestions(self, original_string_keyphrase, suggestions):
        """
        Creates a dictionary mapping of suggestions with their respective similarity scores to the original keyphrase.
        :param original_string_keyphrase: String,
        :param suggestions: List, a list of strings to compare to the original keyphrase
        :return: List, list of similarity scores
        """
        sims = []  # a list of all similarity scores
        mapped_to_sim_tokens = {}  # mapped dictionary; key = suggestion, value = similarity score

        for suggestion in suggestions:
            sim = calc_character_similarity(suggestion, original_string_keyphrase)
            sims.append(sim)
            mapped_to_sim_tokens[suggestion] = sim

        return sims, mapped_to_sim_tokens

    def get_similar_suggestions(self, suggestions, original_tokens):  # TODO add counter to prioritize similarity
        ret = []
        suggestions_tokens = []
        for suggestion in suggestions:
            suggestions_tokens.append(casual_tokenize(suggestion))

        for token in original_tokens:
            for suggestion in suggestions_tokens:
                if token in suggestion:
                    if not suggestion in ret:
                        ret.append(suggestion)

        # out = self.order_list_by_similarity(original_tokens, ret)
        return ret

    def get_category_from_categories(self, phrases_list):
        lst = []
        bad_words = ['articles', 'links', 'containing']
        for phrase in phrases_list:
            phrase_words = phrase.split()
            for word in phrase_words:
                if word not in bad_words and word not in stopwords.words("english"):
                    lst.append(word)
            lst.append('.')

        text = list_to_string(lst)
        rec = Rake()
        rec.extract_keywords_from_text(text)
        cat = rec.get_ranked_phrases()
        return cat

    def get_n_listed_medium_posts(self, n):
        root = []

        c = 0
        keyp_url_content_mapping = []
        for phrase in self.string_phrases_nouns:
            if c > n:
                break
            root_url = "https://medium.com/search?q="
            first = "%20".join(phrase[1].split(" "))
            url = root_url + first
            href = "href='"
            quotation = "'"

            webpages = []

            page = requests.get(url)
            content = str(page.content)
            w_page = html.fromstring(content)
            w_pages = w_page.xpath("//a/@href")

            body_start = re.search('<div class="section-content"><div class="section-inner sectionLayout--insetColumn">', content).end()
            content_body_1 = content[body_start:]
            body_end = re.search('</div class="section-content"></div class="section-inner sectionLayout--insetColumn">', content_body_1).start()
            body = content_body_1[:body_end]

            title_start = re.search('<title>', content).end()
            content_title_1 = content[title_start:]
            title_end = re.search('</title>', content_title_1).start()
            content_title_2 = content_title_1[:title_end]
            title = content

            webpages.append([title, body])

            phrase_pages = []
            # Filter out bad or repeating links
            for p in w_pages:
                if "/@" in p and p.split("/")[-1][0] != "@" and p not in phrase_pages and "responses" not in p:
                    phrase_pages.append([p])

            if phrase_pages != []:
                for p in phrase_pages:
                    # all_pages.append([phrase[1], phrase_pages])

                    root_dict = {}
                    # content = str(requests.get(p).content)
                    # print(p)
                    # content.replace('<.*>', '')
                    # title_start = re.search('<title>')
                    # search_end = re.search('graf--title">', content).end()
                    # content = content[search_end:]
                    # graf_search_end = re.search('graf-after--figure">', content).end()
                    # content = content[graf_search_end:]
                    # print(content)

                    words = p.split("source=search_post")[0].split("-")
                    words = words[:len(words)-1]

                    root_dict["article_url"] = quote(str(p))
                    body = ""
                    category = ""
                    # root_dict["content"] = quote(str(content))  # TODO find text in content
                    root_dict["title"] = quote(str(phrase))
                    keyp_url_content_mapping.append([title, p, body, category])

                    root.append(root_dict)
                    # print(p)
                    # print(content)

                    c += 1

        # mapping = self.write_suggestions_to_json(keyp_url_content_mapping)
        return root

    def get_wiki_urls_top_n_phrases(self, n):
        # TODO get proper categories
        for i in range(n):
            phrase = self.string_phrases_nouns[i][1]
            try:
                url, content, cats = get_wiki_url_and_content_by_keyphrase(phrase)
                category = get_shortest_in_list(cats, phrase)

                root_list = {}
                root_list["article_url"] = url
                root_list["content"] = quote(str(content))
                root_list["title"] = quote(str(list_to_string(url.split('/')[-1].split('_'))))
                root_list["topic"] = quote(str(category))
                self.mapping.append(root_list)
            except Exception:
                try:
                    url, content, cats = get_wiki_url_and_content_by_keyphrase(phrase[:len(phrase.split(' ')) / 2])
                    category = get_shortest_in_list(cats, phrase[:len(phrase.split(' ')) / 2])

                    root_list = {}
                    root_list["article_url"] = url
                    root_list["content"] = quote(str(content))
                    root_list["title"] = quote(str(list_to_string(url.split('/')[-1].split('_'))))
                    root_list["topic"] = quote(str(category))
                    self.mapping.append(root_list)
                except Exception:
                    pass
                pass

    def process_text(self):
        # Remove new lines and turn to lower case
        text = re.sub('\n', ' ', self.text).lower()

        # Extract keyphrases using Rake
        # TODO also possible to extract keywords from sentence
        rake = Rake()
        if self.text_type == 'article':
            rake.extract_keywords_from_text(text)
        elif self.text_type == 'social':
            rake.extract_keywords_from_sentences(text)
        self.all_phrases = rake.get_ranked_phrases_with_scores()
        # word_freq_dist = rake.get_word_frequency_distribution()

        # Tokenize text
        self.article_text_tokenized = word_tokenize(text)

        # Tokenize phrases
        self.all_phrases_tokenized = self.tokenize_phrases()

        # Tag all phrases and remove all but noun words
        self.all_phrases_tagged = pos_tag_phrase_pairs(self.all_phrases)
        self.all_phrases_tagged_nouns = filter_pos(self.all_phrases_tagged, "nouns")

        # Convert list of tagged nouns back to a string phrase
        self.string_phrases_nouns = self.tuple_list_to_string_list()

        # Get the indexes from the non-filtered suggested phrases in the original text
        # self.all_surrounding_tokens, self.all_context_tokens = self.get_all_surrounding_tokens()

    def run(self, text, text_type, sites):
        """
        TODO Improvements:
        1. casual_tokenize can't handle 'words-with-hyphens-like-this' & reduces coverage
        """

        self.text = text
        self.text_type = text_type

        self.process_text()

        self.websites = [site.lower() for site in sites]
        self.mapping = []

        # Get wikipedia urls for top n phrases  TODO takes on average about 20-30 seconds per article of medium length
        if "wiki" in self.websites:
            self.get_wiki_urls_top_n_phrases(5)

        # Get page links on medium by phrase
        if "medium" in self.websites:
            self.get_n_listed_medium_posts(5)

        # Output final mapping of multiple sites
        print(self.mapping)

if __name__ == '__main__':
    # TODO try http://lxml.de/parsing.html for speed when scraping html
    # https://elitedatascience.com/python-web-scraping-libraries#lxml

    parser = argparse.ArgumentParser("Choose which kind of text document is passed to the recommender system.")
    parser.add_argument("--article_text", type=str, default=None)
    parser.add_argument("--social_text", type=str, default=None)
    args = parser.parse_args()

    rec = Recommender()

    # Run following 7 lines when reading from local articles directory, with each article being a single text file
    PATH = str(Path(__file__).resolve().parents[0])
    article_dir = os.path.join(PATH, "articles")
    if os.path.exists(article_dir):
        for file in os.listdir(article_dir):
            start_time = time.time()
            original_text = read_text_file(os.path.join(article_dir, file))
            rec.run(original_text, text_type="article", sites=["wiki"])
            print_time(start_time)

    # Run following 4 lines when passing text as argument value
    # if args.article_text:
    #     rec.run(args.article_text, text_type="article", sites=["wiki"])
    # elif args.social_text:
    #     rec.run(args.social_text, text_type="social", sites=["wiki"])
