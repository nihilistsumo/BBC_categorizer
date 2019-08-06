import argparse, json, spacy

import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
from pathos.threading import ThreadPool

def preprocess_para(paratext, nlp):
    paratext = paratext.lower().replace('\n', ' ').replace('\t', ' ').replace('\xa0', ' ')
    paratext = ' '.join(paratext.split())
    doc = nlp(paratext)
    tokenized_sentences = []
    for s in doc.sents:
        # tokenized_sentences.append([t.text for t in nlp(s.string)])
        tokenized_sentences.append(s.string.split())
    return tokenized_sentences

def preprocessed_paratext(data_dict):
    nlp = spacy.load("en_core_web_sm")
    preproc_doctext_dict = dict()
    for docid in data_dict.keys():
        preproc_doctext_dict[docid] = preprocess_para(data_dict[docid]['title']+'. '+data_dict[docid]['text'], nlp)
    return preproc_doctext_dict

def get_elmo_embeddings(docid):
    # doclist_index_dict = dict()
    # start_index = 0
    # for docid in doclist:
    #     sent_count = len(preproc_doctext_dict[docid])
    #     doclist_index_dict[docid] = (start_index, start_index + sent_count)
    #     start_index += sent_count
    # sentences = []
    # for docid in doclist:
    #     sentences = sentences + preproc_doctext_dict[docid]
    sentences = preproc_doctext_dict[docid]
    elmo = ElmoEmbedder()
    embed_vecs = elmo.embed_sentences(sentences)
    doc_embed_vecs = []
    for i in range(len(sentences)):
        doc_embed_vecs.append(next(embed_vecs))
    doc_embed_dict[docid] = doc_embed_vecs
    # doc_embed_dict[docid] = embed_vecs
    # for docid in doclist_index_dict.keys():
    #     doc_embed_vecs = []
    #     for i in range(doclist_index_dict[docid][0], doclist_index_dict[docid][1]):
    #         doc_embed_vecs.append(next(embed_vecs))
    #     doc_embed_dict[docid] = doc_embed_vecs
    # print("{} documents embedded".format(len(doclist)))

parser = argparse.ArgumentParser(description="Generate ELMo embeddings for docs")
parser.add_argument("-d", "--data_dict", required=True, help="Path to bbc data dict file")
parser.add_argument("-tn", "--thread_count", type=int, required=True, help="No of threads in Thread pool")
parser.add_argument("-o", "--out", required=True, help="Path to output file")
args = vars(parser.parse_args())
bbc_data_dict_file = args["data_dict"]
thread_count = args["thread_count"]
outfile = args["out"]
with open(bbc_data_dict_file, 'r') as dd:
    bbc_data_dict = json.load(dd)
preproc_doctext_dict = preprocessed_paratext(bbc_data_dict)
doc_embed_dict = dict()
print("Data loaded")
doclist = list(preproc_doctext_dict.keys())

with ThreadPool(nodes=thread_count) as pool:
    pool.map(get_elmo_embeddings, doclist)

np.save(outfile, doc_embed_dict)