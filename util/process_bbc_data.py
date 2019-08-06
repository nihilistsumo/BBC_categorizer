import json, os, argparse, random

def organize_from_raw(cat_folder_list):
    doc_data_dict = dict()
    for fold in cat_folder_list:
        cat_name = fold.split('/')[len(fold.split('/')) - 1]
        # doc_data_dict[cat_name] = []
        for f in os.listdir(fold):
            doc_filepath = fold+'/'+f
            with open(doc_filepath, 'r') as d:
                try:
                    docid = cat_name + '.' + f.split('.')[0]
                    text = d.readlines()
                    title = text[0].rstrip('\n')
                    psg = ''
                    for i in range(1, len(text)):
                        if len(text[i]) > 1:
                            psg = psg + text[i].rstrip('\n')
                    doc_data_dict[docid] = {'title': title, 'text': psg, 'cat': cat_name}
                except UnicodeDecodeError:
                    print('Unicode decode error happened for doc: '+f+' in '+cat_name+', skipping...')
        print(cat_name+' done')
    return doc_data_dict

def random_train_test_split(doc_data, train_ratio, train_val_ratio):
    train_docs = random.sample(doc_data.keys(), round(len(doc_data) * train_ratio))
    test_docs = [doc for doc in doc_data.keys() if doc not in train_docs]
    val_docs = random.sample(train_docs, round(len(train_docs) * train_val_ratio))
    train_docs = [doc for doc in train_docs if doc not in val_docs]
    random.shuffle(train_docs)
    random.shuffle(val_docs)
    random.shuffle(test_docs)
    return train_docs, val_docs, test_docs

def main():
    random.seed(27)
    parser = argparse.ArgumentParser(description='Process raw bbc data')
    parser.add_argument('-bbc', '--bbc_folder', required=True, help='Path to BBC dataset folder')
    parser.add_argument('-t', '--train_ratio', type=float, required=True, help='Ratio of train split')
    parser.add_argument('-tv', '--train_val_ratio', type=float, required=True, help='Ratio of val split within train')
    parser.add_argument('-o', '--outdir', required=True, help='Path to output directory')
    args = vars(parser.parse_args())
    bbc_folder_path = args['bbc_folder']
    train_rat = args['train_ratio']
    val_rat = args['train_val_ratio']
    outpath = args['outdir']
    bbc_folders = []

    for dir in os.listdir(bbc_folder_path):
        if os.path.isdir(bbc_folder_path+'/'+dir):
            bbc_folders.append(bbc_folder_path+'/'+dir)
    bbc_data_dict = organize_from_raw(bbc_folders)
    train_docs, val_docs, test_docs = random_train_test_split(bbc_data_dict, train_rat, val_rat)
    doc_split_dict = {'train': train_docs, 'val': val_docs, 'test': test_docs}
    with open(outpath+'/bbc_data_dict.json', 'w') as dat:
        json.dump(bbc_data_dict, dat)
    with open(outpath+'/bbc_data_splits.json', 'w') as spl:
        json.dump(doc_split_dict, spl)

if __name__ == '__main__':
    main()