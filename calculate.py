import string
import pandas as pd
import nltk
import sys
from nltk.corpus import stopwords
nltk.download('stopwords')

def main():
    number_grams = 3

    number_top_keywords = 20
    save_file = open(sys.argv[1], 'a')

    script = open(sys.argv[2], "r").read()
    total_trans = open(sys.argv[3], "r").read()
    names_trans = [str(sys.argv[3]) + "\n"]

    for tran in sys.argv[4:]:
        total_trans += open(tran, "r").read()
        names_trans.append(str(tran) + "\n")

    script_data = ngrams_to_strings(get_n_grams(text_process(script), number_grams))
    script_df = group_in_dataframe(script_data, "Main script")

    script_df_top = script_df.head(number_top_keywords)

    total_trans_data = ngrams_to_strings(get_n_grams(text_process(total_trans), number_grams))
    total_trans_df = group_in_dataframe(total_trans_data, "Transcripts")

    script_trans_df = pd.concat([script_df_top, total_trans_df], axis=1, join="inner")

    script_trans_df = script_trans_df.sort_values("Transcripts", ascending=False)

    string1 = "\nMain script:\n%s" % sys.argv[2]
    string2 = "\nTranscripts:\n"
    string3 = "\nThe top %s key-words in the main script:\n" % number_top_keywords
    string4 = "\nThe top %s key-words in the main script, ranked by appearance in the transcripts:\n" % number_top_keywords

    printlist = [string1, string2] + names_trans + [string3, str(script_df_top), string4, str(script_trans_df)]

    for string in printlist:
        print(string)
        save_file.write(string)

def text_process(text):

    no_punc = [char for char in text if char not in string.punctuation]

    no_punc = ''.join(no_punc)

    no_stopw = [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]

    stemmer = nltk.stem.snowball.EnglishStemmer(no_stopw)
    return [stemmer.stem(i) for i in no_stopw]

def get_n_grams(word_list, n):

    ngrams = []
    count = 1
    while count <= n:
        for i in range(len(word_list)-(count-1)):
            ngrams.append(word_list[i:i+count])
        count += 1
    return ngrams

def ngrams_to_strings(ngrams):

    ngrams_sorted = ([sorted(i) for i in ngrams])
    return [' '.join(i) for i in ngrams_sorted]


def group_in_dataframe(data, column_name):

    df = pd.DataFrame(data=data, columns=["key-word"])
    df = pd.DataFrame(df.groupby("key-word").size().rename(column_name))
    return df.sort_values(column_name, ascending=False)


if __name__ == "__main__":
    main()