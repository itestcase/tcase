import pandas as pd
import string
import sys
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def main():

    gr = 3

    top = 15
    save_file = open(sys.argv[1], 'a')

    script = open(sys.argv[2], "r").read()
    tr = open(sys.argv[3], "r").read()
    names = [str(sys.argv[3]) + "\n"]

    for tran in sys.argv[4:]:
        tr += open(tran, "r").read()
        names.append(str(tran) + "\n")

    sc = ngrams_to_strings(get_n_grams(text_process(script), gr))
    script_df = group_in_dataframe(sc, "Main script")

    top_script = script_df.head(top)

    total_trans_data = ngrams_to_strings(get_n_grams(text_process(tr), gr))
    total_trans_df = group_in_dataframe(total_trans_data, "Transcripts")

    trans = pd.concat([top_script, total_trans_df], axis=1, join="inner")

    trans = trans.sort_values("Transcripts", ascending=False)

    string1 = "\nMain script:\n%s" % sys.argv[2]
    string2 = "\nTranscripts:\n"
    string3 = "\nThe top %s key-words in the main script:\n" % top
    string4 = "\nThe top %s key-words in the main script, ranked by appearance in the transcripts:\n" % top

    printlist = [string1, string2] + names + [string3, str(top_script), string4, str(trans)]

    for string in printlist:
        print(string)
        save_file.write(string)

def text_process(text):

    wop = [char for char in text if char not in string.punctuation]

    wop = ''.join(wop)

    nop = [word for word in wop.split() if word.lower() not in stopwords.words('english')]

    stemmer = nltk.stem.snowball.EnglishStemmer(nop)
    
    return [stemmer.stem(i) for i in nop]

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