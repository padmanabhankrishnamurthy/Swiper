from hand_detection import detect_hands
import os
import string
import enchant

def get_stored_words(save_path):
    stored = set()
    for file in os.listdir(save_path):
        if 'jpg' in file and file[:file.find('_')] not in stored:
            stored.add(file[:file.find('_')].lower())
    return stored

def get_unique_words(path, save_path):
    stored_words = get_stored_words(save_path)

    d = enchant.Dict("en_US")

    file = open(path, 'r')
    words = set()
    for line in file:
        line = line.translate(str.maketrans('', '', string.punctuation))
        for word in line.split():
            word = word.lower()
            if word not in stored_words and word not in words and d.check(word): # we don't want stuff like wooooohooooooo in our lexicon
                words.add(word)
    return list(words)

def main():
    lyrics_path = 'lyrics.txt'
    save_path = 'data'

    words = get_unique_words(lyrics_path, save_path)
    words = [word.lower() for word in words]
    print(words, len(words))

    detect_hands(words, save_path)

if __name__ == '__main__':
    # main()
    # stored_words = get_stored_words('data')
    # unique_words = get_unique_words('lyrics.txt', 'data')
    # unique_words = [word.lower() for word in unique_words]
    # print(len(stored_words), stored_words)
    # print(len(unique_words), unique_words)
    # print([word for word in unique_words if word in stored_words])