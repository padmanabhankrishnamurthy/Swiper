from hand_detection import detect_hands
import os
import string
import enchant

def get_unique_words(path):

    d = enchant.Dict("en_US")

    file = open(path, 'r')
    words = set()
    for line in file:
        line = line.translate(str.maketrans('', '', string.punctuation))
        for word in line.split():
            if word not in words and d.check(word): # we don't want stuff like wooooohooooooo in our lexicon
                words.add(word)
    return list(words)


if __name__ == '__main__':
    lyrics_path = 'lyrics.txt'
    words = get_unique_words(lyrics_path)
    print(words, len(words))

    save_path = 'data'

    detect_hands(words, save_path)