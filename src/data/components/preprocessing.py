import os
import pandas as pd
import re
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect

def process_dat_file(dat_file, folder_img_path):
    df = pd.read_csv(dat_file, engine='python',sep='::', names=['movieid', 'title', 'genre'], encoding='latin-1', index_col=False).set_index('movieid')
    df['genre'] = df.genre.str.split('|')
    df['id'] = df.index
    df['img_path'] = df.apply(lambda row: os.path.join(folder_img_path, f'{row.id}.jpg') if os.path.exists(os.path.join(folder_img_path, f'{row.id}.jpg')) else '', axis=1)
    return df

def random_oversampling(df, target_samples):
    # dataframe
    oversampled_data = df.copy()
    movies_train_copy = df.copy()
    movies_train_copy = movies_train_copy[~movies_train_copy['genre'].apply(lambda x: True if "Drama" in x else False)]
    movies_train_copy = movies_train_copy[~movies_train_copy['genre'].apply(lambda x: True if "Comedy" in x else False)]

    # oversampling
    genres_list = [
        "Film-Noir",
        "Western",
        "Fantasy",
        "Animation",
        "Mystery",
        "Documentary",
        "Musical",
        "War",
        "Crime",
        "Children's",
        "Sci-Fi",
        "Adventure",
        "Horror",
        "Romance",
        "Thriller",
        "Action",
        "Comedy",
        "Drama"
    ]
    for genre in genres_list:
        if genre in ["Drama", "Comedy"]:
            continue
        
        freq = len(oversampled_data[oversampled_data['genre'].apply(lambda x : True if genre in x else False)])
        oversample_factor = target_samples / freq
        if oversample_factor < 1:
            oversample_factor = 1
        genre_data = movies_train_copy[movies_train_copy['genre'].apply(lambda x: True if genre in x else False)]
        oversampled_genre_data = genre_data.sample(int(freq * (oversample_factor - 1)) + 1, replace=True, random_state=1012)
        oversampled_data = pd.concat([oversampled_data, oversampled_genre_data])

    # # print
    # gc = oversampled_data['genre'].explode().value_counts()
    # print("\n__________________ RANDOM OVERSAMPLING ________________")
    # print(f"Number of genres: {len(gc)} \n")
    # print(f"Frequency: \n{gc}")
    # print("\n")
    
    return oversampled_data

def upsampling_all(df, factor):
    upsampling_data = df.copy()
    return pd.concat([upsampling_data] * factor)

def tokenize(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = lower_text(text)
    text = remove_year(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = remove_non_english_words_in_parentheses(text)
    return text

def lower_text(text):
    return text.lower()

def remove_year(text):
    tokens = wordpunct_tokenize(text)
    tokens = tokens[:-1] # remove last token because it is the year which maybe is not useful
    text = ' '.join(tokens)
    return text

def remove_punctuation(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    personal_pronouns = set(['i', 'me', 'my', 'myself', 'you', 'your', 'yours', 'yourself',
                                'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
                                'we', 'us', 'our', 'ours', 'ourselves',
                                'they', 'them', 'their', 'theirs', 'themselves'])

    stop_words_v2 = stop_words - personal_pronouns
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words_v2]
    text = ' '.join(filtered_words)
    return text

non_english_words = []
def remove_non_english_words_in_parentheses(input_str):
    def is_english(text):
      try:
          # Check if the language is English
          if(text.__contains__("a . k . a .")):
              return True

          return detect(text) == 'en'
      except:

          # Handle cases where language detection fails
          return False
    # Use regular expression to find phrases within parentheses
    pattern = r'\((.*?)\)'
    matches = re.findall(pattern, input_str)

    # Filter out non-English phrases
    english_matches = [match.strip() for match in matches if is_english(match.strip())]

    # Replace only non-English phrases within parentheses with an empty string
    for non_english_match in set(matches) - set(english_matches):
        non_english_words.append(non_english_match)
        input_str = input_str.replace(f'({non_english_match})', '')
        
    return input_str.strip()

if __name__ == "__main__":
    tokens = tokenize("Boys from Brazil, The (1978)	")
    print(tokens)