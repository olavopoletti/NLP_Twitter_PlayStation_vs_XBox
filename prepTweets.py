# Imports
import pandas as pd
import re
import string
import glob
from fastparquet import write
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords 
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from collections import defaultdict
import os

# Brands under analysis and respective search terms
brands = dict(
            Xbox=['xbox'],
            Playstation=["ps1", "ps2", "ps3", "ps4", "ps5", "playstation"] 
            )

# List of punctuation
punctuation = [i for i in string.punctuation]

# Instantiate the tokenizer class
tokenizer = TweetTokenizer(preserve_case=False, 
                           strip_handles=True,
                           reduce_len=True)

# Map for the tags based on the solution proposed by 
# Shuchita Banthia on Stack Overflow
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

lematizer = WordNetLemmatizer()

# Dictionary with Language codes and languages
languages = {
            'und': 'undetected',
            'af': 'Afrikaans', 'af-ZA': 'Afrikaans', 'ar': 'Arabic',
            'ar-AE': 'Arabic', 'ar-BH': 'Arabic', 'ar-DZ': 'Arabic', 
            'ar-EG': 'Arabic', 'ar-IQ': 'Arabic', 'ar-JO': 'Arabic',
            'ar-KW': 'Arabic', 'ar-LB': 'Arabic', 'ar-LY': 'Arabic',
            'ar-MA': 'Arabic', 'ar-OM': 'Arabic', 'ar-QA': 'Arabic',
            'ar-SA': 'Arabic', 'ar-SY': 'Arabic', 'ar-SY': 'Arabic',
            'ar-YE': 'Arabic', 'az': 'Azeri', 'az-AZ': 'Azeri',
            'art': 'Artificial languages', 'bn': 'Bengali', 'am': 'Amharic',
            'be': 'Belarusian', 'be-BY': 'Belarusian', 'bg': 'Bulgarian',
            'bg-BG': 'Bulgarian', 'bs-BA': 'Bosnian', 'ca': 'Catalan',
            'ca-ES': 'Catalan', 'cs': 'Czech', 'cs-CZ': 'Czech', 'cy': 'Welsh',
            'cy-GB': 'Welsh','da': 'Danish', 'da-DK': 'Danish', 'de': 'German',
            'ckb': 'Central Kurdish',
            'de-AT': 'German', 'de-CH': 'German', 'de-DE': 'German',
            'de-LI': 'German', 'de-LU': 'German', 'dv': 'Divehi',
            'dv-MV': 'Divehi', 'el': 'Greek', 'el-GR': 'Greek',
            'en': 'English', 'en': 'English', 'en-AU': 'English',
            'en-BZ': 'English', 'en-CA': 'English', 'en-CB': 'English',
            'en-GB': 'English', 'en-IE': 'English', 'en-JM': 'English',
            'en-NZ': 'English', 'en-PH': 'English', 'en-TT': 'English',
            'en-US': 'English', 'en-ZA': 'English', 'en-ZW': 'English',
            'eo': 'Esperanto', 'es': 'Spanish', 'es-AR': 'Spanish',
            'es-BO': 'Spanish', 'es-CL': 'Spanish', 'es-CO': 'Spanish',
            'es-CR': 'Spanish', 'es-DO': 'Spanish', 'es-EC': 'Spanish',
            'es-ES': 'Spanish', 'es-GT': 'Spanish', 'es-HN': 'Spanish',
            'es-MX': 'Spanish', 'es-NI': 'Spanish', 'es-PA': 'Spanish',
            'es-PE': 'Spanish', 'es-PR': 'Spanish', 'es-PY': 'Spanish',
            'es-SV': 'Spanish', 'es-UY': 'Spanish', 'es-VE': 'Spanish',
            'et': 'Estonian', 'et-EE': 'Estonian', 'eu': 'Basque',
            'eu-ES': 'Basque', 'fa': 'Farsi', 'fa-IR': 'Farsi',
            'fi': 'Finnish', 'fi-FI': 'Finnish', 'fo': 'Faroese',
            'fo-FR': 'Faroese', 'fr': 'French', 'fr-BE': 'French',
            'fr-CA': 'French', 'fr-CH': 'French', 'fr-FR': 'French',
            'fr-LU': 'French', 'fr-MC': 'French', 'gl': 'Galician',
            'gl-ES': 'Galician', 'gu': 'Gujarati', 'gu-IN': 'Gujarati',
            'he': 'Hebrew', 'he-IL': 'Hebrew', 'hi': 'Hindi', 'hi-IN': 'Hindi',
            'hr': 'Croatian', 'hr-BA': 'Croatian', 'hr-HR': 'Croatian',
            'ht': 'Haitian', 'hu': 'Hungarian', 'hu-HU': 'Hungarian', 'hy': '',
            'ny': 'Armenian', 'ny-AM': 'Armenian', 'id': 'Indonesian',
            'id-ID': 'Indonesian', 'is': 'Icelandic', 'is-IS': 'Icelandic',
            'it': 'Italian', 'it-CH': 'Italian', 'it-IT': 'Italian',
            'iw': 'Hebrew',
            'ja': 'Japanese', 'ja-JP': 'Japanese', 'ka': 'Georgian',
            'ka-GE': 'Georgian', 'ka': 'Georgian', 'ka-GE': 'Georgian',
            'kk': 'Kazakh', 'kk-KZ': 'Kazakh', 'kn': 'Kannada',
            'kn-IN': 'Kannada', 'in': 'Indonesian', 'ind': 'Indonesian',
            'ko': 'Korean', 'km': 'Khmer',
            'ko-KR': 'Korean', 'kok': 'Konkani', 'kok-IN': 'Konkani',
            'ky': 'Kyrgyz', 'ky-KG': 'Kyrgyz', 'lt': 'Lithuanian',
            'lt-LT': 'Lithuanian', 'lv': 'Latvian', 'lv-LV': 'Latvian',
            'lo': 'Lao', 'ml': 'Malayalam',
            'mi': 'Maori', 'mi-NZ': 'Maori', 'mk': 'FYRO Macedonian',
            'mk-MK': 'FYRO Macedonian', 'mn': 'Mongolian', 'mn-MN': 'Mongolian',
            'mr': 'Marathi', 'mr-IN': 'Marathi', 'ms': 'Malay',
            'ms-BN': 'Malay', 'ms-MY': 'Malay', 'mt': 'Maltese',
            'mt-MT': 'Maltese', 'nb': 'Norwegian', 'no': 'Norwegian',
            'my': 'Malay', 'nb-NO': 'Norwegian', 'ne': 'Nepali',
            'nl': 'Dutch', 'nl-BE': 'Dutch', 'nl-NL': 'Dutch', 
            'nn-NO': 'Norwegian', 'ns': 'Northern Sotho', 'or': 'Oriya',
            'ns-ZA': 'Northern Sotho', 'pa': 'Punjabi', 'pa-IN': 'Punjabi',
            'pl': 'Polish', 'pl-PL': 'Polish', 'ps': 'Pashto', 
            'ps-AR': 'Pashto', 'pt': 'Portuguese', 'pt-BR': 'Portuguese',
            'pt-PT': 'Portuguese', 'qu': 'Quechua', 'qu-BO': 'Quechua',
            'qu-EC': 'Quechua', 'qu-PE': 'Quechua', 'qme': '', 'qht': '',
            'qam': '', 'ro': 'Romanian',
            'qst': 'Relexified Portuguese-Based Creole',
            'ro-RO': 'Romanian', 'ru': 'Russian', 'ru-RU': 'Russian',
            'sa': 'Sanskrit', 'sa-IN': 'Sanskrit', 'se': 'Sami',
            'se-FI': 'Sami', 'se-NO': 'Sami', 'se-SE': 'Sami', 'sk': 'Slovak',
            'si': 'Sinhala', 'sd': 'Sindhi',
            'sk-SK': 'Slovak', 'sl': 'Slovenian', 'sl-SI': 'Slovenian',
            'sq': 'Albanian', 'sq-AL': 'Albanian', 'sr': 'Serbian', 
            'sr-BA': 'Serbian',
            'sr-SP': 'Serbian', 'sv': 'Swedish', 'sv-FI': 'Swedish',
            'sv-SE': 'Swedish', 'sw': 'Swahili', 'sw-KE': 'Swahili',
            'syr': 'Syriac', 'syr-SY': 'Syriac', 'ta': 'Tamil',
            'ta-IN': 'Tamil', 'te': 'Telugu', 'te-IN': 'Telugu', 'th': 'Thai',
            'th-TH': 'Thai', 'tl': 'Tagalog', 'tl': 'Tagalog',
            'tl-PH': 'Tagalog', 'tn': 'Tswana', 'tn--ZA': 'Tswana',
            'tr': 'Turkish', 'tr-TR': 'Turkish', 'tt': 'Tatar',
            'tt-RU': 'Tatar','ts': 'Tsonga', 'uk': 'Ukrainian',
            'uk-UA': 'Ukrainian', 'ur': 'Urdu', 'ur-PK': 'Urdu', 'uz': 'Uzbek',
            'uz-UZ': 'Uzbek', 'vi': 'Vietnamese', 'vi-VI': 'Vietnamese',
            'xh': 'Xhosa', 'xh-ZA': 'Xhosa', 'zh': 'Chinese',
            'zh-CN': 'Chinese', 'zh-HK': 'Chinese', 'zh-MO': 'Chinese',
            'zh-SG': 'Chinese', 'zh-TW': 'Chinese', 'zu': 'Zulu',
            'zu-ZA': 'Zulu', 'zxx': 'No linguistic content'          
            }

# Dictionary with stop words for every language in the data frame
stop_words = dict()
for l in list(set(languages.values())):
    try:
        stop_words[l] = stopwords.words(l)
    except OSError:
        pass

# A function for preprocessing tweets
def processTweets(tweetsDf):
        
    # Create a dataframe
    df = tweetsDf
    
    # Convert the column "lang" to string so we can filter it
    df['lang'] = df.lang.astype('string')
    
    # Keeping the tweets written in Portuguese, Spanish, and English 
    # df = df[df.lang.isin(['pt','es', 'en'])]
    
    # Extracting user name and location from 'user'
    df['name'] = df.user.apply(lambda x: x['displayname'])
    df['location'] = df.user.apply(lambda x: x['location'])
    
    # Dropping column we don't plan using
    df.drop(
            [
            '_type',
            'url',
            'renderedContent',
            'user',
            'retweetedTweet',
            'quotedTweet',
            'inReplyToTweetId',
            'inReplyToUser',
            'mentionedUsers',
            'coordinates',
            'place',
            #'hashtags',
            'cashtags',
            'card',
            'conversationId',
            'sourceUrl',
            'sourceLabel',
            'links',
            'source',
            'media',
            ],
            axis=1,
            inplace=True
            )
    
    # Convert the hashtags into strings
    df['hashtags'] = df.hashtags.astype('string')
    
    # Convert the language codes into strings
    df['lang'] = df.lang.astype('string')
    
    # Convert language codes into standard language
    df['language'] = df.lang.apply(lambda x: languages.get(x, 'Not Found'))
    
    # Convert the body of the tweet into strings
    df['tweet'] = df.rawContent.astype('string')
    
    # Remove retweet text "RT"
    df['clean_tweet'] = df.tweet.apply(lambda x: re.sub(r'^RT[\s]+', '', x))
    
    # Remove hyperlinks
    df['clean_tweet'] = df.clean_tweet.apply(
                                        lambda x:
                                            re.sub(
                                                r'https?:\/\/.*[\r\n]*',
                                                '',
                                                x
                                                )
                                            )
    
    # Remove the hash # sign
    df['clean_tweet'] = df.clean_tweet.apply(lambda x: re.sub(r'#', '', x))
    
    # Tokenize the tweets
    df['token'] = df.clean_tweet.apply(lambda x: tokenizer.tokenize(x))
    
    # Lists of punctuation for each row
    df['punctuation'] = [punctuation] * len(df.token)
    
    # Lists of stop words according with the row language
    df['stop_words'] = df.language.apply(lambda x: stop_words.get(x, []))
    
    # Lists combining punctuation and stop words for each row
    df['cleaner'] = df.stop_words + df.punctuation
    
    # An auxiliary column to help remove
    # stop words and punctuation from the tokens
    df['temp_ct'] = tuple(zip(df.token, df.cleaner))
    
    # Remove stop words and punctuation from the tokes
    df['clean_tokens'] = df.temp_ct.apply(
                                    lambda x:
                                        list(set(x[0]).difference(set(x[1])))
                                        )
    
    # Identify a Pos Tag (a characterization for the word) for each token
    df['tags'] = df.token.apply(lambda x: pos_tag(x))
    
    # Add the lemma (root-meaning) for each token
    df['lemmas'] = df.tags.apply(
                            lambda x:
                                [lematizer.lemmatize(i[0], tag_map[i[1][0]])\
                                    for i in x]
                                )
    
    # An auxiliary column to help remove
    # stop words and punctuation from the lemmas
    df['temp_cl'] = tuple(zip(df.lemmas, df.cleaner))      
    
    # Remove stop words and punctuation from the tokes
    df['clean_lemmas'] = df.temp_cl.apply(
                                    lambda x:
                                        list(set(x[0]).difference(set(x[1])))
                                        )
    df = df.drop(
                columns=[
                    'temp_cl', 'temp_ct', 'cleaner', 'punctuation',
                    'stop_words',
                    ]
                )
    return df

def mergeTweets(brand):
    
    # Listing files for brand's search terms
    files = []
    for f in brands[brand]:
        files.extend(glob.glob('.\\Data\\*{}.json'.format(f)))
    
    # Create the directory
    path = f'.\\data\\{brand}'
    if not os.path.isdir(path):
        os.makedirs(path)

    # Determine if we should append the data frame
    partitions = 0

    # List the ids already processed
    ids = []
    
    for f in files:
              
        # Data frame for the current file
        df = pd.read_json(f, lines=True)
        
        # Remove the duplicate ids
        df = df[~df.id.isin(ids)]
        
        # Process Tweets
        df = processTweets(df)

        # Add the brand to the data frame
        df['Brand'] = brand
        
        # Save the data frame as a parquet file
        df.to_parquet(
                        '{}\\{}.parquet'.format(path, partitions),
                        compression='GZIP',
                        engine='pyarrow',
                        )

        # Update the need for appending a new partition
        partitions += 1
        
        # Update the list of info already considered
        ids.extend(df.id.values)
    os.close(path)        
    os.rename(path, f'.\\Data\\{brand}.parquet')
    
mergeTweets('Playstation')