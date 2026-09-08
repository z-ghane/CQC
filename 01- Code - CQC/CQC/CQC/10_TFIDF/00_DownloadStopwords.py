import nltk

# --------------- download stopwords

#nltk.download("stopwords")

nltk.download(
    'stopwords', 
    download_dir = r'E:\Repository\Prog Lang\python\nltk_data'
    )

nltk.data.path.append(r'E:\Repository\Prog Lang\python\nltk_data')
print(nltk.data.path)

