import pickle

# --- topic model utils --- 

def load_tm(topic_model):
    with open(f'{topic_model}/model.pkl', 'rb') as f: 
        model = pickle.load(f)
    with open(f'{topic_model}/vectorizer.pkl', 'rb') as f: 
        vectorizer = pickle.load(f)
    return model, vectorizer