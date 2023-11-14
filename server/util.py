import pickle
import spacy
nlp = spacy.load("en_core_web_sm")

__model = None

#To remove is_stop or punctuations from the text
def preprocess(text):
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)
    return " ".join(filtered_tokens)


#To get the acutal prediction 
def get_text_prediction(text):
    load_saved_model()
    #Step1 -Preprocess the text using NLP
    preprocessed_text = []
    for item in text:
        preprocessed_text.append(preprocess(item))
    return __model.predict_proba(preprocessed_text)[0]


#Loading the saved pickle model
def load_saved_model():
    global __model

    with open('C:/Users/HP/Desktop/ML & DL/ML Project/Detect AI generated text/server/artifacts/model.pkl', 'rb') as file:
        __model = pickle.load(file)
    print("Loading the saved model.....")
    return None

#Main Function
if __name__ == "__main__":
    load_saved_model()
    # get_text_prediction('text')
    # corpus = ["We present a review of unconstrained face recognition methods that are known in recent past. Park et al.24, proposed an key point detection method which enhanced the accuracy on the 1000 tattoo images. Experimental result has shown that the method is superior than the SIFT key point. A two level illumination estimation framework is proposed by Zhang et al.25. Experimental result shows the satisfactory performance against various datasets. Min et al.26, presented the possibility to explore face recognition in the presence of partial occlusions such as sunglass and scakrf. The proposed method identified the occluded parts t the pixel level and applying face recognition on nonocclued parts only. Experimental result shows that the proposed method is superior to KLD-LGBPHS, S-LNME, OA-LBP and RSC. Cassio et al.27, proposed five approach to determine whether the subject of test sample is enrolled in face gallery. Out of these approaches, three have been presented a good result. The approaches have been identified on feature explored in the data, scalability and accuracy. For face recognition the approaches have been tested on standard databases namely FRGC and Pubfig83. Various assumptions has been made by the authors before the approaches apply on the datasets. Indhumathi et al.28, proposed an algorithm for face detection using distance images. The effectiveness of the recognizing blurred and poorly well-lighted faces are also addressed by the author and shown good recognition rate than LBP operator. Bhattacharjee29, presented an automatic face recognition system using adaptive polar transformation and wavelet based fusion method. In which visual and thermal face images have been effectively combined by the fusion method on robust manner. Experimental result has shown good performance in recognizing against various possible complicacies. A Component based representation in automated face recognition has been demonstrated by Bonnen et al.30. Experimental result shown that the method is robust to change in facial pose, and recognition accuracy on occulded face is enhanced in forensic scenarios."]
    # predictions = get_text_prediction(corpus)
    # print(f"Human Written: {predictions[0][0]}, AI Written: {predictions[0][1]}")
    # print(predictions[0][0], predictions[0][1])