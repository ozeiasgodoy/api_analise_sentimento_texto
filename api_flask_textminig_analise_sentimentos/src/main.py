from flask import Flask, request, send_file
from unidecode import unidecode
import re
import nltk
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
nltk.download('vader_lexicon')
nltk.download('rslp')


app = Flask(__name__)


@app.route("/index", methods=['GET'])
def index():
    return {"message": "Api Funcionando"}, 200

@app.route("/Analise_Sent_Texto_One_Setence", methods=["GET", "POST"])
def pandas_upload():  
    if request.method == "POST":
        texto = request.get_json()["text"]

        new = []
        stemmer = nltk.stem.RSLPStemmer()
        #tudo para minusculo
        texto = texto.lower()
        #remove os acentos
        texto = unidecode(texto)
        #remove ponto e virgula
        texto = texto.replace(".","").replace(",","")
        
        textNew = ''
        for split in texto.split():
            #Deixa apenas o radical das palavras
            split = stemmer.stem(split)
            #Remove as plavras de parada
            if split not in stopwords.words('portuguese'):
                textNew = textNew + split + " "
    
        sid = SentimentIntensityAnalyzer()
        ss = sid.polarity_scores(texto)
        
        try:
            ss.pop('compound')
            plt.bar(range(len(ss)), list(ss.values()), align='center')
            plt.xticks(range(len(ss)), list(ss.keys()))
            plt.savefig('result.png')
            plt.switch_backend('agg')
            return send_file('../result.png', mimetype='image/gif')
        except:
            return {"Erro ao processar a requisição", 404}

@app.route("/Analise_Sent_Texto_Multi_Setence", methods=["GET", "POST"])
def pandas_upload_multi():  
    if request.method == "POST":
        texto = request.get_json()["text"] 

        newText = []
        for line in texto:
            stemmer = nltk.stem.RSLPStemmer()
            #tudo para minusculo
            line = line.lower()
            #remove os acentos
            line = unidecode(line)
            #remove ponto e virgula
            line = line.replace(".","").replace(",","")
            #Deixa apenas o radical das palavras
            lineNew = ''
            for split in line.split():
                split = stemmer.stem(split)
                lineNew = lineNew + split + " "
            
            newText.append(lineNew)

        first = True
        predominancia = {'neg':0, 'neu':0, 'pos':0}

        #for sentence in texto:
        for sentence in newText:
            sid = SentimentIntensityAnalyzer()
            #print(sentence.lower())
            ss = sid.polarity_scores(sentence)

            if not first:
                if ss["compound"] >=0.05:
                    predominancia["pos"] += 1
                else:
                    if ss["compound"] <= -0.05:
                        predominancia["neg"] += 1
                    else:
                        predominancia["neu"] += 1    
                
            first = False

        try:
            plt.bar(range(len(predominancia)), list(predominancia.values()), align='center')
            plt.xticks(range(len(predominancia)), list(predominancia.keys()))
            plt.savefig('result_multi.png')
            plt.switch_backend('agg')
            return send_file('../result_multi.png', mimetype='image/gif')
        except:
            return {"Erro ao processar a requisição", 404}

@app.route("/Analise_Sent_Texto_Multi_Setence_JSON", methods=["GET", "POST"])
def pandas_upload_multi_json():  
    if request.method == "POST":
        texto = request.get_json()["text"]
    
        newLine = []     
        results = []

        for line in texto:
            stemmer = nltk.stem.RSLPStemmer()
            #tudo para minusculo
            line = line.lower()
            #remove os acentos
            line = unidecode(line)
            #remove ponto e virgula
            line = line.replace(".","").replace(",","")
            
            newLine.append(line)

        lineNew = ''
        for sentence in newLine:
            for split in sentence.split():
                split = stemmer.stem(split)
                lineNew = lineNew + split + " "
                sid = SentimentIntensityAnalyzer()
                ss = sid.polarity_scores(lineNew)

            results.append( {"sentenca" : sentence, "score" : ss})
        
    return {"resultado" : results}

if __name__ == '__main__':
    app.run(host="localhost", port=3000)