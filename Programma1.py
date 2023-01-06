#-*- coding: utf-8 -*-
import sys
import codecs
import nltk
import math
from nltk import bigrams

def estraiTestoTokenizzato(frasi): #funzione che estrae e ritorna testo tokenizzato e PoS taggato
	tokensTot = [] 
	tokensPOSTot = []
	for frase in frasi: 
		tokens = nltk.word_tokenize(frase) #testo tokenizzato
		tokensPOS = nltk.pos_tag(tokens) #PoS tag
		tokensTot = tokensTot + tokens
		tokensPOSTot = tokensPOSTot + tokensPOS
	return tokensTot, tokensPOSTot

def lunghezzamedia(frasi, tokens): #funzione che calcola e ritorna lunghezza media di frasi e di parole
	lunghezzaTOTParole = 0
	mediaFrasi = len(tokens)/len(frasi) #lunghezza media delle frasi in termini di token
	for i in tokens:
		lunghezzaTOTParole = lunghezzaTOTParole + len(i) #lunghezza media delle parole in termini di caratteri
	mediaParole = lunghezzaTOTParole/len(tokens) 
	return mediaFrasi, mediaParole

def distribuzionehapaxVocabolario(tokens): #funzione che stampa la distribuzione degli hapax e del vocabolario all'aumentare di 1000 token
	countHapax = 0 #counter di hapax
	arrayHapax = [] #array che conterrà 1000 token alla volta
	arrayVocabolario = []
	salto = 1000 
	for i in range(0, len(tokens), salto): #range che va di mille in mille
		arrayHapax = tokens [i : i + salto] #metto dentro all'array di volta in volta solo i 1000 token che sto scorrendo
		arrayVocabolario = arrayVocabolario + tokens [i : i + salto] #metto dentro all'array di volta in volta 1000 token
		vocabolario = set(arrayVocabolario) 
		print "Grandezza vocabolario in", i + salto, "token:",len(vocabolario)
		for j in arrayHapax: #scorro i mille token alla volta e conto gli hapax
			frequenzaToken = tokens.count(j)
			if (frequenzaToken == 1): 
				countHapax = countHapax + 1
		print "Numero di Hapax in", i + salto,"token:", countHapax

def rapportoSostantiviVerbi(testoPOSTaggato): #funzione che calcola e ritorna il rapporto tra sostantivi e verbi 
	verbiCounter = 0 #counter di verbi
	sostantiviCounter = 0 #counter di sostantivi
	for i in testoPOSTaggato:
		if i[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']: #categorie verbali
			verbiCounter = verbiCounter + 1
		if i[1] in ['NN', 'NNS', 'NNP', 'NNPS']: #categorie sostantivi
			sostantiviCounter = sostantiviCounter + 1
	rapporto = sostantiviCounter *1.0 / (verbiCounter *1.0) #moltiplico per 1.0 per divisione in termini decimali
	return rapporto

def estrazionePOStag(testoAnalizzato): 
	#funzione che estrae solamente la PoS tag dal testo PoS taggato e ritorna due liste di elementi PoS
	listaPOS = []
	for i in testoAnalizzato:
		listaPOS.append(i[1]) #appendo alla lista inizialmente vuota solo la PoS tag
	distribuzione = nltk.FreqDist(listaPOS) #calcolo frequenze degli elementi PoS tag
	listaPOSOrdinata = distribuzione.most_common(10) #ordino e ottengo i primi dieci elementi PoS tag più frequenti
	return listaPOS, listaPOSOrdinata #ritorno lista non ordinata e ordinata

def probabilitaCondizionataMassima(bigrammi, listaPOSTag, bigrammiDiversi): 
	#funzione che calcola per ogni bigramma la sua probabilità condizionata massima e ritorna il tutto in un dizionario ordinato
	dizionario = {} #creo dizionario vuoto
	for bigramma in bigrammiDiversi: #scorro bigrammi 
		freqBigramma = bigrammi.count(bigramma) #calcolo frequenza bigramma
		freqV = listaPOSTag.count(bigramma[1]) #calcolo frequenza del secondo elemento del bigramma
		probabilitaCondizionata = freqBigramma*1.0 / (freqV*1.0) #calcolo probabilità condizionata
		dizionario[bigramma] = probabilitaCondizionata #chiave: bigramma, elemento: probabilità condizionata
	return sorted(dizionario.items(), key = lambda x: x[1], reverse = True) #ritorno dizionario ordinato in maniera decrescente in base alla probabilità condizionata

def forzaAssociativaMassima(bigrammi, listaPOSTag, bigrammiDiversi): 
	#funzione che calcola per ogni bigramma la sua forza associativa massima e ritorna il tutto in un dizionario ordinato
	dizionario = {} #creo dizionario vuoto
	for bigramma in bigrammiDiversi: #scorro bigrammi 
		freqBigramma = bigrammi.count(bigramma) #calcolo frequenza bigramma
		freqU = listaPOSTag.count(bigramma[0]) #calcolo frequenza del primo elemento del bigramma
		freqV = listaPOSTag.count(bigramma[1]) #calcolo frequenza del secondo elemento del bigramma
		probBigramma = freqBigramma*1.0 / (len(bigrammi)*1.0) #calcolo probabilità del bigramma
		probU = freqU*1.0 / (len(listaPOSTag)*1.0) #calcolo probabilità del primo elemento del bigramma
		probV = freqV*1.0 / (len(listaPOSTag)*1.0) #calcolo probabilità del secondo elemento del bigramma
		frazione = probBigramma / (probU*probV) 
		localMutualInformation = freqBigramma * math.log(frazione,2) #calcolo LMI
		dizionario[bigramma] = localMutualInformation #chiave: bigramma, elemento: LMI
	return sorted(dizionario.items(), key = lambda x: x[1], reverse = True) #ritorno dizionario ordinato in maniera decrescente in base alla LMI

def main (file1, file2):
	fileInput1 = codecs.open(file1, "r", "utf-8") 
	fileInput2 = codecs.open(file2, "r", "utf-8")
	raw1= fileInput1.read()
	raw2= fileInput2.read()
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	frasi1 = tokenizer.tokenize(raw1)
	frasi2 = tokenizer.tokenize(raw2)
	testoTokenizzato1, testoPOSTaggato1 = estraiTestoTokenizzato(frasi1) 
	testoTokenizzato2, testoPOSTaggato2 = estraiTestoTokenizzato(frasi2)
	mediaFrasi1, mediaParole1 = lunghezzamedia(frasi1, testoTokenizzato1)
	mediaFrasi2, mediaParole2 = lunghezzamedia(frasi2, testoTokenizzato2)
	vocabolario1 = set(testoTokenizzato1)
	vocabolario2 = set(testoTokenizzato2)
	rapportoSostantiviVerbi1 = rapportoSostantiviVerbi(testoPOSTaggato1)
	rapportoSostantiviVerbi2 = rapportoSostantiviVerbi(testoPOSTaggato2)
	listaPOSTag1,POSTagPiuFrequenti1 = estrazionePOStag(testoPOSTaggato1) 
	listaPOSTag2,POSTagPiuFrequenti2 = estrazionePOStag(testoPOSTaggato2)
	bigrammi1 = list(bigrams(listaPOSTag1)) 
	bigrammiDiversi1 = set(bigrammi1) 
	bigrammi2 = list(bigrams(listaPOSTag2))
	bigrammiDiversi2 = set(bigrammi2)
	probabilitaCondizionata1 = probabilitaCondizionataMassima(bigrammi1, listaPOSTag1, bigrammiDiversi1)
	probabilitaCondizionata2 = probabilitaCondizionataMassima(bigrammi2, listaPOSTag2, bigrammiDiversi2)
	forzaAssociativa1 = forzaAssociativaMassima(bigrammi1, listaPOSTag1, bigrammiDiversi1) 
	forzaAssociativa2 = forzaAssociativaMassima(bigrammi2, listaPOSTag2, bigrammiDiversi2) 
	#testo1, A Tale of Two Cities, by Charles Dickens
	print "Il testo", file1, "ha", len(frasi1), "frasi", "e", len(testoTokenizzato1), "token."
	print "Lunghezza media frasi:", mediaFrasi1 
	print "Lunghezza media parole:", mediaParole1
	print "Lunghezza vocabolario:", len(vocabolario1)
	print
	distribuzionehapaxVocabolario(testoTokenizzato1)
	print "\nRapporto sostantivi-verbi:", rapportoSostantiviVerbi1
	print "\n10 PoS più frequenti:"
	for i in POSTagPiuFrequenti1:
		print "PoS tag:", i[0], "\tfrequenza:", i[1] 
	print "\n10 bigrammi PoS con probabilità condizionata massima:"
	for i in range(10): #scorro i primi dieci bigrammi già ordinati e accedo ai singoli elementi
		print "bigramma: \"", probabilitaCondizionata1[i][0][0], probabilitaCondizionata1[i][0][1], "\"", "\tprobabilità condizionata:", probabilitaCondizionata1[i][1]
	print "\n10 bigrammi PoS con forza associativa massima:"
	for i in range(10): #scorro i primi dieci bigrammi già ordinati e accedo ai singoli elementi
		print "bigramma: \"", forzaAssociativa1[i][0][0], forzaAssociativa1[i][0][1], "\"", "\tforza associativa:", forzaAssociativa1[i][1]
	#testo2, Dracula, by Bram Stoker
	print "\nIl testo", file2, "ha", len(frasi2), "frasi", "e", len(testoTokenizzato2), "token."
	print "Lunghezza media frasi:", mediaFrasi2
	print "Lunghezza media parole:", mediaParole2
	print "Lunghezza vocabolario:", len(vocabolario2)
	print
	distribuzionehapaxVocabolario(testoTokenizzato2)
	print "\nRapporto sostantivi-verbi:", rapportoSostantiviVerbi2
	print "\n10 PoS più frequenti:"
	for i in POSTagPiuFrequenti2:
		print "PoS tag:", i[0], "\tfrequenza:", i[1]
	print "\n10 bigrammi PoS con probabilità condizionata massima:"
	for i in range(10): #scorro i primi dieci bigrammi già ordinati e accedo ai singoli elementi
		print "bigramma: \"", probabilitaCondizionata2[i][0][0], probabilitaCondizionata2[i][0][1], "\"", "\tprobabilità condizionata:", probabilitaCondizionata2[i][1]
	print "\n10 bigrammi PoS con forza associativa massima:"
	for i in range(10): #scorro i primi dieci bigrammi già ordinati e accedo ai singoli elementi
		print "bigramma: \"", forzaAssociativa2[i][0][0], forzaAssociativa2[i][0][1], "\"", "\tforza associativa:", forzaAssociativa2[i][1]

main(sys.argv[1], sys.argv[2])

'''
python2 Programma1.py aTaleOfTwoCities.txt dracula.txt > outputProgramma1.txt

'''
