#-*- coding: utf-8 -*-
import sys
import codecs
import nltk
import re

#funzione che estrae e ritorna testo tokenizzato e PoS taggato
def estraiTestoTokenizzato(frasi):
	tokensTot = []
	for frase in frasi:
		tokens = nltk.word_tokenize(frase) #testo tokenizzato
		tokensTot = tokensTot + tokens
	distFrequenzaToken = nltk.FreqDist(tokensTot) #distribuzione di frequenza del libro
	return tokensTot, distFrequenzaToken

'''funzione che dalle frasi associate nel dizionario ad ogni nome estrae i 10 Luoghi più frequenti, le 10 Persone più frequenti, i 10 Sostantivi più frequenti, i 10 Verbi più frequenti
	le Date, i Mesi e i Giorni della settimana e la frase lunga minimo 8 token e massimo 12 con probabilità più alta.'''
def perOgniNome(dizionario, listaNomiPropri, testo, distFrequenzaToken): 
	array1 = []
	dizionarioVerbi = {} 
	dizionarioSostantivi = {}
	dizionarioPersone = {}
	dizionarioLuoghi = {}
	dizionarioDateMesiGiorni = {}
	dizionarioMarkov = {} 
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	lunghezzaCorpus = len(testo)
	for i in dizionario: #scorro i nomi
		if i in listaNomiPropri: #se il nome è tra i più frequenti
			array1 = dizionario[i] #metto dentro all'array vuoto le frasi del nome
			array1 = tokenizer.tokenize(array1)
			contenitoreVerbi = [] 
			contenitoreSostantivi = []
			contenitorePersone = []
			contenitoreLuoghi = []
			arrayMatch = []
			tempLunga = 0
			arrayLunga = []
			for frase in array1: #scorro le frasi
				#cerco date, mesi e giorni con espressioni regolari
				matchDate = re.findall(r'(?:\s?\d?\d[-\/]\d?\d[-\/]?\d?\d?\d?\d?)|(?:\d?\d(?:th|rd|st|nd)?\s(?:[Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?|[Aa]pr(?:il)?|[Mm]ay|[Jj]un(?:e)?|[Jj]ul(?:y)?|[Aa]ug(?:ust)?|[Ss]ep(?:tember)?|[Oo]ct(?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)(?:\s\d?\d?\d\d)?)|(?:(?:[Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?|[Aa]pr(?:il)?|[Mm]ay|[Jj]un(?:e)?|[Jj]ul(?:y)?|[Aa]ug(?:ust)?|[Ss]ep(?:tember)?|[Oo]ct(?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)\s\d?\d(?:th|rd|st|nd)?,?(?:\s\d?\d?\d\d)?)', frase)
				matchMesi = re.findall(r'\b(?:[Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?|[Aa]pr(?:il)?|May|[Jj]un(?:e)?|[Jj]ul(?:y)?|[Aa]ug(?:ust)?|[Ss]ep(?:tember)?|[Oo]ct(?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)\b', frase) 
				matchGiorni = re.findall(r'\b(?:[Mm]on|[Tt]ues|[Ww]ednes|[Tt]hurs|[Ff]ri|[Ss]atur|[Ss]un)day\b', frase)
				if not matchDate==[]: #se ha trovato delle date
					arrayMatch.append(matchDate)
				if not matchMesi==[]: #se ha trovato dei mesi
					arrayMatch.append(matchMesi)
				if not matchGiorni==[]: #se ha trovato dei giorni
					arrayMatch.append(matchGiorni)
				tokens = nltk.word_tokenize(frase)
				#cerco frase lunga tra 8 e 12 token con probabilità massima calcolata con Markov di ordine 0
				if (len(tokens) >= 8) and (len(tokens)<= 12): #se lunghezza della frase è tra 8 e 12 compresi
					probabilitaFrase = 1.0 #inizializzo probabilità a 1 
					for tok in tokens: #scorro i token
						probabilitaToken = (distFrequenzaToken[tok]*1.0/lunghezzaCorpus*1.0) #calcolo probabilità dei token
						probabilitaFrase = probabilitaFrase * probabilitaToken #calcolo probabilità della frase
					if probabilitaFrase > tempLunga: #se probabilità della frase è maggiore della frase con probabilità più alta incontrata fino a quel punto
						tempLunga = probabilitaFrase 
						arrayLunga = frase #salvo la frase con probabilità più alta
				tokensPOS = nltk.pos_tag(tokens) #PoS tag
				analisi = nltk.ne_chunk(tokensPOS) #NE tag
				#cerco i 10 sostantivi e verbi più frequenti
				for j in tokensPOS: #scorro il testo pos-taggato
					if j[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']: #categorie verbali
						contenitoreVerbi.append(j[0])
					if j[1] in ['NN', 'NNS', 'NNP', 'NNPS']: #categorie sostantivi
						contenitoreSostantivi.append(j[0])
				#cerco i 10 luoghi e persone più frequenti
				for nodo in analisi:
					NEPerson = ''
					NELuogo = ''
					if hasattr(nodo, 'label'):
						if nodo.label() in ["PERSON"]: #se nodo ha attributo persona
							for k in nodo.leaves():
								NEPerson = NEPerson + ' ' + k[0]
							contenitorePersone.append(NEPerson)
						if nodo.label() in ["GPE"]: #se nodo ha attributo luogo
							for k in nodo.leaves():
								NELuogo = NELuogo + ' ' + k[0]
							contenitoreLuoghi.append(NELuogo)
			#distribuzione di frequenza per ogni array
			distribuzioneSostantivi = nltk.FreqDist(contenitoreSostantivi) 
			distribuzioneVerbi = nltk.FreqDist(contenitoreVerbi) 
			distribuzionePersone = nltk.FreqDist(contenitorePersone)
			distribuzioneLuoghi = nltk.FreqDist(contenitoreLuoghi)
			#ordino gli array in ordine di frequenza decrescente e prendo solo i 10 elementi più frequenti
			listaOrdinataSostantivi = distribuzioneSostantivi.most_common(10)
			listaOrdinataVerbi = distribuzioneVerbi.most_common(10)
			listaOrdinataPersone = distribuzionePersone.most_common(10)
			listaOrdinataLuoghi = distribuzioneLuoghi.most_common(10)
			#assegno nei dizionari per ogni nome i rispettivi array
			dizionarioMarkov[i] = arrayLunga
			dizionarioDateMesiGiorni[i] = arrayMatch
			dizionarioSostantivi[i] = listaOrdinataSostantivi
			dizionarioVerbi[i] = listaOrdinataVerbi
			dizionarioPersone[i] = listaOrdinataPersone
			dizionarioLuoghi[i] = listaOrdinataLuoghi
	return dizionarioSostantivi, dizionarioVerbi, dizionarioPersone, dizionarioLuoghi, dizionarioDateMesiGiorni, dizionarioMarkov

#funzione che estrae la frase più lunga e la frase più breve per ogni 10 dei nomi propri
def estraiFraseLungaCorta(dizionario, listaNomiPropri):
	array1 = [] 
	dizionarioLunga = {} 
	dizionarioCorta = {}
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	for i in dizionario: #scorro i nomi
		if i in listaNomiPropri: #se il nome è tra i più frequenti
			array1 = dizionario[i] #metto dentro all'array vuoto le frasi del nome
			array1 = tokenizer.tokenize(array1) 
			arrayLunga = [] 
			tempLunga = 0 
			tempCorta = float("inf")
			arrayCorta = []
			for frase in array1: #scorro le frasi
				tokens = nltk.word_tokenize(frase) 
				if len(tokens) > tempLunga: #se lunghezza della frase è maggiore della più lunga incontrata fino a quel punto
					tempLunga = len(tokens) 
					arrayLunga = frase #salvo la frase più lunga
				if len(tokens) < tempCorta: #se lunghezza della frase è minore della più corta incontrata fino a quel punto
					tempCorta = len(tokens)
					arrayCorta = frase #salvo la frase più corta
			dizionarioLunga[i] = arrayLunga 
			dizionarioCorta[i] = arrayCorta
	return dizionarioLunga, dizionarioCorta

#funzione che estrae i dieci nomi di persona più frequenti e le relative frasi 
def estraiNomiPropriFrasi(frasi): 
	dizionario = {} #chiave = nome, elemnto = frasi
	NamedEntityList = [] #lista nomi
	for frase in frasi:
		tokens = nltk.word_tokenize(frase)
		tokensPOS = nltk.pos_tag(tokens) #PoS tag
		analisi = nltk.ne_chunk(tokensPOS) #NE tag
		#cerco i nomi di persona e le relative frasi in cui è contenuto
		for nodo in analisi:
			NE = ''
			if hasattr(nodo, 'label'): 
				if nodo.label() in ["PERSON"]: #se nodo ha attributo persona
					for i in nodo.leaves():
						NE = NE + ' ' + i[0]
					if not NE in dizionario: #se nome non è presente in dizionario	
						dizionario[NE] = frase.encode('ascii','ignore') #assegno frase a nome, encode per evitare UnicodeEncodeDecode error
					else:
						if not frase in dizionario[NE]: #se frase non è presente legata al nome
							dizionario[NE] = dizionario[NE] + "  " + frase.encode('ascii','ignore') #aggiungo frase alle precedenti
					NamedEntityList.append(NE)
	distribuzione = nltk.FreqDist(NamedEntityList) #distribuzione di frequenza dei nomi
	listaOrdinata = distribuzione.most_common(10) #10 nomi più frequenti
	array = [] 
	for i in listaOrdinata: #salvo la lista dei nomi più frequenti
		array.append(i[0]) 
	return listaOrdinata, dizionario, array

def main (file1, file2):
	fileInput1 = codecs.open(file1, "r", "utf-8") 
	fileInput2 = codecs.open(file2, "r", "utf-8")
	raw1= fileInput1.read()
	raw2= fileInput2.read()
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	frasi1 = tokenizer.tokenize(raw1)
	frasi2 = tokenizer.tokenize(raw2)
	testoTokenizzato1, distFrequenza1 = estraiTestoTokenizzato(frasi1)
	testoTokenizzato2, distFrequenza2 = estraiTestoTokenizzato(frasi2)
	listaNomiPropri1, nomiPropriFrasi1, soloListaNomi1 = estraiNomiPropriFrasi(frasi1) 
	listaNomiPropri2, nomiPropriFrasi2, soloListaNomi2 = estraiNomiPropriFrasi(frasi2)
	fraseLunga1, fraseCorta1 = estraiFraseLungaCorta(nomiPropriFrasi1, soloListaNomi1)
	fraseLunga2, fraseCorta2 = estraiFraseLungaCorta(nomiPropriFrasi2, soloListaNomi2)
	sostantiviFrasiNome1, verbiFrasiNome1, personeFrasiNome1, luoghiFrasiNome1, dateMesiGiorni1, fraseMarkov1 = perOgniNome(nomiPropriFrasi1, soloListaNomi1, testoTokenizzato1, distFrequenza1)
	sostantiviFrasiNome2, verbiFrasiNome2, personeFrasiNome2, luoghiFrasiNome2, dateMesiGiorni2, fraseMarkov2 = perOgniNome(nomiPropriFrasi2, soloListaNomi2, testoTokenizzato2, distFrequenza2)
	#testo1, A Tale of Two Cities, by Charles Dickens
	print "\n10 Nomi propri di persona più frequenti nel testo", file1
	for i in listaNomiPropri1: #scorro i 10 nomi più frequenti
		print "Nome:", i[0], "\nFrequenza:", i[1], "\n\nFrasi che contengono il nome:\n", nomiPropriFrasi1[i[0]], "\n\nFrase lunga:", fraseLunga1[i[0]], "\nFrase corta:", fraseCorta1[i[0]]
		print "\n10 Sostantivi più frequenti:", sostantiviFrasiNome1[i[0]], "\n10 Verbi più frequenti:", verbiFrasiNome1[i[0]]
		print "10 Persone più frequenti:", personeFrasiNome1[i[0]], "\n10 Luoghi più frequenti:", luoghiFrasiNome1[i[0]], "\n\nDate, Mesi e Giorni della settimana:", dateMesiGiorni1[i[0]]
		print "Frase Markov con probabilità più alta lunga minimo 8 token e massimo 12:", fraseMarkov1[i[0]],"\n---------------------------------------------\n"
	#testo2, Dracula, by Bram Stoker
	print "\n10 Nomi propri di persona più frequenti nel testo", file2
	for i in listaNomiPropri2: #scorro i 10 nomi più frequenti
		print "Nome:", i[0], "\nFrequenza:", i[1], "\n\nFrasi che contengono il nome:\n", nomiPropriFrasi2[i[0]], "\n\nFrase lunga:", fraseLunga2[i[0]], "\nFrase corta:", fraseCorta2[i[0]]
		print "\n10 Sostantivi più frequenti:", sostantiviFrasiNome2[i[0]], "\n10 Verbi più frequenti:", verbiFrasiNome2[i[0]]
		print "10 Persone più frequenti:", personeFrasiNome2[i[0]], "\n10 Luoghi più frequenti:", luoghiFrasiNome2[i[0]], "\n\nDate, Mesi e Giorni della settimana:", dateMesiGiorni2[i[0]]
		print "Frase Markov con probabilità più alta lunga minimo 8 token e massimo 12:", fraseMarkov2[i[0]],"\n---------------------------------------------\n"

main(sys.argv[1], sys.argv[2])

'''	
python2 Programma2.py aTaleOfTwoCities.txt dracula.txt > outputProgramma2.txt

'''
