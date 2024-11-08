---
title: Topic Modeling DDI
description: Die Berechnung eines Topic Models auf der Grundlage des DDI-Dokumentenkorpus.
author: MBG
date: 2024-10-31 16:20
categories: [Text Mining]
tags: [topic model, text mining, datenanalyse]
pin: false
math: true
mermaid: true
---

In diesem Post wird erläutert, wie ein Topic Model auf der Grundlage eines Dokumentenkorpus trainiert und visualisiert werden kann. 

Das Berliner Medienunternehmen [Polit-X](https://polit-x.de/de/) stand uns bei der Datenbeschaffung zur Seite und stellte uns einige als XML-Dokument serialisierte Query's an ihre Datenbank zur Verfügung. Wir haben zu diesem Zweck eine Liste von Schlüsselwörtern zusammengestellt, auf deren Grundlage die Query's abgesetzt wurde. Das Ergebnis enthält alle Dokumente der Datenbank, die das Keyword mindestens einmal enthalten.

Zunächst habe ich einige Grundlagenarbeiten geleistet und verschiedene Python-Skripte geschrieben, um das Korpus einzulesen, zu verwalten und zu serialisieren. Diese Funktionen habe ich mit der Klasse `CorpusManager` realisiert. Zudem habe ich einige Methoden implementiert, um die Dokumente des eingelesenen Korpus zu filtern.

Meine Skripte habe ich im Github Repository [Doing_DDI](https://github.com/Nolram567/Doing_DDI) veröffentlicht.

Das Skript `corpus_preprocessor.py` enthält eine Klasse, mit der man das Korpus für die Berechnung eines Themenmodells (und für andere Text-Mining-Methoden) vorverarbeiten kann. Wichtige Hintergründe zu der Vorverarbeitung von Texten für das Topic Modeling in der Politikwissenschaft können beispielsweise bei Denny & Spirling nachgelesen werden[^1].

---

## Die Vorverarbeitung der Daten

Die von mir durchgeführte Vorverarbeitung bestand aus den folgenden Schritten, die ich in der Methode `prepare_for_topic_modeling` der Klasse `CorpusPreprocessor` gebündelt habe:

1. **Vorbereinigung**: Identifikation von fehlenden Leerzeichen, z. B. am Ende von Sätzen.
2. **Lemmatisierung**: Die Texte der Dokumente werden lemmatisiert, d. h. die Terme werden von ihrer ggf. flektierten Form auf die Grundform abgebildet. Diese Aufgabe übernimmt das Sprachmodell `de_core_news_lg` der Python-Bibliothek [spaCy](https://spacy.io/).
3. **Normalisierung**: Die Texte werden zu Kleinschreibung normalisiert.
4. **Tokenisierung**: Die Texte, die als Einzelstrings vorliegen, werden zu Listen aus Termen aufgetrennt.
5. **N-Gram Inklusion**: Multiword Expressions werden zu einem Term konkateniert, sodass sie vom Modell als Bedeutungseinheit erfasst werden.
6. **Bereinigung**: Wir entfernen Stoppwörter, E-Mail-Adressen, Zahlen, sehr seltene Terme, (...).


---

## Die Berechnung des Topic Models

Für die Berechnung des Themenmodells mit Latent Dirichlet Allocation (LDA) nutze ich die etablierte Python-Bibliothek `gensim`[^2]. Nur am Rande möchte ich erwähnen, dass es zahlreiche Algorithmen für die Berechnung von Themenmodellen gibt. Beispielsweise konnten in der Politikwissenschaft mit der `Non-Negative Matrix Factorization` bereits gute Ergebnisse erzielt werden[^3]. Allerdings ist LDA nach wie vor ein bewährter und häufig genutzter Ansatz für die Themenmodellierung.

Bevor wir mit dem Training des LDA-Modells beginnen können, müssen wir unser Korpus in die Vektordarstellung - in ein Bag-of-Words-Model[^4] - transformieren. Zudem müssen wir ein Dictionary erstellen, das alle individuellen Terme auf eine eineindeutige ID abbildet. 

---

### Wie funktioniert Latent Dirichlet Allocation?

LDA ist ein unüberwachtes Lernverfahren für die Themenmodellierung zum Zwecke der Klassifikation (und der Analyse) von Dokumenten. Das Verfahren geht auf den Informatiker David Blei zurück[^5]. Kurzgesagt modellieren wir die Dokumente als Wahrscheinlichkeitsverteilungen.Die Dokumente werden als Verteilungen über den Themen und die Themen als Verteilungen über das Vokabular des Korpus modelliert. Im Ergebnis können wir jedem Thema - mit einer gewissen Wahrscheinlichkeit - die Terme zuweisen, die es konstituieren und jedem Dokument - mit einer gewissen Wahrscheinlichkeit - die Anteile der Themen, die es beinhaltet. Der Begriff "Thema" ist hier nicht im Sinne seiner herkömmlichen Bedeutung zu verstehen. LDA modelliert Themen als Termmengen, die in einem Kontext - hier: Dokumente - häufiger kookkurrieren als andere Terme der Grundgesamtheit.

Die mathematischen und technischen Details von LDA sind komplex. Allen Interessierten kann ich beispielsweise Kapitel 7.6.2. in "Applied Text Mining"[^6] empfehlen sowie die folgenden Videos auf YouTube:

**Eine Einführung in die Themenmodellierung mit R (Kontext: Computational Social Sciences):**

{% include embed/youtube.html id='IUAHUEy1V0Q' %}

**Die mathematischen-technischen Details verständlich erklärt:** 

{% include embed/youtube.html id='T05t-SqKArY' %}

---

### Das Filtern der Dokumente
Da unser Korpus alle Dokumente enthält, die das Wort "Dateninstitut" mindestens einmal enthalten, musste ich zusätzlich einige Dokumente filtern, um ein akkurates Ergebnis zu erzielen. Das Korpus enthält neben langen Dokumenten, die das DI nur am Rande thematisieren - wie etwa Haushaltsgesetze - auch Pressemitteilungen ohne substanziellen Inhalt, die etwa eine Personalie bekannt gegeben. Ich habe daher alle Dokumente mit weniger als 150 Termen aussortiert. Zusätzlich habe ich die irrelevanten Dokumente entfernt. Zu diesem Zweck habe ich die `TF-IDF` für den Term "Dateninstitut" für alle Dokumente berechnet und alle Dokumente entfernt, deren `TF-IDF` für "Dateninstitut" kleiner ist als der Median der Werte. Auf diese Weise wird die weniger relevante Hälfte der Dokumente entfernt.

---

### Die Parametrisierung des Modells
Die Parametrisierung des Models ist eine diffizile Angelegenheit. Grundsätzlich führen unterschiedliche Kombinationen zu akkuraten Ergebnissen. Allerdings sind manche Parameter wichtiger für die Kohärenz und Interpretierbarkeit der Ergebnisse als andere. Ein zentraler Parameter des LDA-Algorithmus ist die Zahl der Themen `k`. Für die Bestimmung eines angemessenen `k` habe ich eine Reihe von Modellen mit differenten `k` berechnet und für jedes Model die semantische Kohärenz C<sub>V</sub>[^7] bestimmt. Die Bibliothek `gensim` implementiert Methoden, um C<sub>V</sub> zu berechnen. Das Model, bei welchem das Maximum der C<sub>V</sub>-Werte gemessen wurde, wird für die qualitative Begutachtung gespeichert und visualisiert.

Weitere wichtige Parameter sind:
* Die Hyperparameter `alpha` und `eta`: Die Parameter `alpha` und `eta` bestimmen die (initiale) Verteilung der Themen in Dokumenten (alpha) und der Wörter in Themen (eta). Beide Parameter können wir vom Algorithmus optimieren lassen, indem wir sie auf `auto` setzen. Alternativ können wir eine Konstante übergeben oder `alpha` auf `asymmetric` setzen. Mit `asymmetric` wird eine asymmetrische Verteilung zugelassen, was bedeutet, dass manche Themen in bestimmten Dokumenten bevorzugt auftreten können.
* `chunksize`: Die Menge der Dokumente, die parallel verarbeitet wird.
* `iterations`: Wie oft wiederholen wir den Trainingsprozess pro Dokument.
* `passes`: Wie oft wiederholen wir den Trainingsprozess für das Korpus.
* (...)

Weitere Einblicke können beispielsweise aus der [Dokumentation](https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html#sphx-glr-auto-examples-tutorials-run-lda-py) von `gensim` gewonnen werden.

Die Parametrisierung, die der Visualisierung des Models am Ende dieses Beitrags zugrunde liegt:

```python
LdaModel(
    corpus=bow_corpus,
    id2word=dictionary,
    num_topics=k,
    iterations=300,  # Check if documents converge with the given parameters.
    passes=30,  # Check if documents converge with the given parameters.
    chunksize=64,  # Choose chunksize according to the available memory and corpus size.
    alpha='asymmetric',  # Topic Distribution per document
    eta='auto',  # Automatic distribution of terms per topic
    eval_every=1,  # Evaluation after every iteration
    random_state=42 # Choose a constant value for better reproducibility
)
```

---

## Die Visualisierung des Models

Für die Visualisierung des Modells habe ich die Python-Bibliothek `pyLDAvis` verwendet[^8]. Die Kreise stellen die Themen und ihr Gewicht im Korpus dar, und beim klicken auf oder hoovern über einen der Kreise kann man die salienten Terme des jeweiligen Themas begutachten. Der blaue Balken veranschaulicht die Salienz des Terms und der rote Balken die Exklusivität. Die Grafik kann [hier](https://nolram567.github.io/Doing_DDI/) im Vollbildmodus erkundet werden.

{% include lda_embeding.html %}

---

[^1]: Denny & Spirling (2018).
[^2]: Řehůřek & Sojka (2010).
[^3]: Vgl. Greene & Cross (2017): S. 29 f.
[^4]: Vgl. Qamar & Raza (2024): S. 91 f.
[^5]: Blei et al. (2001)
[^6]: Vgl. Qamar & Raza (2024): S. 245 f.
[^7]: Röder et al. (2015).
[^8]: Sievert & Shirley (2014).

## Literaturverzeichnis
* Blei, David, et al. “Latent Dirichlet Allocation.” Advances in Neural Information Processing Systems, vol. 14, MIT Press, 2001. Neural Information Processing Systems, [https://proceedings.neurips.cc/paper/2001/hash/296472c9542ad4d4788d543508116cbc-Abstract.html](https://proceedings.neurips.cc/paper/2001/hash/296472c9542ad4d4788d543508116cbc-Abstract.html).
* Denny, Matthew J., and Arthur Spirling. “Text Preprocessing For Unsupervised Learning: Why It Matters, When It Misleads, And What To Do About It.” Political Analysis, vol. 26, no. 2, Apr. 2018, pp. 168–89. Cambridge University Press, [https://doi.org/10.1017/pan.2017.44](https://doi.org/10.1017/pan.2017.44).
* Greene, Derek, and James P. Cross. “Exploring the Political Agenda of the European Parliament Using a Dynamic Topic Modeling Approach.” Political Analysis, vol. 25, no. 1, Jan. 2017, pp. 77–94. Cambridge University Press, [https://doi.org/10.1017/pan.2016.7](https://doi.org/10.1017/pan.2016.7).
* Qamar, Usman, and Muhammad Summair Raza. Applied Text Mining. Springer Nature Switzerland, 2024. DOI.org (Crossref), [https://doi.org/10.1007/978-3-031-51917-8](https://doi.org/10.1007/978-3-031-51917-8).
* Řehůřek, Radim, and Petr Sojka. Software Framework for Topic Modelling with Large Corpora. University of Malta, 2010. is.muni.cz, [https://is.muni.cz/publication/884893/en/Software-Framework-for-Topic-Modelling-with-Large-Corpora/Rehurek-Sojka](https://is.muni.cz/publication/884893/en/Software-Framework-for-Topic-Modelling-with-Large-Corpora/Rehurek-Sojka).
* Röder, Michael, et al. “Exploring the Space of Topic Coherence Measures.” Proceedings of the Eighth ACM International Conference on Web Search and Data Mining, Association for Computing Machinery, 2015, pp. 399–408. ACM Digital Library, [https://doi.org/10.1145/2684822.2685324](https://doi.org/10.1145/2684822.2685324).
* Sievert, Carson, and Kenneth Shirley. “LDAvis: A Method for Visualizing and Interpreting Topics.” Proceedings of the Workshop on Interactive Language Learning, Visualization, and Interfaces, edited by Jason Chuang et al., Association for Computational Linguistics, 2014, pp. 63–70. ACLWeb, [https://doi.org/10.3115/v1/W14-3110](https://doi.org/10.3115/v1/W14-311).

