% Read positive words
fidPositive = fopen(fullfile('opinion-lexicon-English','positive-words.txt'));
C = textscan(fidPositive,'%s','CommentStyle',';');
wordsPositive = string(C{1});

% Read negative words
fidNegative = fopen(fullfile('opinion-lexicon-English','negative-words.txt'));
C = textscan(fidNegative,'%s','CommentStyle',';');
wordsNegative = string(C{1});
fclose all;

words_hash = java.util.Hashtable;
[possize, ~] = size(wordsPositive);
[negsize,~] = size(wordsNegative);
for ii = 1:possize
    words_hash.put(wordsPositive(ii,1),1);
end
for ii = 1:negsize
    words_hash.put(wordsNegative(ii,1),-1);
end

rng('default')
 
emb = fastTextWordEmbedding;
words = [wordsPositive;wordsNegative]; 
labels = categorical(nan(numel(words),1));
labels(1:numel(wordsPositive)) = "Positive";
labels(numel(wordsPositive)+1:end) = "Negative";
data = table(words,labels,'VariableNames',{'Word','Label'});

idx = ~isVocabularyWord(emb,data.Word); 
data(idx,:) = []; 

numWords = size(data,1);
cvp = cvpartition(numWords,'HoldOut',0.1);
dataTrain = data(training(cvp),:);
dataTest = data(test(cvp),:);
wordsTrain = dataTrain.Word;
XTrain = word2vec(emb,wordsTrain);
YTrain = dataTrain.Label;

mdl = fitcsvm(XTrain,YTrain);

filename = "bookreview.xlsx";
tbl = readtable(filename,'TextType','string');
textData = tbl.TextData;
document = preprocessText(textData);
idx = ~isVocabularyWord(emb,document.Vocabulary);
document = removeWords(document,idx);

words = document.Vocabulary;
words(ismember(words,wordsTrain)) = [];

vec = word2vec(emb,words);
[YPred,scores] = predict(mdl,vec);

figure
histogram(YPred);
title("Histogram of the Predicted Sentiment Scores")

figure
subplot(1,2,1)
idx = YPred == "Positive";
wordcloud(words(idx),scores(idx,1));
title("Predicted Positive Sentiment")

subplot(1,2,2)
wordcloud(words(~idx),scores(~idx,2));
title("Predicted Negative Sentiment")

%Sentiment Analiysis
filename = "bookreview.xlsx";
tbl = readtable(filename,'TextType','string');



str = tbl.TextData;
documents = tokenizedDocument(str);


cleanDocuments = addPartOfSpeechDetails(documents);
cleanDocuments = removeStopWords(documents);
cleanDocuments = normalizeWords(cleanDocuments,'Style','lemma');
cleanDocuments = erasePunctuation(cleanDocuments);
cleanDocuments = removeShortWords(cleanDocuments,2);
cleanDocuments = removeLongWords(cleanDocuments,13);


compoundScores = vaderSentimentScores(cleanDocuments);
compoundScores()

idx = compoundScores > 0;
strPositive = str(idx);
strNegative = str(~idx);

figure
histogram(compoundScores);
title("Histogram of Sentiment Scores")

figure
subplot(1,2,1)
wordcloud(strPositive);
title("Positive Sentiment")

subplot(1,2,2)
wordcloud(strNegative);
title("Negative Sentiment")

 
%Clean data and raw data analysis
cleanedBag = bagOfWords(cleanDocuments);


[cleanedBag,idx] = removeEmptyDocuments(cleanedBag);
cleanedBag;

rawDocuments = tokenizedDocument(str);
rawBag = bagOfWords(rawDocuments);
numWordsCleaned = cleanedBag.NumWords;
numWordsRaw = rawBag.NumWords;
reduction = 1 - numWordsCleaned/numWordsRaw;

figure
subplot(1,2,1)
wordcloud(rawBag);
title("Raw Data")
subplot(1,2,2)
wordcloud(cleanedBag);
title("Cleaned Data")

%Score Check
filename = "scoreCheck.xlsx";
tbl = readtable(filename,'TextType','string');
str = tbl.TextData;
documents = tokenizedDocument(str);
cleanDocument = preprocessText(documents);
Scores = vaderSentimentScores(cleanDocument);
Scores()
figure
plot(Scores)
title("Graph to test the Sentiment Scores")

%multiple polarities
filename = "multiplePolarity.xlsx";
tbl = readtable(filename,'TextType','string');
str = tbl.TextData;
documents = tokenizedDocument(str);
cleanDocument = preprocessText(documents);
Score = vaderSentimentScores(cleanDocument);
Score()


function [documents] = preprocessText(textData)
cleanTextData = lower(textData);
documents = tokenizedDocument(cleanTextData);
documents = erasePunctuation(documents);
documents = removeStopWords(documents); 
end