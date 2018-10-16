# “Modern Propositions: Predicting Ballot Measure Outcomes”  

## Problem Statement:
A ballot measure is a piece of proposed legislation (often at the state level) to be approved or rejected by eligible voters. Ballot measures are also known as "propositions" or simply "questions". Ballot measures differ from most legislation passed by representative democracies; ordinarily an elected legislature develops and passes laws. Ballot measures, by contrast, are an example of direct democracy.

The subject of ballot measures varies widely, however, common subjects include, religion, agriculture, taxation, criminal justice, labor unions, and of particular interest to me, cannabis.  

Companies and organizations spend millions each year attempting to forecast the outcomes of various ballot measures. Armies of lobbyists wine-and-dine for tiny amounts of inside information while hordes of researchers keep tabs on legislation in statehouses across the country. Being able to predict a legislative outcome can give an organization a competitive advantage in many strategic domains. Similarly, organizations on either side of the proposition issue may be interested in early detection that their outcome isn’t forecasted so they can take action (e.g. rewrite the legislation, allocate more money lobbying for their side, etc.)

The traditional method to predict ballot measure outcomes is polling. Unfortunately, polling can be expensive ($20k+ for a small phone bank poll), time consuming (2-3 week minimum), and inaccurate (see the 2016 presidential election and Brexit). Can we find a cheaper, quicker, and/or more accuate method?


## Introduction:

In many states, the most important policy changes this year won’t come from congressional legislation, but from ballot initiatives. According to The Economist’s annual Democracy Index report, the United States in 2017 qualified as a “flawed democracy” for the second year in a row. Congressional gridlock is worsening, partisan polarization is increasingthere is rampant gerrymandering. As a result U.S. citizens no longer trust their electoral system and many Americans are cutting out the middleman per se, and turning to ballot initiatives.

With more and more legislative outcomes being determined through the ballot measure process, the value of being able to effectively predict ballot measure outcomes is also increasing: What would you do if you could predict the outcome of a ballot measure?

Maybe:
-Adjust corporate strategy
-Make a lucrative investment
-Make better decisions about where to move

In 2016, Oregonians voted down a measure that would have created a 2.5% sales tax for corporations to help fund education. In 2018 Oregon has the third lowest high school graduation rate in the US.

In 2016, Californians (and a number of other states) voted to legalize the recreational use of cannabis. Since then, Canopy Growth and Tilray, and many other blue chip cannabis stocks are up well over 100%.

In 2018, Nevadians will vote to minimize regulations on energy markets. I bet NV Energy would like to know the outcome of that decision beforehand.

## Methods
In this project I utilize machine learning to attempt to predict ballot measure outcomes using readily available, public information.

### Data
I was able to obtain relevant data from the following sources:

*Followthemoney.org* details the outcomes of over 1,500 ballot measures in U.S. since 2002. It also included the ballot measures actual text, and the amount of money raised to support and oppose the measure.

*The Bureau of Labor Statistics* provides a number of state-specific economic metrics like unemployment by year.

*The Census* provides the state per capita median income infomation by year.

*The National Conference of State Legislatures* provides political ideology scores (e.g. the NOMINATE score) by state by year.

### Features
I extracted a number of features from the data data sets listed above, as well as engineered a few critical features such as whether or not a similar measure has passed in another state in the past (this was done via fuzzy matching). I also created a feature that scored the ideoogical leaning of each measure based on its text (I built a classification model that takes in political text and outputs the probability it is Republican. The model was trained using transcripts from congressional debates and presidential candidate speeches).

### Modeling
I explored all of the common classification algorithms to identify the model that maximized my evaluation metric. I chose the F1 score since variation in ballot measures and interested stakeholders dictate that both true positives and true negatives could be of interest.

The K Nearest Neighbors, Support Vector Classifier, and Random Forest classification models all performed quite well and so I ultimately opted for an ensemble approach, bundling all three models into a Voting Classifier (with a little more weight given to Random Forest).

Overfitting was an issue, I utilized all of the common techniques such as feature engineering and cross-validation to combat it.

SMOTEing was also done to correct for a slight class imbalance (~2:1 passed:failed).

## Results
The final, best fit model had an F1 score of 0.83. This seems to be quite good for a first pass. A dummy classification model was also investigated to provide a baseline. The dummy model had an F1 score of 0.49.

A beta-version of a web application for this model was also created via Flask and can be found here:

## Conclusion
It does appear that machine learning may be a method to help provide a better solution to our ever increasing need to more efficiently predict the outcome of ballot propositions. In the future, I would explore in inclusion of twitter sentiment as a feature in the model as well as some kind of polling metric.
