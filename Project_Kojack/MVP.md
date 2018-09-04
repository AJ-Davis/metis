# “Predicting Ballot Measure Outcomes”  

## Domain: A brief description of the domain you're working in, and a discussion of your familiarity with this domain

Legislation - I am not particularly familiar

### Problem:
A ballot measure is a piece of proposed legislation (often at the state level) to be approved or rejected by eligible voters. Ballot measures are also known as "propositions" or simply "questions". Ballot measures differ from most legislation passed by representative democracies; ordinarily an elected legislature develops and passes laws. Ballot measures, by contrast, are an example of direct democracy.

The subject of ballot measures varies widely, however, common subjects include, religion, agriculture, taxation, criminal justice, labor unions, and of particular interest to me, cannabis.  

Companies and organizations spend millions each year attempting to forecast the outcomes of various ballot measures. Armies of lobbyists wine-and-dine for tiny amounts of inside information while hordes of researchers keep tabs on legislation in statehouses across the country. Being able to predict a legislative outcome can give an organization a competitive advantage in many strategic domains. Similarly, organizations on either side of the proposition issue may be interested in early detection that their outcome isn’t forecasted so they can take action (e.g. rewrite the legislation, allocate more money lobbying for their side, etc.)


### Solution:
A simple web application that allows one to predict a ballot measure outcome based on publicly available information like, campaign financing, the way the measure is written, state demographics and public sentiment.

## Data:  

Followthemoney.org - Contains a database of all the state ballot measures since 2002, including outcomes, descriptions, some financing information

Twitter – for public sentiment.

## Approach:
- Use NLP techniques to turn social media and actual proposition text into useful features.

- Include historical outcomes from similar legislation as a feature.

- Utilize campaign financing as features (followthemoney.org details financing for both pro and opp)

- Utilize state demographics as features (political party density)

- Employ standard classification algorithms and scoring metrics to make predictions

- Build a web application that would allow anyone to make predictions for newly introduced legislation and tweak the various features to identify actions to potentially change the course.

## Known unknowns: A list of items with an unclear level of effort, or which will require special attention

- Will there be enough data? Appears to be ~1500 labeled ballot measure outcomes in the followthemoney.org data set

- Will the bill text provide meaningful features?
