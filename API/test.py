import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from openai import OpenAI
import os
client = OpenAI(
    api_key='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'
)


# Load the dataset
df = pd.read_csv('AI_EarthHack_Dataset.csv', encoding='latin-1')

# Remove rows with NaN values
df = df.dropna()

# Combine problem and solution text for analysis
combined_text = df['problem'] + " " + df['solution']

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(combined_text)

# Topic Modeling with NMF
nmf = NMF(n_components=5, random_state=1).fit(tfidf)
'''
# Displaying the topics
for i, topic in enumerate(nmf.components_):
    print(f"Topic #{i+1}:")
    print(" ".join([tfidf_vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-11:-1]]))
    print("\n")
'''

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Quesiton1: The construction industry is indubitably one of the significant contributors to global waste, contributing approximately 1.3 billion tons of waste annually, exerting significant pressure on our landfills and natural resources. Traditional construction methods entail single-use designs that require frequent demolitions, leading to resource depletion and wastage.   "},
    {"role": "user", "content": "Solution2: Herein, we propose an innovative approach to mitigate this problem: Modular Construction. This method embraces recycling and reuse, taking a significant stride towards a circular economy.   Modular construction involves utilizing engineered components in a manufacturing facility that are later assembled on-site. These components are designed for easy disassembling, enabling them to be reused in diverse projects, thus significantly reducing waste and conserving resources.  Not only does this method decrease construction waste by up to 90%, but it also decreases construction time by 30-50%, optimizing both environmental and financial efficiency. This reduction in time corresponds to substantial financial savings for businesses. Moreover, the modular approach allows greater flexibility, adapting to changing needs over time.  We believe, by adopting modular construction, the industry can transit from a 'take, make and dispose' model to a more sustainable 'reduce, reuse, and recycle' model, driving the industry towards a more circular and sustainable future. The feasibility of this concept is already being proven in markets around the globe, indicating its potential for scalability and real-world application."},
    {"role": "user", "content": "Is this sustainable, circular economy business ideas, new idea and feasibility the answer format sustainable: xxxxx, circular economy business ideas: xxxxx, new idea: xxxxxx, feasibility: xxxxx, Industry prospects: xxxxxx,  Estimated scale of industry: include money range xxxxxxx "}
  ]
)

print(completion.choices[0].message)

