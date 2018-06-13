import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

def rmse(predictions, targets):
  return np.sqrt(((predictions - targets)**2).mean())

module_url = "https://tfhub.dev/google/universal-sentence-encoder/1" #@param ["https://tfhub.dev/google/universal-sentence-encoder/1", "https://tfhub.dev/google/universal-sentence-encoder-large/1"]

# Compute a representation for each message, showing various lengths supported.
messages = ["What's the {OTHER:COMMON} for this afternoon?",
            "What's the {OTHER:COMMON} like on my {OTHER:COMMON}?",
            "Show me a {OTHER:COMMON} on {LOCATION:PROPER} and {LOCATION:PROPER}.",
            "Can you find me a {LOCATION:COMMON} with {LOCATION:COMMON} nearby?",
            "Find a {LOCATION:COMMON} along {LOCATION:COMMON}.",
            "Find the cheapest indoor {OTHER:COMMON} within 500 meters of my {OTHER:COMMON}.",
            "Okay, can you find me a {LOCATION:COMMON} on my {OTHER:COMMON} that has a {LOCATION:COMMON}?",
            "Find {OTHER:COMMON} near {LOCATION:COMMON} that accepts {OTHER:COMMON} and has a {OTHER:COMMON}.",
            
            "Navigate to {LOCATION:PROPER}.",
            "What's my {OTHER:PROPER} to {LOCATION:COMMON}?",
            "Show me alternative {OTHER:COMMON}.",
            "Reroute using {OTHER:PROPER}.",
            
            
            "Drive to {LOCATION:PROPER}.",
            "What's my {OTHER:COMMON}?",
            "Can I make tomorrow's 10am {EVENT:COMMON} without recharging?",
            "What's {OTHER:COMMON} like on the {LOCATION:PROPER}?",
            "Are there any {OTHER:COMMON} on my {OTHER:COMMON}?",
            "Will it rain tomorrow in {LOCATION:PROPER}?"
            ]

# Compute a representation for each message, showing various lengths supported.
messages2 = ["What's the weather forecast for this afternoon?",
            "What's the traffic like on my route?",
            "Show me a Traffic Camera on US-101 and Bayshore Blvd.",
            "Can you find me a gas station with restroom facilities nearby?",
            "Find a coffee shop along route",
            "Find the cheapest indoor parking within 500 meters of my destination.",
            "Okay, can you find me a supermarket on my route that has a charging station?",
            "Find parking near destination that accepts credit cards and has a valet service.",
            "Navigate to San Francisco Museum of Modern Art.",
            "What's my ETA to destination?",
            "Show me alternative routes.",
            "Reroute using I-580 East.",
  
            "Drive to Downtown Berkeley.",
            "What's my drive range?",
            "Can I make tomorrow's 10am meeting without recharging?",
            "What's traffic like on the Bay Bridge?",
            "Are there any speed cameras on my route?",
            "Will it rain tomorrow in Oakland?"
            ]

scripts = ["[SEARCH FROM:WEATHERFORECAST WHERE:HERE WHEN:AFTERNOON]",
        "[SEARCH FROM:TRAFFIC WHERE:ONROUTE]",
        "[SEARCH FROM:TRAFFICCAMERA WHERE:[SEARCH GEOCODE WHERE:US-101 and Bayshore Blvd]]",
         "[SEARCH FROM:GASSTATION WHERE:NEARBY WITH:RESTROOM]",
         "[SEARCH ONE FROM:COFFEESHOP WHERE:ALONGROUTE]",
         "[SEARCH ONE FROM:OFFROADPARKING WHERE:DESTINATION RANGE:500M WITH:[SORT PRICE ASC]]",
         "[SEARCH ONE FROM:SUPERMARKET WHERE:ONROUTE WITH:CHARGINGSTATION]",
         "[SEARCH ONE FROM:PARKING WITH:CREDITCARD WITH:VALETSERVICE]",
         "[ROUTE TO:[SEARCH KEYWORD:San Francisco Museum of Modern Artl]]",
         "[ROUTE INFO:ETA]",
         "[ROUTE ALTROUTE]",
         "[ROUTE ALTROUTE USE:[SEARCH LINKS:ROUTE]]",
         
         "[MODE GUIDANCE WITH:[ROUTE TO:[SEARCH KEYWORD:Downtown Berkeley]]]",
         "[MODE DRIVERANGE]",
         "[MODE DRIVERANGE TO:[SEARCH KEYWORD:10AM MEETING FROM:SCHEDULE WHEN:10AM] WITH:[VOICERESPONSE TEMPLATE:YES/NO*]]",
         "[MODE TRAFFIC [SEARCH FROM:TRAFFIC WHERE:[SEARCH KEYWORD:Bay Bridge]] WITH:[VOICERESPONSE TEMPLATE:*]",
         "[MODE SPEEDCAMERA WHERE:ONROUTE WITH:[VOICERESPONSE TEMPLATE:*]]",
         "[MODE WEATHERFORECAST WHERE:[SEARCH KEYWORD:Oakland] WHEN:TOMORROW]"
          ]

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

# 품사 태깅 문장 임베딩
with tf.Session() as session:
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  message_embeddings = session.run(embed(messages))
  
#일반 문장 임베딩
with tf.Session() as session:
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  message_embeddings2 = session.run(embed(messages2))

for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
    print("Message: {}".format(messages[i]))
    print("Embedding size: {}".format(len(message_embedding)))
    message_embedding_snippet = ", ".join(
        (str(x) for x in message_embedding[:3]))
    print("Embedding: [{}, ...]\n".format(message_embedding_snippet))

# Compute a representation for each message, showing various lengths supported.
test_message = ["Where is a {OTHER:COMMON} with a {OTHER:COMMON} nearby?",
               "Find routes to {LOCATION:PROPER}.",
               "Find routes to {OTHER:COMMON}.",
               "How is {OTHER:COMMON} in {LOCATION:PROPER} now?",
               "Find a {LOCATION:COMMON} near the {LOCATION:COMMON} on my {OTHER:COMMON}."]

test_message2 = ["Where is a gas station with a restroom facilities nearby?",
               "Find routes to San Francisco Museum of Modern Art.",
               "Find routes to gas station.",
               "How is traffic in Bay Bridge now?",
               "Find a supermarket near the charging station on my route."]

test_labels = [3, 8, 8, 15, 6]

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

#테스트용 품사 태깅 문장 임베딩
with tf.Session() as session:
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  test_message_embeddings = session.run(embed(test_message))
  
#테스트용 일반 문장 임베딩
with tf.Session() as session:
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  test_message_embeddings2 = session.run(embed(test_message2))

# Word class with symbols
print("Word class with symbols")
for test_message_embedding, test_label in zip(test_message_embeddings, test_labels):
  minimum = 100
  minimum_index = 0
  for i, message_embedding in enumerate(message_embeddings):
    error = rmse(np.array(message_embedding), np.array(test_message_embedding))
    if minimum > error:
      minimum = error
      minimum_index = i

  print("Minimum RMSE value: {}".format(minimum))
  print("Most similar script: {}".format(scripts[minimum_index]))
  print("Estimation: {}".format(minimum_index))
  print("Answer: {}\n".format(test_label))

# Common nouns
print("Common nouns")
for test_message_embedding, test_label in zip(test_message_embeddings2, test_labels):
  minimum = 100
  minimum_index = 0
  for i, message_embedding in enumerate(message_embeddings2):
    error = rmse(np.array(message_embedding), np.array(test_message_embedding))
    if minimum > error:
      minimum = error
      minimum_index = i

  print("Minimum RMSE value: {}".format(minimum))
  print("Most similar script: {}".format(scripts[minimum_index]))
  print("Estimation: {}".format(minimum_index))
  print("Answer: {}\n".format(test_label))

