library(tidytext)
library(dplyr)
library(ggplot2)

# Einfaches Sample-Textset
texts <- data.frame(
  id = 1:4,
  text = c("I love this movie", "This movie is terrible", "Absolutely fantastic and fun", "Worst movie ever")
)

# Tokenisieren
words <- texts %>%
  unnest_tokens(word, text)

# Bing Sentiment-Lexikon
bing <- get_sentiments("bing")

# Join mit Sentiment
sentiment_words <- words %>%
  inner_join(bing, by = "word") %>%
  count(sentiment, sort = TRUE)

# Plotten
ggplot(sentiment_words, aes(x = sentiment, y = n, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  labs(title = "Sentiment im Mini-Demo",
       x = "Sentiment", y = "Anzahl Wörter") +
  theme_minimal()
