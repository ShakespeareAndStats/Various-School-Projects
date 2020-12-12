library(ggformula)
library(tidyverse)
library(knitr)
library(tidyverse)
library(tidytext)
library(rsample)

library(gutenbergr)
full_text7 <- gutenberg_download(11889)
#Calling it full_text3 because it is from Volume 7

cleaned_text7_0 <- full_text7[-c(1:458),]

cleaned_text1 <- cleaned_text7_0

names_full <- c("Clarissa", "Harlowe", "Robert", "Lovelace", "Arabella", "Bella", "Anna", "Howe", "Roger", "Solmes", "James", "John", "Belford", "Leman", "Esq", "Antony", "Clary", "Hervey", "Betty", "Norton", "Joseph", "Mr", "Miss", "Mrs", "Hickman", "Belton", "Dorcas", "Sinclair", "Charlotte", "Betty", "Sarah", "Montague", "Patty")
shakes_stop <- c("thee", "thou", "thy", "tis", "Tis", "hath", "hast", "Enter", "twill", "art", "thyself", "ere", "whence", "Exeunt", "twixt", "Exit", "thine", "canst", "o'er", "is't", "on't", "wherefore", "wither", "wilt", "shalt", "shouldst", "wouldst", "nay", "yea", "Ay", "ay", "twere", "thence", "ye", "twas", "prithee", "doth", "th", "hither", "Act", "ACT", "Scene","II", "III", "IV", "V", "VI", "VII", "1")

library(tm)
cleaned_text1$text <- unlist(lapply(cleaned_text1$text, FUN=removeWords, words=names_full))
cleaned_text1$text <- unlist(lapply(cleaned_text1$text, FUN=removeWords, words=shakes_stop))
cleaned_text1$title <- "Clarissa"

Pride <- gutenberg_download(1342)

names_pride <- c("Elizabeth", "Lizzy", "Bennet", "Fitzwilliam", "Darcy", "Mr.", "Mrs.", "Gardiner", "Jane", "Mary", "Catherine", "Kitty", "Lydia", "Charles", "Bingley", "Caroline", "George", "Wickham", "William", "Collins", "Lady", "de Bourgh", "Edward", "Georgiana", "Charlotte", "Lucas", "Colonel", "Chapter", "said", "Mr", "s", "Miss", "Netherfield", "Longbourn", "Pemberley")
Pride$text <- unlist(lapply(Pride$text, FUN=removeWords, words=names_pride))
Pride$title <- "Pride and Prejudice"

books <- rbind(cleaned_text1, Pride)
books$document <- 1:nrow(books)

tidy_book <- cleaned_text1 %>%
  mutate(line = row_number()) %>%
  unnest_tokens(word, text)

tidy_book %>%
  count(word, sort = TRUE)

get_stopwords()
get_stopwords(source = "smart")

tidy_books <- books %>%
  unnest_tokens(word, text) %>%
  group_by(word) %>%
  filter(n() > 10) %>%
  ungroup()

tidy_books %>%
  count(title, word, sort = TRUE) %>%
  anti_join(get_stopwords()) %>%
  group_by(title) %>%
  top_n(10) %>%
  ungroup() %>%
  ggplot(aes(reorder_within(word, n, title), n,
             fill = title
  )) +
  geom_col(alpha = 0.8, show.legend = FALSE) +
  scale_x_reordered() +
  coord_flip() +
  facet_wrap(~title, scales = "free") +
  scale_y_continuous(expand = c(0, 0)) +
  labs(
    x = NULL, y = "Word count",
    title = "Most frequent words after removing stop words",
    subtitle = "Comparing Clarissa and Pride and Prejudice"
  )

books_split <- books %>%
  select(document) %>%
  initial_split()
train_data <- training(books_split)
test_data <- testing(books_split) 

sparse_words <- tidy_books %>%
  count(document, word) %>%
  inner_join(train_data) %>%
  cast_sparse(document, word, n)

class(sparse_words)

dim(sparse_words)

word_rownames <- as.integer(rownames(sparse_words))

books_joined <- data_frame(document = word_rownames) %>%
  left_join(books %>%
              select(document, title))

library(glmnet)
library(doParallel)
registerDoParallel(cores = 3)

is_clar <- books_joined$title == "Clarissa"
model <- cv.glmnet(sparse_words, is_clar,
                   family = "binomial",
                   parallel = TRUE, keep = TRUE
)

coefs <- model$glmnet.fit %>%
  tidy() %>%
  filter(lambda == model$lambda.1se)

coefs %>%
  group_by(estimate > 0) %>%
  top_n(5, abs(estimate)) %>%
  ungroup() %>%
  ggplot(aes(fct_reorder(term, estimate), estimate, fill = estimate > 0)) +
  geom_col(alpha = 0.8, show.legend = FALSE) +
  coord_flip() +
  labs(
    x = NULL,
    title = "Coefficients that increase/decrease probability the most",
    subtitle = "Comparing Clarissa and Pride and Prejudice"
  )

intercept <- coefs %>%
  filter(term == "(Intercept)") %>%
  pull(estimate)

classifications <- tidy_books %>%
  inner_join(test_data) %>%
  inner_join(coefs, by = c("word" = "term")) %>%
  group_by(document) %>%
  summarize(score = sum(estimate)) %>%
  mutate(probability = plogis(intercept + score))

library(yardstick)

comment_classes <- classifications %>%
  left_join(books %>%
              select(title, document), by = "document") %>%
  mutate(title = as.factor(title))

comment_classes %>%
  roc_curve(title, probability) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_line(
    color = "midnightblue",
    size = 1.5
  ) +
  geom_abline(
    lty = 2, alpha = 0.5,
    color = "gray50",
    size = 1.2
  ) +
  labs(
    title = "ROC curve for text classification using regularized regression",
    subtitle = "Predicting whether text was written by William Shakespeare or Jane Austen"
  )

comment_classes %>%
  roc_auc(title, probability)

comment_classes %>%
  mutate(
    prediction = case_when(
      probability > 0.5 ~ "Clarissa",
      TRUE ~ "Pride and Prejudice"
    ),
    prediction = as.factor(prediction)
  ) %>%
  conf_mat(title, prediction)

comment_classes %>%
  filter(
    probability > .8, 
    title == "Clarissa"
  ) %>%
  sample_n(10) %>% 
  inner_join(books %>%
               select(document, text)) %>%
  select(probability, text) 

comment_classes %>%
  filter(
    probability < .3,
    title == "Pride and Prejudice"
  ) %>%
  sample_n(10) %>%
  inner_join(books %>%
               select(document, text)) %>%
  select(probability, text) 

comment_classes %>%
  filter(
    probability < .5,
    title == "Clarissa"
  ) %>%
  sample_n(10) %>%
  inner_join(books %>%
               select(document, text)) %>%
  select(probability, text) 

comment_classes %>%
  filter(
    probability > .8, 
    title == "Pride and Prejudice"
  ) %>%
  sample_n(10) %>% 
  inner_join(books %>%
               select(document, text)) %>%
  select(probability, text) 

comment_classes %>%
  filter(
    probability < .3,
    title == "Clarissa"
  ) %>%
  sample_n(10) %>%
  inner_join(books %>%
               select(document, text)) %>%
  select(probability, text) 
