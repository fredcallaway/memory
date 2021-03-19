knitr::opts_chunk$set(warning=FALSE, message=FALSE)
library("tidyverse")
library("lme4")
library("jtools")
library(magrittr)
library(purrr)
library(rmdformats) 
library(patchwork)
library(jsonlite)
library(tidyjson)

options(
    "summ-model.info"=FALSE, 
    "summ-model.fit"=FALSE, 
    "summ-re.table"=FALSE, 
    "summ-groups.table"=FALSE,
    "jtools-digits"=3
)

RED =  "#E41A1C" 
BLUE =  "#377EB8" 
GREEN =  "#4DAF4A" 
PURPLE =  "#984EA3" 
ORANGE =  "#FF7F00" 
YELLOW =  "#FFDD47" 
GRAY = "#8E8E8E"
BLACK = "#1B1B1B"
theme_set(theme_classic(base_size = 18))
kable = knitr::kable
glue = glue::glue

response_type_colors = scale_colour_manual(values=c(
    GREEN,
    ORANGE,
    RED,
    GRAY,
    BLACK
), aesthetics=c("fill", "colour"), name="Response Type")

word_type_colors = scale_colour_manual(values=c(
    BLUE,
    YELLOW
), aesthetics=c("fill", "colour"), name="Word Memorability")

knitr::opts_chunk$set(out.width="60%", fig.align="center")

json_to_columns <- function(df, column){
    json_df = df %>% 
        pull({{column}}) %>% 
        spread_all %>%
        as_tibble %>%
        select(-document.id)
    df %>% 
        select(-{{column}}) %>% 
        bind_cols(json_df)
}

zscore = function(x) (x - mean(x, na.rm=T)) / sd(x, na.rm=T)



# %% ==================== this ====================

