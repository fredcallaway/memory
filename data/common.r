# assumes VERSION is already defined
suppressPackageStartupMessages(source('../analysis/base.r'))


load_data = function(type) {
    VERSIONS %>% 
    map(~ read_csv(glue('../data/full/{.x}/{type}.csv'), col_types = cols())) %>% 
    bind_rows
}

preprocess_recall = . %>% mutate(
    response_type = factor(response_type, 
        levels=c("correct", "intrusion", "other", "timeout", "empty"),
    ),
    correct = response_type == "correct",
    # base_rt = rt,
    rt = typing_rt
)


summarise_pretest = .  %>%
    filter(!practice & block == max(block)) %>% 
    # strength is for an analysis approach that we pre-registered but later
    # discovered problems with
    mutate(strength = -if_else(correct, log(rt), log(15000))) %>% 
    group_by(wid, word) %>% 
    summarise(across(c(correct, strength), mean, na.rm=T)) %>%
    rename(pretest_accuracy = correct) %>% 
    group_by(wid) %>%
    mutate(strength = zscore(strength))
