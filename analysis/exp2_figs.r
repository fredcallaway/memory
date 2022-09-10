suppressPackageStartupMessages(source("setup.r"))
SIZE = 2.7
MAKE_PDF = TRUE
STEP_SIZE = .1

RUN = opt_get("run", default="sep7")
OUT = opt_get("out", default="exp2")
MODELS = opt_get("models", default="fixed_optimal,flexible") %>% 
    strsplit(",") %>% 
    unlist

savefig = function(name, width, height) {
    fig(glue("{OUT}/{name}"), width*SIZE, height*SIZE, pdf=MAKE_PDF)
}
system(glue('mkdir -p figs/{OUT}'))


# %% ==================== load data ====================
WIDTH = 1.5 + length(MODELS)

pretest = read_csv('../data/processed/exp2/pretest.csv', col_types = cols())
df = load_model_human(RUN, "exp2", "trials", MODELS) %>% 
    filter(response_type == "correct") %>% 
    mutate(rel_pretest_accuracy = pretest_accuracy_first - pretest_accuracy_second)

fixations = load_model_human(RUN, "exp2", "fixations", MODELS) %>%
    filter(response_type == "correct") %>% 
    mutate(
        duration = duration / 1000,
        last_fix = as.numeric(presentation == n_pres),
        fix_first = presentation %% 2,
        fix_stronger = case_when(
            pretest_accuracy_first == pretest_accuracy_second ~ NaN,
            pretest_accuracy_first > pretest_accuracy_second ~ 1*fix_first,
            pretest_accuracy_first < pretest_accuracy_second ~ 1*!fix_first
        ),
        fixated = case_when(
            mod(presentation, 2) == 1 ~ pretest_accuracy_first,
            mod(presentation, 2) == 0 ~ pretest_accuracy_second,
        ),
        nonfixated = case_when(
            mod(presentation, 2) == 1 ~ pretest_accuracy_second,
            mod(presentation, 2) == 0 ~ pretest_accuracy_first,
        ),
        relative = fixated - nonfixated
    )

our_check = df %>% filter(name == "Human") %>% with(sum(rt)) %>% floor %>% as.integer
model_check = glue("../model/results/{RUN}_exp2/checksum") %>% read_file %>% as.integer
stopifnot(our_check == model_check)

# %% ==================== nonfinal fixation durations ====================

nonfinal = fixations %>% 
    filter(last_fix == 0) %>% 
    mutate(type="Non-Final")

short_names = .  %>% mutate(name = recode_factor(name, 
    "Optimal Metamemory" = "Optimal", 
    "No Meta-Level Control" = "No Control",
    "Empirical No Control" = "Emp No Control",
    "Human" = "Human"
), ordered=T)

plt_fixated = nonfinal %>% 
    short_names %>% 
    # group_by(name, wid) %>% mutate(duration = scale(duration, center=T, scale=F)) %>% 
    mutate(duration=duration)  %>% 
    plot_effect(fixated, duration, type, median) +
    labs(x="Pretest Accuracy of Fixated Cue", y="Fixation Duration (s)", colour="Fixation Type")

plt_nonfixated = nonfinal %>% 
    short_names %>% 
    filter(presentation > 1) %>% 
    # group_by(name, wid) %>% mutate(duration = scale(duration, center=T, scale=F)) %>%
    mutate(duration=duration)  %>% 
    plot_effect(nonfixated, duration, type, median) +
    labs(x="Pretest Accuracy of Non-Fixated Cue", y="Fixation Duration (s)", colour="Fixation Type") +
    theme(
        axis.title.y=element_blank(),
        axis.text.y=element_text(color="white"),
        axis.ticks.y=element_blank(),
     ) 

final_colors =  scale_colour_manual(values=c(
    "Final"="#3B77B3", 
    "Non-Final"="#EAC200"
), aesthetics=c("fill", "colour"), name="Fixation Type")

nonfinal = (plt_fixated + plt_nonfixated) + 
    plot_layout(ncol=2) +
    plot_annotation(tag_levels = 'A') & 
    theme(plot.tag.position = c(0, 1), legend.position="none") &
    coord_cartesian(xlim=c(NULL), ylim=join_limits(plt_fixated, plt_nonfixated)) &
    scale_x_continuous(n.breaks=3) &
    final_colors

savefig("nonfinal", WIDTH, 1)

# plt_last_duration + scale_colour_manual(values=c(
#         "Final"="#83C57A", 
#         "Non-Final"="#AF7BDC"
#     ), aesthetics=c("fill", "colour"), name="Fixation Type")

# savefig("final_histograms", 3.5, 1)

# %% ==================== commitment ====================

cutoff = fixations %>% 
    filter(name == "Human" & last_fix == 1) %>% 
    with(quantile(duration, .95, na.rm=T))

final_nonfinal = fixations %>% 
    filter(duration <= cutoff) %>%
    mutate(type=if_else(last_fix==1, "Final", "Non-Final")) %>% 
    ggplot(aes(duration, fill=type, y = ..width..*..density..)) +
    geom_histogram(position="identity", breaks=seq(0, cutoff, length.out=30), alpha=0.5) +
    facet_grid(~name) +
    # theme(legend.position="None") +
    scale_colour_manual(values=c(
        "Final"="#3B77B3", 
        "Non-Final"="#EAC200"
        # "Final"="#87DE7A", 
        # "Non-Final"="#AF7BDC"
    ), aesthetics=c("fill", "colour"), name="Fixation Type") +
    labs(x="Fixation Duration (s)", y="Proportion")
    # scale_x_continuous(breaks=seq(-1,5,1))

# savefig("final_nonfinal", WIDTH, 1)
cuts = seq(0, cutoff, length=7)
n_bin = length(cuts)-1
labs = cuts[-1] - (diff(cuts)[1] / 2)

X = fixations %>% 
    filter(presentation > 1) %>% 
    mutate(
        fixlen = factor(cut(duration, cuts, labels=F), levels=seq(n_bin), labels=labs),
        pretest= factor(fixated - nonfixated, levels=seq(1, -1, -0.5))
    ) %>% 
    count(name, wid, fixlen, pretest, .drop=FALSE) %>% 
    group_by(name, wid, fixlen) %>% 
    mutate(prop = n / sum(n)) %>% 
    filter(as.numeric(as.character(pretest)) < 0)  %>% 
    mutate(fixlen = as.numeric(as.character(fixlen))) %>% 
    drop_na()

ebardat = X %>% 
    group_by(name, wid, fixlen) %>% 
    summarise(prop=sum(prop))

pretest_data = df %>% 
    transmute(name, pretest=pretest_accuracy_first-pretest_accuracy_second) %>% 
    count(name,pretest) %>% 
    group_by(name) %>% 
    mutate(pretest_prop = cumsum(n) / sum(n)) %>% 
    select(-n) %>% drop_na() %>% 
    filter(pretest < 0)

problong = X %>% 
    ggplot(aes(x=fixlen, group=pretest, fill=pretest, y = prop)) +
    # geom_hline(aes(yintercept=pretest_prop, color=factor(pretest)), data=pretest_data) +
    stat_summary(fun=mean, geom="bar", position="stack") +
    stat_summary(aes(group=NULL, fill=NULL), data=ebardat, fun.data=mean_cl_boot, geom="errorbar", width = 0.15, size=.3) +
    scale_colour_manual(
        values=mute(c(
            `-0.5`="#fdae61",
            `-1`="#d7191c"
        )),
        aesthetics=c("colour", "fill"),
        # guide = guide_legend(reverse = TRUE), 
        name="Relative\nPretest\nAccuracy"
    ) +
    labs(x="Fixation Duration", fill="Pretest Accuracy", y="Proportion") +
    facet_grid(~name)

(final_nonfinal / problong) + plot_annotation(tag_levels = 'A')
savefig("commitment", WIDTH, 2)

# # %% ==================== overall proportion and timecourse ====================

timecourse = fixations %>% 
    # filter(n_pres >= 2) %>%
    # filter(n_pres <= 3) %>% 
    transmute(trial_id, name, wid, fix_first, duration,
              rel_pretest_accuracy = pretest_accuracy_first - pretest_accuracy_second) %>%
    group_by(trial_id) %>%
    mutate(n_step = diff(c(0, round(cumsum(duration/STEP_SIZE))))) %>% 
    uncount(n_step) %>% 
    group_by(trial_id) %>% 
    mutate(time = (STEP_SIZE)*row_number())

style = list(
    coord_cartesian(xlim=c(NULL), ylim=c(0, 1)),
    scale_colour_manual(
        values=c(
            "#d7191c",
            "#fdae61",
            "#ADADAD",
            "#abd9e9",
            "#2c7bb6"
        ), 
        guide = guide_legend(reverse = TRUE), 
        aesthetics=c("colour"),
        name="Relative\nPretest\nAccuracy"
    )
)

plt_overall = df %>%
    filter(n_pres >= 2) %>%
    mutate(x = factor(rel_pretest_accuracy), y = total_first / (total_first + total_second)) %>% 
    collapse_participants(median, y, x) %>% 
    ggplot(aes(x, y)) +
    stat_summary(fun=mean, group=0, geom="line", colour="#DADADA") +
    stat_summary(aes(color=x), fun.data=mean_cl_boot) +
    facet_grid(~name) +
    theme(legend.position="none") +
    style +
    labs(x="Relative Pretest Accuracy of First Cue", y="Proportion Fixation\nTime on First Cue")
# savefig("overall", 3.2, 1)

cutoff = df %>% filter(name == "Human") %>% with(quantile(rt, .95, na.rm=T) / 1000)
plt_timecourse = timecourse %>% 
    filter(time < cutoff) %>% 
    plot_effect_continuous(time, fix_first, rel_pretest_accuracy, mean) +
    style +
    labs(x="Time (s)", y="Probability Fixate First Cue")
# savefig("timecourse", 3.5, 1)

(plt_overall / plt_timecourse) +
    plot_layout(guides = "collect") +
    plot_annotation(tag_levels = 'A') & 
    theme(plot.tag.position = c(0, 1))

savefig("overall_timecourse", WIDTH, 2)
