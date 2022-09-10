suppressPackageStartupMessages(source("base.r"))
library(optigrab)


SIZE = 2.7
savefig = function(name, width, height) {
    fig(name, width*SIZE, height*SIZE, path=OUT, make_pdf=TRUE)
}

load_human = function(exp, file) {
    read_csv(glue('../data/processed/{exp}/{file}.csv')) %>% 
        mutate(name = 'Human')
}

load_model = function(run, exp, file, name, n=1) {
    glue('../model/results/{run}/{exp}/simulations/{name}_{file}/{n}.csv') %>% 
        read_csv(col_types = cols()) %>% 
        mutate(name=name, wid = glue('{name}-{n}'))
}

load_model_human = function(run, exp, file, models, n=1) {
    print(models)
    bind_rows(
        load_human(exp, file),
        map(models, ~ load_model(run, exp, file, .))
    ) %>% 
    mutate(name = recode_factor(name, 
        "optimal" = "Optimal Metamemory", 
        "fixed_optimal" = "Optimal Metamemory",
        "flexible" = "No Meta-Level Control",
        "empirical_old" = "Empirical No Control",
        "fixed_empirical_old" = "Empirical No Control",
        "human" = "Human"
    ), ordered=T)
}
