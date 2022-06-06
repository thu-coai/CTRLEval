from scipy.stats import pearsonr, spearmanr, kendalltau
from ctrleval import CTRLEval

# evaluation tasks: sentiment-controlled / topic-controlled text generation
task_name = ['senti', 'topic']

for task in task_name:
    with open('./data/data_{}.txt'.format(task), 'r') as f:
        raw_data = [line.strip().split('\t') for line in f.readlines()]
    # generated text
    data = [data_ele[2] for data_ele in raw_data]
    # content prefix
    prefix = [data_ele[1] for data_ele in raw_data]
    # attribute label
    label = [data_ele[0] for data_ele in raw_data]
    # coherence score (human)
    coh_human = [float(data_ele[3]) for data_ele in raw_data]
    # consistency score (human)
    cons_human = [float(data_ele[4]) for data_ele in raw_data]
    # attribute relevance score (human)
    ar_human = [float(data_ele[5]) for data_ele in raw_data]

    # note: in addition to set the model name, you can also download PEGASUS and set the model path.
    # https://huggingface.co/google/pegasus-large
    scorer = CTRLEval(iwf_dir='iwf_full.txt', prompt_dir='./prompt/prompt_{}.txt'.format(task),
                      verbal_dir='./prompt/verbal_{}.txt'.format(task), model_name_or_path='google/pegasus-large')

    # compute the evaluation results for coherence, consistency, and attribute relevance
    coh_result = scorer.score(aspect='coh', data=data, batch_size=8)
    cons_result = scorer.score(aspect='cons', data=data, prefix=prefix, batch_size=8)
    ar_result = scorer.score(aspect='ar', data=data, label=label, batch_size=8)

    print('Result for {} task:'.format(task))
    print('Coherence (r/rho/tau): ', pearsonr(coh_result, coh_human), '\t', spearmanr(coh_result, coh_human),
          '\t', kendalltau(coh_result, coh_human))
    print('Consistency (r/rho/tau): ', pearsonr(cons_result, cons_human), '\t', spearmanr(cons_result, cons_human),
          '\t', kendalltau(cons_result, cons_human))
    print('Attribute Relevance (r/rho/tau): ', pearsonr(ar_result, ar_human), '\t', spearmanr(ar_result, ar_human),
          '\t', kendalltau(ar_result, ar_human))
