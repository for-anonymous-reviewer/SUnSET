# This code is from news-tls github repo.
# https://github.com/complementizer/news-tls


def get_scores(metric_desc, pred_tl, groundtruth, evaluator):

    if metric_desc == "concat":
        return evaluator.evaluate_concat(pred_tl, groundtruth)
    elif metric_desc == "agreement":
        return evaluator.evaluate_agreement(pred_tl, groundtruth)
    elif metric_desc == "align_date_costs":
        return evaluator.evaluate_align_date_costs(pred_tl, groundtruth)
    elif metric_desc == "align_date_content_costs":
        return evaluator.evaluate_align_date_content_costs(
            pred_tl, groundtruth)
    elif metric_desc == "align_date_content_costs_many_to_one":
        return evaluator.evaluate_align_date_content_costs_many_to_one(
            pred_tl, groundtruth)


def zero_scores():
    return {'f_score': 0., 'precision': 0., 'recall': 0.}


def sort_dates(dates):
    date = sorted(list(dates))
    new_date = [i.strftime("%d-%b-%Y") for i in date]
    return new_date

def evaluate_dates(pred, ground_truth):
    pred_dates = pred.get_dates()
    ref_dates = ground_truth.get_dates()
    shared = pred_dates.intersection(ref_dates)
    n_shared = len(shared)
    n_pred = len(pred_dates)
    n_ref = len(ref_dates)
    prec = n_shared / n_pred
    rec = n_shared / n_ref
    if prec + rec == 0:
        f_score = 0
    else:
        f_score = 2 * prec * rec / (prec + rec)

    pred_dates = set(pred_dates)
    ref_dates = set(ref_dates)
    unmatch = (pred_dates | ref_dates) - shared

    date_info = {
        "topic": "",
        "topic_tls_no": 0,
        "target_date": "\n".join(sort_dates(ref_dates)),
        "target_date_count": len(ref_dates),
        "predict_date": "\n".join(sort_dates(pred_dates)),
        "predict_date_count": len(pred_dates),
        "match_date": "\n".join(sort_dates(shared)),
        "match_date_count": len(shared),
        "unmatch_date": "\n".join(sort_dates(unmatch)),
        "unmatch_date_count": len(unmatch),
        "precision": prec,
        "recall": rec,
        "f1": f_score
    }

    return {
        'precision': prec,
        'recall': rec,
        'f_score': f_score,
    }, date_info


def get_average_results(tmp_results):
    rouge_1 = zero_scores()
    rouge_2 = zero_scores()
    date_prf = zero_scores()
    for rouge_res, date_res, _ in tmp_results:
        metrics = [m for m in date_res.keys() if m != 'f_score']
        for m in metrics:
            rouge_1[m] += rouge_res['rouge_1'][m]
            rouge_2[m] += rouge_res['rouge_2'][m]
            date_prf[m] += date_res[m]
    n = len(tmp_results)
    for result in [rouge_1, rouge_2, date_prf]:
        for k in ['precision', 'recall']:
            result[k] /= n
        prec = result['precision']
        rec = result['recall']
        if prec + rec == 0:
            result['f_score'] = 0.
        else:
            result['f_score'] = (2 * prec * rec) / (prec + rec)
    return rouge_1, rouge_2, date_prf