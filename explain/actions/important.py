"""Important features operation."""
import inflect
import numpy as np
import statsmodels.stats.api as sms

from explain.utils import add_to_dict_lists

from explain.actions.utils import gen_parse_op_text

inflect_engine = inflect.engine()


def gen_feature_name_to_rank_dict(data, explanations):
    """Generates a dictionary that maps feature name -> rank -> ids.

    This dictionary contains a mapping from a feature name to a rank to a list
    of ids that have that feature name at that rank.

    Arguments:
        data: the dataset
        explanations: the explanations for the dataset
    Returns:
        feature_name_to_rank: the dictionary with the mapping described above
    """
    feature_name_to_rank = {}
    for feature_name in data.columns:
        # Dictionary mapping rank (i.e., first most important to a list of ids at that rank)
        # to ids that have that feature at that rank
        rank_to_ids = {}
        for i, current_id in enumerate(explanations):
            list_exp = explanations[current_id].list_exp
            for rank, tup in enumerate(list_exp):
                # tup[0] is the feature name in the explanation
                # also, lime will store the value for categorical features
                # at the end of the explanation, i.e., race=0 or race=1
                # so we need to check startswith
                if tup[0].startswith(feature_name):
                    add_to_dict_lists(rank, current_id, rank_to_ids)
                    # Feature name must appear once per explanation so we can break
                    break
        feature_name_to_rank[feature_name] = rank_to_ids
    return feature_name_to_rank


def compute_rank_stats(data, feature_name_to_rank):
    """Compute stats about the feature rankings."""
    max_ranks = {}
    avg_ranks = {}
    ci_95s = {}
    print(feature_name_to_rank)
    for feature_name in data.columns:
        # Get the ranks of each feature name
        rank_to_ids = feature_name_to_rank[feature_name]

        # If the feature isn't very important and never
        # ends up getting included
        if len(feature_name_to_rank[feature_name]) == 0:
            continue

        max_rank = sorted(rank_to_ids.keys())[0]
        max_ranks[feature_name] = max_rank

        rank_list = []
        for key in rank_to_ids:
            rank_list.extend([key] * len(rank_to_ids[key]))
        rank_list = np.array(rank_list) + 1
        avg_ranking = np.mean(rank_list)

        # in case there is only one instance
        if len(rank_list) == 1:
            ci_95 = None
        else:
            ci_95 = sms.DescrStatsW(rank_list).tconfint_mean()

        avg_ranks[feature_name] = avg_ranking
        ci_95s[feature_name] = ci_95
    return max_ranks, avg_ranks, ci_95s


def compute_quart_description(all_rankings, avg_ranking):
    """Compute the ranking quartiles"""
    quartiles = np.percentile(all_rankings, [25, 50, 75])

    if avg_ranking < quartiles[0]:
        describe_imp = "highly"
    elif avg_ranking < quartiles[1]:
        describe_imp = "fairly"
    elif avg_ranking < quartiles[2]:
        describe_imp = "somewhat"
    else:
        describe_imp = "not very"

    return describe_imp


def individual_feature_importance(avg_ranks,
                                  conversation,
                                  ci_95s,
                                  parsed_feature_name,
                                  max_ranks,
                                  data,
                                  ids,
                                  parse_op,
                                  return_s,
                                  feature_name_to_rank,
                                  max_ids):
    """TODO(dylan): docstring"""

    # Get the ranking for the particular feature name
    avg_ranking = avg_ranks[parsed_feature_name]

    # Format CI's for the feature name
    ci_95 = ci_95s[parsed_feature_name]
    if ci_95 is not None:
        present_ci_low = round(ci_95[0], conversation.rounding_precision)
        present_ci_high = round(ci_95[1], conversation.rounding_precision)
    else:
        present_ci_low, present_ci_high = None, None

    # Format ranking
    presen_avg_ranking = round(avg_ranking, conversation.rounding_precision)

    # Add inflections for the max ranking
    # presen_max_rank = inflect_engine.ordinal(max_ranks[parsed_feature_name] + 1)
    # max_rank = max_ranks[parsed_feature_name]

    # Format the ranking
    if len(ids) > 1:
        return_s += f" the {parsed_feature_name} feature is ranked on average <b>{presen_avg_ranking}</b> "
    else:
        return_s += f" the {parsed_feature_name} feature is ranked {presen_avg_ranking} "

    # Add CI if more than 1 instance
    if ci_95 is not None:
        return_s += f"(95% CI [{present_ci_low}, {present_ci_high}])."

    # compute description of rankings
    if data.shape[1] > 1:
        plural = "s"
    else:
        plural = ""
    return_s += f" Here, rank 1 is the most important feature (out of {data.shape[1]} feature{plural})."

    # Add max ranking
    if len(ids) > 1:
        all_rankings = [avg_ranks[f_name] for f_name in avg_ranks.keys()]
        describe_imp = compute_quart_description(all_rankings, avg_ranking)

        if len(parse_op) == 0:
            return_s += "<br><br>Compared to other instances in the data,"
        else:
            return_s += f" Compared to other instances where {parse_op},"

        return_s += f" {parsed_feature_name} is a <b>{describe_imp} important feature</b>.<br><br>"

    return_s += "\n\n"

    return return_s


def topk_feature_importance(avg_ranks, conversation, parse_op, return_s, topk):

    if topk == len(avg_ranks):
        return_s += (" the importance of the features have the following ranking, where 1 is the "
                     "most important feature:<br><br>")
    else:
        return_s += (f" the <b>top {topk}</b> most important features are as follows, where 1 is the most "
                     "important feature:<br><br>")
    avg_ranks = list((key, avg_ranks[key]) for key in avg_ranks)
    avg_ranks = sorted(avg_ranks, key=lambda x: x[1])
    for i in range(topk):
        return_s += f"<b>{i+1}:</b> {avg_ranks[i][0]}<br>"
    conversation.store_followup_desc("")
    return return_s


def important_operation(conversation, parse_text, i, **kwargs):
    """Important features operation.

    For a given feature, this operation finds explanations where it is important.

    TODO(dylan): resolve complexity of this function
    """
    # The maximum number of ids to show in the initial explanation
    # This should be a hyperparam, but not haven't updated it yet
    MAXIDS = 5

    data = conversation.temp_dataset.contents['X']
    if len(conversation.temp_dataset.contents['X']) == 0:
        # In the case that filtering has removed all the instances
        return 'There are no instances that meet this description!', 0
    ids = list(data.index)

    # The feature, all, or topk that is being evaluated for importance
    parsed_feature_name = parse_text[i+1]

    # Generate the text for the filtering operation
    parse_op = gen_parse_op_text(conversation)

    # Get the explainer
    mega_explainer_exp = conversation.get_var('mega_explainer').contents

    # If there's ids to regenerate from a previous operation, i.e., by changing the feature values
    regen = conversation.temp_dataset.contents['ids_to_regenerate']

    # Get the explanations
    explanations = mega_explainer_exp.get_explanations(ids, data, ids_to_regenerate=regen)

    # Generate feature name to frequency of ids at rank mapping
    feature_name_to_rank = gen_feature_name_to_rank_dict(data, explanations)

    # Compute rank stats for features including max rank of feature, its average rank
    # and the 95% ci's
    max_ranks, avg_ranks, ci_95s = compute_rank_stats(data, feature_name_to_rank)

    # Start formatting response into a string
    if len(parse_op) == 0:
        # In the case that no filtering operations have been applied
        return_s = "For the model's predictions across <b>all</b> the data,"
    else:
        return_s = f"For the model's predictions on instance with <b>{parse_op}</b>,"

    # Cases for showing all the most important features, topk most important or importance of an
    # individual feature
    if parsed_feature_name == "all":
        return_s = topk_feature_importance(avg_ranks,
                                           conversation,
                                           parse_op,
                                           return_s,
                                           topk=len(avg_ranks))
    elif parsed_feature_name == "topk":
        topk = int(parse_text[i+2])
        return_s = topk_feature_importance(avg_ranks, conversation, parse_op, return_s, topk)
    else:
        # Individual feature importance case
        return_s = individual_feature_importance(avg_ranks,
                                                 conversation,
                                                 ci_95s,
                                                 parsed_feature_name,
                                                 max_ranks,
                                                 data,
                                                 ids,
                                                 parse_op,
                                                 return_s,
                                                 feature_name_to_rank,
                                                 MAXIDS)

    return return_s, 1
