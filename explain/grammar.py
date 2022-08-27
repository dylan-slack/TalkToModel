GRAMMAR = r"""
?start: action
action: operation done | operation join action | followup done
operation: explanation | filter | predictions | whatami | lastturnfilter | lastturnop | data | impfeatures | show | whatif | likelihood | modeldescription | function | score | ndatapoints | interact | label | mistakes | fstats | define | labelfilter | predfilter

labelfilter: " labelfilter" class
predfilter: " predictionfilter" class

fstats: fstatsword (allfeaturenames | " target")
fstatsword: " statistic"

define: defineword allfeaturenames
defineword: " define"

ndatapoints: " countdata"

mistakes: mistakesword mistakestypes
mistakesword: " mistake"
mistakestypes: " typical" | " count" | " sample"

correct: correctword correcttypes
correctword: " correct"
correcttypes: " typical" | " count" | " sample"

label: " label"

join: and | or
and: " and"
or: " or"
filterword: " filter"

filter: filterword featuretype
featuretype: {avaliablefeaturetypes}

explanation: explainword explaintype
explainword: " explain"
explaintype: featureimportance | lime | cfe
featureimportance: " features"
lime: " lime"
cfe: " cfe"

predictions: " predict"

whatami: " self"

interact: " interact"

data: " data"
modeldescription: " model"
function: " function"

score: scoreword metricword
scoreword: " score"
metricword: " default" | " accuracy" | " f1" | " roc" | " precision" | " recall" | " sensitivity" | " specificity" | " ppv" | " npv"
testword: " test"

followup: " followup"

whatif: whatifword ( ( numfeaturenames numupdates adhocnumvalues ) | catnames )
whatifword: " change"

show: " show"

likelihood: likelihoodword
likelihoodword: " likelihood"

lastturnfilter: " previousfilter"
lastturnop: " previousoperation"

impfeatures: impfeaturesword (allfeaturenames | allfeaturesword | topk)
allfeaturesword: " all"
topk: topkword ( {topkvalues} )
topkword: " topk"


impfeaturesword: " important"
numupdates: " increase" | " set" | " decrease"

done: " [e]"
"""  # noqa: E501

# append the cat feature name and
# the values in another nonterminal
CAT_FEATURES = r"""
catnames: {catfeaturenames}
"""

TARGET_VAR = r"""
class: {classes}
"""

# numfeaturenames are the numerical feature names
# and numvalues are the potential numeric values
NUM_FEATURES = r"""
numnames: {numfeaturenames}
equality: gt | lt | gte | lte | eq | ne
gt: " greater than"
gte: " greater equal than"
lt: " less than"
lte: " less equal than"
eq: " equal to"
ne: " not equal to"
"""
