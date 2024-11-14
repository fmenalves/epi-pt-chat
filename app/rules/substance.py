from medspacy.ner import TargetRule

# Define the rules for the diabetes NER component
substance_rules = [
    # Literal matching for common terms
    TargetRule(
        literal="diabetes", category="CONDITION", attributes={"code": "73211009"}
    ),
    TargetRule(
        literal="diabetic", category="CONDITION", attributes={"code": "73211009"}
    ),
    TargetRule(
        literal="type 1 diabetes", category="CONDITION", attributes={"code": "46635009"}
    ),
    TargetRule(
        literal="type 2 diabetes", category="CONDITION", attributes={"code": "44054006"}
    ),
    TargetRule(
        literal="type I diabetes", category="CONDITION", attributes={"code": "46635009"}
    ),
    TargetRule(
        literal="type II diabetes",
        category="CONDITION",
        attributes={"code": "44054006"},
    ),
    TargetRule(literal="DM1", category="CONDITION", attributes={"code": "46635009"}),
    TargetRule(literal="DM2", category="CONDITION", attributes={"code": "44054006"}),
    TargetRule(
        literal="insulin resistance",
        pattern=[
            {"LOWER": "insulin"},
            {"IS_PUNCT": True, "OP": "?"},
            {"LOWER": "resistance"},
        ],
        category="CONDITION",
        attributes={"code": "763325000"},
    ),
    TargetRule(
        literal="hyperglycemia", category="CONDITION", attributes={"code": "80394007"}
    ),
    TargetRule(
        literal="hypoglycemia", category="CONDITION", attributes={"code": "302866003"}
    ),
    TargetRule(
        literal="glucose intolerance",
        pattern=[
            {"LOWER": "glucose"},
            {"IS_PUNCT": True, "OP": "?"},
            {"LOWER": "intolerance"},
        ],
        category="CONDITION",
        attributes={"code": "267426009"},
    ),
    TargetRule(
        literal="gestational diabetes",
        pattern=[{"LOWER": "gestational"}, {"LOWER": "diabetes"}],
        category="CONDITION",
        attributes={"code": "11687002"},
    ),
    TargetRule(
        literal="pre diabetes",
        pattern=[
            {"LOWER": "pre"},
            {"IS_PUNCT": True, "OP": "?"},
            {"LOWER": "diabetes"},
        ],
        category="CONDITION",
        attributes={"code": "714628002"},
    ),
    TargetRule(
        literal="prediabetes",
        pattern=[{"LOWER": "prediabetes"}],
        category="CONDITION",
        attributes={"code": "714628002"},
    ),
    # Medication and treatment-related rules
    TargetRule(
        literal="metformin",
        category="TREATMENT",
        attributes={"code": "6809", "system": "RxNorm"},
    ),
    TargetRule(
        literal="insulin",
        category="TREATMENT",
        attributes={"code": "5856", "system": "RxNorm"},
    ),
    TargetRule(
        literal="insulin therapy",
        category="TREATMENT",
        attributes={"code": "345041000000101"},
    ),
    TargetRule(
        literal="glucose monitoring",
        category="TREATMENT",
        attributes={"code": "698472009"},
    ),
    # Complications associated with diabetes
    TargetRule(
        literal="diabetic neuropathy",
        category="COMPLICATION",
        attributes={"code": "866003"},
    ),
    TargetRule(
        literal="diabetic retinopathy",
        category="COMPLICATION",
        attributes={"code": "155107006"},
    ),
    TargetRule(
        literal="diabetic nephropathy",
        category="COMPLICATION",
        attributes={"code": "127013003"},
    ),
    TargetRule(
        literal="diabetic foot",
        pattern=[
            {"LOWER": "diabetic"},
            {"IS_PUNCT": True, "OP": "?"},
            {"LOWER": "foot"},
        ],
        category="COMPLICATION",
        attributes={"code": "280137006"},
    ),
    TargetRule(
        literal="hyperosmolar hyperglycemic state",
        pattern=[
            {"LOWER": "hyperosmolar"},
            {"IS_PUNCT": True, "OP": "?"},
            {"LOWER": "hyperglycemic"},
            {"IS_PUNCT": True, "OP": "?"},
            {"LOWER": "state"},
        ],
        category="COMPLICATION",
        attributes={"code": "310505005"},
    ),
    TargetRule(
        literal="diabetic ketoacidosis",
        pattern=[
            {"LOWER": "diabetic"},
            {"IS_PUNCT": True, "OP": "?"},
            {"LOWER": "ketoacidosis"},
        ],
        category="COMPLICATION",
        attributes={"code": "420422005"},
    ),
]
