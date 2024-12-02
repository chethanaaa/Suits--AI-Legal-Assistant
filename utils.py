# Function to query rules based on a condition
def query_rules_by_condition(kb, condition):
    results = []
    for rule, details in kb['rules'].items():
        if condition in details['conditions']:
            results.append({
                "rule": rule,
                "action": details['action']
            })
    return results

# Function to get the action for a specific rule
def get_action_by_rule(kb, rule_name):
    rule = kb['rules'].get(rule_name)
    if rule:
        return rule['action']
    else:
        return "Rule not found."

def get_compensation_details(kb, rule_name):
    rule = kb['rules'].get(rule_name)
    if not rule:
        return {"error": "Rule not found."}

    compensation = rule.get("compensation")
    if not compensation:
        return {"error": "Compensation details not defined for this rule."}

    return compensation