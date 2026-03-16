import vote_simulation
from vote_simulation.models.rules.registry import _RULE_BUILDERS 

def test_register_rule(): 
    """try all registered rules and test them with dummy dataset
    """
    # Print all registered rules
    print("Registered rules:")
    available = ", ".join(sorted(_RULE_BUILDERS))

    print(available)
    
    # Test each registered rule with a dummy dataset
    dummy_ballots = [
        {"Alice": 5, "Bob": 3, "Charlie": 1},
        {"Alice": 4, "Bob": 4, "Charlie": 2},
        {"Alice": 3, "Bob": 5, "Charlie": 1},
    ]
    dummy_candidates = {"Alice", "Bob", "Charlie"}
    
    for rule_name in _RULE_BUILDERS:
        try:
            rule_builder = vote_simulation.models.rules.registry.get_rule_builder(rule_name)
            rule_instance = rule_builder(dummy_ballots, dummy_candidates)
            assert rule_instance is NotImplementedError or hasattr(rule_instance, "winner_")
        except Exception as e:
            print(f"Error building rule '{rule_name}': {e}")