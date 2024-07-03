from paibox.collector import Collector


def test_Collector_operations():
    # Just check the typing hint
    d = {"1": 1, "2": "Tom", "3": False, "4": [1, 2, 3]}

    c1 = Collector(d).subset(str)  # Collector[Any, str]
    c2 = Collector(d).exclude(bool)  # Collector[Any, Any]
