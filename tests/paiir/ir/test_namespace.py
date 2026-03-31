import pytest

from paibox.paiir.ir._namespace import IRNamespace


class Dummy:
    pass


class TestIRNamespace:
    def test_create_name_is_stable_for_same_object(self):
        namespace = IRNamespace()
        obj = Dummy()

        name1 = namespace.create_name(obj)
        name2 = namespace.create_name(obj)

        assert name1 == "Dummy_0"
        assert name2 == name1

    def test_create_name_uses_candidate_and_suffixes_uniquely(self):
        namespace = IRNamespace()

        name1 = namespace.create_name("GeneralAddOp", Dummy())
        name2 = namespace.create_name("GeneralAddOp", Dummy())

        assert name1 == "GeneralAddOp_0"
        assert name2 == "GeneralAddOp_1"

    def test_create_name_normalizes_invalid_candidate(self):
        namespace = IRNamespace()

        name = namespace.create_name("9 bad-name", Dummy())

        assert name == "_9_bad_name_0"

    def test_associate_name_with_obj_rejects_duplicate_name(self):
        namespace = IRNamespace()
        namespace.associate_name_with_obj("Node_7", Dummy())

        with pytest.raises(ValueError, match="already used"):
            namespace.associate_name_with_obj("Node_7", Dummy())

    def test_rename_rejects_conflicting_name(self):
        namespace = IRNamespace()
        a = Dummy()
        b = Dummy()
        namespace.create_name("Node", a)
        namespace.create_name("Node", b)

        with pytest.raises(ValueError, match="already used"):
            namespace.rename(b, "Node_0")

    def test_rename_accepts_new_unique_name(self):
        namespace = IRNamespace()
        obj = Dummy()
        namespace.create_name("Node", obj)

        namespace.rename(obj, "SpecializedAdd_9")

        assert namespace.create_name(obj) == "SpecializedAdd_9"
