import json


class TestCoreConfigDict:
    def test_CoreConfigDict_instance(self, ensure_dump_dir, MockCoreConfigDict):
        c = MockCoreConfigDict.export()

        with open(ensure_dump_dir / "core_config.json", "w") as f:
            json.dump(c, f, indent=4, ensure_ascii=True)


class TestNeuronConfig:
    def test_NeuronConfig_instance(self, ensure_dump_dir, MockNeuronConfig):
        with open(ensure_dump_dir / "neu_config.json", "w") as f:
            json.dump(MockNeuronConfig.__json__(), f, indent=4, ensure_ascii=True)


class TestCorePlacementConfig:
    def test_CorePlacementConfig_instance(
        self, ensure_dump_dir, MockCorePlacementConfig
    ):
        with open(ensure_dump_dir / "core_placement.json", "w") as f:
            json.dump(
                MockCorePlacementConfig.__json__(), f, indent=4, ensure_ascii=True
            )

    def test_EmptyCorePlacementConfig_instance(
        self, ensure_dump_dir, MockEmptyCorePlacementConfig
    ):
        with open(ensure_dump_dir / "empty_core_placement.json", "w") as f:
            json.dump(
                MockEmptyCorePlacementConfig.__json__(), f, indent=4, ensure_ascii=True
            )
