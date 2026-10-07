import importlib.util

from PIL import Image


def load_script(project):
    path = project / "scripts/curated_bc_visual.py"
    spec = importlib.util.spec_from_file_location("curated_bc_visual", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_class_selection_is_balanced_unique_and_reproducible(project, tmp_path):
    script = load_script(project)
    for synset, _ in script.CLASSES:
        folder = tmp_path / synset
        folder.mkdir()
        for index in range(8):
            (folder / f"val_{index:02d}.JPEG").touch()
    first = script.select_examples(tmp_path, 6, 217)
    second = script.select_examples(tmp_path, 6, 217)
    assert [item["sample_id"] for item in first] == [item["sample_id"] for item in second]
    assert len(first) == len({item["sample_id"] for item in first}) == 36
    assert all(sum(item["synset"] == synset for item in first) == 6 for synset, _ in script.CLASSES)


def test_contact_sheet_contains_six_paired_examples(project, tmp_path):
    script = load_script(project)
    synset, label = script.CLASSES[0]
    for arm, color in (("original", "red"), ("B", "green"), ("C", "blue")):
        folder = tmp_path / synset / arm
        folder.mkdir(parents=True)
        for index in range(6):
            Image.new("RGB", (256, 256), color).save(folder / f"{index:02d}.png")
    script.make_sheet(tmp_path, synset, label, 6)
    with Image.open(tmp_path / f"{synset}.png") as sheet:
        assert sheet.size == (832, 1779)
        assert sheet.getpixel((25, 80)) == (255, 0, 0)
        assert sheet.getpixel((293, 80)) == (0, 128, 0)
        assert sheet.getpixel((561, 80)) == (0, 0, 255)
