from tests.golden_voltage import manifest


def test_undecodable_media_fall_back_to_raw_hashes(monkeypatch, tmp_path):
    png_a = tmp_path / "a.png"
    png_b = tmp_path / "b.png"
    mp4_a = tmp_path / "a.mp4"
    mp4_b = tmp_path / "b.mp4"
    for path, content in (
        (png_a, b"bad png a"),
        (png_b, b"bad png b"),
        (mp4_a, b"bad mp4 a"),
        (mp4_b, b"bad mp4 b"),
    ):
        path.write_bytes(content)

    monkeypatch.setattr(manifest.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(OSError()))

    assert manifest._png_decoded(png_a) is None
    assert manifest._mp4_decoded(mp4_a) is None
    for kind, path_a, path_b in (("png", png_a, png_b), ("mp4", mp4_a, mp4_b)):
        entry_a = {"kind": kind, "sha256": manifest._sha_file(path_a), "decoded": None}
        entry_b = {"kind": kind, "sha256": manifest._sha_file(path_b), "decoded": None}
        assert manifest._digest(entry_a, manifest.POLICY) != manifest._digest(entry_b, manifest.POLICY)
