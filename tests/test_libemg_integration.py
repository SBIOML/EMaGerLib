"""Smoke test for the libemg / libemg-fork integration surface.

Validates that the libemg symbols and function signatures that emagerlib
depends on are present and shaped as expected. Useful when switching
between fork branches (e.g. emager-v3 vs emager-v3-update) to detect
breaking changes without needing hardware.

Run with: emager-run-tests   (or: python -m unittest tests.test_libemg_integration)
"""
import inspect
import unittest


class TestLibemgImports(unittest.TestCase):
    """Modules and symbols emagerlib imports from libemg must exist."""

    def test_streamers_emagerv3_streamer_importable(self):
        from libemg.streamers import emagerv3_streamer  # noqa: F401

    def test_data_handler_OnlineDataHandler_importable(self):
        from libemg.data_handler import OnlineDataHandler  # noqa: F401

    def test_gui_GUI_importable(self):
        from libemg.gui import GUI  # noqa: F401


class TestLibemgSignatures(unittest.TestCase):
    """Function/class signatures must match how emagerlib calls them."""

    def test_emagerv3_streamer_accepts_baud_rate(self):
        from libemg.streamers import emagerv3_streamer
        sig = inspect.signature(emagerv3_streamer)
        # Accept either an explicit baud_rate parameter, or **kwargs
        # (the current fork passes baud_rate through emager_kwargs).
        has_explicit = "baud_rate" in sig.parameters
        has_var_keyword = any(
            p.kind is inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        self.assertTrue(
            has_explicit or has_var_keyword,
            f"emagerv3_streamer must accept baud_rate (explicit or via "
            f"**kwargs). Got: {sig}",
        )

    def test_emagerv3_streamer_callable_with_no_args(self):
        from libemg.streamers import emagerv3_streamer
        sig = inspect.signature(emagerv3_streamer)
        # All parameters must have defaults (so emagerv3_streamer() works,
        # as called in emagerlib/utils/streamer_utils.py).
        required = [
            name for name, p in sig.parameters.items()
            if p.default is inspect.Parameter.empty
            and p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                               inspect.Parameter.VAR_KEYWORD)
        ]
        self.assertEqual(
            required, [],
            f"emagerv3_streamer() must be callable with no args. "
            f"Required params: {required}",
        )

    def test_OnlineDataHandler_accepts_shared_memory_items(self):
        from libemg.data_handler import OnlineDataHandler
        sig = inspect.signature(OnlineDataHandler)
        self.assertIn(
            "shared_memory_items", sig.parameters,
            f"OnlineDataHandler must accept shared_memory_items. Got: {sig}",
        )

    def test_GUI_accepts_expected_kwargs(self):
        from libemg.gui import GUI
        sig = inspect.signature(GUI)
        for kw in ("args", "debug", "width", "height"):
            self.assertIn(
                kw, sig.parameters,
                f"GUI must accept {kw} kwarg. Got: {sig}",
            )


class TestEmagerlibCanUseLibemg(unittest.TestCase):
    """Verify our own modules that depend on libemg actually import."""

    def test_streamer_utils_imports(self):
        from emagerlib.utils import streamer_utils  # noqa: F401

    def test_gesture_decoder_imports(self):
        from emagerlib.control import gesture_decoder  # noqa: F401


if __name__ == "__main__":
    unittest.main()
