"""Annotator frontend package.

Marker so ``meta_evaluator.annotator.frontend`` is an importable package whose
directory ships the prebuilt ``dist/`` bundle. Hosts (e.g. meta-evaluator-platform)
locate the bundle via ``importlib.import_module(...).__file__``.
"""
