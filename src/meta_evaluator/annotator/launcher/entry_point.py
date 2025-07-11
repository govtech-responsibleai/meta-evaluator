"""Script to launch the Streamlit annotation interface."""

import sys
from pathlib import Path
from meta_evaluator.annotator.interface import StreamlitAnnotator
from meta_evaluator.annotator.launcher import StreamlitLauncher

if len(sys.argv) != 2:
    raise ValueError("Usage: python launch_streamlit_app.py <config_file_path>")

tmp_dir = sys.argv[1]

# tempfile.TemporaryDirectory was created in the same directory as annotations_dir. Extract directory from tmp_dir.
annotations_dir = str(Path(tmp_dir).parent)

# Load files for annotations
eval_task, eval_data = StreamlitLauncher.load_files_for_annotations(tmp_dir)

# Launch annotator
annotator = StreamlitAnnotator(
    eval_data=eval_data,
    eval_task=eval_task,
    annotations_dir=annotations_dir,
)
annotator.build_streamlit_app()
