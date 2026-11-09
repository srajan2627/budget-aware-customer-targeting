# Project presentations

`Progress_Report_8_Presentation.pptx` contains 16 slides with speaker notes on
every slide. Open the Notes pane or Presenter View to read the spoken explanation
and source links. The four tables and six charts remain editable.

The deck covers the complete targeting workflow, original campaign estimates,
Report 8 reliability experiments, segmentation, the current GenAI prototype,
completion by deliverable, and remaining work. It gives no arbitrary overall
completion percentage. It dates the presentation November 9, 2026, while keeping
the November 4 research evidence separate from later presentation work.

The authoring source is `code/reporting/build_report8_slides.mjs`. It requires
the supplied Node/Artifact Tool presentation runtime, not the scientific Python
environment. Set `REPORT8_SLIDES_WORKSPACE` to an absolute private build directory
and run it from the repository root. `REPORT8_SLIDES_REVISION` selects a new
output revision; the finalizer will not overwrite an existing final output.
`RUNTIME_NODE_MODULES` and `PRESENTATIONS_SKILL_DIR` can override the default
bundled-runtime locations. Previews and validation records stay in the private
build directory. The scientific analysis does not depend on this authoring tool.

`CIS631_Project_Progress_Report.pptx` is the retained earlier presentation.
