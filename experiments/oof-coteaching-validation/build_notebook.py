import json
from pathlib import Path
root=Path(__file__).resolve().parent
base=(root.parent/"physical-paired-validation"/"physical_paired_validation.py").read_text(encoding="utf-8")
base=base.replace('if __name__=="__main__": main()',"")
extension=(root/"oof_coteaching_validation.py").read_text(encoding="utf-8")
source=base+"\n\n"+extension
nb={"cells":[{"id":"oof-coteaching-intro","cell_type":"markdown","metadata":{},"source":["# RSNA Knee nested OOF co-teaching validation\n","Private cached-feature experiment. All organizer-labelled studies remain final image-only evaluation.\n"]},{"id":"oof-coteaching-run","cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":source.splitlines(keepends=True)}],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11"}},"nbformat":4,"nbformat_minor":5}
(root/"oof-coteaching-validation.ipynb").write_text(json.dumps(nb),encoding="utf-8")
