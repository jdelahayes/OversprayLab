"""Exports TEED (github.com/xavysp/TEED, MIT) to the ONNX model used by overspraylab/teed.py.

Only needed to regenerate overspraylab/edge_model/teed_biped7.onnx; OversprayLab itself does not need
PyTorch. In a separate environment:

    python -m venv teed_venv
    teed_venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    teed_venv/Scripts/python -m pip install onnx
    git clone https://github.com/xavysp/TEED.git
    teed_venv/Scripts/python tools/export_teed_onnx.py TEED overspraylab/edge_model/teed_biped7.onnx

Exported from TEED commit 40fa4b1 (checkpoints/BIPED/7/7_model.pth): opset 17, input "input"
(1x3xHxW, BGR minus the BIPED mean, H and W multiples of 8), output "edges" = fused logits.
"""
import os
import sys
import types

repo, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
stub = types.ModuleType("utils.img_processing")       # avoids kornia / scikit-image / scikit-learn
stub.count_parameters = lambda m=None: 0
sys.modules["utils.img_processing"] = stub

import torch                                            # noqa: E402
from ted import TED                                     # noqa: E402


class Fused(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)[-1]                            # fused edge map, logits


model = TED()
state = torch.load(os.path.join(repo, "checkpoints", "BIPED", "7", "7_model.pth"), map_location="cpu",
                   weights_only=True)
print(model.load_state_dict(state))
fused = Fused(model.eval()).eval()
torch.onnx.export(fused, torch.randn(1, 3, 352, 352), out, input_names=["input"], output_names=["edges"],
                  opset_version=17, dynamic_axes={"input": {2: "h", 3: "w"}, "edges": {2: "h", 3: "w"}},
                  dynamo=False)
print("exported", out, os.path.getsize(out), "bytes")
