import subprocess
import os
import re

BASE_URL = "https://v1t7amu4ku8gm0-8000.proxy.runpod.net"
OUTPUT_DIR = "application/web/core/static/core/meshes/multi_pieces"

raw_data = """
1,/workspace/everyday/Vase/23f305e2ca3a3007ee106795de609335/fractured_61,3,1.1435,0.019752,2.23,0.037122
2,/workspace/everyday/Bowl/6a772d12b98ab61dc26651d9d35b77ca/fractured_46,3,2.266,0.010764,3.6867,0.018049
3,/workspace/everyday/Cup/106f8b5d2f2c777535c291801eaf5463/fractured_53,3,4.3794,0.014228,7.141,0.028276
4,/workspace/everyday/Bowl/6a772d12b98ab61dc26651d9d35b77ca/fractured_31,4,14.7894,0.30446,30.0828,0.629439
5,/workspace/everyday/Mirror/2f9722272370e78cbe172933e1937bf6/fractured_48,3,19.5206,0.164959,37.1452,0.408738
6,/workspace/everyday/Bottle/48bacbf60318db537a0c402e1ca31f3/fractured_57,3,21.112,0.656733,57.3173,0.998854
7,/workspace/everyday/Cookie/e1bc84ee7b9ef7f331999dc8823fe0d/fractured_1,3,22.4574,0.231438,48.3667,0.606162
8,/workspace/everyday/Mug/336122c3105440d193e42e2720468bf0/fractured_48,3,28.2282,0.145692,73.3076,0.305536
9,/workspace/everyday/Vase/23f305e2ca3a3007ee106795de609335/fractured_6,4,28.9825,0.068901,109.8826,0.230018
1,/workspace/everyday/Vase/23f305e2ca3a3007ee106795de609335/fractured_61,3,1.1435,0.019752,2.23,0.037122
2,/workspace/everyday/Bowl/6a772d12b98ab61dc26651d9d35b77ca/fractured_46,3,2.1263,0.010208,3.6793,0.016732
3,/workspace/everyday/Cup/106f8b5d2f2c777535c291801eaf5463/fractured_53,3,4.3794,0.014228,7.141,0.028276
4,/workspace/everyday/Bowl/6a772d12b98ab61dc26651d9d35b77ca/fractured_31,4,14.8077,0.304481,30.047,0.629577
5,/workspace/everyday/Mirror/2f9722272370e78cbe172933e1937bf6/fractured_48,3,19.5206,0.164959,37.1452,0.408738
6,/workspace/everyday/Cookie/e1bc84ee7b9ef7f331999dc8823fe0d/fractured_1,3,22.6455,0.231631,48.4806,0.607575
7,/workspace/everyday/Vase/d2f113f579d2c4c9b7e6746928016c6b/fractured_52,4,27.9156,0.15275,74.2094,0.340822
8,/workspace/everyday/Mug/336122c3105440d193e42e2720468bf0/fractured_48,3,28.2282,0.145692,73.3076,0.305536
9,/workspace/everyday/Mug/57f73714cbc425e44ae022a8f6e258a7/fractured_56,4,33.9067,0.210934,95.872,0.554065
10,/workspace/everyday/Vase/23f305e2ca3a3007ee106795de609335/fractured_6,4,35.1483,0.054593,130.2838,0.16519
1,/workspace/everyday/Vase/d2f113f579d2c4c9b7e6746928016c6b/fractured_46,4,13.5477,0.040748,19.6891,0.073647
2,/workspace/everyday/Bottle/d74bc917899133e080c257afea181fa2/fractured_28,4,16.3126,0.022053,49.209,0.04918
3,/workspace/everyday/Vase/1b370d5326cb7da75318625c74026d6/fractured_42,4,29.3511,0.237344,63.5107,0.348875
4,/workspace/everyday/Mirror/b5787ebfe1c5b31f68661782af60b711/fractured_19,4,32.5258,0.145439,90.1799,0.288114
5,/workspace/everyday/Plate/bf4e419f863966f1bcbb3468b0bfc2ac/fractured_48,4,33.3443,0.119681,106.4401,0.242904
6,/workspace/everyday/Mirror/b5787ebfe1c5b31f68661782af60b711/fractured_38,4,36.2026,0.137767,92.2875,0.331894
7,/workspace/everyday/Cup/3a3b19614d545834cd321262c762dc9/fractured_41,4,38.2355,0.101227,100.6927,0.19986
"""
paths = set(re.findall(r"/workspace/everyday/[^\s,]+", raw_data))

for p in sorted(paths):
    relative = p.replace("/workspace/everyday/", "")
    url = f"{BASE_URL}/everyday/{relative}/"
    local_dir = os.path.join(OUTPUT_DIR, relative)
    os.makedirs(local_dir, exist_ok=True)

    print(f"\nDownloading: {url}")
    print(f"  -> {local_dir}")

    cmd = [
        "wget",
        "-r", "-np", "-nd",
        "--reject", "index.html*",
        "-A", "*.obj",
        "-P", local_dir,
        url,
    ]
    subprocess.run(cmd)
