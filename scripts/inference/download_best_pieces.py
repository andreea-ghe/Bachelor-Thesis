
import subprocess
import os
import re

BASE_URL = "https://xg1pivgkgne8cd-8000.proxy.runpod.net"
OUTPUT_DIR = "application/web/core/static/core/meshes/multi_pieces"

raw_data = """
1,/workspace/everyday/Vase/23f305e2ca3a3007ee106795de609335/fractured_61,3,1.1435,0.019752,2.23,0.037122
2,/workspace/everyday/Bowl/6a772d12b98ab61dc26651d9d35b77ca/fractured_46,3,2.266,0.010764,3.6867,0.018049
3,/workspace/everyday/Cup/106f8b5d2f2c777535c291801eaf5463/fractured_53,3,4.3794,0.014228,7.141,0.028276
97,/workspace/everyday/Mirror/d8051303d474fff937ac8d648045f9c5/fractured_23,4,121.3075,0.264675,172.9985,0.650986
98,/workspace/everyday/ToyFigure/2acb67458c17af08c4b494041bc63422/fractured_74,4,122.4858,0.119689,174.8353,0.210162
99,/workspace/everyday/ToyFigure/7cc11810eaa479e177e21aa92afd81b0/fractured_14,4,128.342,0.215783,175.1219,0.489388
100,/workspace/everyday/ToyFigure/7134e227c856685d50f3a2fc91e9b3c5/fractured_46,4,132.3074,0.163908,178.5406,0.252515
90,/workspace/everyday/ToyFigure/7f691e0ea1e529a83a26deafa9169642/fractured_23,4,116.7562,0.466393,171.5426,0.720022
91,/workspace/everyday/Mirror/51feed1c1b21444828b4c5ec471dc963/fractured_2,4,116.8938,0.340419,177.7727,0.790134
92,/workspace/everyday/WineBottle/908e85e13c6fbde0a1ca08763d503f0e/fractured_42,4,117.9004,0.344374,165.7074,0.608395
93,/workspace/everyday/WineBottle/908e85e13c6fbde0a1ca08763d503f0e/fractured_75,3,117.9863,0.30428,179.1648,0.672328
94,/workspace/everyday/Mirror/28f1c6c292c0190480360680c1602c7d/fractured_40,3,118.1927,0.275776,177.9597,0.568276
95,/workspace/everyday/Mirror/d8051303d474fff937ac8d648045f9c5/fractured_23,4,121.3075,0.264675,172.9985,0.650986
96,/workspace/everyday/ToyFigure/2acb67458c17af08c4b494041bc63422/fractured_74,4,122.4858,0.119689,174.8353,0.210162
97,/workspace/everyday/WineBottle/109d55a137c042f5760315ac3bf2c13e/fractured_28,4,122.7258,0.097023,173.1748,0.18391
98,/workspace/everyday/ToyFigure/7cc11810eaa479e177e21aa92afd81b0/fractured_14,4,127.8315,0.215438,174.7251,0.48792
99,/workspace/everyday/Plate/bf4e419f863966f1bcbb3468b0bfc2ac/fractured_5,4,131.3119,0.394518,177.9017,0.596231
100,/workspace/everyday/ToyFigure/7134e227c856685d50f3a2fc91e9b3c5/fractured_46,4,132.4968,0.163734,179.3484,0.252518
95,/workspace/everyday/ToyFigure/29a37eb86f4677163f6b1b708c713b4a/fractured_17,4,122.9527,0.336562,171.4868,0.647519
96,/workspace/everyday/WineBottle/109d55a137c042f5760315ac3bf2c13e/fractured_55,4,123.3148,0.205876,175.9704,0.407108
97,/workspace/everyday/ToyFigure/7134e227c856685d50f3a2fc91e9b3c5/fractured_15,4,125.2621,0.208326,177.8731,0.330299
98,/workspace/everyday/Cup/29e8781a8f6fdf1f13323fc4d5700bec/fractured_47,4,129.1546,0.085816,179.8389,0.134719
99,/workspace/everyday/Cookie/e1bc84ee7b9ef7f331999dc8823fe0d/fractured_19,4,130.0801,0.354342,177.9286,0.698303
100,/workspace/everyday/Plate/bf4e419f863966f1bcbb3468b0bfc2ac/fractured_12,4,133.2645,0.342019,179.6809,0.703618
"""
paths = set(re.findall(r"/workspace/[^\s,]+", raw_data))

for p in sorted(paths):
    relative = p.replace("/workspace", "")
    url = f"{BASE_URL}{relative}/"
    parts = relative.split("/")
    idx = parts.index("everyday")
    subpath = "/".join(parts[idx + 1:])
    output_path = os.path.join(OUTPUT_DIR)
    print(f"\nDownloading: {url}")
    cmd = [
        "wget",
        "-r",
        "-np",
        "-nH",
        "--cut-dirs=1",
        "--reject", "index.html*",
        "-A", "*.obj",
        "-P", output_path,
        url
    ]
    subprocess.run(cmd)