# 3D model credits

| Hat in the app | Model | Author | Licence | Source |
| --- | --- | --- | --- | --- |
| Beanie | Black Knit Beanie | Arkify | CC BY 4.0 | https://sketchfab.com/3d-models/black-knit-beanie-edb7f2c044b64d5da40484495239fee3 |
| Bucket | Bucket Hat | oluwaphemi | CC BY 4.0 | https://sketchfab.com/3d-models/bucket-hat-5206aa93bce448a8b936ce763cfc245d |
| Bowler | Bowler Hat | Joycestick (piperno.b) | CC BY 4.0 | https://sketchfab.com/3d-models/bowler-hat-3c3f5bfb98284924a70cf410ab532bc9 |
| Cap | Baseball Cap | Scott VanArsdale (vanart) | CC BY 4.0 | https://sketchfab.com/3d-models/baseball-cap-1c1d34d73fd94e6b9e8f82b1eb7194a0 |
| Fedora | Classic Fedora Hat | ghostmachinesing | CC BY 4.0 | https://sketchfab.com/3d-models/classic-fedora-hat-f0e1a3f3cbc94b3187b618c325334bad |
| Straw hat | Unisex Summer Straw Hat | arloopa | Sketchfab Standard (not CC: may be used in an app, but the model file may not be shared on its own) | https://sketchfab.com/3d-models/unisex-summer-straw-hat-be7bdce89a9942b6bb1dd6bef9632652 |
| Officer | Military Hat | aSocialCrab | CC BY 4.0 | https://sketchfab.com/3d-models/military-hat-6d1a59873e34488e8ea0bc7732afd7d1 |

CC BY 4.0: https://creativecommons.org/licenses/by/4.0/

Changes made for this app: textures reduced to 1024 px WebP, meshes welded and simplified, the cap's
specular-glossiness material converted to metallic-roughness. The shapes are otherwise unchanged.

## Head shape model

`models/headshape.bin` / `.json` are derived from the identity model of the **ICT Face Model (ICT-FaceKit)**,
https://github.com/ICT-VGL/ICT-FaceKit, aligned to MediaPipe's canonical face (`tools/build_headshape.py`) and
calibrated on real MediaPipe landmarks (`tools/calibrate_headshape.py`). Licence of the source:

    MIT License

    Copyright (c) 2020 USC Institute for Creative Technologies

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

MediaPipe's canonical face model (Apache 2.0) was used only to build it and is not shipped.
