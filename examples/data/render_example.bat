:: Copyright (c) Facebook, Inc. and its affiliates.
::
:: This source code is licensed under the MIT license found in the
:: LICENSE file in the root directory of this source tree.

set BLENDER="C:\\Users\\jgu\\Downloads\\bunny01.blend"
set codepath="\\wsl$\\Ubuntu\\home\\jgu\\work\\NSVF"
set OUTPUT="C:\\Users\\jgu\\Downloads\\bunny"

blender --background %BLENDER% --python %codepath%\examples\data\nerf_render_ori.py -- %OUTPUT%

pause