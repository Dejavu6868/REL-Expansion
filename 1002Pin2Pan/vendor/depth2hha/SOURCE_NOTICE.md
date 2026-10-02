# Depth2HHA-python（第三方，MIT）

来源：<https://github.com/charlesCXK/Depth2HHA-python>（master，2026-10-02 下载）。CMX README 指定用它从深度图生成 HHA。

保留的文件为 `getHHA.py`、`utils/rgbd_util.py`、`utils/util.py` 和 `LICENSE`。只做了打包改动，数值语句未改：

- `getHHA.py`：`from utils.rgbd_util import *` 改为相对导入；删除 `from utils.getCameraParam import *`（只有 `__main__` 示例用到，因此示例无法直接运行）。
- `utils/rgbd_util.py`：`from utils.util import *` 改为相对导入。
- 新增空的 `__init__.py`。
- `utils/util.py` `getRMatrix`：在旋转轴归一化之后加入 `ax = np.ravel(ax)`。原代码中 `ax` 是 3×1 列向量，numpy ≥1.24 无法用它构造 `s_hat`，会直接报错。展平后数值不变。
