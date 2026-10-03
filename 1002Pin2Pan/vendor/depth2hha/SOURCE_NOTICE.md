# Depth2HHA-python（第三方，MIT）

来源：<https://github.com/charlesCXK/Depth2HHA-python>（提交 `6c7c14bd151d927d9ea15639556a83b3b179b75e`，2026-10-02 下载，2026-10-03 逐文件核对）。CMX README 指定用它从深度图生成 HHA。

保留的文件为 `getHHA.py`、`utils/rgbd_util.py`、`utils/util.py` 和 `LICENSE`。HHA 定义不变；打包与数值类型兼容改动如下：

- `getHHA.py`：`from utils.rgbd_util import *` 改为相对导入；删除 `from utils.getCameraParam import *`（只有 `__main__` 示例用到，因此示例无法直接运行）。
- `utils/rgbd_util.py`：`from utils.util import *` 改为相对导入。
- 新增空的 `__init__.py`。
- `utils/util.py` `getRMatrix`：在旋转轴归一化之后加入 `ax = np.ravel(ax)`。原代码中 `ax` 是 3×1 列向量，numpy ≥1.24 无法用它构造 `s_hat`，会直接报错。展平后数值不变。
- `utils/util.py` `getYDirHelper`：对选中的重力特征向量使用 `np.real_if_close`。部分 NumPy/LAPACK 构建会把实特征向量返回为复数类型，导致后续 `np.degrees` 报错；只接受虚部为数值零的向量，仍含实质虚部时明确报错。
