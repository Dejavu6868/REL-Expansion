# Third-party source notice

REL+ v2.1 的编码顺序与几何 helper 延续自服务器上已审计的公开
`SrtaEstrella/REL-SF4PASS` 副本：

- 权威 ERP REL：`/home/zhuzhaoziao/RELPlus/CMX-REL/third_party/rel_original`
- 公开 perspective helper：`/home/zhuzhaoziao/RELPlus/REL-SF4PASS-reference/utils/rgbd_util.py`
- 该公开文件缺失的兼容 helper：`/data/bxh_copy/Pano_MA_Seg/utils/hha_util.py`

兼容 helper 已与权威 vendored `hha_util.py` 做逐字节对拍。项目没有修改这些源文件，
核心运行时也不导入它们；只有测试在服务器上读取 live source 做回归。

上游副本声明 MIT License。其许可证文本中的 copyright holder 仍为上游占位符
`[XXX]`，本项目不擅自替上游补写身份信息。
